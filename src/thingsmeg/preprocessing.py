"""MEG preprocessing (Weeks 1-2).

Pipeline: load CTF .ds -> pick MEG channels -> filter -> fit ICA -> drop EOG/ECG
components -> read events from BIDS sidecar -> epoch -> baseline -> resample.

Facts from inspect_dataset.py (BIGMEG1, ses-01, run-01):
  - CTF system, .ds directories
  - 310 channels: 272 mag (MEG gradiometers), 28 ref_meg, 10 misc
  - sfreq: 1200 Hz raw -> resampled to 200 Hz post-epoching
  - Events in *_events.tsv sidecar (no STI channel): columns onset, duration,
    trial_type, things_category_nr, value, sample
  - trial_type == 'exp' are the experimental trials; 'catch'/'test' are checks
  - ~226 trials/run, ~10 runs/session, 12 sessions -> ~27k trials/subject
  - things_category_nr is the stimulus label (int) used for decoding and RDMs
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


def load_raw(meg_path: str | Path) -> mne.io.BaseRaw:
    """Read one CTF .ds recording and pick only the 272 MEG gradiometers.

    Reference channels (ref_meg) are kept during ICA and dropped afterwards.
    Misc channels (trigger/EOG proxies) are dropped at load time.
    """
    raw = mne.io.read_raw_ctf(str(meg_path), preload=False, verbose="WARNING")
    # Keep MEG + ref_meg for now; drop misc
    raw.pick(picks=["mag", "ref_meg"])
    return raw


def filter_raw(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Band-pass filter and optional notch per config."""
    raw.load_data()
    raw.filter(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        method="fir",
        fir_window="hamming",
        verbose="WARNING",
    )
    if cfg.notch is not None:
        raw.notch_filter(freqs=cfg.notch, verbose="WARNING")
    return raw


def fit_ica(raw: mne.io.BaseRaw, cfg: Config) -> mne.preprocessing.ICA:
    """Fit ICA on the filtered raw and auto-label EOG/ECG components.

    IMPORTANT: inspect components manually for at least the first subject.
    Automatic detection (find_bads_eog / find_bads_ecg) is a starting point, not
    ground truth. The CTF system has no dedicated EOG/ECG channels — MNE will
    attempt to infer from signal shape; review the topographies.
    """
    ica = mne.preprocessing.ICA(
        n_components=cfg.ica_n_components,
        method=cfg.ica_method,
        random_state=42,
        verbose="WARNING",
    )
    # ICA is fit on MEG channels only (not ref channels)
    ica.fit(raw, picks="mag", verbose="WARNING")
    return ica


def apply_ica(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    subject: str = "unknown",
) -> tuple[mne.io.BaseRaw, list[int]]:
    """Exclude artifact ICA components; return cleaned raw + excluded indices.

    Manually verified exclusions (inspected 2026-07-08/09):

    BIGMEG1 (ses-01 run-01, inspected 2026-07-03):
      ICA026 — eye movement / left temporal muscle
      ICA028 — heartbeat (diffuse bilateral, sub-2 Hz dominant)
      ICA030 — heartbeat clearest (~1 Hz rhythm, harmonic spectrum)
      ICA043 — heartbeat second instance

    BIGMEG2 (FastICA 40-comp, 4000-epoch subsample, inspected 2026-07-09):
      ICA003 — eye blink / vertical EOG (frontal-superior gradient)
      ICA004 — eye (same frontal pattern as ICA003)
      ICA009 — cardiac (bilateral symmetric temporal, classic MEG heartbeat)

    BIGMEG3 (infomax 60-comp, full data, inspected 2026-07-09):
      ICA000 — cardiac (regular ~1 Hz spikes, right-temporal dipole)
      ICA002 — slow drift / movement (sub-Hz sweep, strongly sub-Hz spectrum)
      ICA005 — probable cardiac 2nd (~2 Hz rhythmic, focal left-temporal)

    BIGMEG4 (infomax 31-comp, full data, inspected 2026-07-09):
      ICA000 — cardiac (sparse ~1 Hz spikes, anterolateral topography)
      ICA001 — probable cardiac 2nd (same ~1 Hz rhythm, same topographic family)
    """
    # Per-subject manually verified exclusions
    manual: dict[str, list[int]] = {
        "BIGMEG1": [26, 28, 30, 43],
        "BIGMEG2": [3, 4, 9],
        "BIGMEG3": [0, 2, 5],
        "BIGMEG4": [0, 1],
    }

    if subject in manual:
        excluded = manual[subject]
        log.info("ICA: using manually verified exclusions for %s: %s", subject, excluded)
    else:
        # Fallback: auto-detection (CTF has no dedicated EOG/ECG channel — heuristic only)
        excluded = []
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw, method="correlation", verbose="WARNING")
            excluded.extend(ecg_idx)
        except Exception:  # noqa: BLE001
            log.warning("ECG component detection failed — skipping")
        try:
            eog_idx, _ = ica.find_bads_eog(raw, verbose="WARNING")
            excluded.extend(eog_idx)
        except Exception:  # noqa: BLE001
            log.warning("EOG component detection failed — skipping")
        excluded = list(set(excluded))
        log.warning("ICA: auto-detected exclusions for %s (unverified): %s", subject, excluded)

    ica.exclude = excluded
    raw_clean = ica.apply(raw.copy(), verbose="WARNING")
    raw_clean.pick("mag")
    return raw_clean, excluded


def read_events_tsv(meg_path: Path) -> pd.DataFrame:
    """Read the BIDS *_events.tsv sidecar for this run (experimental trials only).

    Returns a DataFrame with columns: onset_sample (int), things_category_nr (int).
    Only rows with trial_type == 'exp' are kept.
    """
    tsv = meg_path.parent / meg_path.name.replace("_meg.ds", "_events.tsv")
    if not tsv.exists():
        raise FileNotFoundError(f"Events sidecar not found: {tsv}")
    ev = pd.read_csv(tsv, sep="\t")
    ev = ev[ev["trial_type"] == "exp"].copy()
    # 'sample' column is the onset sample at the raw sfreq; use it directly
    ev["onset_sample"] = ev["sample"].astype(int)
    ev["things_category_nr"] = ev["things_category_nr"].astype(int)
    return ev[["onset_sample", "things_category_nr"]].reset_index(drop=True)


def make_epochs(
    raw: mne.io.BaseRaw,
    events_df: pd.DataFrame,
    cfg: Config,
) -> tuple[mne.Epochs, np.ndarray]:
    """Epoch the cleaned raw around stimulus onset; return (epochs, labels).

    labels is a 1-D int array of things_category_nr aligned to epochs rows.
    Epochs are baseline-corrected and resampled to cfg.resample_sfreq.
    """
    # Build MNE events array: (n_events, 3) — [sample, 0, event_id]
    # We use a single event_id=1 for all experimental trials; the label array
    # carries the per-trial identity.
    mne_events = np.column_stack([
        events_df["onset_sample"].values,
        np.zeros(len(events_df), dtype=int),
        np.ones(len(events_df), dtype=int),
    ])
    epochs = mne.Epochs(
        raw,
        events=mne_events,
        event_id={"exp": 1},
        tmin=cfg.tmin,
        tmax=cfg.tmax,
        baseline=tuple(cfg.baseline),
        picks="mag",
        preload=True,
        reject=None,  # no hard threshold; rely on ICA + later visual inspection
        verbose="WARNING",
    )
    epochs.resample(cfg.resample_sfreq, verbose="WARNING")
    labels = events_df["things_category_nr"].values[epochs.selection]
    return epochs, labels


def _group_by_session(meg_files: list[Path]) -> dict[str, list[Path]]:
    """Group a flat list of run paths by session label (e.g. 'ses-01')."""
    sessions: dict[str, list[Path]] = {}
    for p in meg_files:
        # Extract ses-XX from path parts
        ses = next((part for part in p.parts if part.startswith("ses-")), "ses-unknown")
        sessions.setdefault(ses, []).append(p)
    return sessions


def preprocess_subject(subject: str, cfg: Config) -> dict[str, Any]:
    """End-to-end preprocessing for one subject; writes epochs to derivatives/.

    ICA is fitted once per session (on the first run of that session) and
    applied to all runs in that session — ~10× faster than per-run ICA.

    Returns a report dict for logging and the methods section.
    """
    from .io import find_meg_files

    subject_raw_dir = Path(cfg.path_raw) / f"sub-{subject}"
    meg_files = find_meg_files(subject_raw_dir)
    if not meg_files:
        raise FileNotFoundError(f"No MEG files found for {subject} under {subject_raw_dir}")

    sessions = _group_by_session(meg_files)

    all_epochs: list[mne.Epochs] = []
    all_labels: list[np.ndarray] = []
    all_run_indices: list[np.ndarray] = []
    n_excluded_components: list[int] = []
    skipped_runs: list[str] = []

    for ses_label, ses_runs in sorted(sessions.items()):
        # Fit ICA once on the first good run of this session
        ica = None
        excluded: list[int] = []
        for candidate in ses_runs:
            try:
                log.info("Fitting ICA for %s %s on %s", subject, ses_label, candidate.name)
                raw_ica = load_raw(candidate)
                raw_ica = filter_raw(raw_ica, cfg)
                ica = fit_ica(raw_ica, cfg)
                _, excluded = apply_ica(raw_ica, ica, subject=subject)
                ica.exclude = excluded
                break
            except Exception as e:  # noqa: BLE001
                log.warning("ICA fit failed on %s — %s; trying next run", candidate.name, e)

        if ica is None:
            log.warning("Could not fit ICA for %s %s — skipping entire session", subject, ses_label)
            skipped_runs.extend(p.name for p in ses_runs)
            continue

        n_excluded_components.append(len(excluded))

        for run_idx_in_ses, meg_path in enumerate(ses_runs):
            log.info("Processing %s", meg_path.name)
            try:
                raw = load_raw(meg_path)
                raw = filter_raw(raw, cfg)
                raw_clean = ica.apply(raw.copy(), verbose="WARNING")
                raw_clean.pick("mag")
                events_df = read_events_tsv(meg_path)
                epochs, labels = make_epochs(raw_clean, events_df, cfg)
                all_epochs.append(epochs)
                all_labels.append(labels)
                # Track which global run index each trial belongs to
                global_run_idx = len(all_epochs) - 1
                all_run_indices.append(np.full(len(labels), global_run_idx, dtype=int))
            except Exception as e:  # noqa: BLE001
                log.warning("Skipping %s — %s", meg_path.name, e)
                skipped_runs.append(meg_path.name)

    # Align head position transforms to first run before concatenating.
    # CTF THINGS-MEG records separate dev_head_t per session; mismatches block
    # concatenation. We snap all to run-0's transform (standard practice; actual
    # spatial realignment would require Maxwell filtering which CTF data supports
    # only after tsss — not needed for RSA/decoding which work in sensor space).
    ref_dev_head_t = all_epochs[0].info["dev_head_t"]
    for ep in all_epochs[1:]:
        ep.info["dev_head_t"] = ref_dev_head_t

    epochs_cat = mne.concatenate_epochs(all_epochs, verbose="WARNING")
    labels_cat = np.concatenate(all_labels)
    runs_cat = np.concatenate(all_run_indices)

    # Save
    out_dir = Path(cfg.path_derivatives) / f"sub-{subject}"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_cat.save(out_dir / "epochs.fif", overwrite=True, verbose="WARNING")
    np.save(out_dir / "labels.npy", labels_cat)
    np.save(out_dir / "runs.npy", runs_cat)

    report = {
        "subject": subject,
        "n_runs_total": len(meg_files),
        "n_runs_used": len(meg_files) - len(skipped_runs),
        "n_runs_skipped": len(skipped_runs),
        "skipped_runs": skipped_runs,
        "n_trials_total": len(epochs_cat),
        "n_trials_per_label": int(np.unique(labels_cat, return_counts=True)[1].mean()),
        "n_times": len(epochs_cat.times),
        "times_ms": (float(epochs_cat.times[0] * 1000), float(epochs_cat.times[-1] * 1000)),
        "sfreq_final": epochs_cat.info["sfreq"],
        "ica_components_excluded_per_run": n_excluded_components,
    }
    log.info("Preprocessing done: %s", report)
    return report
