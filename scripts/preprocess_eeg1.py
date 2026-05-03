"""Preprocess THINGS-EEG1 for a single subject.

Pipeline (mirrors THINGS-MEG preprocessing as closely as possible):
  1. Load .set file via MNE (EEGLAB format)
  2. Bandpass filter 0.1–40 Hz
  3. Re-reference to average
  4. Epoch: 0 to 500 ms relative to stimulus onset (RSVP; no pre-stim baseline)
     NOTE: RSVP at 10 Hz means stimulus SOA = 100 ms — pre-stim period is
     occupied by the previous stimulus. Use 0–200 ms to avoid overlap.
  5. Baseline: subtract mean of first 10 ms (no clean pre-stim available)
  6. Resample to 200 Hz
  7. Save epochs as .fif

Output: /Volumes/MEG/things-eeg1/derivatives/preprocessed/<sub>/
  - <sub>_eeg_epo.fif.gz  (epochs, all trials)
  - <sub>_condition_means.npy  (1854 × n_times × n_channels, float32)

Memory note: EEG epochs are much smaller than MEG (63 ch vs 272 ch, fewer trials).
Each subject: 63 × ~22248 trials × 40 timepoints × 4 bytes ≈ 225 MB.

Usage:
    python scripts/preprocess_eeg1.py --subject sub-01
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import mne

RAW_DIR  = Path("/Volumes/MEG/things-eeg1")
DERIV    = RAW_DIR / "derivatives" / "preprocessed"

L_FREQ   = 0.1
H_FREQ   = 40.0
SFREQ    = 200       # resample target (Hz)
TMIN     = -0.05     # 50 ms pre-stim (within RSVP stream — limited baseline)
TMAX     = 0.50      # 500 ms post-stim
BASELINE = (-0.05, 0.0)


def run(subject: str) -> None:
    out_dir = DERIV / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Find EEG file (BrainVision .vhdr or EEGLAB .set) ─────────────────────
    eeg_dir = RAW_DIR / subject / "eeg"
    vhdr_files = sorted(p for p in eeg_dir.glob("*_task-rsvp_eeg.vhdr") if not p.name.startswith("._"))
    set_files  = sorted(p for p in eeg_dir.glob("*_task-rsvp_eeg.set")  if not p.name.startswith("._"))
    if vhdr_files:
        raw_path = vhdr_files[0]
        print(f"Loading BrainVision {raw_path.name}")
        raw = mne.io.read_raw_brainvision(str(raw_path), preload=True, verbose=False)
    elif set_files:
        raw_path = set_files[0]
        print(f"Loading EEGLAB {raw_path.name}  ({raw_path.stat().st_size/1e6:.0f} MB)")
        raw = mne.io.read_raw_eeglab(str(raw_path), preload=True, verbose=False)
    else:
        raise FileNotFoundError(f"No EEG file found in {eeg_dir}")
    print(f"  Channels: {len(raw.ch_names)}, sfreq: {raw.info['sfreq']:.0f} Hz, duration: {raw.times[-1]/60:.1f} min")

    # ── Filter ────────────────────────────────────────────────────────────────
    print("  Filtering 0.1–40 Hz...")
    raw.filter(L_FREQ, H_FREQ, fir_design="firwin", verbose=False)

    # ── Re-reference to average ───────────────────────────────────────────────
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ── Find stimulus events from annotations ─────────────────────────────────
    # THINGS-EEG1 uses EEGLAB events — extract via MNE events_from_annotations
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    print(f"  Events found: {len(events)}, unique IDs: {len(event_id)}")

    # ── Load events TSV ───────────────────────────────────────────────────────
    import pandas as pd
    events_tsvs = sorted(p for p in (RAW_DIR / subject / "eeg").glob("*_task-rsvp_events.tsv") if not p.name.startswith("._"))
    if not events_tsvs:
        raise FileNotFoundError(f"No events TSV found for {subject}")
    ev_df = pd.concat([pd.read_csv(p, sep="\t") for p in events_tsvs], ignore_index=True)
    print(f"  Trials in TSV: {len(ev_df)}, unique concepts: {ev_df['objectnumber'].nunique()}")

    # objectnumber is 0-indexed (0–1853) → convert to 1-indexed to match THINGS-MEG
    concept_nrs = (ev_df["objectnumber"].values + 1).astype(int)

    # ── Build MNE events array from TSV sample column ─────────────────────────
    # `sample` column = sample index at raw sfreq; `onset` = sample/sfreq (seconds)
    sfreq_raw = raw.info["sfreq"]
    if "sample" in ev_df.columns:
        samples = ev_df["sample"].values.astype(int)
    else:
        samples = (ev_df["onset"].values * sfreq_raw).astype(int)

    valid       = (samples >= 0) & (samples < raw.n_times)
    samples     = samples[valid]
    concept_nrs = concept_nrs[valid]

    mne_events  = np.column_stack([samples, np.zeros(len(samples), int), concept_nrs])
    unique_ids  = {str(int(c)): int(c) for c in np.unique(concept_nrs)}

    print(f"  Epoching {len(mne_events)} trials, {len(unique_ids)} unique concepts...")
    epochs = mne.Epochs(
        raw, mne_events, event_id=unique_ids,
        tmin=TMIN, tmax=TMAX,
        baseline=BASELINE,
        preload=True, verbose=False,
    )
    print(f"  Epochs after reject: {len(epochs)}, shape: {epochs.get_data().shape}")

    # ── Resample ──────────────────────────────────────────────────────────────
    if epochs.info["sfreq"] != SFREQ:
        print(f"  Resampling {epochs.info['sfreq']:.0f} → {SFREQ} Hz...")
        epochs.resample(SFREQ, verbose=False)

    # ── Save epochs ───────────────────────────────────────────────────────────
    epo_path = out_dir / f"{subject}_eeg_epo.fif.gz"
    epochs.save(str(epo_path), overwrite=True, verbose=False)
    print(f"  Saved: {epo_path.name}  ({epo_path.stat().st_size/1e6:.0f} MB)")

    # ── Compute per-concept condition means ───────────────────────────────────
    print("  Computing condition means (1854 × n_times × n_ch)...")
    n_times = len(epochs.times)
    n_ch    = len(epochs.ch_names)
    means   = np.zeros((1854, n_times, n_ch), dtype=np.float32)
    counts  = np.zeros(1854, dtype=int)

    data  = epochs.get_data()   # (n_trials, n_ch, n_times)
    ecodes = epochs.events[:, 2]  # concept nrs (1-indexed)

    for trial_idx, cnr in enumerate(ecodes):
        idx = int(cnr) - 1   # 0-indexed
        if 0 <= idx < 1854:
            means[idx] += data[trial_idx].T.astype(np.float32)  # (n_times, n_ch)
            counts[idx] += 1

    # Average (divide by count; leave zeros where no trials)
    valid_mask = counts > 0
    means[valid_mask] /= counts[valid_mask, None, None]

    means_path = out_dir / f"{subject}_condition_means.npy"
    np.save(str(means_path), means)
    print(f"  Saved: {means_path.name}  ({means_path.stat().st_size/1e6:.0f} MB)")
    print(f"  Concepts with ≥1 trial: {valid_mask.sum()}/1854")
    print(f"  Times: {epochs.times[0]*1e3:.0f}–{epochs.times[-1]*1e3:.0f} ms ({n_times} points)")

    return epochs.times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="e.g. sub-01")
    args = parser.parse_args()

    if not (RAW_DIR / args.subject).exists():
        print(f"ERROR: {RAW_DIR / args.subject} not found.")
        print(f"Download first: python scripts/download_things_eeg1.py --subject {args.subject}")
        sys.exit(1)

    times = run(args.subject)
    print(f"\nDone. Times: {times[0]*1e3:.0f}–{times[-1]*1e3:.0f} ms")
