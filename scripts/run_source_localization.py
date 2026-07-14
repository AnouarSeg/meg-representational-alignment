"""MEG source localization: sensor-space condition means → source-space activity.

Pipeline (no FreeSurfer required):
  1. Load preprocessed condition means (n_cat, n_times, n_ch) per subject
  2. Automated fiducial coregistration to fsaverage
  3. Forward solution (fsaverage ico-4 source space, sphere BEM)
  4. MNE minimum-norm inverse operator
  5. Apply inverse to condition means → source means
  6. Save per-subject; delete intermediate objects to free RAM

Memory budget: ~400 MB peak per subject. Subjects processed sequentially.

Run with things-meg conda env:
  /opt/anaconda3/envs/things-meg/bin/python3 scripts/run_source_localization.py

Output:
  results/source/sub-{NAME}_source_means.npy   (n_cat, n_times, n_src)
  results/source/sub-{NAME}_vertices.npz       (lh, rh vertex arrays)
  results/source/times_ms.npy
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

RESULTS   = Path("results")
SRC_DIR   = Path("results/source")
SUBJECTS  = ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]
MNE_DATA  = Path("/Users/anouarseghir/mne_data/MNE-fsaverage-data")
FS_DIR    = str(MNE_DATA)       # subjects_dir
FS_SUB    = "fsaverage"
LAMBDA2   = 1.0 / 9.0           # SNR=3 → lambda2=1/9 (standard for evoked)
ICO       = 4                   # ico-4: 2562 vertices/hemisphere (manageable RAM)


def load_condition_means(subject: str):
    """Load (n_cat, n_times, n_ch) from results/condition_means.npz."""
    f = RESULTS / "condition_means.npz"
    d = np.load(str(f), allow_pickle=True)
    times = d["times_ms"].astype(np.float32)
    data  = d[subject]   # stored as (n_cat, n_ch, n_times) → transpose to (n_cat, n_times, n_ch)
    if data.shape[1] > data.shape[2]:   # n_ch (272) > n_times (180) → need to transpose
        data = data.transpose(0, 2, 1)
    return data.astype(np.float32), times


def get_raw_info(subject: str):
    """Load MEG info (channel layout + digitization) from first session raw file."""
    import mne, glob
    raw_dir = Path(f"/Volumes/MEG/things-meg/raw/sub-{subject}")
    # Find any .ds CTF file
    ds_files = sorted(p for p in raw_dir.rglob("*task-main_run-01*.ds")
                      if not p.name.startswith("._") and p.is_dir())
    if not ds_files:
        ds_files = sorted(p for p in raw_dir.rglob("*.ds")
                          if not p.name.startswith("._") and p.is_dir())
    if not ds_files:
        raise FileNotFoundError(f"No .ds files found for {subject}")
    raw = mne.io.read_raw_ctf(str(ds_files[0]), preload=False, verbose=False)
    # Keep only MEG channels — match our 272-ch condition means
    raw.pick_types(meg=True, ref_meg=False, verbose=False)
    print(f"  Info: {len(raw.ch_names)} MEG channels, {len(raw.info['dig'])} dig points")
    return raw.info


def coregister_to_fsaverage(info):
    """Automated fiducial-based coregistration to fsaverage. Returns trans."""
    import mne
    coreg = mne.coreg.Coregistration(info, FS_SUB, subjects_dir=FS_DIR,
                                     fiducials="auto")
    coreg.fit_fiducials(verbose=False)
    print(f"  Coregistration done (fiducial-based)")
    return coreg.trans


def make_forward(info, trans):
    """Forward solution: fsaverage ico-4 surface source space + sphere BEM."""
    import mne, os
    # Source space
    src_path = os.path.join(FS_DIR, FS_SUB, "bem", f"{FS_SUB}-ico-{ICO}-src.fif")
    if os.path.exists(src_path):
        src = mne.read_source_spaces(src_path, verbose=False)
    else:
        src = mne.setup_source_space(FS_SUB, spacing=f"ico{ICO}",
                                     subjects_dir=FS_DIR, verbose=False)

    # BEM solution
    bem_path = os.path.join(FS_DIR, FS_SUB, "bem",
                            f"{FS_SUB}-5120-5120-5120-bem-sol.fif")
    bem = mne.read_bem_solution(bem_path, verbose=False)

    fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                    meg=True, eeg=False, verbose=False)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True, verbose=False)
    n_src = fwd["nsource"]
    print(f"  Forward: {n_src} sources")
    return fwd


def make_inverse(info, fwd, noise_cov=None):
    """MNE minimum-norm inverse operator with diagonal noise covariance."""
    import mne
    if noise_cov is None:
        # Identity noise cov — equivalent to assuming equal noise on all channels
        noise_cov = mne.make_ad_hoc_cov(info, verbose=False)
    inv = mne.minimum_norm.make_inverse_operator(info, fwd, noise_cov,
                                                  loose=0.2, depth=0.8,
                                                  verbose=False)
    return inv


def apply_inverse_to_means(condition_means, inv, info, times):
    """
    Apply MNE inverse to all condition means using public apply_inverse API.
    condition_means: (n_cat, n_times, n_ch)
    Returns: source_means (n_cat, n_times, n_src), vertices (lh, rh)
    """
    import mne
    n_cat, n_times, n_ch = condition_means.shape
    nave = 12

    # Stack all (cat × time) as one big EvokedArray, apply inverse once
    # Shape: (n_ch, n_cat * n_times)
    all_data = condition_means.reshape(-1, n_ch).T.astype(np.float64)
    fake_evoked = mne.EvokedArray(all_data, info, tmin=0.0, nave=nave, verbose=False)

    stc = mne.minimum_norm.apply_inverse(
        fake_evoked, inv, lambda2=LAMBDA2, method="MNE",
        pick_ori="normal", verbose=False
    )
    # stc.data: (n_src, n_cat * n_times)
    n_src = stc.data.shape[0]
    source_means = stc.data.T.reshape(n_cat, n_times, n_src).astype(np.float32)
    print(f"  Source means: {source_means.shape}")

    vertices = [inv["src"][0]["vertno"], inv["src"][1]["vertno"]]
    return source_means, vertices


if __name__ == "__main__":
    import mne
    mne.set_log_level("WARNING")
    SRC_DIR.mkdir(parents=True, exist_ok=True)

    times_saved = False
    for subject in SUBJECTS:
        out_file = SRC_DIR / f"sub-{subject}_source_means.npy"
        if out_file.exists():
            print(f"\n[{subject}] Already done, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"[{subject}]")

        print("  Loading condition means...")
        cond_means, times = load_condition_means(subject)
        n_cat, n_times, n_ch = cond_means.shape
        print(f"  Shape: {cond_means.shape}")

        print("  Loading MEG info...")
        info = get_raw_info(subject)

        # Verify channel count matches (n_ch is last dim after transpose)
        if len(info.ch_names) != n_ch:
            print(f"  Trimming info: {len(info.ch_names)} → {n_ch} channels")
            picks = mne.pick_types(info, meg=True, ref_meg=False)[:n_ch]
            info  = mne.pick_info(info, picks)

        print("  Coregistering to fsaverage...")
        trans = coregister_to_fsaverage(info)

        print("  Building forward solution...")
        fwd = make_forward(info, trans)

        print("  Building inverse operator...")
        inv = make_inverse(info, fwd)

        print("  Applying inverse to condition means...")
        source_means, vertices = apply_inverse_to_means(cond_means, inv, info, times)
        print(f"  Source means: {source_means.shape}")

        # Save
        np.save(str(out_file), source_means)
        np.savez(str(SRC_DIR / f"sub-{subject}_vertices.npz"),
                 lh=vertices[0], rh=vertices[1])
        if not times_saved:
            np.save(str(SRC_DIR / "times_ms.npy"), times)
            times_saved = True

        print(f"  Saved {out_file}")

        # Free memory
        del cond_means, fwd, inv, source_means, trans
        gc.collect()

    print("\n=== Source localization complete ===")
    print(f"Output: {SRC_DIR}")
