"""Batch download and preprocess THINGS-EEG2 (ds004197, n=200).

Strategy: download raw data for N subjects, preprocess immediately,
save condition means + RDMs, then delete raw to free space.
This keeps peak disk usage to ~1-2 GB/subject instead of ~100 GB total.

Usage:
  python scripts/download_things_eeg2.py --batch-size 10 --max-subjects 200
  python scripts/download_things_eeg2.py --batch-size 5 --max-subjects 20   # pilot

Requirements:
  pip install openneuro-py  (or use datalad/aws s3)

Output:
  data/derivatives/eeg2/sub-{N:04d}_condition_means.npy   (n_cat, n_times, n_ch)
  data/derivatives/eeg2/sub-{N:04d}_rdm_times.npy         (n_times,)
  results/eeg2/alignment_complexity.npz                    (after all subjects done)
"""
from __future__ import annotations
import argparse
import subprocess
import shutil
from pathlib import Path
import numpy as np

RAW_DIR    = Path("/Volumes/MEG/things-eeg2-raw")   # temporary, deleted after processing
DERIV_DIR  = Path("data/derivatives/eeg2")
RESULTS    = Path("results/eeg2")
DATASET_ID = "ds004197"
N_SUBJECTS = 200

# EEG2 paradigm parameters
SFREQ_TARGET = 200    # resample to match EEG1
T_MIN        = -0.1   # epoch start (s)
T_MAX        = 0.995  # epoch end (s)
BASELINE     = (-0.1, 0.0)
L_FREQ       = 0.1
H_FREQ       = 40.0


def available_space_gb(path: Path) -> float:
    import shutil as sh
    stat = sh.disk_usage(str(path.anchor))
    return stat.free / 1e9


def download_subject(sub_id: int, raw_dir: Path) -> bool:
    """Download one subject's raw EEG via openneuro-py or aws s3."""
    sub_str = f"sub-{sub_id:02d}"
    out_dir = raw_dir / sub_str
    if out_dir.exists() and any(out_dir.rglob("*.bdf")):
        print(f"  {sub_str}: already downloaded")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)

    result = subprocess.run([
        "python3", "-m", "openneuro", "download",
        "--dataset", DATASET_ID,
        "--include", f"{sub_str}/eeg",
        "--target-dir", str(raw_dir),
        "--max-concurrent-downloads", "3",
    ], capture_output=True, text=True, timeout=1800)

    if result.returncode == 0:
        return True

    print(f"  openneuro-py failed (exit {result.returncode})")
    if result.stderr:
        print(f"  stderr: {result.stderr[:300]}")
    return False


def preprocess_subject(sub_id: int, raw_dir: Path, deriv_dir: Path) -> bool:
    """Preprocess one subject: filter, epoch, average per concept, save means."""
    import mne
    mne.set_log_level("WARNING")

    sub_str = f"sub-{sub_id:02d}"
    out_file = deriv_dir / f"{sub_str}_condition_means.npy"
    if out_file.exists():
        print(f"  {sub_str}: already preprocessed")
        return True

    # Find raw EEG file
    eeg_dir = raw_dir / sub_str / "eeg"
    bdf_files = list(eeg_dir.glob("*.bdf")) + list(eeg_dir.glob("*.set"))
    if not bdf_files:
        print(f"  {sub_str}: no EEG file found in {eeg_dir}")
        return False

    raw_path = bdf_files[0]
    print(f"  {sub_str}: loading {raw_path.name}")

    try:
        raw = mne.io.read_raw(str(raw_path), preload=True, verbose=False)

        # Drop non-EEG channels
        raw.pick_types(eeg=True, stim=False, exclude="bads")

        # Bandpass
        raw.filter(L_FREQ, H_FREQ, method="fir", verbose=False)

        # Average reference
        raw.set_eeg_reference("average", projection=False, verbose=False)

        # Find events from stimulus channel / annotations
        try:
            events = mne.find_events(raw, stim_channel="Status", verbose=False)
        except Exception:
            events, _ = mne.events_from_annotations(raw, verbose=False)

        # Epoch
        epochs = mne.Epochs(
            raw, events, tmin=T_MIN, tmax=T_MAX,
            baseline=BASELINE, preload=True,
            reject=dict(eeg=200e-6),   # 200 µV rejection
            verbose=False
        )

        # Resample
        if raw.info["sfreq"] != SFREQ_TARGET:
            epochs.resample(SFREQ_TARGET, verbose=False)

        # Get event IDs (concept indices 1–1854 + test images 1855–2054)
        concept_ids = [eid for eid in epochs.event_id.values() if eid <= 1854]

        # Condition means: average epochs per concept
        # Shape: (n_concepts, n_times, n_channels)
        times = epochs.times * 1000  # → ms
        n_ch  = len(epochs.ch_names)
        n_t   = len(times)

        means = np.zeros((len(concept_ids), n_t, n_ch), dtype=np.float32)
        valid_ids = []
        for i, cid in enumerate(sorted(concept_ids)):
            mask = epochs.events[:, 2] == cid
            if mask.sum() == 0:
                continue
            means[i] = epochs.get_data()[mask].mean(axis=0).T
            valid_ids.append(cid)

        valid_ids = np.array(valid_ids)
        means     = means[:len(valid_ids)]

        np.save(str(out_file), means)
        np.save(str(deriv_dir / f"{sub_str}_times.npy"), times.astype(np.float32))
        np.save(str(deriv_dir / f"{sub_str}_valid_concepts.npy"), valid_ids)
        print(f"  {sub_str}: saved {means.shape} condition means, {len(valid_ids)} concepts")
        return True

    except Exception as e:
        print(f"  {sub_str}: preprocessing failed — {e}")
        return False


def delete_raw(sub_id: int, raw_dir: Path):
    sub_str = f"sub-{sub_id:02d}"
    d = raw_dir / sub_str
    if d.exists():
        shutil.rmtree(str(d))
        print(f"  {sub_str}: raw deleted to free space")


def check_space(needed_gb: float = 2.0):
    free = available_space_gb(RAW_DIR if RAW_DIR.exists() else Path("/Volumes/MEG"))
    if free < needed_gb:
        raise RuntimeError(f"Only {free:.1f} GB free — need {needed_gb} GB. Aborting.")
    print(f"  Disk free: {free:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size",    type=int, default=5,
                        help="subjects per batch (download→preprocess→delete)")
    parser.add_argument("--max-subjects",  type=int, default=20,
                        help="stop after N subjects (200 for full dataset)")
    parser.add_argument("--start-subject", type=int, default=1)
    parser.add_argument("--keep-raw",      action="store_true",
                        help="do not delete raw after preprocessing")
    parser.add_argument("--preprocess-only", action="store_true",
                        help="skip download, only preprocess already-downloaded data")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIV_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    subjects = list(range(args.start_subject,
                          min(args.start_subject + args.max_subjects, N_SUBJECTS + 1)))

    print(f"=== THINGS-EEG2 batch pipeline ===")
    print(f"Subjects: {subjects[0]}–{subjects[-1]} (n={len(subjects)})")
    print(f"Batch size: {args.batch_size}")
    print(f"Raw dir: {RAW_DIR}")
    print(f"Deriv dir: {DERIV_DIR}\n")

    successful = []
    failed     = []

    for i in range(0, len(subjects), args.batch_size):
        batch = subjects[i:i + args.batch_size]
        print(f"\n--- Batch {i//args.batch_size + 1}: subjects {batch} ---")

        check_space(needed_gb=args.batch_size * 1.5)

        for sub_id in batch:
            print(f"\n[sub-{sub_id:02d}]")

            if not args.preprocess_only:
                ok = download_subject(sub_id, RAW_DIR)
                if not ok:
                    print(f"  Download failed, skipping")
                    failed.append(sub_id)
                    continue

            ok = preprocess_subject(sub_id, RAW_DIR, DERIV_DIR)
            if ok:
                successful.append(sub_id)
            else:
                failed.append(sub_id)

            if not args.keep_raw and not args.preprocess_only:
                delete_raw(sub_id, RAW_DIR)

    print(f"\n=== Done ===")
    print(f"Successful: {len(successful)} subjects")
    print(f"Failed:     {len(failed)} subjects {failed if failed else ''}")
    print(f"\nNext step: run scripts/run_alignment_eeg2.py to compute alignment")
