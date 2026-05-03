"""Download THINGS-EEG1 from OpenNeuro (ds003825).

50 subjects, 63-channel EEG, RSVP at 10 Hz, 1854 THINGS concepts × 12 images each.
~1.1 GB per subject compressed. Download one subject first to inspect, then all.

Usage:
    # Single subject (inspect first):
    python scripts/download_things_eeg1.py --subject sub-01

    # All subjects (50 × ~1 GB ≈ 55 GB):
    python scripts/download_things_eeg1.py --all

    # Specific range:
    python scripts/download_things_eeg1.py --subjects sub-01 sub-02 sub-03

Storage: /Volumes/MEG/things-eeg1/  (uses MEG drive)
"""

import argparse
import sys
from pathlib import Path

import openneuro

DATASET  = "ds003825"
TAG      = "1.2.0"   # latest snapshot
OUT_DIR  = Path("/Volumes/MEG/things-eeg1")

ALWAYS_INCLUDE = [
    "participants.tsv",
    "participants.json",
    "dataset_description.json",
    "task-rsvp_eeg.json",
    "task-rsvp_events.json",
    "README",
    "CHANGES",
]

def download_subject(sub: str) -> None:
    print(f"\nDownloading {sub} from {DATASET} v{TAG} → {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patterns = ALWAYS_INCLUDE + [f"{sub}/**"]
    openneuro.download(
        dataset=DATASET,
        tag=TAG,
        target_dir=OUT_DIR,
        include=patterns,
        max_concurrent_downloads=4,
    )
    print(f"Done: {sub}")


def list_subjects() -> list[str]:
    return [f"sub-{i:02d}" for i in range(1, 51)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject",  help="Single subject, e.g. sub-01")
    group.add_argument("--subjects", nargs="+", help="List of subjects")
    group.add_argument("--all",      action="store_true", help="All 50 subjects")
    args = parser.parse_args()

    if not Path("/Volumes/MEG").exists():
        print("ERROR: /Volumes/MEG not mounted. Plug in the MEG drive first.")
        sys.exit(1)

    if args.all:
        subjects = list_subjects()
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = [args.subject]

    print(f"Subjects to download: {subjects}")
    print(f"Estimated size: {len(subjects) * 1.1:.1f} GB")
    print(f"Target: {OUT_DIR}")
    print()

    for sub in subjects:
        download_subject(sub)

    print(f"\nAll done. Data in {OUT_DIR}")
