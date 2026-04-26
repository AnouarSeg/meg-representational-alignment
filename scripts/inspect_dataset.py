#!/usr/bin/env python
"""Inspect the downloaded THINGS-MEG dataset before writing any preprocessing.

Prints a layout summary and, if MNE can read a recording, a one-recording report
(channel types, sampling rate, duration, events). The point is to replace
assumptions about the raw files with facts:

    python scripts/inspect_dataset.py

Run this after downloading at least one subject. The preprocessing module is
intentionally left as stubs until its output has been read.
"""

from __future__ import annotations

import json

from thingsmeg.config import load_config
from thingsmeg.io import find_meg_files, read_participants, summarize_layout


def main() -> None:
    cfg = load_config()
    summary = summarize_layout(cfg.path_raw)

    print("=" * 70)
    print("DATASET LAYOUT SUMMARY")
    print("=" * 70)
    print(json.dumps(summary, indent=2))

    participants = read_participants(cfg.path_raw)
    if participants:
        print("\nparticipants.tsv:\n" + participants)

    meg_files = find_meg_files(cfg.path_raw)
    if not meg_files:
        print("\nNo MEG recordings found yet. Download a subject first:")
        print("  python scripts/download_things_meg.py --subject <label>")
        return

    # Try to read the first recording with MNE for a concrete report.
    first = meg_files[0]
    print("\n" + "=" * 70)
    print(f"PROBING FIRST RECORDING: {first.name}")
    print("=" * 70)
    try:
        import mne

        raw = mne.io.read_raw(str(first), preload=False, verbose="ERROR")
        print(f"sfreq        : {raw.info['sfreq']} Hz")
        print(f"n_channels   : {raw.info['nchan']}")
        ch_types = raw.get_channel_types()
        type_counts = {t: ch_types.count(t) for t in sorted(set(ch_types))}
        print(f"channel types: {type_counts}")
        print(f"duration     : {raw.times[-1]:.1f} s")
        print(f"highpass/low : {raw.info['highpass']} / {raw.info['lowpass']} Hz")
        # Events live in the BIDS sidecar *_events.tsv, not in STI channels
        events_tsv = first.parent / (first.name.replace("_meg.ds", "_events.tsv"))
        if events_tsv.exists():
            import pandas as pd
            ev = pd.read_csv(events_tsv, sep="\t")
            print(f"events.tsv   : {len(ev)} rows, columns: {list(ev.columns)}")
            if "trial_type" in ev.columns:
                print(f"trial_types  : {ev['trial_type'].unique()[:10].tolist()}")
            if "stim_file" in ev.columns:
                print(f"stim_file ex : {ev['stim_file'].dropna().iloc[0]}")
            print(ev.head(3).to_string(index=False))
        else:
            print(f"events.tsv   : not found at {events_tsv}")
    except Exception as e:  # noqa: BLE001
        print(f"MNE could not read this file directly ({e}).")
        print("Inspect the format manually and pick the right mne.io reader.")


if __name__ == "__main__":
    main()
