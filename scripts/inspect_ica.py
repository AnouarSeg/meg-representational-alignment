"""Interactive ICA inspection for one run of THINGS-MEG.

Fits ICA on BIGMEG1 ses-01 run-01, opens the MNE component browser so you can
manually identify and mark EOG/ECG components before committing to automated
rejection across all subjects.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/inspect_ica.py

Keys in the ICA browser:
    click component  -> toggle exclusion
    left/right arrow -> scroll through components
    close window     -> prints the marked exclusions to stdout
"""

from __future__ import annotations

from pathlib import Path

import mne

from thingsmeg.config import load_config
from thingsmeg.preprocessing import filter_raw, fit_ica, load_raw

mne.set_log_level("WARNING")

cfg = load_config()

# First run of BIGMEG1
meg_path = (
    Path(cfg.path_raw)
    / "sub-BIGMEG1/ses-01/meg"
    / "sub-BIGMEG1_ses-01_task-main_run-01_meg.ds"
)

print(f"Loading: {meg_path.name}")
raw = load_raw(meg_path)

print("Filtering (0.1–40 Hz)…")
raw = filter_raw(raw, cfg)

print("Fitting ICA (this takes ~1–2 min)…")
ica = fit_ica(raw, cfg)

print(f"\nICA fitted: {ica.n_components_} components")
print("Opening component browser — click components to mark for exclusion, then close.")

# Plot all component topographies for a quick overview
ica.plot_components(picks=range(min(40, ica.n_components_)))

# Interactive time-series browser: scroll through components and mark bad ones
ica.plot_sources(raw)

print("\nComponents marked for exclusion:", ica.exclude)
print("Re-run with these indices hardcoded in apply_ica() if auto-detection misses them.")
