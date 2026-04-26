"""Deep-dive plot for a single ICA component.

Shows the three diagnostic panels used to confirm artifact identity:
  1. Topography (spatial pattern)
  2. Time series (when was it active?)
  3. Power spectrum (what frequencies dominate?)
  4. Overlay on raw epochs (what does it look like in context?)

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/inspect_ica_component.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne

from thingsmeg.config import load_config
from thingsmeg.preprocessing import filter_raw, fit_ica, load_raw

mne.set_log_level("WARNING")

cfg = load_config()

meg_path = (
    Path(cfg.path_raw)
    / "sub-BIGMEG1/ses-01/meg"
    / "sub-BIGMEG1_ses-01_task-main_run-01_meg.ds"
)

print("Loading + filtering…")
raw = load_raw(meg_path)
raw = filter_raw(raw, cfg)

print("Fitting ICA…")
ica = fit_ica(raw, cfg)

# ── 1. Zoomed topography for ICA001 alone ──────────────────────────────────
print("\nPlotting ICA001 diagnostics…")
ica.plot_components(picks=[1], title="ICA001 topography")

# ── 2. Time series — 30s window so blink spikes are visible ────────────────
# get_sources returns a Raw-like object with one channel per component
sources = ica.get_sources(raw)
sources.plot(
    picks=[1],          # ICA001 only
    duration=30,
    start=60,           # skip first 60s (participant settling)
    title="ICA001 time series (30s window) — look for irregular spikes = blinks",
    scalings={"misc": 3e-3},
)

# ── 3. Power spectrum — blinks are broadband; heartbeat peaks at ~1 Hz ─────
fig, ax = plt.subplots(figsize=(7, 3))
sources_data = sources.get_data(picks=[1])[0]   # 1-D time series
sfreq = sources.info["sfreq"]
from scipy.signal import welch
freqs, psd = welch(sources_data, fs=sfreq, nperseg=int(sfreq * 4))
ax.semilogy(freqs, psd)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power")
ax.set_title("ICA001 power spectrum")
ax.set_xlim(0, 40)
ax.axvline(1.0, color="r", linestyle="--", label="~1 Hz (heartbeat)")
ax.axvline(10, color="g", linestyle="--", label="10 Hz (alpha)")
ax.legend()
plt.tight_layout()
plt.show()

print("Done. Compare:")
print("  Blink:     broadband power, irregular spikes in time series, FRONTAL topography")
print("  Heartbeat: sharp ~1 Hz peak in spectrum, rhythmic time series, GLOBAL topography")
print("  Neural:    peaked spectrum (alpha/beta), oscillatory time series, FOCAL dipole topo")
