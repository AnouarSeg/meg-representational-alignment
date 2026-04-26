"""Save diagnostic plots for all ICA components as PNGs.

For each component: topography + 30s time series + power spectrum saved to
figures/ica_inspection/ICA{n:03d}.png — no GUI interaction needed.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/save_ica_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves files, no GUI needed
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch

from thingsmeg.config import load_config
from thingsmeg.preprocessing import filter_raw, fit_ica, load_raw

mne.set_log_level("WARNING")

cfg = load_config()

out_dir = Path("figures/ica_inspection")
out_dir.mkdir(parents=True, exist_ok=True)

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
n_components = ica.n_components_
print(f"Fitted: {n_components} components. Saving plots…")

sources = ica.get_sources(raw)
sfreq = sources.info["sfreq"]
all_sources = sources.get_data()  # (n_components, n_times)

# 30s window starting at 60s (skip settling period)
t_start = int(60 * sfreq)
t_end = int(90 * sfreq)

for i in range(n_components):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"ICA{i:03d}", fontsize=14, fontweight="bold")

    # ── Panel 1: topography ──────────────────────────────────────────────
    topo_fig = ica.plot_components(picks=[i], show=False)
    # grab the rendered topography and paste it into our axes
    topo_fig.canvas.draw()
    buf = np.frombuffer(topo_fig.canvas.tostring_argb(), dtype=np.uint8)
    buf = buf.reshape(topo_fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, 1:]  # ARGB -> RGB
    axes[0].imshow(buf)
    axes[0].axis("off")
    axes[0].set_title("Topography")
    plt.close(topo_fig)

    # ── Panel 2: 30s time series ─────────────────────────────────────────
    ts = all_sources[i, t_start:t_end]
    times = np.arange(len(ts)) / sfreq
    axes[1].plot(times, ts, linewidth=0.6, color="steelblue")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude (a.u.)")
    axes[1].set_title("Time series (30s window)")

    # ── Panel 3: power spectrum ──────────────────────────────────────────
    freqs, psd = welch(all_sources[i], fs=sfreq, nperseg=int(sfreq * 4))
    mask = freqs <= 40
    axes[2].semilogy(freqs[mask], psd[mask], color="darkred")
    axes[2].axvline(1.0, color="orange", linestyle="--", linewidth=1, label="1 Hz (ECG)")
    axes[2].axvline(10, color="green", linestyle="--", linewidth=1, label="10 Hz (alpha)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("Power")
    axes[2].set_title("Power spectrum (0–40 Hz)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_dir / f"ICA{i:03d}.png", dpi=100)
    plt.close(fig)
    print(f"  saved ICA{i:03d}.png")

print(f"\nDone. All plots in {out_dir}/")
