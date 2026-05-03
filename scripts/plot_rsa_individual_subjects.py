"""Plot individual subject RSA curves alongside group mean.

Shows inter-subject variability for both MEG (n=4) and EEG (n=48).

Output: figures/figure4c_rsa_individual_subjects.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS     = Path("results")
RESULTS_EEG = Path("results/eeg1")


def plot_meg_individual(ax, data_path: Path) -> None:
    d = np.load(str(data_path))
    times = d["times_ms"]
    nc_up = d["nc_upper"]
    nc_lo = d["nc_lower"]

    models  = ["SPOSE", "CLIP_image"]
    colors  = {"SPOSE": "#e41a1c", "CLIP_image": "#377eb8"}
    labels  = {"SPOSE": "SPOSE", "CLIP_image": "CLIP ViT-B/32"}

    ax.fill_between(times, nc_lo, nc_up, alpha=0.10, color="gray")
    ax.text(times[-1], float(nc_up[-30:].mean()), "NC", fontsize=7, color="gray", va="center")

    for m in models:
        if m not in d:
            continue
        vals = d[m]   # (n_sub, n_times)
        c = colors[m]
        for i in range(vals.shape[0]):
            ax.plot(times, vals[i], color=c, alpha=0.25, lw=1)
        ax.plot(times, vals.mean(0), color=c, lw=2.5, label=labels[m])

    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_title("MEG (n=4): individual subjects")
    ax.set_ylabel("Spearman r")
    ax.legend(fontsize=8)


def plot_eeg_individual(ax, data_path: Path) -> None:
    if not data_path.exists():
        ax.text(0.5, 0.5, "EEG RSA not yet computed", transform=ax.transAxes,
                ha="center", va="center")
        return

    d = np.load(str(data_path))
    times  = d["times"]
    nc_up  = d["nc_upper"]
    nc_lo  = d["nc_lower"]

    models = ["SPOSE", "CLIP_image"]
    colors = {"SPOSE": "#e41a1c", "CLIP_image": "#377eb8"}
    labels = {"SPOSE": "SPOSE", "CLIP_image": "CLIP ViT-B/32"}

    ax.fill_between(times, nc_lo, nc_up, alpha=0.10, color="gray")
    ax.text(float(times[-1]), float(nc_up[-5:].mean()), "NC", fontsize=7,
            color="gray", va="center")

    for m in models:
        if m not in d:
            continue
        vals = d[m]   # (n_sub, n_times)
        c = colors[m]
        # Too many subjects to plot individual lines legibly — show median ± IQR
        med = np.median(vals, axis=0)
        q25 = np.percentile(vals, 25, axis=0)
        q75 = np.percentile(vals, 75, axis=0)
        ax.fill_between(times, q25, q75, alpha=0.15, color=c)
        ax.plot(times, med, color=c, lw=2.5, label=f"{labels[m]} (median ± IQR)")

    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_title(f"EEG (n={d[list(d.keys())[3]].shape[0] if 'SPOSE' in d else '?'}): median ± IQR")
    ax.set_ylabel("Spearman r")
    ax.legend(fontsize=8)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=False)

    plot_meg_individual(axes[0], RESULTS / "brain_model_rsa_full1854.npz")
    plot_eeg_individual(axes[1], RESULTS_EEG / "brain_model_rsa_eeg1.npz")

    for ax in axes:
        ax.set_xlabel("Time relative to stimulus onset (ms)")

    fig.suptitle("Brain–model RSA: inter-subject variability", fontsize=11)
    fig.tight_layout()
    out = Path("figures/figure4c_rsa_individual_subjects.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
