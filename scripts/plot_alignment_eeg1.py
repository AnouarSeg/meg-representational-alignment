"""Plot alignment complexity results from THINGS-EEG1.

Input:  results/eeg1/alignment_complexity_eeg1.npz
Output: figures/figure3b_alignment_complexity_eeg1.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS  = Path("results/eeg1/alignment_complexity_eeg1.npz")
FIG_OUT  = Path("figures/figure3b_alignment_complexity_eeg1.png")
SFREQ    = 200   # Hz after resampling
TMIN_MS  = -50
TMAX_MS  = 495

def main() -> None:
    if not RESULTS.exists():
        print(f"ERROR: {RESULTS} not found. Run run_alignment_eeg1.py first.")
        return

    d = np.load(str(RESULTS), allow_pickle=True)
    proc_all = d["procrustes_acc"]   # (n_pairs, n_times)
    ridg_all = d["ridge_acc"]
    comp_all = d["complexity"]
    n_subj   = int(d["n_subjects"])
    n_pairs  = proc_all.shape[0]

    n_times = proc_all.shape[1]
    times   = np.linspace(TMIN_MS, TMAX_MS, n_times)

    # Bootstrap CIs over pairs
    rng = np.random.default_rng(42)
    n_boot = 1000
    proc_boot = np.zeros((n_boot, n_times))
    ridg_boot = np.zeros((n_boot, n_times))
    comp_boot = np.zeros((n_boot, n_times))
    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, n_pairs)
        proc_boot[b] = proc_all[idx].mean(0)
        ridg_boot[b] = ridg_all[idx].mean(0)
        comp_boot[b] = comp_all[idx].mean(0)

    proc_m  = proc_all.mean(0)
    ridg_m  = ridg_all.mean(0)
    comp_m  = comp_all.mean(0)
    proc_lo, proc_hi = np.percentile(proc_boot, [2.5, 97.5], axis=0)
    ridg_lo, ridg_hi = np.percentile(ridg_boot, [2.5, 97.5], axis=0)
    comp_lo, comp_hi = np.percentile(comp_boot, [2.5, 97.5], axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Panel A: transfer accuracy
    ax = axes[0]
    ax.fill_between(times, proc_lo * 100, proc_hi * 100, alpha=0.2, color="#1f77b4")
    ax.fill_between(times, ridg_lo * 100, ridg_hi * 100, alpha=0.2, color="#ff7f0e")
    ax.plot(times, proc_m * 100, label="Procrustes", color="#1f77b4", lw=2)
    ax.plot(times, ridg_m * 100, label="Ridge", color="#ff7f0e", lw=2)
    ax.axhline(100 / 1854, color="gray", ls="--", lw=1, label="Chance")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title(f"THINGS-EEG1 cross-subject alignment (n={n_subj} subjects, {n_pairs} pairs)")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)

    # Panel B: complexity (ridge - procrustes)
    ax = axes[1]
    ax.fill_between(times, comp_lo * 100, comp_hi * 100, alpha=0.2, color="#2ca02c")
    ax.plot(times, comp_m * 100, label="Ridge − Procrustes (complexity)", color="#2ca02c", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time relative to stimulus onset (ms)")
    ax.set_ylabel("Complexity gain (%)")
    ax.legend(fontsize=9)

    # Annotate mean complexity
    post_mask = times > 0
    post_mean = comp_m[post_mask].mean() * 100
    ax.text(0.98, 0.05,
            f"Post-stim mean: {post_mean:+.3f}%",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(str(FIG_OUT), dpi=150, bbox_inches="tight")
    print(f"Saved {FIG_OUT}")

    print(f"\nResults summary:")
    print(f"  Subjects: {n_subj}, pairs: {n_pairs}")
    print(f"  Procrustes peak: {proc_m.max()*100:.2f}% at {times[proc_m.argmax()]:.0f} ms")
    print(f"  Ridge peak:      {ridg_m.max()*100:.2f}% at {times[ridg_m.argmax()]:.0f} ms")
    print(f"  Post-stim complexity mean: {post_mean:+.3f}%  (95% CI: {comp_lo[post_mask].mean()*100:+.3f} to {comp_hi[post_mask].mean()*100:+.3f})")
    print(f"  Max complexity: {comp_m.max()*100:+.3f}% at {times[comp_m.argmax()]:.0f} ms")

if __name__ == "__main__":
    main()
