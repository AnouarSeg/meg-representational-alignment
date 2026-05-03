"""Cluster permutation test on alignment complexity null (EEG1).

Tests H0: complexity time course = 0 everywhere post-stimulus.
Uses MNE-style cluster permutation: observed t-statistic vs permuted
distribution of max-cluster-mass.

Input:  results/eeg1/alignment_complexity_eeg1.npz
Output: results/eeg1/alignment_complexity_permutation.npz
        figures/figure3d_alignment_complexity_permutation.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS = Path("results/eeg1")
N_PERM  = 1000
ALPHA   = 0.05
TMIN_MS = -50
TMAX_MS = 495


def cluster_mass(t_vals: np.ndarray, threshold: float) -> float:
    """Sum of t-values in clusters exceeding threshold."""
    in_cluster = np.abs(t_vals) > threshold
    if not in_cluster.any():
        return 0.0
    # Find contiguous clusters
    from itertools import groupby
    mass = 0.0
    for key, group in groupby(enumerate(in_cluster), lambda x: x[1]):
        if key:
            indices = [i for i, _ in group]
            mass = max(mass, np.abs(t_vals[indices]).sum())
    return mass


if __name__ == "__main__":
    d = np.load(str(RESULTS / "alignment_complexity_eeg1.npz"))
    comp = d["complexity"]   # (n_pairs, n_times)
    n_pairs, n_times = comp.shape
    times = np.linspace(TMIN_MS, TMAX_MS, n_times)

    # Only test post-stimulus window
    post_mask = times > 0
    comp_post = comp[:, post_mask]
    times_post = times[post_mask]
    n_post = comp_post.shape[1]

    # Observed t-statistic (one-sample t vs 0) at each timepoint
    t_obs, _ = stats.ttest_1samp(comp_post, 0, axis=0)
    df = n_pairs - 1
    threshold = stats.t.ppf(1 - ALPHA / 2, df)   # two-tailed

    obs_mass = cluster_mass(t_obs, threshold)
    print(f"Observed max cluster mass: {obs_mass:.3f}  (threshold t={threshold:.3f})")

    # Permutation: flip signs randomly (sign-flip test — valid for zero-mean H0)
    rng = np.random.default_rng(42)
    perm_masses = np.zeros(N_PERM)
    for p in range(N_PERM):
        signs = rng.choice([-1, 1], size=n_pairs)
        comp_perm = comp_post * signs[:, None]
        t_perm, _ = stats.ttest_1samp(comp_perm, 0, axis=0)
        perm_masses[p] = cluster_mass(t_perm, threshold)
        if (p+1) % 100 == 0:
            print(f"  Permutation {p+1}/{N_PERM}", end="\r")

    p_value = (perm_masses >= obs_mass).mean()
    print(f"\nCluster permutation p-value: {p_value:.4f}")
    print(f"Mean post-stim complexity: {comp_post.mean()*100:.4f}%")
    print(f"Mean t-stat: {t_obs.mean():.3f},  range [{t_obs.min():.3f}, {t_obs.max():.3f}]")

    np.savez(str(RESULTS / "alignment_complexity_permutation.npz"),
             times_post=times_post,
             t_obs=t_obs,
             perm_masses=perm_masses,
             obs_mass=obs_mass,
             p_value=p_value,
             threshold=threshold)

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: t-statistic time course
    ax = axes[0]
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axhline(threshold, color="r", ls="--", lw=1, label=f"Threshold (t={threshold:.2f})")
    ax.axhline(-threshold, color="r", ls="--", lw=1)
    ax.plot(times_post, t_obs, color="#2ca02c", lw=2, label="Observed t")
    ax.set_xlabel("Time post-stimulus (ms)")
    ax.set_ylabel("t-statistic")
    ax.set_title(f"Complexity null: t-statistic\n(p={p_value:.3f}, n={n_pairs} pairs)")
    ax.legend(fontsize=9)

    # Right: permutation distribution
    ax = axes[1]
    ax.hist(perm_masses, bins=40, color="gray", alpha=0.7, label="Permuted max cluster mass")
    ax.axvline(obs_mass, color="#2ca02c", lw=2, label=f"Observed ({obs_mass:.3f})")
    ax.set_xlabel("Max cluster mass")
    ax.set_ylabel("Count")
    ax.set_title(f"Permutation distribution (n={N_PERM})\np={p_value:.3f}")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = Path("figures/figure3d_alignment_complexity_permutation.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
