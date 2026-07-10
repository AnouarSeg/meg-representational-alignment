"""Cross-subject alignment complexity on THINGS-EEG2 (n=200).

Loads preprocessed condition means from data/derivatives/eeg2/,
runs Procrustes vs ridge alignment across all subject pairs,
reports complexity over time.

Designed to run after download_things_eeg2.py has processed subjects.

Output:
  results/eeg2/alignment_complexity.npz
  figures/figure_eeg2_alignment.png
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
import numpy as np

DERIV   = Path("data/derivatives/eeg2")
RESULTS = Path("results/eeg2")
FIGS    = Path("figures")

N_TRAIN  = 160   # training concepts per fold
N_TEST   = 40
CV_FOLDS = 5
ALPHA    = 1.0   # ridge regularisation


def procrustes_map(X, Y):
    """Orthogonal Procrustes: find R minimising ||Y - XR||_F."""
    U, _, Vt = np.linalg.svd(X.T @ Y)
    return U @ Vt


def ridge_map(X, Y, alpha=ALPHA):
    """Ridge regression: W = (X'X + αI)^{-1} X'Y."""
    n, d = X.shape
    return np.linalg.solve(X.T @ X + alpha * np.eye(d), X.T @ Y)


def transfer_accuracy(X_train, Y_train, X_test, Y_test, W):
    """1-NN accuracy of transformed X_test vs Y_test."""
    X_hat = X_test @ W
    dists = np.sum((X_hat[:, None] - Y_test[None]) ** 2, axis=-1)
    preds = dists.argmin(axis=1)
    return (preds == np.arange(len(X_test))).mean() * 100


def align_pair(A, B, times):
    """
    A, B: (n_cat, n_times, n_ch) condition means for two subjects.
    Returns: procrustes_ts, ridge_ts — transfer accuracy per timepoint.
    """
    n_cat, n_times, n_ch = A.shape
    proc_ts  = np.zeros(n_times)
    ridge_ts = np.zeros(n_times)

    rng  = np.random.default_rng(42)
    idx  = np.arange(n_cat)

    fold_size = n_cat // CV_FOLDS
    for fold in range(CV_FOLDS):
        te_idx = idx[fold * fold_size:(fold + 1) * fold_size]
        tr_idx = np.setdiff1d(idx, te_idx)

        for t in range(n_times):
            Atr, Ate = A[tr_idx, t], A[te_idx, t]
            Btr, Bte = B[tr_idx, t], B[te_idx, t]

            # z-score
            mu_a, sd_a = Atr.mean(0), Atr.std(0) + 1e-12
            mu_b, sd_b = Btr.mean(0), Btr.std(0) + 1e-12
            Atr_z = (Atr - mu_a) / sd_a;  Ate_z = (Ate - mu_a) / sd_a
            Btr_z = (Btr - mu_b) / sd_b;  Bte_z = (Bte - mu_b) / sd_b

            R = procrustes_map(Atr_z, Btr_z)
            W = ridge_map(Atr_z, Btr_z)

            proc_ts[t]  += transfer_accuracy(Atr_z, Btr_z, Ate_z, Bte_z, R)
            ridge_ts[t] += transfer_accuracy(Atr_z, Btr_z, Ate_z, Bte_z, W)

    return proc_ts / CV_FOLDS, ridge_ts / CV_FOLDS


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Find preprocessed subjects
    mean_files = sorted(DERIV.glob("sub-*_condition_means.npy"))
    if not mean_files:
        raise FileNotFoundError(f"No preprocessed subjects found in {DERIV}")

    subjects = [f.stem.replace("_condition_means", "") for f in mean_files]
    print(f"Found {len(subjects)} preprocessed subjects: {subjects[:5]}...")

    # Load all subjects
    data = {}
    times = None
    for sub in subjects:
        means = np.load(str(DERIV / f"{sub}_condition_means.npy"))
        data[sub] = means.astype(np.float32)
        if times is None:
            t_file = DERIV / f"{sub}_times.npy"
            times  = np.load(str(t_file)) if t_file.exists() else np.linspace(-100, 995, means.shape[1])

    n_cat   = min(d.shape[0] for d in data.values())
    n_times = len(times)
    print(f"Categories (min across subjects): {n_cat}, timepoints: {n_times}")

    # Trim all to shared concept count
    for sub in data:
        data[sub] = data[sub][:n_cat]

    # Run alignment across all pairs (sample up to 500 pairs if n > 32)
    pairs = list(combinations(subjects, 2))
    max_pairs = 500
    if len(pairs) > max_pairs:
        rng   = np.random.default_rng(0)
        pairs = [pairs[i] for i in rng.choice(len(pairs), max_pairs, replace=False)]
    print(f"Running alignment on {len(pairs)} subject pairs...")

    proc_all  = np.zeros((len(pairs), n_times))
    ridge_all = np.zeros((len(pairs), n_times))

    for i, (s1, s2) in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Pair {i}/{len(pairs)}: {s1} vs {s2}")
        proc_all[i], ridge_all[i] = align_pair(data[s1], data[s2], times)

    proc_mean  = proc_all.mean(0)
    ridge_mean = ridge_all.mean(0)
    complexity = ridge_mean - proc_mean

    # Bootstrap CI on complexity
    rng = np.random.default_rng(1)
    boot = np.array([
        (ridge_all[rng.integers(0, len(pairs), len(pairs))].mean(0) -
         proc_all[ rng.integers(0, len(pairs), len(pairs))].mean(0))
        for _ in range(1000)
    ])
    ci_lo = np.percentile(boot, 2.5, axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)

    post  = times > 0
    print(f"\n=== EEG2 Results ===")
    print(f"Procrustes peak: {proc_mean[post].max():.3f}% at {times[post][proc_mean[post].argmax()]:.0f}ms")
    print(f"Ridge peak:      {ridge_mean[post].max():.3f}% at {times[post][ridge_mean[post].argmax()]:.0f}ms")
    print(f"Complexity (post-stim mean): {complexity[post].mean():.4f}%")
    print(f"95% CI: [{ci_lo[post].mean():.4f}, {ci_hi[post].mean():.4f}]%")

    np.savez(str(RESULTS / "alignment_complexity.npz"),
             times=times, procrustes=proc_mean, ridge=ridge_mean,
             complexity=complexity, ci_lo=ci_lo, ci_hi=ci_hi,
             n_pairs=len(pairs), n_subjects=len(subjects))
    print(f"Saved results/eeg2/alignment_complexity.npz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.plot(times, proc_mean,  "#2980b9", lw=2, label=f"Procrustes (peak {proc_mean[post].max():.2f}%)")
    ax.plot(times, ridge_mean, "#e74c3c", lw=2, label=f"Ridge      (peak {ridge_mean[post].max():.2f}%)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title(f"EEG2 cross-subject alignment (n={len(subjects)} subjects, {len(pairs)} pairs)")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(times, complexity, "k", lw=2, label="Ridge − Procrustes")
    ax.fill_between(times, ci_lo, ci_hi, color="k", alpha=0.15, label="95% CI")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Complexity (%)")
    ax.set_title("Alignment complexity (EEG2)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(FIGS / "figure_eeg2_alignment.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure_eeg2_alignment.png")
