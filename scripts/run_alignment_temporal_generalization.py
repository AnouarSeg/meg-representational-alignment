"""Temporal generalization of cross-subject alignment (EEG1).

Train Procrustes map at time T, apply at time T'.
Reveals whether the alignment geometry is time-stable (sustained, off-diagonal)
or time-specific (transient, diagonal only).

Memory: uses condition means only (not epochs).
  Load pairs of subjects' means: 2 × 1854 × 110 × 63 × 4B = ~103 MB per pair.
  Run on a random sample of 50 pairs to keep runtime under 2h.

Output: results/eeg1/alignment_tgm.npz
        figures/figure3e_alignment_tgm.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

DERIV   = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
RESULTS = Path("results/eeg1")
N_PAIRS_SAMPLE = 10   # small sample — TGM is O(T^2) expensive
N_FOLDS = 5
N_CAT   = 200   # subsample categories for speed (still meaningful geometry)
MODAL_CH = 63


def procrustes_transfer_tgm(X_tr, Y_tr, X_te_all, Y_te_all):
    """Fit on train split at one timepoint, evaluate at all timepoints.

    X_tr, Y_tr: (n_train, n_ch) — training data at time T_train
    X_te_all:   (n_cat, n_times, n_ch) — test data at all timepoints
    Y_te_all:   (n_cat, n_times, n_ch)

    Returns: (n_times,) accuracy
    """
    try:
        R, _ = orthogonal_procrustes(X_tr, Y_tr)
    except np.linalg.LinAlgError:
        return np.full(X_te_all.shape[1], np.nan)

    n_te   = X_te_all.shape[0]
    n_times = X_te_all.shape[1]
    acc = np.zeros(n_times)

    for t_test in range(n_times):
        pred = X_te_all[:, t_test, :] @ R
        tgt  = Y_te_all[:, t_test, :]
        correct = sum(np.argmin(np.linalg.norm(pred[i] - tgt, axis=1)) == i
                      for i in range(n_te))
        acc[t_test] = correct / n_te
    return acc


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Find 63-ch subjects
    available = sorted(
        p.name for p in DERIV.glob("sub-*/")
        if (DERIV / p.name / f"{p.name}_condition_means.npy").exists()
    )
    subjects = [s for s in available
                if np.load(DERIV/s/f"{s}_condition_means.npy", mmap_mode="r").shape[2] == MODAL_CH]
    print(f"{len(subjects)} subjects available")

    from itertools import combinations
    all_pairs = list(combinations(subjects, 2))
    rng = np.random.default_rng(42)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS_SAMPLE, replace=False)]
    print(f"Running {len(pairs)} random pairs")

    n_times = None
    tgm_all = []

    fold_size = N_CAT // N_FOLDS
    folds = [np.arange(k*fold_size, (k+1)*fold_size) for k in range(N_FOLDS)]
    print(f"Using {N_CAT} categories, {N_PAIRS_SAMPLE} pairs, {n_times if n_times else '?'} timepoints")

    cat_idx = rng.choice(1854, N_CAT, replace=False)
    cat_idx.sort()

    for pi, (sA, sB) in enumerate(pairs):
        A_full = np.load(DERIV/sA/f"{sA}_condition_means.npy")
        B_full = np.load(DERIV/sB/f"{sB}_condition_means.npy")
        A = A_full[cat_idx]
        B = B_full[cat_idx]
        del A_full, B_full
        if n_times is None:
            n_times = A.shape[1]
            times = np.linspace(-50, 495, n_times)

        tgm = np.zeros((n_times, n_times))

        for t_train in range(n_times):
            fold_acc = np.zeros(n_times)
            for fi in range(N_FOLDS):
                te_idx = folds[fi]
                tr_idx = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fi])
                acc = procrustes_transfer_tgm(
                    A[tr_idx, t_train, :], B[tr_idx, t_train, :],
                    A[te_idx], B[te_idx]
                )
                fold_acc += acc / N_FOLDS
            tgm[t_train] = fold_acc

            if t_train % 20 == 0:
                print(f"  Pair {pi+1}/{len(pairs)}, t_train={t_train}/{n_times}", end="\r")

        tgm_all.append(tgm)
        del A, B

    tgm_mean = np.mean(tgm_all, axis=0)
    np.savez(str(RESULTS / "alignment_tgm.npz"),
             tgm_mean=tgm_mean, times=times, n_pairs=len(pairs))
    print(f"\nSaved results/eeg1/alignment_tgm.npz")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chance = 1 / N_CAT

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    vmax = max(abs(tgm_mean.max()-chance), 0.001)
    im = ax.imshow(tgm_mean - chance, origin="lower",
                   extent=[times[0], times[-1], times[0], times[-1]],
                   aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.axhline(0, color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.plot([times[0],times[-1]], [times[0],times[-1]], "k-", lw=0.5, alpha=0.4)
    ax.set_xlabel("Test time (ms)")
    ax.set_ylabel("Train time (ms)")
    ax.set_title(f"Alignment TGM (Procrustes)\nn={len(pairs)} pairs, accuracy − chance")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    ax.plot(times, np.diag(tgm_mean)*100, color="#1f77b4", lw=2, label="Diagonal (T=T')")
    ax.plot(times, tgm_mean.mean(0)*100, color="#ff7f0e", lw=1.5, ls="--",
            label="Row mean (generalization)")
    ax.axhline(chance*100, color="gray", ls=":", lw=1, label="Chance")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Test time (ms)")
    ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title("Diagonal vs generalization")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = Path("figures/figure3e_alignment_tgm.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
