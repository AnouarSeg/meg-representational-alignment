"""PCA dimensionality reduction before alignment (EEG1).

Tests whether the complexity null is due to low EEG dimensionality (63ch).
Hypothesis: projecting to k << 63 dimensions reduces noise and may reveal
an alignment complexity gradient (ridge > Procrustes).

For each k in [5, 10, 20, 30, 50, 63]:
  - PCA on concatenated condition means (fit on train, apply to test)
  - Procrustes and ridge transfer accuracy (5-fold CV)
  - Complexity = ridge - Procrustes

Memory: 10 pairs × 2 subjects × 1854 × 110 × 63 × 4B ≈ 103 MB peak — safe.

Output: results/eeg1/alignment_pca.npz
        figures/figure3f_alignment_pca.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.linalg import orthogonal_procrustes

DERIV    = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
RESULTS  = Path("results/eeg1")
K_VALUES = [5, 10, 20, 30, 50, 63]
N_PAIRS  = 20
N_FOLDS  = 5
N_CAT    = 1854
MODAL_CH = 63
RIDGE_A  = 1.0


def transfer_accuracy(Y_pred: np.ndarray, Y_test: np.ndarray) -> float:
    n = len(Y_pred)
    return sum(np.argmin(np.linalg.norm(Y_pred[i] - Y_test, axis=1)) == i
               for i in range(n)) / n


def run_pair_pca(A: np.ndarray, B: np.ndarray, k: int) -> dict:
    """Alignment complexity at all timepoints after projecting to k PCs.

    A, B: (N_CAT, n_times, n_ch)
    """
    n_times = A.shape[1]
    fold_size = N_CAT // N_FOLDS
    folds = [np.arange(f*fold_size, (f+1)*fold_size) for f in range(N_FOLDS)]

    proc_acc = np.zeros(n_times)
    ridg_acc = np.zeros(n_times)

    for t in range(n_times):
        At = A[:, t, :]   # (N_CAT, n_ch)
        Bt = B[:, t, :]

        p_acc = r_acc = 0.0
        for fi in range(N_FOLDS):
            te = folds[fi]
            tr = np.concatenate([folds[j] for j in range(N_FOLDS) if j != fi])

            # Fit PCA on training data from both subjects combined
            pca = PCA(n_components=k)
            pca.fit(np.concatenate([At[tr], Bt[tr]], axis=0))

            A_tr = pca.transform(At[tr])
            A_te = pca.transform(At[te])
            B_tr = pca.transform(Bt[tr])
            B_te = pca.transform(Bt[te])

            # Procrustes
            try:
                R, _ = orthogonal_procrustes(A_tr, B_tr)
                p_acc += transfer_accuracy(A_te @ R, B_te)
            except np.linalg.LinAlgError:
                p_acc += transfer_accuracy(A_te, B_te)

            # Ridge
            reg = Ridge(alpha=RIDGE_A, fit_intercept=False)
            reg.fit(A_tr, B_tr)
            r_acc += transfer_accuracy(reg.predict(A_te), B_te)

        proc_acc[t] = p_acc / N_FOLDS
        ridg_acc[t] = r_acc / N_FOLDS

    return {"procrustes": proc_acc, "ridge": ridg_acc,
            "complexity": ridg_acc - proc_acc}


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    available = sorted(
        p.name for p in DERIV.glob("sub-*/")
        if (DERIV / p.name / f"{p.name}_condition_means.npy").exists()
    )
    subjects = [s for s in available
                if np.load(DERIV/s/f"{s}_condition_means.npy",
                           mmap_mode="r").shape[2] == MODAL_CH]

    from itertools import combinations
    all_pairs = list(combinations(subjects, 2))
    rng = np.random.default_rng(0)
    pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), N_PAIRS, replace=False)]
    print(f"Running {len(pairs)} pairs × {len(K_VALUES)} k values")

    results = {k: [] for k in K_VALUES}   # k → list of (proc, ridge, complexity) arrays

    for pi, (sA, sB) in enumerate(pairs):
        A = np.load(DERIV/sA/f"{sA}_condition_means.npy")
        B = np.load(DERIV/sB/f"{sB}_condition_means.npy")
        print(f"Pair {pi+1}/{len(pairs)}: {sA} × {sB}")

        for k in K_VALUES:
            r = run_pair_pca(A, B, k)
            results[k].append(r)
            post = np.linspace(-50, 495, A.shape[1]) > 0
            print(f"  k={k:2d}: proc={np.array([r['procrustes'][post].max() for r in results[k]]).mean()*100:.3f}%  "
                  f"complexity={np.array([r['complexity'][post].mean() for r in results[k]]).mean()*100:.4f}%")

        del A, B

    times = np.linspace(-50, 495, 110)
    post  = times > 0

    save_dict = {"times": times, "k_values": np.array(K_VALUES)}
    for k in K_VALUES:
        proc = np.array([r["procrustes"] for r in results[k]])
        ridg = np.array([r["ridge"]      for r in results[k]])
        comp = np.array([r["complexity"] for r in results[k]])
        save_dict[f"proc_k{k}"] = proc
        save_dict[f"ridg_k{k}"] = ridg
        save_dict[f"comp_k{k}"] = comp

    np.savez(str(RESULTS / "alignment_pca.npz"), **save_dict)

    print("\n=== PCA Alignment Results ===")
    print(f"{'k':>4}  {'Proc peak':>10}  {'Ridge peak':>10}  {'Post-stim complexity':>22}")
    for k in K_VALUES:
        proc = np.array([r["procrustes"] for r in results[k]]).mean(0)
        ridg = np.array([r["ridge"]      for r in results[k]]).mean(0)
        comp = np.array([r["complexity"] for r in results[k]])
        print(f"{k:>4}  {proc[post].max()*100:>10.3f}%  {ridg[post].max()*100:>10.3f}%  "
              f"{comp[:,post].mean()*100:>+10.4f}% ± {comp[:,post].std()*100:.4f}%")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = viridis(np.linspace(0.15, 0.9, len(K_VALUES)))

    ax = axes[0]
    for ki, k in enumerate(K_VALUES):
        proc_m = np.array([r["procrustes"] for r in results[k]]).mean(0)
        ridg_m = np.array([r["ridge"]      for r in results[k]]).mean(0)
        ax.plot(times, proc_m*100, color=colors[ki], lw=2, label=f"k={k} Proc")
        ax.plot(times, ridg_m*100, color=colors[ki], lw=1.5, ls="--")
    ax.axhline(1/N_CAT*100, color="gray", ls=":", lw=1, label="Chance")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title("Procrustes (solid) vs Ridge (dashed)\nby PCA dimension k")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for ki, k in enumerate(K_VALUES):
        comp_m = np.array([r["complexity"] for r in results[k]]).mean(0)
        ax.plot(times, comp_m*100, color=colors[ki], lw=2, label=f"k={k}")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Complexity: ridge − Procrustes (%)")
    ax.set_title("Alignment complexity by PCA dimension")
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = Path("figures/figure3f_alignment_pca.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
