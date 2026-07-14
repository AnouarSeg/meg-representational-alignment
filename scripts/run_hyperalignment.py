"""Hyperalignment vs Procrustes cross-subject alignment (EEG1, n=48).

Hyperalignment (Haxby et al. 2011): iterative Procrustes to a common reference.
At each iteration, each subject is aligned to the current group mean (reference),
then the reference is updated. Converges in ~10 iterations.

This is the classical method and often outperforms pairwise Procrustes by finding
a better shared representational space rather than independent pairwise maps.

Uses EEG1 (n=48) for statistical power. Runs sequentially, low memory.

Output:
  results/eeg1/hyperalignment_complexity.npz
  figures/figure_hyperalignment.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

RESULTS   = Path("results/eeg1")
EEG1_DIR  = Path("data/derivatives/eeg1")
FIGS      = Path("figures")
CV_FOLDS  = 5
ALPHA     = 1.0
N_ITER    = 10        # hyperalignment iterations
MAX_PAIRS = 200       # sample for pairwise comparison
RNG       = np.random.default_rng(42)


def procrustes(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def ridge(X, Y, alpha=ALPHA):
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def transfer_acc(X_tr, Y_tr, X_te, Y_te, W):
    pred = X_te @ W
    d    = np.sum((pred[:, None] - Y_te[None]) ** 2, axis=-1)
    return (d.argmin(1) == np.arange(len(X_te))).mean() * 100


def hyperalign(data_list, n_iter=N_ITER):
    """
    data_list: list of (n_cat, n_features) arrays — one per subject.
    Returns: list of aligned arrays in common space.
    """
    # Initialise reference as first subject
    aligned = [d.copy() for d in data_list]
    for it in range(n_iter):
        reference = np.mean(aligned, axis=0)
        reference /= (np.linalg.norm(reference, axis=1, keepdims=True) + 1e-12)
        new_aligned = []
        for d in aligned:
            R = procrustes(d, reference)
            new_aligned.append(d @ R)
        prev_ref = reference
        aligned  = new_aligned
        reference = np.mean(aligned, axis=0)
        # Check convergence
        change = np.linalg.norm(reference - prev_ref) / (np.linalg.norm(prev_ref) + 1e-12)
        if change < 1e-4:
            print(f"    Converged at iteration {it+1}")
            break
    return aligned


def load_eeg1_subjects():
    """Load EEG1 condition means. Returns dict sub_id → (n_cat, n_times, n_ch)."""
    import glob
    files = sorted(f for f in EEG1_DIR.glob("sub-*_condition_means.npy")
                   if not Path(f).name.startswith("._"))
    if not files:
        # Try loading from alignment results
        f = RESULTS / "alignment_complexity_eeg1.npz"
        if f.exists():
            return None, np.load(str(f), allow_pickle=True)["times"]
        raise FileNotFoundError(f"No EEG1 condition means in {EEG1_DIR}")
    subjects = {}
    times    = None
    for f in files:
        sub = Path(f).stem.replace("_condition_means", "")
        d   = np.load(str(f)).astype(np.float32)
        if d.ndim == 3 and d.shape[1] < d.shape[2]:
            d = d.transpose(0, 2, 1)   # → (n_cat, n_times, n_ch)
        subjects[sub] = d
        if times is None:
            t_file = EEG1_DIR / f"{sub}_times.npy"
            times  = np.load(str(t_file)) if t_file.exists() else np.linspace(-50, 495, d.shape[1])
    return subjects, np.array(times)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("=== Hyperalignment vs Procrustes ===\n")

    subjects, times = load_eeg1_subjects()
    if subjects is None:
        print("EEG1 condition means not found — cannot run hyperalignment.")
        print("Run batch_preprocess_and_align.sh first to generate condition means.")
        import sys; sys.exit(0)

    n_times = len(times)
    sub_list = list(subjects.keys())
    n_sub    = len(sub_list)
    print(f"Subjects: {n_sub}, timepoints: {n_times}")

    n_cat = min(d.shape[0] for d in subjects.values())
    n_ch  = min(d.shape[2] for d in subjects.values())
    print(f"Categories: {n_cat}, channels: {n_ch}")

    # 5-fold CV: split categories into train/test
    idx       = np.arange(n_cat)
    fold_size = n_cat // CV_FOLDS
    post      = times > 0

    hyper_all = np.zeros((CV_FOLDS, n_times))
    proc_all  = np.zeros((CV_FOLDS, n_times))

    for fold in range(CV_FOLDS):
        te_idx = idx[fold * fold_size:(fold + 1) * fold_size]
        tr_idx = np.setdiff1d(idx, te_idx)

        print(f"\nFold {fold+1}/{CV_FOLDS} — train: {len(tr_idx)}, test: {len(te_idx)}")

        hyper_ts = np.zeros(n_times)
        proc_ts  = np.zeros(n_times)
        n_pairs  = 0

        # Sample subject pairs
        from itertools import combinations
        pairs = list(combinations(range(n_sub), 2))
        if len(pairs) > MAX_PAIRS:
            sel = RNG.choice(len(pairs), MAX_PAIRS, replace=False)
            pairs = [pairs[i] for i in sel]

        for t in range(n_times):
            # Build training data matrices for hyperalignment
            train_mats = [subjects[sub][:n_cat, t, :n_ch][tr_idx]
                          for sub in sub_list]
            # Z-score each subject
            train_mats = [(m - m.mean(0)) / (m.std(0) + 1e-12) for m in train_mats]

            # Hyperalign training data
            aligned_tr = hyperalign(train_mats, n_iter=N_ITER)

            # For each pair: compute Procrustes and hyperalignment transfer acc on test set
            for i, j in pairs:
                A_tr = train_mats[i];  A_te_raw = subjects[sub_list[i]][:n_cat, t, :n_ch][te_idx]
                B_tr = train_mats[j];  B_te_raw = subjects[sub_list[j]][:n_cat, t, :n_ch][te_idx]

                # Z-score test using train stats
                mu_a = subjects[sub_list[i]][:n_cat, t, :n_ch][tr_idx].mean(0)
                sd_a = subjects[sub_list[i]][:n_cat, t, :n_ch][tr_idx].std(0) + 1e-12
                mu_b = subjects[sub_list[j]][:n_cat, t, :n_ch][tr_idx].mean(0)
                sd_b = subjects[sub_list[j]][:n_cat, t, :n_ch][tr_idx].std(0) + 1e-12
                A_te = (A_te_raw - mu_a) / sd_a
                B_te = (B_te_raw - mu_b) / sd_b

                # Pairwise Procrustes (standard)
                R_p   = procrustes(A_tr, B_tr)
                proc_ts[t] += transfer_acc(A_tr, B_tr, A_te, B_te, R_p)

                # Hyperalignment: map i and j to common space, then 1-NN
                A_hyp_tr = aligned_tr[i];  B_hyp_tr = aligned_tr[j]
                # Get test transform: align test data using train hyperalignment maps
                # Proxy: reuse the train Procrustes to common space
                ref = np.mean(aligned_tr, axis=0)
                R_ha = procrustes(A_tr, ref)
                R_hb = procrustes(B_tr, ref)
                A_te_h = A_te @ R_ha
                B_te_h = B_te @ R_hb
                d = np.sum((A_te_h[:, None] - B_te_h[None]) ** 2, axis=-1)
                hyper_ts[t] += (d.argmin(1) == np.arange(len(te_idx))).mean() * 100
                n_pairs += 1

            if t % 30 == 0:
                print(f"  t={t} ({times[t]:.0f}ms): hyper={hyper_ts[t]/len(pairs):.3f}%  proc={proc_ts[t]/len(pairs):.3f}%")

        hyper_all[fold] = hyper_ts / len(pairs)
        proc_all[fold]  = proc_ts  / len(pairs)

    hyper_mean = hyper_all.mean(0)
    proc_mean  = proc_all.mean(0)
    complexity = hyper_mean - proc_mean

    print(f"\n=== Results ===")
    print(f"Procrustes peak: {proc_mean[post].max():.3f}% at {times[post][proc_mean[post].argmax()]:.0f}ms")
    print(f"Hyperalign peak: {hyper_mean[post].max():.3f}% at {times[post][hyper_mean[post].argmax()]:.0f}ms")
    print(f"Complexity (hyper-proc, post mean): {complexity[post].mean():.4f}%")

    np.savez(str(RESULTS / "hyperalignment_complexity.npz"),
             times=times, procrustes=proc_mean, hyperalign=hyper_mean,
             complexity=complexity, n_subjects=n_sub, n_pairs=len(pairs))
    print("Saved results/eeg1/hyperalignment_complexity.npz")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ax = axes[0]
    ax.plot(times, proc_mean,  "#2980b9", lw=2, label=f"Procrustes (peak {proc_mean[post].max():.3f}%)")
    ax.plot(times, hyper_mean, "#e74c3c", lw=2, label=f"Hyperalign (peak {hyper_mean[post].max():.3f}%)")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title(f"Hyperalignment vs Procrustes (EEG1, n={n_sub})")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.plot(times, complexity, "k", lw=2, label="Hyper − Procrustes")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Δ accuracy (%)")
    ax.set_title("Hyperalignment gain over Procrustes")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(str(FIGS / "figure_hyperalignment.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure_hyperalignment.png")
