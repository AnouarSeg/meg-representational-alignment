"""Cross-subject alignment complexity for THINGS-EEG1 (up to 50 subjects).

Same Procrustes vs. ridge pipeline as run_alignment.py but operating on
EEG1 condition means. With 50 subjects → 1225 pairs → overwhelming power.

Input:  /Volumes/MEG/things-eeg1/derivatives/preprocessed/sub-*/sub-*_condition_means.npy
Output: results/eeg1/alignment_complexity_eeg1.npz

Key difference from MEG version:
- 1854 categories (vs. 100 in MEG version — more robust geometry)
- 63 channels (vs. 272) — less risk of overfitting ridge
- RSVP paradigm → time axis is 0–500 ms from image onset within 10 Hz stream
  (pre-stim contaminated by previous image)

Memory: 50 subjects × 1854 × n_times × 63 × 4 bytes ≈ load one pair at a time.
"""

from __future__ import annotations
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from scipy.linalg import orthogonal_procrustes

DERIV   = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
RESULTS = Path("results/eeg1")

N_CAT    = 1854
N_FOLDS  = 5
RIDGE_A  = 1.0
MAX_SUBJ = 50   # cap; use all available


def procrustes_transfer(X_train, Y_train, X_test):
    """Fit orthogonal Procrustes on train, evaluate on test."""
    try:
        R, _ = orthogonal_procrustes(X_train, Y_train)
    except np.linalg.LinAlgError:
        # SVD non-convergence on degenerate data — fall back to identity
        R = np.eye(X_train.shape[1])
    Y_pred = X_test @ R
    return Y_pred


def ridge_transfer(X_train, Y_train, X_test):
    """Fit ridge regression on train, evaluate on test."""
    reg = Ridge(alpha=RIDGE_A, fit_intercept=False)
    reg.fit(X_train, Y_train)
    return reg.predict(X_test)


def transfer_accuracy(Y_pred, Y_test):
    """Rank-based accuracy: how often is the true item closest to its prediction."""
    n = len(Y_pred)
    correct = 0
    for i in range(n):
        dists = np.linalg.norm(Y_pred[i] - Y_test, axis=1)
        correct += (np.argmin(dists) == i)
    return correct / n


def run_pair(A: np.ndarray, B: np.ndarray) -> dict:
    """Run alignment complexity analysis for one subject pair.

    Parameters
    ----------
    A, B : (N_CAT, n_times, n_ch) float32

    Returns
    -------
    dict with keys: procrustes_acc, ridge_acc, complexity — each (n_times,)
    """
    n_times = A.shape[1]
    proc_acc = np.zeros(n_times)
    ridg_acc = np.zeros(n_times)

    fold_size = N_CAT // N_FOLDS
    folds = [np.arange(k * fold_size, (k + 1) * fold_size) for k in range(N_FOLDS)]

    for t in range(n_times):
        At = A[:, t, :]   # (N_CAT, n_ch)
        Bt = B[:, t, :]

        p_acc, r_acc = 0.0, 0.0
        for fold_idx in range(N_FOLDS):
            test_idx  = folds[fold_idx]
            train_idx = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fold_idx])

            X_tr, X_te = At[train_idx], At[test_idx]
            Y_tr, Y_te = Bt[train_idx], Bt[test_idx]

            p_acc += transfer_accuracy(procrustes_transfer(X_tr, Y_tr, X_te), Y_te)
            r_acc += transfer_accuracy(ridge_transfer(X_tr, Y_tr, X_te), Y_te)

        proc_acc[t] = p_acc / N_FOLDS
        ridg_acc[t] = r_acc / N_FOLDS

        if t % 20 == 0:
            print(f"    t={t}/{n_times}  proc={proc_acc[t]:.3f}  ridge={ridg_acc[t]:.3f}", end="\r")

    return {"procrustes_acc": proc_acc, "ridge_acc": ridg_acc,
            "complexity": ridg_acc - proc_acc}


def load_means(subject: str) -> np.ndarray | None:
    p = DERIV / subject / f"{subject}_condition_means.npy"
    if not p.exists():
        return None
    return np.load(str(p))   # (1854, n_times, n_ch)


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Find all preprocessed subjects
    available = sorted(p.name for p in DERIV.glob("sub-*/") if
                       (DERIV / p.name / f"{p.name}_condition_means.npy").exists())
    print(f"Found {len(available)} preprocessed EEG1 subjects: {available}")

    if len(available) < 2:
        print("Need at least 2 subjects. Run preprocess_eeg1.py first.")
        sys.exit(1)

    subjects = available[:MAX_SUBJ]
    pairs    = list(combinations(subjects, 2))
    print(f"Running {len(pairs)} pairs ({len(subjects)} subjects)")

    # Filter to subjects with the modal channel count (handles outlier caps)
    ch_counts = {}
    for s in subjects:
        p = DERIV / s / f"{s}_condition_means.npy"
        ch_counts[s] = np.load(str(p), mmap_mode="r").shape[2]
    modal_ch = max(set(ch_counts.values()), key=list(ch_counts.values()).count)
    subjects = [s for s in subjects if ch_counts[s] == modal_ch]
    pairs    = list(combinations(subjects, 2))
    print(f"After channel filter ({modal_ch} ch): {len(subjects)} subjects, {len(pairs)} pairs")

    # Load all condition means into memory
    # subjects × 1854 × 110 timepoints × n_ch × 4 bytes — load lazily
    CHECKPOINT = RESULTS / "alignment_complexity_eeg1_checkpoint.npz"
    all_results = []
    pair_labels = []

    # Resume from checkpoint if it exists
    start_pair = 0
    if CHECKPOINT.exists():
        ck = np.load(str(CHECKPOINT), allow_pickle=True)
        pair_labels = list(ck["pair_labels"])
        proc_ck = ck["procrustes_acc"]
        ridg_ck = ck["ridge_acc"]
        for k in range(len(pair_labels)):
            all_results.append({
                "procrustes_acc": proc_ck[k],
                "ridge_acc":      ridg_ck[k],
                "complexity":     ridg_ck[k] - proc_ck[k],
            })
        start_pair = len(pair_labels)
        print(f"Resuming from checkpoint: {start_pair} pairs already done")

    for i, (sA, sB) in enumerate(pairs):
        if i < start_pair:
            continue
        print(f"\nPair {i+1}/{len(pairs)}: {sA} × {sB}")
        A = load_means(sA)
        B = load_means(sB)
        if A is None or B is None:
            print(f"  Skipping — missing data")
            continue

        result = run_pair(A, B)
        all_results.append(result)
        pair_labels.append(f"{sA}-{sB}")
        del A, B

        # Save checkpoint every 50 pairs
        if len(all_results) % 50 == 0:
            n_t = len(all_results[0]["procrustes_acc"])
            np.savez(str(CHECKPOINT),
                     procrustes_acc=np.array([r["procrustes_acc"] for r in all_results]),
                     ridge_acc=np.array([r["ridge_acc"] for r in all_results]),
                     pair_labels=np.array(pair_labels))
            print(f"  [checkpoint saved: {len(all_results)} pairs]")

    if not all_results:
        print("No pairs processed.")
        sys.exit(1)

    n_times = len(all_results[0]["procrustes_acc"])
    proc_all = np.array([r["procrustes_acc"] for r in all_results])
    ridg_all = np.array([r["ridge_acc"]      for r in all_results])
    comp_all = np.array([r["complexity"]      for r in all_results])

    out = RESULTS / "alignment_complexity_eeg1.npz"
    np.savez(str(out),
             procrustes_acc=proc_all,
             ridge_acc=ridg_all,
             complexity=comp_all,
             pair_labels=np.array(pair_labels),
             n_subjects=len(subjects))
    print(f"\nSaved {out}")
    print(f"Post-stim complexity: {comp_all.mean():.4f} ± {comp_all.std():.4f}")
    print(f"Procrustes peak: {proc_all.mean(0).max()*100:.2f}%")
