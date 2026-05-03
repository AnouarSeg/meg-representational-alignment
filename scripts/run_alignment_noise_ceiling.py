"""Noise ceiling for cross-subject alignment (EEG1, n=48).

Split-half reliability: split each subject's trials into two halves,
compute condition means for each half, then measure how well half-A of
subject i can be aligned to half-B of subject i (upper bound — same subject).
Compare to cross-subject alignment (lower bound — different subjects).

Output: results/eeg1/alignment_noise_ceiling.npz
"""

from __future__ import annotations
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy.linalg import orthogonal_procrustes

DERIV   = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
RAW_DIR = Path("/Volumes/MEG/things-eeg1")
RESULTS = Path("results/eeg1")
N_SPLITS = 5   # number of random splits for stability
N_BOOT   = 500
MODAL_CH = 63


def load_epochs_split(subject: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load .fif.gz epochs and return two random half-split condition means.

    Returns (means_A, means_B) each (1854, n_times, n_ch) float32.
    """
    import mne
    epo_path = DERIV / subject / f"{subject}_eeg_epo.fif.gz"
    if not epo_path.exists():
        return None

    epochs = mne.read_epochs(str(epo_path), preload=True, verbose=False)
    data   = epochs.get_data()        # (n_trials, n_ch, n_times)
    codes  = epochs.events[:, 2]      # concept nrs (1-indexed)
    n_times = data.shape[2]
    n_ch    = data.shape[1]

    rng = np.random.default_rng(42)
    means_A = np.zeros((1854, n_times, n_ch), dtype=np.float32)
    means_B = np.zeros((1854, n_times, n_ch), dtype=np.float32)

    for cnr in np.unique(codes):
        idx = np.where(codes == cnr)[0]
        rng.shuffle(idx)
        half = max(1, len(idx) // 2)
        idxA, idxB = idx[:half], idx[half:2*half]
        i = int(cnr) - 1
        if 0 <= i < 1854:
            if len(idxA):
                means_A[i] = data[idxA].mean(0).T.astype(np.float32)
            if len(idxB):
                means_B[i] = data[idxB].mean(0).T.astype(np.float32)

    return means_A, means_B


def transfer_accuracy_procrustes(A_tr, B_tr, A_te, B_te) -> float:
    try:
        R, _ = orthogonal_procrustes(A_tr, B_tr)
    except np.linalg.LinAlgError:
        R = np.eye(A_tr.shape[1])
    pred = A_te @ R
    n = len(pred)
    correct = sum(np.argmin(np.linalg.norm(pred[i] - B_te, axis=1)) == i for i in range(n))
    return correct / n


def within_subject_ceiling(subject: str, n_times: int) -> np.ndarray | None:
    """Upper bound: align half-A to half-B of the same subject."""
    result = load_epochs_split(subject)
    if result is None:
        return None
    A, B = result

    N_FOLDS = 5
    N_CAT   = 1854
    fold_size = N_CAT // N_FOLDS
    folds = [np.arange(k*fold_size, (k+1)*fold_size) for k in range(N_FOLDS)]

    acc = np.zeros(n_times)
    for t in range(n_times):
        At, Bt = A[:, t, :], B[:, t, :]
        fold_acc = 0.0
        for fi in range(N_FOLDS):
            te = folds[fi]
            tr = np.concatenate([folds[k] for k in range(N_FOLDS) if k != fi])
            fold_acc += transfer_accuracy_procrustes(At[tr], Bt[tr], At[te], Bt[te])
        acc[t] = fold_acc / N_FOLDS
        if t % 20 == 0:
            print(f"    t={t}/{n_times}", end="\r")
    return acc


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Find 63-ch subjects
    available = sorted(
        p.name for p in DERIV.glob("sub-*/")
        if (DERIV / p.name / f"{p.name}_condition_means.npy").exists()
    )
    subjects_63 = []
    for s in available:
        arr = np.load(DERIV / s / f"{s}_condition_means.npy", mmap_mode="r")
        if arr.shape[2] == MODAL_CH:
            subjects_63.append(s)
    print(f"63-ch subjects: {len(subjects_63)}")

    # Load cross-subject results for reference
    cs = np.load(RESULTS / "alignment_complexity_eeg1.npz")
    proc_cs = cs["procrustes_acc"]   # (n_pairs, n_times)
    n_times  = proc_cs.shape[1]
    times    = np.linspace(-50, 495, n_times)

    # Compute within-subject ceiling for a subset (10 subjects — loading epochs is heavy)
    # Use first 10 subjects with 63 ch
    ceiling_subjects = subjects_63[:10]
    print(f"Computing within-subject ceiling for {len(ceiling_subjects)} subjects...")

    ceiling_accs = []
    for s in ceiling_subjects:
        print(f"  {s}...")
        acc = within_subject_ceiling(s, n_times)
        if acc is not None:
            ceiling_accs.append(acc)
            print(f"    peak: {acc.max()*100:.3f}%")

    ceiling_accs = np.array(ceiling_accs)   # (n_ceiling_subs, n_times)

    # Cross-subject mean (lower bound)
    cs_mean = proc_cs.mean(0)
    ceiling_mean = ceiling_accs.mean(0)

    # Bootstrap CIs
    rng = np.random.default_rng(0)
    cs_boot = np.array([proc_cs[rng.integers(0, len(proc_cs), len(proc_cs))].mean(0)
                        for _ in range(N_BOOT)])
    ceil_boot = np.array([ceiling_accs[rng.integers(0, len(ceiling_accs), len(ceiling_accs))].mean(0)
                          for _ in range(N_BOOT)])

    np.savez(str(RESULTS / "alignment_noise_ceiling.npz"),
             times=times,
             cross_subject_mean=cs_mean,
             ceiling_mean=ceiling_mean,
             ceiling_accs=ceiling_accs,
             cross_subject_lo=np.percentile(cs_boot, 2.5, axis=0),
             cross_subject_hi=np.percentile(cs_boot, 97.5, axis=0),
             ceiling_lo=np.percentile(ceil_boot, 2.5, axis=0),
             ceiling_hi=np.percentile(ceil_boot, 97.5, axis=0))

    post = times > 0
    pct = cs_mean[post].max() / ceiling_mean[post].max() * 100
    print(f"\nCross-subject peak: {cs_mean.max()*100:.3f}%")
    print(f"Within-subject ceiling peak: {ceiling_mean.max()*100:.3f}%")
    print(f"Cross-subject / ceiling: {pct:.1f}%")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(times,
                    np.percentile(ceil_boot, 2.5, axis=0)*100,
                    np.percentile(ceil_boot, 97.5, axis=0)*100,
                    alpha=0.15, color="gray")
    ax.plot(times, ceiling_mean*100, color="gray", lw=2, ls="--",
            label=f"Within-subject ceiling (n={len(ceiling_accs)})")
    ax.fill_between(times,
                    np.percentile(cs_boot, 2.5, axis=0)*100,
                    np.percentile(cs_boot, 97.5, axis=0)*100,
                    alpha=0.2, color="#1f77b4")
    ax.plot(times, cs_mean*100, color="#1f77b4", lw=2,
            label=f"Cross-subject Procrustes (n=1128 pairs)")
    ax.axhline(1/1854*100, color="k", ls=":", lw=1, label="Chance")
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Transfer accuracy (%)")
    ax.set_title("Alignment noise ceiling: cross-subject vs. within-subject (EEG1)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/figure3c_alignment_noise_ceiling.png", dpi=150, bbox_inches="tight")
    print("Saved figures/figure3c_alignment_noise_ceiling.png")
