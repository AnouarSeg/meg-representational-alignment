"""Cross-subject alignment complexity analysis (Weeks 4-5, Figures 2-3).

Core scientific question: does the complexity of the linear map needed to align
subjects increase over post-stimulus time, mirroring a processing hierarchy?

Approach:
  At each timepoint, for each subject pair (A, B):
    1. Fit Procrustes (constrained: orthogonal) and Ridge (unconstrained linear)
       from A's condition means to B's condition means, on a training split.
    2. Measure transfer accuracy on held-out conditions.
    3. Complexity = Ridge gain over Procrustes.
       Near zero → early: rigid rotation suffices (simple, shared geometry)
       Positive  → late:  flexible map needed (complex, idiosyncratic geometry)

Figure 2: transfer accuracy over time (with vs without alignment)
Figure 3: alignment complexity over time (headline figure)

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_alignment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.alignment import ProcrustesAligner, williams_shape_distance
from thingsmeg.config import load_config

mne.set_log_level("WARNING")

cfg = load_config()
subjects = cfg.raw["dataset"]["subjects"]
deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

N_CATS = 100
N_FOLDS = 5
RIDGE_ALPHA = 1.0

# ── Load / cache condition means ──────────────────────────────────────────────
means_cache = results_dir / "condition_means.npz"
subject_data = {}
times_ms = None

if means_cache.exists():
    print("Loading cached condition means…")
    cache = np.load(str(means_cache), allow_pickle=True)
    times_ms = cache["times_ms"]
    for subject in subjects:
        if subject in cache:
            subject_data[subject] = cache[subject]
            print(f"  [{subject}] {subject_data[subject].shape}")
else:
    for subject in subjects:
        epochs_path = deriv_dir / f"sub-{subject}" / "epochs.fif"
        labels_path = deriv_dir / f"sub-{subject}" / "labels.npy"
        if not epochs_path.exists():
            print(f"[SKIP] {subject}")
            continue
        print(f"[{subject}] Loading epochs…")
        epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
        labels = np.load(str(labels_path))
        if times_ms is None:
            times_ms = epochs.times * 1000
        cats, counts = np.unique(labels, return_counts=True)
        top_cats = np.sort(cats[np.argsort(counts)[-N_CATS:]])
        X = epochs.get_data()
        cat_means = np.stack([X[labels == c].mean(axis=0) for c in top_cats])
        mu = cat_means.mean(axis=0, keepdims=True)
        sd = cat_means.std(axis=0, keepdims=True) + 1e-10
        subject_data[subject] = (cat_means - mu) / sd
        print(f"  [{subject}] {subject_data[subject].shape}")
        del epochs, X
    np.savez(str(means_cache), times_ms=times_ms, **subject_data)
    print(f"Cached to {means_cache}")

n_times = next(iter(subject_data.values())).shape[2]
subjects_avail = list(subject_data.keys())
pairs = [(i, j) for i in range(len(subjects_avail)) for j in range(i + 1, len(subjects_avail))]
pair_labels = [f"{subjects_avail[i]}-{subjects_avail[j]}" for i, j in pairs]
n_pairs = len(pairs)

# ── Per-timepoint analysis ────────────────────────────────────────────────────
# Results arrays: (n_pairs, n_times)
shape_dist      = np.zeros((n_pairs, n_times))
procrustes_acc  = np.zeros((n_pairs, n_times))
ridge_acc       = np.zeros((n_pairs, n_times))
complexity      = np.zeros((n_pairs, n_times))  # ridge_acc - procrustes_acc
ridge_eff_rank  = np.zeros((n_pairs, n_times))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

print(f"\nRunning alignment analysis: {n_pairs} pairs × {n_times} timepoints…")
for t in range(n_times):
    if t % 20 == 0:
        print(f"  t={t}/{n_times} ({times_ms[t]:.0f} ms)")

    for pi, (i, j) in enumerate(pairs):
        sa, sb = subjects_avail[i], subjects_avail[j]
        A = subject_data[sa][:, :, t]  # (N_CATS, n_ch)
        B = subject_data[sb][:, :, t]

        shape_dist[pi, t] = williams_shape_distance(A, B)

        # Cross-validated transfer accuracy
        p_accs, r_accs, r_ranks = [], [], []
        for train_idx, test_idx in kf.split(A):
            A_tr, A_te = A[train_idx], A[test_idx]
            B_tr, B_te = B[train_idx], B[test_idx]

            # Procrustes: orthogonal map A_tr → B_tr
            proc = ProcrustesAligner().fit(A_tr, B_tr)
            A_te_proc = proc.transform(A_te)
            # Accuracy: nearest-neighbour in B space (cosine-like)
            dists_proc = np.linalg.norm(A_te_proc[:, None, :] - B_te[None, :, :], axis=2)
            p_accs.append(np.mean(np.argmin(dists_proc, axis=1) == np.arange(len(test_idx))))

            # Ridge: unconstrained linear map A_tr → B_tr
            scaler = StandardScaler()
            A_tr_s = scaler.fit_transform(A_tr)
            A_te_s = scaler.transform(A_te)
            ridge = Ridge(alpha=RIDGE_ALPHA)
            ridge.fit(A_tr_s, B_tr)
            A_te_ridge = ridge.predict(A_te_s)
            dists_ridge = np.linalg.norm(A_te_ridge[:, None, :] - B_te[None, :, :], axis=2)
            r_accs.append(np.mean(np.argmin(dists_ridge, axis=1) == np.arange(len(test_idx))))

            # Effective rank of ridge weight matrix
            svs = np.linalg.svd(ridge.coef_, compute_uv=False)
            r_ranks.append(float(svs.sum() ** 2 / (svs ** 2).sum()) if svs.sum() > 0 else 0.0)

        procrustes_acc[pi, t] = np.mean(p_accs)
        ridge_acc[pi, t] = np.mean(r_accs)
        complexity[pi, t] = np.mean(r_accs) - np.mean(p_accs)
        ridge_eff_rank[pi, t] = np.mean(r_ranks)

np.savez(
    results_dir / "alignment_complexity.npz",
    times_ms=times_ms,
    pair_labels=pair_labels,
    shape_distance=shape_dist,
    procrustes_acc=procrustes_acc,
    ridge_acc=ridge_acc,
    complexity=complexity,
    ridge_eff_rank=ridge_eff_rank,
)
print("Saved results/alignment_complexity.npz")

def smooth(x, n=7):
    return np.convolve(x, np.ones(n) / n, mode="same")

# ── Figure 2: transfer accuracy over time ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mean_proc = procrustes_acc.mean(axis=0)
mean_ridge = ridge_acc.mean(axis=0)
sem_proc = procrustes_acc.std(axis=0) / np.sqrt(n_pairs)
sem_ridge = ridge_acc.std(axis=0) / np.sqrt(n_pairs)

ax = axes[0]
ax.plot(times_ms, smooth(mean_proc), color="steelblue", linewidth=2, label="Procrustes")
ax.fill_between(times_ms, smooth(mean_proc - sem_proc), smooth(mean_proc + sem_proc),
                color="steelblue", alpha=0.2)
ax.plot(times_ms, smooth(mean_ridge), color="darkorange", linewidth=2, label="Ridge")
ax.fill_between(times_ms, smooth(mean_ridge - sem_ridge), smooth(mean_ridge + sem_ridge),
                color="darkorange", alpha=0.2)
ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(1 / N_CATS, color="gray", linestyle="--", linewidth=1, label=f"Chance (1/{N_CATS})")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Cross-subject transfer accuracy (nearest-neighbour)")
ax.set_title("Cross-subject decoding after alignment")
ax.legend()

ax2 = axes[1]
mean_shape = shape_dist.mean(axis=0)
sem_shape = shape_dist.std(axis=0) / np.sqrt(n_pairs)
ax2.plot(times_ms, smooth(mean_shape), color="seagreen", linewidth=2)
ax2.fill_between(times_ms, smooth(mean_shape - sem_shape), smooth(mean_shape + sem_shape),
                 color="seagreen", alpha=0.2)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Williams shape distance (lower = more aligned)")
ax2.set_title("Cross-subject representational geometry similarity")

fig.suptitle("Figure 2: Cross-subject alignment over time", fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "figure2_alignment.png", dpi=150)
print("Saved figures/figure2_alignment.png")

# ── Figure 3: complexity over time (headline) ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

mean_comp = complexity.mean(axis=0)
sem_comp = complexity.std(axis=0) / np.sqrt(n_pairs)

ax = axes[0]
ax.plot(times_ms, smooth(mean_comp), color="crimson", linewidth=2.5)
ax.fill_between(times_ms, smooth(mean_comp - sem_comp), smooth(mean_comp + sem_comp),
                color="crimson", alpha=0.2)
ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Ridge gain over Procrustes (accuracy)")
ax.set_title("Alignment complexity over time\n(positive = flexible map needed beyond rotation)")

mean_rank = ridge_eff_rank.mean(axis=0)
sem_rank = ridge_eff_rank.std(axis=0) / np.sqrt(n_pairs)
ax2 = axes[1]
ax2.plot(times_ms, smooth(mean_rank), color="purple", linewidth=2.5)
ax2.fill_between(times_ms, smooth(mean_rank - sem_rank), smooth(mean_rank + sem_rank),
                 color="purple", alpha=0.2)
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Effective rank of alignment map")
ax2.set_title("Effective dimensionality of alignment\n(higher = richer transform needed)")

fig.suptitle("Figure 3: Does alignment complexity increase with time?", fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "figure3_complexity.png", dpi=150)
print("Saved figures/figure3_complexity.png")
