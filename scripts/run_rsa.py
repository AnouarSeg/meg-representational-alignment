"""Compute time-resolved crossnobis RDMs for all subjects (Week 3).

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_rsa.py

Outputs:
    results/rdms_{subject}.npz   — rdms (n_windows, n_cond, n_cond) + times
    figures/figure_rsa.png       — mean RDM at peak time + RDM structure over time
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.config import load_config
from thingsmeg.rsa import rdm_correlation, time_resolved_rdms

mne.set_log_level("WARNING")

cfg = load_config()
subjects = cfg.raw["dataset"]["subjects"]
deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Use top-100 categories (same as decoding) for tractable RDMs
N_CATS = 100

all_rdm_times = None
subject_rdms = {}  # subject -> (n_windows, n_cond, n_cond)

for subject in subjects:
    epochs_path = deriv_dir / f"sub-{subject}" / "epochs.fif"
    labels_path = deriv_dir / f"sub-{subject}" / "labels.npy"
    if not epochs_path.exists():
        print(f"[SKIP] {subject}")
        continue

    print(f"[{subject}] Loading…")
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    labels = np.load(str(labels_path))

    # Load run indices (120 runs); map to 12 sessions (10 runs/session)
    # to keep the crossnobis partition-pair loop at 66 iterations.
    run_ids = np.load(str(deriv_dir / f"sub-{subject}" / "runs.npy"))
    session_descriptor = run_ids // 10  # runs 0-9 → ses 0, 10-19 → ses 1, …

    # Subset to top-N_CATS categories
    cats, counts = np.unique(labels, return_counts=True)
    top_cats = cats[np.argsort(counts)[-N_CATS:]]
    mask = np.isin(labels, top_cats)
    X = epochs.get_data()[mask]            # (n_trials_sub, n_channels, n_times)
    y = labels[mask]
    cv = session_descriptor[mask]

    # Remap labels to 0..N_CATS-1
    label_map = {c: i for i, c in enumerate(sorted(top_cats))}
    y = np.array([label_map[c] for c in y])

    # Z-score across trials per channel per timepoint — removes scale differences
    # between subjects and makes crossnobis distances interpretable
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-10)

    print(f"[{subject}] Computing crossnobis RDMs ({X.shape[0]} trials, {N_CATS} categories)…")
    rdms, t_idx = time_resolved_rdms(X, y, session_descriptor[mask], cfg, metric="crossnobis")
    subject_rdms[subject] = rdms

    if all_rdm_times is None:
        sfreq = epochs.info["sfreq"]
        tmin = epochs.times[0]
        all_rdm_times = tmin + t_idx / sfreq  # seconds -> use as-is
        all_rdm_times_ms = all_rdm_times * 1000

    np.savez(
        results_dir / f"rdms_{subject}.npz",
        rdms=rdms,
        times_ms=all_rdm_times_ms,
        categories=sorted(top_cats),
    )
    print(f"[{subject}] Saved — RDMs shape: {rdms.shape}")

# ── Figure: RDM dissimilarity over time (mean off-diagonal per window) ────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: mean RDM structure over time (mean off-diagonal = "how distinct are reps")
ax = axes[0]
for subject, rdms in subject_rdms.items():
    idx = np.triu_indices(rdms.shape[1], k=1)
    mean_dissim = np.array([rdm[idx].mean() for rdm in rdms])
    # smooth
    mean_dissim = np.convolve(mean_dissim, np.ones(3)/3, mode="same")
    ax.plot(all_rdm_times_ms, mean_dissim, linewidth=1.5, label=subject)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Mean crossnobis distance")
ax.set_title("RDM dissimilarity over time")
ax.legend(fontsize=9)

# Panel 2: mean RDM at the peak window across subjects
peak_t_idx = np.argmax([
    np.mean([subject_rdms[s][t].mean() for s in subject_rdms])
    for t in range(len(all_rdm_times_ms))
])
mean_rdm_at_peak = np.mean([subject_rdms[s][peak_t_idx] for s in subject_rdms], axis=0)

ax2 = axes[1]
im = ax2.imshow(mean_rdm_at_peak, cmap="RdBu_r", aspect="auto")
ax2.set_title(f"Mean RDM at {all_rdm_times_ms[peak_t_idx]:.0f} ms (averaged across subjects)")
ax2.set_xlabel("Category")
ax2.set_ylabel("Category")
plt.colorbar(im, ax=ax2, label="Crossnobis distance")

fig.tight_layout()
fig.savefig(figures_dir / "figure_rsa.png", dpi=150)
print("Saved figures/figure_rsa.png")
