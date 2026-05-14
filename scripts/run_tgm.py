"""Temporal Generalization Matrix (King & Dehaene 2014).

Train at time T, test at time T' — reveals whether representations are transient
(diagonal only) or sustained (off-diagonal bands).

Memory budget: one subject at a time, 100-category subsample.
  1200 trials × 272 ch × 180 times × float32 = ~236 MB peak per subject.
  TGM output per subject: 180 × 180 × float32 = 126 KB — tiny.

Output: results/tgm_scores.npz
        figures/figure1b_tgm.png
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config
from thingsmeg.decoding import temporal_generalization

mne.set_log_level("WARNING")

cfg        = load_config()
subjects   = cfg.raw["dataset"]["subjects"]
deriv_dir  = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

all_tgm = []
times   = None

for subject in subjects:
    epochs_path = deriv_dir / f"sub-{subject}" / "epochs.fif"
    labels_path = deriv_dir / f"sub-{subject}" / "labels.npy"
    if not epochs_path.exists():
        print(f"[SKIP] {subject}")
        continue

    print(f"[{subject}] Loading epochs…")
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    labels = np.load(str(labels_path))
    X = epochs.get_data()
    y = labels
    if times is None:
        times = epochs.times * 1000

    # Subsample to 100 categories — same as decoding for comparability
    rng = np.random.default_rng(42)
    cats, counts = np.unique(y, return_counts=True)
    top_cats = cats[np.argsort(counts)[-100:]]
    mask = np.isin(y, top_cats)
    X_sub, y_sub = X[mask], y[mask]
    print(f"[{subject}] {X_sub.shape[0]} trials, {X_sub.shape[1]} ch, {X_sub.shape[2]} times")

    tgm = temporal_generalization(X_sub, y_sub, cfg)   # (n_times, n_times)
    all_tgm.append(tgm)
    print(f"[{subject}] TGM peak: {tgm.max():.4f} at train={times[tgm.argmax()//len(times)]:.0f}ms, test={times[tgm.argmax()%len(times)]:.0f}ms")

    # Free memory immediately
    del epochs, X, X_sub
    import gc; gc.collect()

tgm_mean = np.mean(all_tgm, axis=0)
np.savez(str(results_dir / "tgm_scores.npz"),
         tgm_mean=tgm_mean,
         tgm_subjects=np.array(all_tgm),
         times=times)
print(f"Saved results/tgm_scores.npz  shape={tgm_mean.shape}")

# Plot
chance = 1 / 100
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: group mean TGM
ax = axes[0]
vmax = max(abs(tgm_mean.max() - chance), abs(tgm_mean.min() - chance))
im = ax.imshow(tgm_mean - chance, origin="lower",
               extent=[times[0], times[-1], times[0], times[-1]],
               aspect="auto", cmap="RdBu_r",
               vmin=-vmax, vmax=vmax)
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.plot([times[0], times[-1]], [times[0], times[-1]], "k-", lw=0.5, alpha=0.4)
ax.set_xlabel("Test time (ms)")
ax.set_ylabel("Train time (ms)")
ax.set_title(f"TGM — mean across {len(all_tgm)} subjects\n(accuracy − chance)")
plt.colorbar(im, ax=ax, label="Accuracy − chance")

# Panel B: individual subject diagonals (time-resolved decoding)
ax = axes[1]
for i, tgm_s in enumerate(all_tgm):
    diag = np.diag(tgm_s)
    ax.plot(times, diag, alpha=0.4, lw=1, label=f"S{i+1}")
ax.plot(times, np.diag(tgm_mean), "k-", lw=2.5, label="Mean")
ax.axhline(chance, color="gray", ls="--", lw=1, label="Chance")
ax.axvline(0, color="k", ls=":", lw=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Accuracy")
ax.set_title("Diagonal (standard decoding)")
ax.legend(fontsize=8)

fig.tight_layout()
out = figures_dir / "figure1b_tgm.png"
fig.savefig(str(out), dpi=150, bbox_inches="tight")
print(f"Saved {out}")
