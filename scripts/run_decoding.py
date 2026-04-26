"""Run time-resolved decoding for all subjects and save Figure 1.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_decoding.py

Outputs:
    results/decoding_scores.npz   — scores (n_subjects, n_times) + times array
    figures/figure1_decoding.png  — decoding accuracy over time, one line per subject
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.config import load_config
from thingsmeg.decoding import time_resolved_decode

mne.set_log_level("WARNING")

cfg = load_config()
subjects = cfg.raw["dataset"]["subjects"]
deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

all_scores = []
times = None

for subject in subjects:
    epochs_path = deriv_dir / f"sub-{subject}" / "epochs.fif"
    labels_path = deriv_dir / f"sub-{subject}" / "labels.npy"

    if not epochs_path.exists():
        print(f"[SKIP] {subject} — epochs.fif not found")
        continue

    print(f"[{subject}] Loading epochs…")
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    labels = np.load(str(labels_path))

    X = epochs.get_data()  # (n_trials, n_channels, n_times)
    y = labels

    if times is None:
        times = epochs.times * 1000  # ms

    print(f"[{subject}] Decoding {X.shape[0]} trials × {X.shape[1]} channels × {X.shape[2]} times…")
    # Subsample to 100 categories (max 2 trials/category used) for tractable
    # multiclass decoding. Full 1854-way is underpowered with ~12 trials/category.
    rng = np.random.default_rng(42)
    cats, counts = np.unique(y, return_counts=True)
    # Pick 100 categories with most trials
    top_cats = cats[np.argsort(counts)[-100:]]
    mask = np.isin(y, top_cats)
    X_sub, y_sub = X[mask], y[mask]
    print(f"[{subject}] Subsampled to {len(top_cats)} categories, {mask.sum()} trials")
    scores = time_resolved_decode(X_sub, y_sub, cfg)
    all_scores.append(scores)
    print(f"[{subject}] Peak accuracy: {scores.max():.3f} at {times[scores.argmax()]:.0f} ms (chance={1/len(top_cats):.3f})")

all_scores = np.array(all_scores)  # (n_subjects, n_times)

# Save results
np.savez(results_dir / "decoding_scores.npz", scores=all_scores, times=times, subjects=subjects)
print(f"Saved results/decoding_scores.npz — shape {all_scores.shape}")

# Figure 1
fig, ax = plt.subplots(figsize=(10, 5))
chance = 1.0 / 100  # 100-way classification

# Smooth with a 50ms sliding average (10 samples at 200 Hz) to reduce noise
def smooth(x, n=10):
    return np.convolve(x, np.ones(n) / n, mode="same")

for i, subject in enumerate(subjects[:len(all_scores)]):
    ax.plot(times, smooth(all_scores[i]), linewidth=1.5, label=subject)

ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance ({chance:.4f})")
ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
ax.set_xlabel("Time relative to stimulus onset (ms)")
ax.set_ylabel("Decoding accuracy")
ax.set_title("Time-resolved object category decoding (ridge classifier, 5-fold CV)")
ax.legend(fontsize=9)
ax.set_xlim(times[0], times[-1])
fig.tight_layout()
fig.savefig(figures_dir / "figure1_decoding.png", dpi=150)
print("Saved figures/figure1_decoding.png")
