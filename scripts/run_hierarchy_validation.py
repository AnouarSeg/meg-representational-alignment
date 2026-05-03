"""Hierarchy validation using official THINGS-MEG derivative time courses.

Tests the early-sensory / late-semantic gradient directly using four
pre-computed time courses from the official dataset derivatives:

  - fMRI V1 → MEG regression   (early visual cortex, expected peak ~80-120ms)
  - fMRI FFA → MEG regression  (face/object area, expected peak ~150-250ms)
  - Animacy ratings             (categorical dimension, expected late)
  - Size ratings                (physical dimension, expected early or flat)

Plus: SPOSE brain-model RSA from the existing 200-category analysis.

No heavy computation: all inputs are CSVs or a pre-existing .npz.
Peak RAM: < 50 MB.

Usage:
    python scripts/run_hierarchy_validation.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

cfg         = load_config()
deriv_dir   = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

SUBJ_LABELS = ["P1", "P2", "P3", "P4"]
times_ms    = np.arange(-100, 1305, 5, dtype=float)   # 281 timepoints


def load_tc(prefix: str) -> np.ndarray | None:
    """Load 4 per-subject CSVs → (4, 281) mean-over-sessions array."""
    rows = []
    for s in SUBJ_LABELS:
        p = deriv_dir / f"validation_{prefix}_{s}.csv"
        if not p.exists():
            print(f"  Missing: {p.name}")
            return None
        rows.append(pd.read_csv(p, index_col=0).values.mean(axis=1))
    return np.stack(rows)   # (4, 281)


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def peak_info(tc: np.ndarray, label: str) -> None:
    mean = tc.mean(0)
    post = times_ms >= 0
    pk   = int(np.argmax(mean))
    print(f"  {label}: peak={mean[pk]:.4f} at {times_ms[pk]:.0f}ms  "
          f"post-stim mean={mean[post].mean():.4f}")


# ── Load time courses ────────────────────────────────────────────────────────
print("Loading validation time courses...")
animacy_tc = load_tc("animacy")
size_tc    = load_tc("size")
v1_tc      = load_tc("fmri_meg_v1")
ffa_tc     = load_tc("fmri_meg_ffa")

for name, tc in [("Animacy", animacy_tc), ("Size", size_tc),
                  ("V1 fMRI→MEG", v1_tc), ("FFA fMRI→MEG", ffa_tc)]:
    if tc is not None:
        peak_info(tc, name)

# ── Load existing SPOSE brain-model RSA (200-cat) ───────────────────────────
print("\nLoading SPOSE brain-model RSA (200-cat)...")
bm = np.load(str(results_dir / "brain_model_rsa_official.npz"), allow_pickle=True)
spose_corrs = bm["SPOSE"]       # (4, 281)
nc_upper    = bm["nc_upper"]    # (281,)
nc_lower    = bm["nc_lower"]    # (281,)
bm_times    = bm["times_ms"]    # should match times_ms
peak_info(spose_corrs, "SPOSE (200-cat)")

# ── Peak latency comparison (hierarchy test) ─────────────────────────────────
print("\n--- Hierarchy gradient (peak latencies) ---")
results_rows = []
for name, tc in [("V1 fMRI→MEG", v1_tc), ("Size", size_tc),
                  ("SPOSE 200-cat", spose_corrs),
                  ("Animacy", animacy_tc), ("FFA fMRI→MEG", ffa_tc)]:
    if tc is None:
        continue
    mean = tc.mean(0)
    pk   = times_ms[np.argmax(mean[times_ms >= 0] if times_ms[0] < 0 else mean)]
    # Find first sig exceedance above baseline
    baseline_std = mean[times_ms < 0].std()
    baseline_mean = mean[times_ms < 0].mean()
    onset_mask = (times_ms >= 0) & (mean > baseline_mean + 2 * baseline_std)
    onset = times_ms[onset_mask][0] if onset_mask.any() else np.nan
    print(f"  {name:25s}: peak={times_ms[np.argmax(mean[bm_times>=0] + 0) if False else np.argmax(mean)]:.0f}ms  onset≈{onset:.0f}ms")
    results_rows.append({"model": name, "peak_ms": float(times_ms[np.argmax(mean)])})

# ── Figure: two-panel hierarchy plot ─────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

# Panel A: fMRI ROI time courses (early V1 vs late FFA = direct hierarchy test)
ax = axes[0]
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5)

for label, tc, color, ls in [
    ("V1 fMRI→MEG (early visual)", v1_tc,  "forestgreen", "-"),
    ("FFA fMRI→MEG (object/face)",  ffa_tc, "crimson",     "-"),
]:
    if tc is None:
        continue
    mean = tc.mean(0)
    sem  = tc.std(0) / np.sqrt(len(tc))
    ms   = smooth(mean)
    ax.plot(bm_times, ms, color=color, lw=2.5, ls=ls, label=label)
    ax.fill_between(bm_times, smooth(mean - sem), smooth(mean + sem),
                    alpha=0.15, color=color)

ax.set_ylabel("MEG–fMRI regression score")
ax.set_title("A  Cortical hierarchy: V1 (early) vs FFA (late) alignment with MEG")
ax.legend(fontsize=10)

# Panel B: semantic / perceptual model time courses + SPOSE
ax2 = axes[1]
ax2.axhline(0, color="gray", lw=0.5, ls="--")
ax2.axvline(0, color="gray", lw=0.5)
ax2.fill_between(bm_times, nc_lower, nc_upper, alpha=0.12, color="gray",
                 label="Noise ceiling (SPOSE, 200-cat)")

for label, tc, color in [
    ("SPOSE human similarity (200-cat)", spose_corrs, "steelblue"),
    ("Animacy ratings",                  animacy_tc,  "purple"),
    ("Size ratings",                     size_tc,     "saddlebrown"),
]:
    if tc is None:
        continue
    mean = tc.mean(0)
    sem  = tc.std(0) / np.sqrt(len(tc))
    ms   = smooth(mean)
    ax2.plot(bm_times, ms, color=color, lw=2, label=label)
    ax2.fill_between(bm_times, smooth(mean - sem), smooth(mean + sem),
                     alpha=0.12, color=color)

ax2.set_xlabel("Time from stimulus onset (ms)")
ax2.set_ylabel("Spearman r / regression score")
ax2.set_title("B  Model time courses: semantic & perceptual dimensions")
ax2.legend(fontsize=9)

plt.tight_layout()
out = str(figures_dir / "figure5_hierarchy_validation.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {out}")

# ── Save results ─────────────────────────────────────────────────────────────
save = dict(times_ms=bm_times, SPOSE=spose_corrs, nc_upper=nc_upper, nc_lower=nc_lower)
for name, tc in [("animacy", animacy_tc), ("size", size_tc),
                  ("V1_fmri", v1_tc), ("FFA_fmri", ffa_tc)]:
    if tc is not None:
        save[name] = tc
np.savez(str(results_dir / "hierarchy_validation.npz"), **save)
print("Saved results/hierarchy_validation.npz")
