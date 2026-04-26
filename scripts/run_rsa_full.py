"""Full-resolution RSA: all 1854 categories, vectorised 2-fold crossnobis (Week 3+).

Memory-efficient design:
  - float32 throughout (8.8 GB → 4.4 GB for epochs, 4.6 GB → 2.3 GB for RDMs)
  - Condition means computed once, then epochs freed before RDM loop
  - RDMs written to np.memmap on disk — never the full array in RAM at once
  - One subject at a time, explicitly gc.collect() between subjects
  - Saves effective_rank without storing all RDMs in RAM

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_rsa_full.py [SUBJECT]
    e.g.:  python scripts/run_rsa_full.py BIGMEG3
    or run all:  python scripts/run_rsa_full.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

mne.set_log_level("WARNING")

cfg = load_config()
all_subjects = cfg.raw["dataset"]["subjects"]

# Allow running a single subject from CLI: python run_rsa_full.py BIGMEG3
if len(sys.argv) > 1:
    subjects = [sys.argv[1]]
else:
    subjects = all_subjects

deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

N_COND = 1854


def crossnobis_2fold(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Vectorised 2-fold crossnobis. A, B: (n_valid, n_ch) float32."""
    M = A @ B.T           # (n_valid, n_valid) — single BLAS call
    diag = np.diag(M)
    rdm = diag[:, None] - M - M.T + diag[None, :]
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_effective_rank(rdm: np.ndarray) -> float:
    """Participation ratio of RDM singular values."""
    svs = np.abs(np.linalg.svd(rdm.astype(np.float64), compute_uv=False))
    s = svs.sum()
    return float(s ** 2 / (svs ** 2).sum()) if s > 0 else 0.0


all_eff_ranks: dict[str, np.ndarray] = {}
all_times_ms: np.ndarray | None = None

for subject in subjects:
    out_path = results_dir / f"rdms_full_{subject}.npz"
    if out_path.exists():
        print(f"[SKIP] {subject} — {out_path.name} already exists")
        # Still load effective_rank for the figure
        d = np.load(str(out_path))
        all_eff_ranks[subject] = d["effective_rank"]
        if all_times_ms is None:
            all_times_ms = d["times_ms"]
        continue

    epochs_path = deriv_dir / f"sub-{subject}" / "epochs.fif"
    if not epochs_path.exists():
        print(f"[SKIP] {subject} — no epochs.fif")
        continue

    print(f"\n[{subject}] Loading epochs…")
    epochs = mne.read_epochs(str(epochs_path), preload=True, verbose=False)
    labels  = np.load(str(deriv_dir / f"sub-{subject}" / "labels.npy"))
    run_ids = np.load(str(deriv_dir / f"sub-{subject}" / "runs.npy"))
    times_ms = (epochs.times * 1000).astype(np.float32)
    if all_times_ms is None:
        all_times_ms = times_ms

    # Cast to float32 immediately (saves ~4 GB vs float64)
    X = epochs.get_data().astype(np.float32)  # (n_trials, n_ch, n_times)
    n_trials, n_ch, n_times = X.shape
    print(f"  Trials={n_trials}, channels={n_ch}, times={n_times}")
    print(f"  X size: {X.nbytes / 1e9:.1f} GB (float32)")

    # Free the MNE object — keep only the numpy array
    del epochs
    gc.collect()

    odd_mask  = (run_ids % 2 == 0)
    even_mask = (run_ids % 2 == 1)

    print(f"  Computing condition means ({N_COND} conds × 2 folds)…")
    all_conds = np.arange(1, N_COND + 1)
    means_odd  = np.zeros((N_COND, n_ch, n_times), dtype=np.float32)
    means_even = np.zeros((N_COND, n_ch, n_times), dtype=np.float32)
    valid = np.zeros(N_COND, dtype=bool)

    for ci, c in enumerate(all_conds):
        m_o = (labels == c) & odd_mask
        m_e = (labels == c) & even_mask
        if m_o.sum() > 0 and m_e.sum() > 0:
            means_odd[ci]  = X[m_o].mean(axis=0)
            means_even[ci] = X[m_e].mean(axis=0)
            valid[ci] = True

    n_valid = int(valid.sum())
    print(f"  {n_valid}/{N_COND} conditions valid in both folds")

    # Free raw trials — no longer needed
    del X
    gc.collect()
    print(f"  Freed X. means_odd+even size: "
          f"{(means_odd.nbytes + means_even.nbytes) / 1e9:.1f} GB")

    # Z-score across conditions per channel per time (in-place)
    for fold in [means_odd, means_even]:
        mu = fold[valid].mean(axis=0, keepdims=True)
        sd = fold[valid].std(axis=0, keepdims=True) + 1e-8
        fold[valid] = (fold[valid] - mu) / sd

    # Slice to valid conditions only to keep arrays small
    A_all = means_odd[valid]    # (n_valid, n_ch, n_times)
    B_all = means_even[valid]
    del means_odd, means_even
    gc.collect()

    # Compute crossnobis one timepoint at a time, stream to memmap on disk
    mmap_path = str(results_dir / f"_rdms_tmp_{subject}.dat")
    rdm_shape = (n_times, n_valid, n_valid)
    rdms_mm = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=rdm_shape)
    eff_ranks = np.zeros(n_times, dtype=np.float32)

    print(f"  Crossnobis at {n_times} timepoints (memmap, float32)…")
    for t in range(n_times):
        A = A_all[:, :, t]   # (n_valid, n_ch)
        B = B_all[:, :, t]
        rdm = crossnobis_2fold(A, B)
        rdms_mm[t] = rdm.astype(np.float32)
        eff_ranks[t] = rdm_effective_rank(rdm)
        if t % 40 == 0:
            rdms_mm.flush()
            print(f"    t={t}/{n_times}  eff_rank={eff_ranks[t]:.2f}")

    rdms_mm.flush()
    del A_all, B_all
    gc.collect()

    # Save as compressed npz (reads from memmap, stream-compresses)
    print(f"  Saving {out_path.name}…")
    valid_cats = all_conds[valid]
    np.savez_compressed(
        str(out_path),
        times_ms=times_ms,
        valid_categories=valid_cats,
        effective_rank=eff_ranks,
        # Store RDMs as well — compressed they'll be ~1-2 GB
        rdms=np.array(rdms_mm),
    )
    print(f"  Saved {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")

    # Clean up memmap temp file
    del rdms_mm
    Path(mmap_path).unlink(missing_ok=True)
    gc.collect()

    all_eff_ranks[subject] = eff_ranks


# ── Figure: effective rank over time (all subjects) ──────────────────────────
if all_eff_ranks and all_times_ms is not None:
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    for (subject, eff_ranks), color in zip(all_eff_ranks.items(), colors):
        smooth = np.convolve(eff_ranks.astype(float), np.ones(7) / 7, mode="same")
        ax.plot(all_times_ms, smooth, linewidth=1.8, label=subject, color=color)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("RDM effective rank")
    ax.set_title("Representational dimensionality over time\n"
                 "(all 1854 categories, 2-fold crossnobis)")
    ax.legend()
    fig.tight_layout()
    out_fig = figures_dir / "figure_rsa_full.png"
    fig.savefig(str(out_fig), dpi=150)
    print(f"\nSaved {out_fig}")
    plt.close(fig)

print("\nDone.")
