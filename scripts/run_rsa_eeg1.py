"""Crossnobis RDMs for THINGS-EEG1 subjects.

Same pipeline as run_rsa_full.py but adapted for EEG1:
  - Input: per-subject condition_means.npy (1854 × n_times × n_ch)
  - Output: rdms_eeg1_<sub>.npz  (1854 × 1854 × n_times, float32, upper-tri only)

Memory: 63 channels much smaller than MEG 272 — full 1854×1854 fits in RAM easily.
One subject: ~1854² × n_times × 4 bytes ≈ 500 MB.

Usage:
    python scripts/run_rsa_eeg1.py --subject sub-01
    # or all available:
    python scripts/run_rsa_eeg1.py --all
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf

DERIV      = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
RESULTS    = Path("results/eeg1")

N_CONCEPTS = 1854
N_FOLDS    = 2


def crossnobis_rdm(means: np.ndarray) -> np.ndarray:
    """Compute crossnobis RDM from condition means.

    Parameters
    ----------
    means : (n_concepts, n_times, n_ch) float32

    Returns
    -------
    rdm : (n_concepts, n_concepts, n_times) float32  -- upper triangle only via triu_indices
          Actually returns full (n_concepts, n_concepts) per timepoint
    """
    n_concepts, n_times, n_ch = means.shape
    rdm = np.zeros((n_concepts, n_concepts, n_times), dtype=np.float32)

    for t in range(n_times):
        X = means[:, t, :]   # (n_concepts, n_ch)

        # Estimate precision matrix via Ledoit-Wolf
        lw = LedoitWolf(assume_centered=True)
        lw.fit(X - X.mean(0))
        precision = lw.precision_.astype(np.float32)

        # Crossnobis: D(i,j) = (xi - xj) @ precision @ (xi - xj)^T / 2
        # Efficient: precompute X @ precision
        XP = X @ precision   # (n_concepts, n_ch)
        # D[i,j] = XP[i] @ X[i] - XP[i] @ X[j] - XP[j] @ X[i] + XP[j] @ X[j]
        diag = (XP * X).sum(1)   # (n_concepts,) = XP[i] @ X[i]
        gram  = XP @ X.T          # (n_concepts, n_concepts)
        D = diag[:, None] - gram - gram.T + diag[None, :]
        rdm[:, :, t] = (D / 2).astype(np.float32)

    return rdm


def run(subject: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"rdms_eeg1_{subject}.npz"
    if out_path.exists():
        print(f"{out_path.name} already exists — skipping")
        return

    means_path = DERIV / subject / f"{subject}_condition_means.npy"
    if not means_path.exists():
        print(f"ERROR: {means_path} not found. Run preprocess_eeg1.py first.")
        sys.exit(1)

    print(f"Loading condition means for {subject}...")
    means = np.load(str(means_path))   # (1854, n_times, n_ch)
    print(f"  Shape: {means.shape}")

    # Z-score across concepts at each timepoint
    mean_m = means.mean(0, keepdims=True)
    std_m  = means.std(0, keepdims=True) + 1e-8
    means  = (means - mean_m) / std_m

    print(f"Computing crossnobis RDMs ({means.shape[1]} timepoints)...")
    rdm = crossnobis_rdm(means)   # (1854, 1854, n_times)

    idx_upper = np.triu_indices(N_CONCEPTS, k=1)
    rdm_upper = rdm[idx_upper[0], idx_upper[1], :]   # (n_pairs, n_times)
    print(f"  RDM upper triangle: {rdm_upper.shape}")

    np.savez_compressed(str(out_path), rdm_upper=rdm_upper.astype(np.float32))
    print(f"Saved {out_path}  ({out_path.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--subject", help="e.g. sub-01")
    grp.add_argument("--all",     action="store_true")
    args = parser.parse_args()

    if args.all:
        available = sorted(DERIV.glob("sub-*/"))
        subjects  = [p.name for p in available]
        print(f"Found {len(subjects)} preprocessed subjects: {subjects}")
    else:
        subjects = [args.subject]

    for sub in subjects:
        run(sub)
