"""Representational similarity analysis (Week 3).

Time-resolved, cross-validated RDMs per subject. Uses crossnobis (cross-validated
Mahalanobis, Walther et al. 2016) via rsatoolbox. Crossnobis is unbiased: expected
value is zero when two conditions are identical, positive when distinguishable.
"""

from __future__ import annotations

import numpy as np
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm import calc_rdm
from scipy.stats import spearmanr

from .config import Config


def compute_rdm(
    X: np.ndarray,       # (n_trials, n_channels) — data at one timepoint/window
    labels: np.ndarray,  # (n_trials,) integer condition labels
    cv_descriptor: np.ndarray | None = None,  # (n_trials,) fold/run indices for crossnobis
    metric: str = "crossnobis",
) -> np.ndarray:
    """Return a symmetric (n_conditions, n_conditions) RDM.

    crossnobis: leave-one-partition-out cross-validated Mahalanobis (Walther 2016).
    Implemented directly to avoid rsatoolbox's requirement that all conditions appear
    in every partition. Expected value = 0 under the null (conditions identical).
    """
    conditions = np.unique(labels)
    n_cond = len(conditions)
    cond_idx = {c: i for i, c in enumerate(conditions)}

    if metric != "crossnobis" or cv_descriptor is None:
        # Fallback: per-condition means, squared Euclidean
        means = np.array([X[labels == c].mean(axis=0) for c in conditions])
        rdm = np.zeros((n_cond, n_cond))
        for i in range(n_cond):
            for j in range(i + 1, n_cond):
                d = float(np.sum((means[i] - means[j]) ** 2))
                rdm[i, j] = rdm[j, i] = d
        return rdm

    # Crossnobis (vectorised): for each pair of partitions (a, b),
    # contribution to RDM[i,j] = (μ_a_i - μ_a_j) · (μ_b_i - μ_b_j)
    # Use leave-one-partition-out; partition pairs averaged at the end.
    partitions = np.unique(cv_descriptor)
    n_parts = len(partitions)
    n_ch = X.shape[1]

    # part_means: (n_parts, n_cond, n_ch) — NaN where condition absent
    part_means = np.full((n_parts, n_cond, n_ch), np.nan)
    for pi, p in enumerate(partitions):
        mask_p = cv_descriptor == p
        for c in conditions:
            mask_c = (labels == c) & mask_p
            if mask_c.sum() > 0:
                part_means[pi, cond_idx[c]] = X[mask_c].mean(axis=0)

    rdm_accum = np.zeros((n_cond, n_cond))
    count_accum = np.zeros((n_cond, n_cond))

    for pi in range(n_parts):
        for pj in range(pi + 1, n_parts):
            ma = part_means[pi]  # (n_cond, n_ch)
            mb = part_means[pj]

            # valid[i] = True if condition i present in both partitions
            valid_mask = ~np.any(np.isnan(ma), axis=1) & ~np.any(np.isnan(mb), axis=1)
            if valid_mask.sum() < 2:
                continue

            ma_v = np.where(valid_mask[:, None], ma, 0.0)
            mb_v = np.where(valid_mask[:, None], mb, 0.0)

            # diff_a[i,j] = ma[i] - ma[j]; crossnobis[i,j] = diff_a · diff_b
            # Using: diff_a · diff_b = (ma_i - ma_j)·(mb_i - mb_j)
            # = ma_i·mb_i - ma_i·mb_j - ma_j·mb_i + ma_j·mb_j
            # Vectorised as outer products of dot-product matrix
            dot = ma_v @ mb_v.T  # (n_cond, n_cond)
            diag = np.diag(dot)
            cross = diag[:, None] - dot - dot.T + diag[None, :]

            outer_valid = valid_mask[:, None] & valid_mask[None, :]
            rdm_accum += cross * outer_valid
            count_accum += outer_valid.astype(float)

    valid = count_accum > 0
    rdm_accum[valid] /= count_accum[valid]
    np.fill_diagonal(rdm_accum, 0.0)
    return rdm_accum


def time_resolved_rdms(
    X: np.ndarray,       # (n_trials, n_channels, n_times)
    labels: np.ndarray,  # (n_trials,)
    cv_descriptor: np.ndarray,  # (n_trials,) run indices
    cfg: Config,
    metric: str = "crossnobis",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute RDMs in a sliding window; return (rdms, times).

    rdms shape: (n_windows, n_conditions, n_conditions)
    Sliding window defined by rsa.time_window_ms and rsa.time_step_ms in config.
    """
    sfreq = float(cfg.raw["preprocessing"]["resample_sfreq"])
    win_samples = int(cfg.raw["rsa"]["time_window_ms"] / 1000 * sfreq)
    step_samples = int(cfg.raw["rsa"]["time_step_ms"] / 1000 * sfreq)
    n_times = X.shape[2]

    rdm_list = []
    center_times = []

    start = 0
    while start + win_samples <= n_times:
        X_win = X[:, :, start:start + win_samples].mean(axis=2)  # (n_trials, n_channels)
        rdm = compute_rdm(X_win, labels, cv_descriptor=cv_descriptor, metric=metric)
        rdm_list.append(rdm)
        center_times.append(start + win_samples // 2)
        start += step_samples

    return np.array(rdm_list), np.array(center_times)


def rdm_correlation(
    rdm_a: np.ndarray,  # (n_conditions, n_conditions)
    rdm_b: np.ndarray,
    method: str = "spearman",
) -> float:
    """Second-order similarity between two RDMs (upper triangle only, no diagonal)."""
    idx = np.triu_indices(rdm_a.shape[0], k=1)
    vec_a = rdm_a[idx]
    vec_b = rdm_b[idx]
    if method == "spearman":
        return float(spearmanr(vec_a, vec_b).statistic)
    elif method == "pearson":
        return float(np.corrcoef(vec_a, vec_b)[0, 1])
    else:
        raise ValueError(f"Unknown method: {method}")
