"""Statistical controls (Week 7).

Non-negotiables: cluster-based permutation tests, noise ceilings, bootstrap CIs,
label-shuffle null distributions. Honest reporting where effects don't clear these
bars is a feature, not a failure.
"""

from __future__ import annotations

import numpy as np
from mne.stats import permutation_cluster_1samp_test
from scipy.stats import spearmanr

from .config import Config


def cluster_permutation_1d(
    scores: np.ndarray,  # (n_subjects, n_times) or (n_pairs, n_times)
    chance: float,
    cfg: Config,
    tail: int = 1,  # 1=positive, -1=negative, 0=two-tailed
) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    """Cluster-based permutation test over time vs. chance (Maris & Oostenveld 2007).

    Returns (t_obs, clusters, cluster_pv, H0) from mne.stats.
    Input scores shape: (n_observations, n_times).
    """
    n_permutations = int(cfg.raw["stats"]["n_permutations"])
    alpha = float(cfg.raw["stats"]["cluster_alpha"])
    seed = int(cfg.raw["stats"]["random_seed"])

    X = scores - chance  # test against chance level
    t_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=n_permutations,
        threshold=None,  # auto t-threshold at p<0.05
        tail=tail,
        n_jobs=1,
        seed=seed,
        verbose=False,
    )
    return t_obs, clusters, cluster_pv, H0


def noise_ceiling(
    rdms: np.ndarray,  # (n_subjects, n_conditions, n_conditions)
) -> tuple[float, float]:
    """Upper/lower noise-ceiling bounds for cross-subject RDM correlation (Nili et al. 2014).

    Upper bound: correlate each subject's RDM with the group mean including itself.
    Lower bound: leave-one-subject-out mean (excludes the subject being evaluated).
    Both are reported as a shaded band behind RSA curves.
    """
    n_subj = rdms.shape[0]
    idx = np.triu_indices(rdms.shape[1], k=1)
    vecs = rdms[:, idx[0], idx[1]]  # (n_subj, n_pairs)

    upper_rs, lower_rs = [], []
    for i in range(n_subj):
        mean_all = vecs.mean(axis=0)
        mean_loo = (vecs.sum(axis=0) - vecs[i]) / (n_subj - 1)
        upper_rs.append(spearmanr(vecs[i], mean_all).statistic)
        lower_rs.append(spearmanr(vecs[i], mean_loo).statistic)

    return float(np.mean(upper_rs)), float(np.mean(lower_rs))


def noise_ceiling_over_time(
    rdms_per_subject: dict[str, np.ndarray],  # {subject: (n_times, n_cond, n_cond)}
) -> tuple[np.ndarray, np.ndarray]:
    """Noise ceiling at each timepoint. Returns (upper, lower) arrays of shape (n_times,)."""
    subjects = list(rdms_per_subject.keys())
    n_times = next(iter(rdms_per_subject.values())).shape[0]

    upper = np.zeros(n_times)
    lower = np.zeros(n_times)

    for t in range(n_times):
        rdms_t = np.stack([rdms_per_subject[s][t] for s in subjects])  # (n_subj, n_c, n_c)
        upper[t], lower[t] = noise_ceiling(rdms_t)

    return upper, lower


def bootstrap_ci(
    scores: np.ndarray,   # (n_observations, n_times) — resample over observations
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap CI over the observation axis (subjects or pairs).

    Returns (lower, upper) arrays of shape (n_times,).
    """
    rng = np.random.default_rng(seed)
    n_obs = scores.shape[0]
    boot_means = np.zeros((n_boot, scores.shape[1]))

    for b in range(n_boot):
        idx = rng.integers(0, n_obs, size=n_obs)
        boot_means[b] = scores[idx].mean(axis=0)

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha, axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha), axis=0)
    return lower, upper


def shuffle_null_rsa(
    brain_vecs: np.ndarray,   # (n_times, n_pairs) brain RDM upper triangle
    model_vec: np.ndarray,    # (n_pairs,) model RDM upper triangle
    n_perm: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Label-shuffle null distribution for brain-model RSA.

    Vectorised: pre-ranks brain_vecs once, then each permutation is a single
    matrix-vector multiply rather than n_times separate spearmanr calls.
    ~200× faster than the naive loop (seconds instead of 20+ minutes).

    Returns (n_perm, n_times) null array of Spearman r values.
    """
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    n_times, n_pairs = brain_vecs.shape

    # Rank brain_vecs across the pairs axis once — Spearman r = Pearson r of ranks
    brain_ranked = np.apply_along_axis(rankdata, 1, brain_vecs).astype(np.float64)
    brain_ranked -= brain_ranked.mean(axis=1, keepdims=True)
    brain_norms = np.linalg.norm(brain_ranked, axis=1, keepdims=True) + 1e-12
    brain_ranked /= brain_norms   # (n_times, n_pairs) normalised ranks

    null = np.zeros((n_perm, n_times), dtype=np.float64)
    for p in range(n_perm):
        shuffled = rankdata(rng.permutation(model_vec)).astype(np.float64)
        shuffled -= shuffled.mean()
        norm = np.linalg.norm(shuffled) + 1e-12
        shuffled /= norm
        null[p] = brain_ranked @ shuffled   # (n_times,) — Pearson r on ranks = Spearman r

    return null
