"""Time-resolved single-subject decoding (Week 1-2, Figure 1).

Linear (ridge) decoders of object category at each post-stimulus timepoint, with
proper cross-validation. Figure 1 is time-resolved decoding accuracy per subject
with cluster-based permutation statistics (see stats.py).
"""

from __future__ import annotations

import numpy as np
from mne.decoding import GeneralizingEstimator, SlidingEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config


def _make_clf() -> object:
    # LDA with shrinkage (Ledoit-Wolf) — optimal for high-dim MEG, few trials
    return make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto"),
    )


def time_resolved_decode(
    X: np.ndarray,  # (n_trials, n_channels, n_times)
    y: np.ndarray,  # (n_trials,) integer category labels
    cfg: Config,
) -> np.ndarray:
    """Return decoding accuracy of shape (n_times,).

    Uses mne.decoding.SlidingEstimator which vectorises the per-timepoint fit
    and supports n_jobs parallelism — much faster than a Python timepoint loop.
    Manual CV loop used because cross_val_score flattens array-output scores.
    """
    n_folds = int(cfg.raw["decoding"]["cv_folds"])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    for train_idx, test_idx in cv.split(X[:, :, 0], y):
        sliding = SlidingEstimator(_make_clf(), scoring="accuracy", n_jobs=1, verbose=False)
        sliding.fit(X[train_idx], y[train_idx])
        fold_scores.append(sliding.score(X[test_idx], y[test_idx]))  # (n_times,)

    return np.mean(fold_scores, axis=0)  # (n_times,)


def temporal_generalization(
    X: np.ndarray,  # (n_trials, n_channels, n_times)
    y: np.ndarray,  # (n_trials,) integer category labels
    cfg: Config,
) -> np.ndarray:
    """Train-time x test-time generalization matrix (King & Dehaene 2014).

    Returns array of shape (n_times_train, n_times_test) — mean accuracy across
    folds. Captures whether representations generalise across time (sustained =
    diagonal band; transient = diagonal only).
    """
    n_folds = int(cfg.raw["decoding"]["cv_folds"])
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_scores = []
    for train_idx, test_idx in cv.split(X[:, :, 0], y):
        gen = GeneralizingEstimator(_make_clf(), scoring="accuracy", n_jobs=-1, verbose=False)
        gen.fit(X[train_idx], y[train_idx])
        fold_scores.append(gen.score(X[test_idx], y[test_idx]))  # (n_times, n_times)

    return np.mean(fold_scores, axis=0)  # (n_times, n_times)
