"""Cross-subject and brain-to-model alignment (Weeks 4-6, Figures 2-4).

Methods: Procrustes, CCA, SRM (brainiak fallback), hyperalignment-style iterative.
Core contribution (Week 5, Figure 3): alignment complexity as a function of time —
early windows align via simpler (closer-to-orthogonal) transforms than late windows.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.preprocessing import StandardScaler

from .config import Config


# ── Williams shape distance ───────────────────────────────────────────────────

def williams_shape_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Generalized shape distance between representational geometries (Williams et al. 2021 NeurIPS).

    Both matrices (n_conditions, n_features); invariant to orthogonal transforms and scaling.
    """
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    K_A = A @ A.T
    K_B = B @ B.T
    K_A = K_A / (np.linalg.norm(K_A, "fro") + 1e-12)
    K_B = K_B / (np.linalg.norm(K_B, "fro") + 1e-12)
    M = K_A.T @ K_B
    U, _, Vt = np.linalg.svd(M)
    K_B_aligned = K_B @ Vt.T @ U.T
    return float(np.linalg.norm(K_A - K_B_aligned, "fro"))


# ── Alignment methods ─────────────────────────────────────────────────────────

class ProcrustesAligner:
    """Orthogonal Procrustes: W = V U^T from SVD of B^T A."""

    def __init__(self) -> None:
        self.W: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> "ProcrustesAligner":
        M = target.T @ source
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        self.W = (U @ Vt).T  # (n_src, n_tgt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W


class CCAAligner:
    """Canonical correlation analysis — projects both spaces to shared latents."""

    def __init__(self, n_components: int = 50) -> None:
        self.n_components = n_components
        self.cca: CCA | None = None
        self.scaler_s: StandardScaler = StandardScaler()
        self.scaler_t: StandardScaler = StandardScaler()

    def fit(self, source: np.ndarray, target: np.ndarray) -> "CCAAligner":
        n_comp = min(self.n_components, source.shape[1], target.shape[1], source.shape[0] - 1)
        self.cca = CCA(n_components=n_comp)
        s = self.scaler_s.fit_transform(source)
        t = self.scaler_t.fit_transform(target)
        self.cca.fit(s, t)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        s = self.scaler_s.transform(X)
        Xs, _ = self.cca.transform(s, s)  # project source into shared space
        return Xs


class SRMAligner:
    """Shared Response Model (Chen et al. 2015) — in-repo fallback.

    Iterative algorithm: find shared response S and per-subject orthogonal W_i
    such that X_i ≈ S W_i^T. Here we align source → target by learning W_source
    and W_target jointly on training data, then map source via W_source W_target^T.
    """

    def __init__(self, n_components: int = 50, n_iter: int = 20) -> None:
        self.n_components = n_components
        self.n_iter = n_iter
        self.W_s: np.ndarray | None = None
        self.W_t: np.ndarray | None = None

    def fit(self, source: np.ndarray, target: np.ndarray) -> "SRMAligner":
        n_comp = min(self.n_components, source.shape[1], target.shape[1])
        # Initialise shared response as PCA of concatenated data
        concat = np.hstack([source, target])
        pca = PCA(n_components=n_comp)
        S = pca.fit_transform(concat)[:, :n_comp]

        for _ in range(self.n_iter):
            # Update W_s: orthonormal basis aligning source → S
            M_s = S.T @ source
            U, _, Vt = np.linalg.svd(M_s, full_matrices=False)
            self.W_s = (U @ Vt)  # (n_comp, n_src)
            # Update W_t
            M_t = S.T @ target
            U, _, Vt = np.linalg.svd(M_t, full_matrices=False)
            self.W_t = (U @ Vt)  # (n_comp, n_tgt)
            # Update S
            S = (source @ self.W_s.T + target @ self.W_t.T) / 2

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # Map source into shared space then back to target space
        return X @ self.W_s.T @ self.W_t


def fit_alignment(
    source: np.ndarray,  # (n_conditions, n_features)
    target: np.ndarray,
    method: str,
    cfg: Config,
) -> ProcrustesAligner | CCAAligner | SRMAligner:
    """Fit a mapping source -> target with the named method."""
    n_comp = int(cfg.raw["alignment"]["n_components"])
    if method == "procrustes":
        return ProcrustesAligner().fit(source, target)
    elif method == "cca":
        return CCAAligner(n_components=n_comp).fit(source, target)
    elif method == "srm":
        return SRMAligner(n_components=n_comp).fit(source, target)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


# ── Transfer decoding ─────────────────────────────────────────────────────────

def transfer_decode(
    aligner,
    X_train_source: np.ndarray,  # (n_trials, n_features) — source subject train
    y_train: np.ndarray,
    X_test_source: np.ndarray,   # source subject test trials
    y_test: np.ndarray,
    X_train_target: np.ndarray,  # target subject train trials (for fitting clf)
    y_train_target: np.ndarray,
) -> float:
    """Cross-subject decoding: train on target, test on aligned source.

    Returns accuracy on held-out source trials after alignment to target space.
    """
    X_train_aligned = aligner.transform(X_train_target)
    X_test_aligned = aligner.transform(X_test_source)
    clf = make_pipeline_clf()
    clf.fit(X_train_aligned, y_train_target)
    return float(clf.score(X_test_aligned, y_test))


def make_pipeline_clf():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))


# ── Alignment complexity ──────────────────────────────────────────────────────

def alignment_complexity(aligner) -> dict[str, float]:
    """Summarise the complexity of a fitted transform.

    Primary metric: orthogonality_gap = ||W^T W - I||_F
      - 0 = perfect rotation (rigid, simplest possible transform)
      - >0 = scaling/shearing involved (more complex)

    Secondary: effective_rank = (sum sv)^2 / sum(sv^2)  [participation ratio]
    """
    if isinstance(aligner, ProcrustesAligner):
        W = aligner.W
    elif isinstance(aligner, SRMAligner):
        W = aligner.W_s.T @ aligner.W_t  # effective mapping matrix
    elif isinstance(aligner, CCAAligner):
        # Use the x-weights as proxy for the mapping
        W = aligner.cca.x_weights_
    else:
        return {}

    svs = np.linalg.svd(W, compute_uv=False)
    n = min(W.shape)
    WtW = W.T @ W
    ortho_gap = float(np.linalg.norm(WtW - np.eye(WtW.shape[0]), "fro"))
    eff_rank = float(svs.sum() ** 2 / (svs ** 2).sum()) if svs.sum() > 0 else 0.0

    return {
        "orthogonality_gap": ortho_gap,
        "effective_rank": eff_rank,
        "singular_values": svs.tolist(),
    }


# ── Time-resolved alignment ───────────────────────────────────────────────────

def time_resolved_alignment(
    subject_data: dict[str, np.ndarray],  # {subject: (n_conditions, n_channels, n_times)}
    method: str,
    cfg: Config,
) -> dict[str, np.ndarray]:
    """Run alignment per timepoint between all subject pairs.

    Returns dict with:
      'times_idx'           (n_times,)
      'shape_distance'      (n_pairs, n_times) — Williams distance
      'orthogonality_gap'   (n_pairs, n_times) — complexity primary metric
      'effective_rank'      (n_pairs, n_times)
      'pair_labels'         list of (subj_a, subj_b) strings
    """
    subjects = list(subject_data.keys())
    pairs = [(i, j) for i in range(len(subjects)) for j in range(i + 1, len(subjects))]
    n_times = next(iter(subject_data.values())).shape[2]

    shape_dist = np.zeros((len(pairs), n_times))
    ortho_gap = np.zeros((len(pairs), n_times))
    eff_rank = np.zeros((len(pairs), n_times))

    for t in range(n_times):
        for pi, (i, j) in enumerate(pairs):
            sa = subjects[i]
            sb = subjects[j]
            A = subject_data[sa][:, :, t]  # (n_conditions, n_channels)
            B = subject_data[sb][:, :, t]

            shape_dist[pi, t] = williams_shape_distance(A, B)

            try:
                aligner = fit_alignment(A, B, method, cfg)
                metrics = alignment_complexity(aligner)
                ortho_gap[pi, t] = metrics.get("orthogonality_gap", np.nan)
                eff_rank[pi, t] = metrics.get("effective_rank", np.nan)
            except Exception:  # noqa: BLE001
                ortho_gap[pi, t] = np.nan
                eff_rank[pi, t] = np.nan

    pair_labels = [f"{subjects[i]}-{subjects[j]}" for i, j in pairs]

    return {
        "times_idx": np.arange(n_times),
        "shape_distance": shape_dist,
        "orthogonality_gap": ortho_gap,
        "effective_rank": eff_rank,
        "pair_labels": pair_labels,
    }
