"""Brain-model RSA on THINGS-EEG1 (n=48 subjects).

Uses per-subject condition means to compute crossnobis RDMs on-the-fly
(memory-efficient: one subject at a time), then correlates with model RDMs.

With 48 subjects the noise ceiling is much tighter, giving a clearer picture
of model-brain alignment than the MEG n=4 analysis.

Output: results/eeg1/brain_model_rsa_eeg1.npz
        figures/figure4b_brain_model_rsa_eeg1.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.io
import scipy.stats

DERIV   = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
DERIV_M = Path("data/derivatives")
RESULTS = Path("results/eeg1")
FIG_OUT = Path("figures/figure4b_brain_model_rsa_eeg1.png")
MODAL_CH = 63
N_BOOT   = 1000

MODELS = {
    "SPOSE":      DERIV_M / "spose_similarity.mat",
    "CLIP_image": DERIV_M / "clip_image_rdm.npz",
    "DINOv2":     DERIV_M / "dinov2_rdm.npz",
    "ResNet50":   DERIV_M / "resnet50_rdm.npz",
    "CLIP_L14":   DERIV_M / "clip_large_rdm.npz",
}


def upper_tri(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


def spearman_r(a, b):
    ra = scipy.stats.rankdata(a).astype(np.float32)
    rb = scipy.stats.rankdata(b).astype(np.float32)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float(np.dot(ra, rb) / d) if d > 0 else 0.0


def crossnobis_upper(means: np.ndarray, t: int) -> np.ndarray:
    """Crossnobis dissimilarity upper triangle at timepoint t."""
    from sklearn.covariance import LedoitWolf
    X = means[:, t, :]   # (N, ch)
    lw = LedoitWolf(assume_centered=True)
    lw.fit(X - X.mean(0))
    P  = lw.precision_.astype(np.float32)
    XP = X @ P
    diag = (XP * X).sum(1)
    gram = XP @ X.T
    D = (diag[:, None] - gram - gram.T + diag[None, :]) / 2
    return upper_tri(D.astype(np.float32))


def load_model_rdms() -> dict[str, np.ndarray]:
    """Load full 1854×1854 model RDMs, return upper triangles."""
    out = {}
    # SPOSE
    sim = scipy.io.loadmat(DERIV_M / "spose_similarity.mat")["spose_sim"]
    out["SPOSE"] = upper_tri((1 - sim).astype(np.float32))
    # npz models
    for name, path in MODELS.items():
        if name == "SPOSE" or not path.exists():
            continue
        d = np.load(str(path))
        rdm = d["rdm"] if "rdm" in d else d[list(d.keys())[0]]
        out[name] = upper_tri(rdm.astype(np.float32))
    print(f"Models loaded: {list(out.keys())}")
    return out


def get_subjects() -> list[str]:
    subs = []
    for p in sorted(DERIV.glob("sub-*/")):
        means_p = p / f"{p.name}_condition_means.npy"
        if not means_p.exists():
            continue
        arr = np.load(str(means_p), mmap_mode="r")
        if arr.shape[2] == MODAL_CH:
            subs.append(p.name)
    return subs


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    model_rdms = load_model_rdms()
    subjects   = get_subjects()
    print(f"Subjects: {len(subjects)}")

    # Determine n_times from first subject
    s0 = np.load(str(DERIV / subjects[0] / f"{subjects[0]}_condition_means.npy"),
                 mmap_mode="r")
    n_times = s0.shape[1]
    times   = np.linspace(-50, 495, n_times)

    n_sub  = len(subjects)
    mod_names = list(model_rdms.keys())

    # Per-subject RSA: (n_sub, n_times) per model
    rsa = {m: np.zeros((n_sub, n_times), dtype=np.float32) for m in mod_names}

    for si, sub in enumerate(subjects):
        means = np.load(str(DERIV / sub / f"{sub}_condition_means.npy"))
        print(f"[{si+1}/{n_sub}] {sub}  shape={means.shape}")
        means = (means - means.mean(0, keepdims=True)) / (means.std(0, keepdims=True) + 1e-8)

        for t in range(n_times):
            brain_vec = crossnobis_upper(means, t)
            for m, model_vec in model_rdms.items():
                rsa[m][si, t] = spearman_r(brain_vec, model_vec)

        if (si + 1) % 5 == 0:
            print(f"  {si+1}/{n_sub} done, SPOSE peak so far: {rsa['SPOSE'][:si+1].mean(0).max():.4f}")
        del means

    # Noise ceiling at all timepoints — use mean brain RDM across subjects approach
    # (too expensive for all timepoints; compute at 20 representative timepoints)
    # Noise ceiling: reuse already-computed RSA arrays
    # Upper NC: correlate each subject's brain RDM with mean of all subjects
    # Lower NC: correlate each subject's brain RDM with mean of all others
    # We approximate this using the RSA values themselves at representative timepoints
    # by treating the model as the "average brain" — but for a proper NC we need
    # brain-brain correlations. Compute at 10 sparse timepoints using stored brain vecs.
    print("Computing noise ceiling (10 sparse timepoints from stored brain RDMs)...")
    nc_times_idx = np.linspace(0, n_times-1, 10, dtype=int)
    nc_up_sparse = np.zeros(10)
    nc_lo_sparse = np.zeros(10)

    # Load all subjects' condition means once, compute brain vecs at sparse timepoints
    all_means = []
    for sub in subjects:
        m = np.load(str(DERIV / sub / f"{sub}_condition_means.npy"))
        m = (m - m.mean(0, keepdims=True)) / (m.std(0, keepdims=True) + 1e-8)
        all_means.append(m)
    print(f"  Loaded {len(all_means)} subjects into memory")

    for ki, ti in enumerate(nc_times_idx):
        brain_vecs = np.array([crossnobis_upper(m, int(ti)) for m in all_means])
        mean_all = brain_vecs.mean(0)
        nc_up_sparse[ki] = float(np.mean([spearman_r(v, mean_all) for v in brain_vecs]))
        lo_rs = [spearman_r(brain_vecs[si],
                            np.delete(brain_vecs, si, axis=0).mean(0))
                 for si in range(n_sub)]
        nc_lo_sparse[ki] = float(np.mean(lo_rs))
        print(f"  t={times[ti]:.0f}ms: NC_up={nc_up_sparse[ki]:.3f}, NC_lo={nc_lo_sparse[ki]:.3f}")

    del all_means

    nc_up = np.interp(np.arange(n_times), nc_times_idx, nc_up_sparse).astype(np.float32)
    nc_lo = np.interp(np.arange(n_times), nc_times_idx, nc_lo_sparse).astype(np.float32)

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    rsa_boots = {m: np.percentile(
        np.array([rsa[m][rng.integers(0, n_sub, n_sub)].mean(0) for _ in range(N_BOOT)]),
        [2.5, 97.5], axis=0) for m in mod_names}

    print("\n=== EEG1 Brain-Model RSA Results ===")
    for m in mod_names:
        m_mean = rsa[m].mean(0)
        print(f"  {m:12s}: peak r={m_mean.max():.4f} at {times[m_mean.argmax()]:.0f}ms  "
              f"({m_mean.max()/nc_up.max()*100:.1f}% of NC upper)")

    np.savez(str(RESULTS / "brain_model_rsa_eeg1.npz"),
             times=times, nc_upper=nc_up, nc_lower=nc_lo,
             **{m: rsa[m] for m in mod_names})

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"SPOSE":"#e41a1c","CLIP_image":"#377eb8","DINOv2":"#ff7f00",
              "ResNet50":"#4daf4a","CLIP_L14":"#984ea3"}
    labels = {"SPOSE":"SPOSE","CLIP_image":"CLIP ViT-B/32","DINOv2":"DINOv2 ViT-B/14",
              "ResNet50":"ResNet-50","CLIP_L14":"CLIP ViT-L/14"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(times, nc_lo, nc_up, alpha=0.12, color="gray", label="Noise ceiling")
    for m in mod_names:
        m_mean = rsa[m].mean(0)
        lo, hi = rsa_boots[m]
        c = colors.get(m, "black")
        ax.fill_between(times, lo, hi, alpha=0.15, color=c)
        ax.plot(times, m_mean, label=labels.get(m, m), color=c, lw=2)
    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time relative to stimulus onset (ms)")
    ax.set_ylabel("Spearman r (brain–model)")
    ax.set_title(f"EEG1 brain–model RSA: n={n_sub} subjects, 1854 categories")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(str(FIG_OUT), dpi=150, bbox_inches="tight")
    print(f"Saved {FIG_OUT}")
