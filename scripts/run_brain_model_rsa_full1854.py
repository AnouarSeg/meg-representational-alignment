"""Brain-model RSA using full 1854-category crossnobis RDMs.

Uses our own computed crossnobis RDMs (results/rdms_full_*.npz) — full 1854
categories, avoiding the HDF5 bottleneck of the official 30.9 GB file.

Models:
  - SPOSE: human similarity judgements (1854×1854)
  - CLIP-image: ViT-B/32 image embeddings (1854×1854)
  - DINOv2: ViT-B/14 image embeddings (1854×1854)
  - ResNet50: supervised ImageNet embeddings (built here if needed)
  - CLIP-L: ViT-L/14 image embeddings (built here if needed)

Output: results/brain_model_rsa_full1854.npz
        figures/figure4_brain_model_rsa_full1854.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.io
import scipy.stats

SUBJECTS = ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]
RESULTS  = Path("results")
DERIV    = Path("data/derivatives")
FIG_OUT  = Path("figures/figure4_brain_model_rsa_full1854.png")

N_BOOT   = 1000


def rank_norm(x: np.ndarray) -> np.ndarray:
    """Rank-normalize a vector (Spearman as Pearson on ranks)."""
    return scipy.stats.rankdata(x).astype(np.float32)


def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank_norm(a), rank_norm(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float(np.dot(ra, rb) / denom) if denom > 0 else 0.0


def upper_tri(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0], k=1)
    return M[idx]


def load_brain_rdms() -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load per-subject RDMs. Returns (rdms_list, cats_list, times)."""
    rdms_list, cats_list = [], []
    times = None
    for sub in SUBJECTS:
        d = np.load(RESULTS / f"rdms_full_{sub}.npz", allow_pickle=True)
        rdms_list.append(d["rdms"].astype(np.float32))   # (T, N, N)
        cats_list.append(d["valid_categories"])
        if times is None:
            times = d["times_ms"]
    return rdms_list, cats_list, times


def intersect_categories(cats_list: list[np.ndarray]) -> np.ndarray:
    """Return sorted category indices present in all subjects."""
    common = set(cats_list[0].tolist())
    for c in cats_list[1:]:
        common &= set(c.tolist())
    return np.array(sorted(common), dtype=int)


def build_model_rdms(common_cats: np.ndarray) -> dict[str, np.ndarray]:
    """Load and subset model RDMs to common categories (0-indexed)."""
    idx = common_cats - 1   # convert to 0-indexed

    # SPOSE
    spose_sim = scipy.io.loadmat(DERIV / "spose_similarity.mat")["spose_sim"]
    spose_rdm = 1 - spose_sim
    spose_sub = upper_tri(spose_rdm[np.ix_(idx, idx)])

    # CLIP-image
    clip_d = np.load(DERIV / "clip_image_rdm.npz")
    clip_rdm = clip_d["rdm"]
    clip_sub = upper_tri(clip_rdm[np.ix_(idx, idx)])

    # DINOv2
    dino_d = np.load(DERIV / "dinov2_rdm.npz")
    dino_key = [k for k in dino_d.keys()][0]
    dino_rdm = dino_d[dino_key]
    dino_sub = upper_tri(dino_rdm[np.ix_(idx, idx)])

    models = {"SPOSE": spose_sub, "CLIP_image": clip_sub, "DINOv2": dino_sub}

    # ResNet-50 (build if not cached)
    resnet_path = DERIV / "resnet50_rdm.npz"
    if resnet_path.exists():
        r50 = np.load(resnet_path)["rdm"]
        models["ResNet50"] = upper_tri(r50[np.ix_(idx, idx)])
    else:
        print("  ResNet50 RDM not found — run build_resnet50_embeddings.py first")

    # CLIP ViT-L/14 (build if not cached)
    clip_l_path = DERIV / "clip_large_rdm.npz"
    if clip_l_path.exists():
        cl = np.load(clip_l_path)["rdm"]
        models["CLIP_L14"] = upper_tri(cl[np.ix_(idx, idx)])
    else:
        print("  CLIP-L/14 RDM not found — run build_clip_large_embeddings.py first")

    return models


def compute_rsa(rdms_list, cats_list, common_cats, model_rdms, times):
    """Compute Spearman r between brain and model RDMs per timepoint per subject."""
    n_t   = rdms_list[0].shape[0]
    n_sub = len(rdms_list)
    n_mod = len(model_rdms)
    mod_names = list(model_rdms.keys())

    rsa = {m: np.zeros((n_sub, n_t), dtype=np.float32) for m in mod_names}

    for si, (sub_rdm, sub_cats) in enumerate(zip(rdms_list, cats_list)):
        # Index into this subject's RDM for common categories
        sub_cat_list = sub_cats.tolist()
        sub_idx = np.array([sub_cat_list.index(c) for c in common_cats])
        print(f"  {SUBJECTS[si]}: {len(sub_idx)} categories")

        for t in range(n_t):
            brain_vec = upper_tri(sub_rdm[t][np.ix_(sub_idx, sub_idx)])
            for m, model_vec in model_rdms.items():
                rsa[m][si, t] = spearman_r(brain_vec, model_vec)

        if (si + 1) % 1 == 0:
            print(f"    done subject {si+1}/{n_sub}")

    return rsa


def noise_ceiling(rdms_list, cats_list, common_cats, times):
    """Nili et al. leave-one-out noise ceiling."""
    n_t   = rdms_list[0].shape[0]
    n_sub = len(rdms_list)
    nc_up = np.zeros(n_t, dtype=np.float32)
    nc_lo = np.zeros(n_t, dtype=np.float32)

    # Subset indices per subject
    sub_indices = []
    for sub_cats in cats_list:
        sub_cat_list = sub_cats.tolist()
        sub_indices.append(np.array([sub_cat_list.index(c) for c in common_cats]))

    for t in range(n_t):
        brain_vecs = [upper_tri(rdms_list[si][t][np.ix_(sub_indices[si], sub_indices[si])])
                      for si in range(n_sub)]

        # Upper: correlate each subject with mean of all
        mean_all = np.mean(brain_vecs, axis=0)
        nc_up[t] = np.mean([spearman_r(v, mean_all) for v in brain_vecs])

        # Lower: correlate each subject with mean of all others
        lo_rs = []
        for si in range(n_sub):
            others = [brain_vecs[j] for j in range(n_sub) if j != si]
            mean_others = np.mean(others, axis=0)
            lo_rs.append(spearman_r(brain_vecs[si], mean_others))
        nc_lo[t] = np.mean(lo_rs)

    return nc_up, nc_lo


def plot_results(rsa, nc_up, nc_lo, times, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"SPOSE": "#e41a1c", "CLIP_image": "#377eb8",
              "DINOv2": "#ff7f00", "ResNet50": "#4daf4a", "CLIP_L14": "#984ea3"}
    labels = {"SPOSE": "SPOSE (human similarity)", "CLIP_image": "CLIP ViT-B/32 (image)",
              "DINOv2": "DINOv2 ViT-B/14", "ResNet50": "ResNet-50 (supervised)",
              "CLIP_L14": "CLIP ViT-L/14 (image)"}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(times, nc_lo, nc_up, alpha=0.15, color="gray", label="Noise ceiling")

    for m, vals in rsa.items():
        m_mean = vals.mean(0)
        # Bootstrap CI over subjects
        rng = np.random.default_rng(0)
        boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean(0)
                          for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        c = colors.get(m, "black")
        ax.fill_between(times, lo, hi, alpha=0.15, color=c)
        ax.plot(times, m_mean, label=labels.get(m, m), color=c, lw=2)

    ax.axhline(0, color="k", ls=":", lw=0.8)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time relative to stimulus onset (ms)")
    ax.set_ylabel("Spearman r (brain–model)")
    ax.set_title(f"Brain–model RSA: full 1854 categories, n={len(SUBJECTS)} MEG subjects")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(times[0], times[-1])

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close()


if __name__ == "__main__":
    print("Loading brain RDMs...")
    rdms_list, cats_list, times = load_brain_rdms()

    print("Finding common categories...")
    common_cats = intersect_categories(cats_list)
    print(f"  Common categories: {len(common_cats)}")

    print("Loading model RDMs...")
    model_rdms = build_model_rdms(common_cats)
    print(f"  Models: {list(model_rdms.keys())}")

    print("Computing noise ceiling...")
    nc_up, nc_lo = noise_ceiling(rdms_list, cats_list, common_cats, times)
    print(f"  NC upper peak: {nc_up.max():.3f}, lower peak: {nc_lo.max():.3f}")

    print("Computing brain-model RSA...")
    rsa = compute_rsa(rdms_list, cats_list, common_cats, model_rdms, times)

    print("\n=== Results ===")
    for m, vals in rsa.items():
        m_mean = vals.mean(0)
        peak_r = m_mean.max()
        peak_t = float(times[m_mean.argmax()])
        pct_nc  = peak_r / nc_up.max() * 100
        print(f"  {m:12s}: peak r={peak_r:.4f} at {peak_t:.0f}ms  ({pct_nc:.1f}% of NC upper)")

    print("\nSaving results...")
    np.savez(str(RESULTS / "brain_model_rsa_full1854.npz"),
             times_ms=times,
             nc_upper=nc_up,
             nc_lower=nc_lo,
             common_cats=common_cats,
             **{m: v for m, v in rsa.items()})

    plot_results(rsa, nc_up, nc_lo, times, FIG_OUT)
    print("Done.")
