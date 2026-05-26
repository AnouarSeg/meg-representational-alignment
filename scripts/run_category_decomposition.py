"""Category decomposition of the brain-model RSA supervision gradient (MEG).

Asks: which *categories* (and category types) drive CLIP-B/32's advantage over DINOv2?
Approach:
  1. Load per-category RSA: for each timepoint T and each pair of categories (i,j),
     compute Spearman correlation between brain RDM slice and model RDM slice over subjects.
     Then aggregate by: animate/inanimate split, coarse super-ordinate (THINGS taxonomy),
     and fine-grained within-category.
  2. Split-by-type: animate vs inanimate (from THINGS category metadata, approximated by
     concept label heuristics if metadata not available).
  3. CLIP-DINOv2 difference map: categories where CLIP > DINOv2 vs vice versa.

Memory plan:
  - brain_model_rsa_full1854.npz has pre-computed per-subject per-timepoint Spearman r
    (shape: n_subjects × n_times — that's the mean across all pairs).
  - We need per-category decomposition: reload the full RDMs and compute
    category-level RSA by comparing one row of the brain RDM to the model.

  Actually simpler: use the THINGS concept list + per-concept decoding accuracy
  and correlate with model RDM distances at the peak RSA timepoint.

  Concrete: at the peak CLIP timepoint (~325ms), extract for each category i:
    - brain_row_i = mean_subjects mean over brain RDM row i (dissimilarity to all others)
    - clip_row_i  = CLIP RDM row i
    - dino_row_i  = DINOv2 RDM row i
  Then regress (brain_row_i) on (clip_row_i, dino_row_i) per-category to see
  which category *families* drive the correlation.

Output: figures/figure4b_category_decomposition.png
        results/category_decomposition.npz
"""

from __future__ import annotations
from pathlib import Path

import numpy as np

RESULTS = Path("results")
DERIV   = Path("data/derivatives")

# ── Load brain RDMs (mean over 4 MEG subjects at each timepoint) ─────────────
# Shape of rdms_full_BIGMEG*.npz: 'rdm' key, (n_times, n_cat, n_cat)
def load_brain_rdm_mean():
    rdms = []
    times = None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists():
            print(f"  Missing {f}, skipping")
            continue
        d = np.load(str(f))
        key = "rdm" if "rdm" in d else "rdms"
        rdm = d[key].astype(np.float32)    # (n_times, n_cat, n_cat)
        rdms.append(rdm)
        if times is None:
            times = d.get("times", d.get("times_ms", np.linspace(-100, 795, rdm.shape[0])))
        print(f"  {sub}: {rdm.shape}")
    if not rdms:
        raise RuntimeError("No brain RDMs found")
    # Align to common category count
    n_cat_min = min(r.shape[1] for r in rdms)
    rdms = [r[:, :n_cat_min, :n_cat_min] for r in rdms]
    brain_mean = np.mean(rdms, axis=0)  # (n_times, n_cat, n_cat)
    return brain_mean, np.array(times)


# ── Load model RDMs ───────────────────────────────────────────────────────────
def load_model_rdm(name: str) -> np.ndarray:
    """Returns (n_cat, n_cat) condensed or full RDM."""
    f = DERIV / f"{name}_rdm.npz"
    if not f.exists():
        raise FileNotFoundError(f)
    d = np.load(str(f))
    key = "rdm" if "rdm" in d else list(d.keys())[0]
    return d[key].astype(np.float32)


# ── Animate / inanimate heuristic from concept names ─────────────────────────
ANIMATE_KEYWORDS = {
    "bird", "cat", "dog", "fish", "horse", "cow", "pig", "sheep", "elephant",
    "bear", "tiger", "lion", "monkey", "gorilla", "chimpanzee", "whale", "dolphin",
    "shark", "insect", "butterfly", "ant", "bee", "spider", "crab", "lobster",
    "frog", "turtle", "snake", "lizard", "crocodile", "person", "human", "baby",
    "child", "woman", "man", "face", "rabbit", "squirrel", "raccoon", "deer",
    "goat", "duck", "chicken", "penguin", "owl", "eagle", "parrot", "seahorse",
    "octopus", "jellyfish", "beetle", "fly", "worm", "fox", "wolf", "zebra",
    "giraffe", "panda", "koala", "kangaroo", "puppy", "kitten", "hamster", "mouse",
    "rat", "bat", "hedgehog",
}

def get_animacy(concepts):
    """Returns bool array: True = animate."""
    animate = np.zeros(len(concepts), dtype=bool)
    for i, c in enumerate(concepts):
        c_lower = c.lower().replace("_", " ").replace("-", " ")
        animate[i] = any(kw in c_lower for kw in ANIMATE_KEYWORDS)
    return animate


# ── Per-category RSA: brain row vs model row ──────────────────────────────────
def per_category_rsa(brain_t: np.ndarray, model_rdm: np.ndarray, n_cat: int) -> np.ndarray:
    """RSA score per category: Spearman r between brain RDM row and model RDM row.

    brain_t, model_rdm: (n_cat, n_cat)
    Returns: (n_cat,) per-category RSA
    """
    from scipy.stats import spearmanr
    scores = np.zeros(n_cat)
    for i in range(n_cat):
        mask = np.ones(n_cat, dtype=bool)
        mask[i] = False
        r, _ = spearmanr(brain_t[i, mask], model_rdm[i, mask])
        scores[i] = r
    return scores


if __name__ == "__main__":
    print("Loading brain RDMs...")
    brain_mean, times = load_brain_rdm_mean()
    n_times, n_cat, _ = brain_mean.shape
    print(f"Brain RDMs: {brain_mean.shape}, times: {times[[0,-1]]} ms")

    print("Loading model RDMs...")
    clip_rdm = load_model_rdm("clip_image")   # clip ViT-B/32
    dino_rdm = load_model_rdm("dinov2")

    # Align to shared categories (brain has 1854 or 1852; model may differ)
    n_shared = min(n_cat, clip_rdm.shape[0], dino_rdm.shape[0])
    brain_m  = brain_mean[:, :n_shared, :n_shared]
    clip_m   = clip_rdm[:n_shared, :n_shared]
    dino_m   = dino_rdm[:n_shared, :n_shared]
    print(f"Using {n_shared} shared categories")

    # Load concept names if available
    concepts_path = Path("data/things_concepts.txt")
    if concepts_path.exists():
        with open(str(concepts_path)) as fh:
            concepts = [line.strip() for line in fh if line.strip()][:n_shared]
    else:
        concepts = [f"cat_{i}" for i in range(n_shared)]

    animate = get_animacy(concepts)
    print(f"Animate: {animate.sum()}, Inanimate: {(~animate).sum()}")

    # Find peak timepoints for each model
    rsa_f = RESULTS / "brain_model_rsa_full1854.npz"
    if rsa_f.exists():
        d = np.load(str(rsa_f))
        clip_ts = d.get("clip_ts", None)
        dino_ts = d.get("dino_ts", None)
        rsa_times = d.get("times", times[:len(clip_ts)] if clip_ts is not None else times)
        if clip_ts is not None:
            t_clip_peak = int(np.argmax(clip_ts))
            t_dino_peak = int(np.argmax(dino_ts))
            t_clip_ms   = rsa_times[t_clip_peak]
            t_dino_ms   = rsa_times[t_dino_peak]
            print(f"CLIP peak: t={t_clip_peak} ({t_clip_ms:.0f} ms), DINOv2 peak: t={t_dino_peak} ({t_dino_ms:.0f} ms)")
        else:
            t_clip_peak = t_dino_peak = int(np.argmin(np.abs(times - 325)))
    else:
        t_clip_peak = t_dino_peak = int(np.argmin(np.abs(times - 325)))

    # Per-category RSA at peak timepoints
    print(f"Computing per-category RSA at CLIP peak t={t_clip_peak}...")
    brain_at_clip = brain_m[t_clip_peak]
    brain_at_dino = brain_m[t_dino_peak]

    clip_cat_rsa = per_category_rsa(brain_at_clip, clip_m, n_shared)
    dino_cat_rsa = per_category_rsa(brain_at_dino, dino_m, n_shared)
    clip_adv     = clip_cat_rsa - dino_cat_rsa

    print(f"CLIP mean cat RSA: {clip_cat_rsa.mean():.4f}")
    print(f"DINOv2 mean cat RSA: {dino_cat_rsa.mean():.4f}")
    print(f"CLIP advantage: {clip_adv.mean():.4f}")
    print(f"  Animate:   CLIP={clip_cat_rsa[animate].mean():.4f}  DINOv2={dino_cat_rsa[animate].mean():.4f}")
    print(f"  Inanimate: CLIP={clip_cat_rsa[~animate].mean():.4f}  DINOv2={dino_cat_rsa[~animate].mean():.4f}")

    # Time courses: split-by-animacy RSA using upper-triangle Spearman
    from scipy.stats import spearmanr
    clip_time_anim   = np.zeros(n_times)
    clip_time_inanim = np.zeros(n_times)
    dino_time_anim   = np.zeros(n_times)
    dino_time_inanim = np.zeros(n_times)

    print("Computing animacy-split time courses...")
    anim_idx   = np.where(animate)[0]
    inanim_idx = np.where(~animate)[0]

    def submatrix_corr(brain_t, model_rdm, idx):
        n = len(idx)
        b = brain_t[np.ix_(idx, idx)]
        m = model_rdm[np.ix_(idx, idx)]
        triu = np.triu_indices(n, k=1)
        r, _ = spearmanr(b[triu], m[triu])
        return r

    for t in range(n_times):
        bt = brain_m[t]
        clip_time_anim[t]   = submatrix_corr(bt, clip_m, anim_idx)
        clip_time_inanim[t] = submatrix_corr(bt, clip_m, inanim_idx)
        dino_time_anim[t]   = submatrix_corr(bt, dino_m, anim_idx)
        dino_time_inanim[t] = submatrix_corr(bt, dino_m, inanim_idx)

    np.savez(
        str(RESULTS / "category_decomposition.npz"),
        times=times, concepts=np.array(concepts), animate=animate,
        clip_cat_rsa=clip_cat_rsa, dino_cat_rsa=dino_cat_rsa,
        clip_time_anim=clip_time_anim, clip_time_inanim=clip_time_inanim,
        dino_time_anim=dino_time_anim, dino_time_inanim=dino_time_inanim,
    )
    print("Saved results/category_decomposition.npz")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Animacy-split time courses
    ax = axes[0]
    ax.plot(times, clip_time_anim*100, color="#1f77b4", lw=2, label="CLIP animate")
    ax.plot(times, clip_time_inanim*100, color="#1f77b4", lw=2, ls="--", label="CLIP inanimate")
    ax.plot(times, dino_time_anim*100, color="#ff7f0e", lw=2, label="DINOv2 animate")
    ax.plot(times, dino_time_inanim*100, color="#ff7f0e", lw=2, ls="--", label="DINOv2 inanimate")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r × 100")
    ax.set_title("Animacy split: CLIP vs DINOv2")
    ax.legend(fontsize=8)

    # Panel 2: Per-category CLIP advantage distribution (animate vs inanimate)
    ax = axes[1]
    ax.hist(clip_adv[animate],   bins=30, alpha=0.7, color="#1f77b4", label=f"Animate (n={animate.sum()})")
    ax.hist(clip_adv[~animate],  bins=30, alpha=0.7, color="#ff7f0e", label=f"Inanimate (n={(~animate).sum()})")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("CLIP − DINOv2 per-category RSA")
    ax.set_ylabel("Count")
    ax.set_title("CLIP advantage distribution by animacy")
    ax.legend(fontsize=9)

    # Panel 3: Scatter — per-category CLIP vs DINOv2 RSA
    ax = axes[2]
    ax.scatter(dino_cat_rsa[~animate], clip_cat_rsa[~animate],
               s=4, alpha=0.3, color="#ff7f0e", label="Inanimate", rasterized=True)
    ax.scatter(dino_cat_rsa[animate], clip_cat_rsa[animate],
               s=8, alpha=0.6, color="#1f77b4", label="Animate", rasterized=True)
    lim = max(abs(clip_cat_rsa).max(), abs(dino_cat_rsa).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.7, alpha=0.4)
    ax.set_xlabel("DINOv2 per-category RSA")
    ax.set_ylabel("CLIP per-category RSA")
    ax.set_title("Per-category brain-model alignment")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = Path("figures/figure4b_category_decomposition.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
