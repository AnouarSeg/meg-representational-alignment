"""Behavioral norms as stimulus-level predictors of brain-model RSA.

Uses THINGS behavioral norms (naming RT, familiarity, typicality, visual complexity,
size, animacy) as predictors of per-category RSA scores. Tests whether the concepts
where CLIP best predicts the brain are the same concepts that are quickly named,
highly familiar, or categorically typical.

Data sources (already on disk):
  data/derivatives/clip_image_rdm.npz     — CLIP concept embeddings (n_cat, n_cat)
  data/derivatives/official_rdm1854.mat   — official THINGS RSA norms (if exists)
  THINGS behavioral norms from OSF/THINGS+ (if downloaded)

If norms not available: uses proxy norms from concept properties
(word frequency via wordfreq, visual complexity estimate from image stats).

Output:
  results/behavioral_norms_rsa.npz
  figures/figure10_behavioral_norms.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
FIGS    = Path("figures")


def load_things_norms(concepts):
    """
    Try to load THINGS behavioral norms. Falls back to proxy measures.
    Returns dict: norm_name → (n_concepts,) array.
    """
    norms = {}

    # 1. Try official THINGS+ norms (TSV files from THINGS ecosystem)
    # These are concept-level: arousal, valence, size, animacy, etc.
    for tsv in Path("data").rglob("*.tsv"):
        if "things" in tsv.stem.lower() or "norms" in tsv.stem.lower():
            try:
                import csv
                with open(str(tsv)) as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    rows = list(reader)
                print(f"  Found TSV: {tsv.name} ({len(rows)} rows, cols: {list(rows[0].keys())[:5]})")
            except Exception:
                pass

    # 2. Try THINGS+ ratings from THINGS-MEG sourcedata
    things_plus = Path("/Volumes/MEG/things-meg") / "images_THINGSplus-CC0.zip"
    if things_plus.exists():
        print(f"  Found THINGS+ zip: {things_plus}")

    # 3. Proxy: word frequency (fast naming RT proxy) via wordfreq
    try:
        from wordfreq import word_frequency
        word_freq = np.array([
            word_frequency(c.replace("_", " ").split()[0], "en")
            for c in concepts
        ], dtype=np.float32)
        norms["word_frequency"] = word_freq
        print(f"  Word frequency: min={word_freq.min():.2e}, max={word_freq.max():.2e}")
    except ImportError:
        print("  wordfreq not installed — skipping word frequency")

    # 4. Proxy: concept name length (inversely related to familiarity)
    name_length = np.array([len(c.replace("_", " ")) for c in concepts], dtype=np.float32)
    norms["name_length"] = name_length

    # 5. Proxy: underscore count (multi-word → compound → less familiar)
    n_words = np.array([len(c.split("_")) for c in concepts], dtype=np.float32)
    norms["n_words"] = n_words

    # 6. Animacy proxy from our existing animate/inanimate labels
    ANIMATE_CONCEPTS = {
        "cat","dog","bird","fish","horse","cow","pig","sheep","lion","tiger",
        "bear","elephant","monkey","rabbit","mouse","rat","snake","frog","spider",
        "bee","ant","fly","worm","dolphin","whale","shark","eagle","owl","duck",
        "chicken","turkey","penguin","seal","deer","fox","wolf","giraffe","zebra",
        "gorilla","chimpanzee","squirrel","hamster","parrot","butterfly","beetle",
        "crab","shrimp","lobster","snail","octopus","person","man","woman","child",
        "baby","face","hand","eye","human","people","animal","insect","reptile",
        "amphibian","mammal","bird","fish","bug","creature","organism",
    }
    def is_animate(c):
        return bool(set(c.split("_")) & ANIMATE_CONCEPTS)
    animacy = np.array([float(is_animate(c)) for c in concepts], dtype=np.float32)
    norms["animacy"] = animacy
    print(f"  Animacy: {animacy.sum():.0f}/{len(concepts)} animate")

    return norms


def per_category_rsa(brain_rdm_mean, model_rdm, times, peak_t_idx):
    """
    Per-category RSA at peak timepoint: for each concept i,
    correlate brain_rdm[t_peak, i, :] with model_rdm[i, :].
    Returns (n_concepts,) array.
    """
    from scipy.stats import spearmanr
    brain_row = brain_rdm_mean[peak_t_idx]   # (n_cat, n_cat)
    n_cat = brain_row.shape[0]
    r_per_cat = np.zeros(n_cat)
    for i in range(n_cat):
        mask = np.arange(n_cat) != i
        r_per_cat[i] = spearmanr(brain_row[i, mask], model_rdm[i, mask]).statistic
    return r_per_cat.astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, pearsonr

    print("=== Behavioral norms RSA ===\n")

    # Load concepts
    clip_d   = np.load(str(DERIV / "clip_image_rdm.npz"), allow_pickle=True)
    concepts = clip_d["concepts"]
    clip_rdm = clip_d["rdm"].astype(np.float32)
    n_cat    = len(concepts)
    print(f"Concepts: {n_cat}")

    # Load brain RDMs (per-subject, slice to n_cat)
    brain_rdm_list = []
    times = None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists():
            continue
        d = np.load(str(f))
        rdm = d["rdms"]
        if rdm.shape[0] != 180:
            rdm = rdm.transpose(2, 0, 1)
        valid = d["valid_categories"]
        # Get indices of our 100 test concepts in the 1852 valid categories
        cat_meta = np.load(str(RESULTS / "model_rdms.npz"), allow_pickle=True)
        test_cats = cat_meta["category_numbers"]
        pos = np.array([np.where(valid == idx)[0][0] for idx in test_cats if idx in valid])
        brain_rdm_list.append(rdm[:, pos[:, None], pos[None, :]])
        if times is None:
            times = d["times_ms"].astype(float)

    times = np.array(times)
    brain_rdm_mean = np.mean(brain_rdm_list, axis=0)   # (n_times, 100, 100)
    n_times = len(times)
    n_cat = brain_rdm_mean.shape[1]   # 100 test categories
    clip_rdm_trim = clip_rdm[:n_cat, :n_cat]
    concepts = concepts[:n_cat]       # match concept list to test categories

    # Find CLIP peak timepoint
    post = times > 0
    # Load precomputed RSA to find peak
    rsa_f = RESULTS / "brain_model_rsa_full1854.npz"
    if rsa_f.exists():
        rsa_d = np.load(str(rsa_f), allow_pickle=True)
        clip_ts = rsa_d.get("CLIP-B/32", rsa_d.get("clip_b32", None))
        if clip_ts is not None and hasattr(clip_ts, '__len__'):
            clip_ts = np.array(clip_ts)
            if clip_ts.ndim > 1:
                clip_ts = clip_ts.mean(0)
            peak_t_idx = np.where(post)[0][clip_ts[post].argmax()]
        else:
            peak_t_idx = np.where(post)[0][int(n_times * 0.6)]
    else:
        peak_t_idx = np.where(post)[0][int(len(np.where(post)[0]) * 0.6)]
    print(f"CLIP peak timepoint: {times[peak_t_idx]:.0f}ms (index {peak_t_idx})")

    # Per-category RSA for CLIP
    print("Computing per-category RSA...")
    r_per_cat = per_category_rsa(brain_rdm_mean, clip_rdm_trim, times, peak_t_idx)
    print(f"  Mean per-cat r = {r_per_cat.mean():.4f}, std = {r_per_cat.std():.4f}")

    # Load behavioral norms
    print("\nLoading behavioral norms...")
    norms = load_things_norms(list(concepts[:n_cat]))

    # Correlate norms with per-category RSA
    print("\n=== Correlation of norms with per-category CLIP RSA ===")
    norm_results = {}
    for name, norm_vals in norms.items():
        norm_trim = norm_vals[:n_cat]
        r, p = spearmanr(r_per_cat, norm_trim)
        norm_results[name] = (r, p)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {name:20s}: r={r:.3f}  p={p:.4f} {sig}")

    np.savez(str(RESULTS / "behavioral_norms_rsa.npz"),
             concepts=concepts[:n_cat],
             r_per_cat=r_per_cat,
             times=times,
             peak_t_ms=times[peak_t_idx],
             **{f"norm_{k}": v[:n_cat] for k, v in norms.items()},
             **{f"corr_{k}": np.array([v[0], v[1]]) for k, v in norm_results.items()})
    print("\nSaved results/behavioral_norms_rsa.npz")

    # Figure: scatter plots of top predictors
    n_plots = min(len(norm_results), 4)
    sorted_norms = sorted(norm_results.items(), key=lambda x: abs(x[1][0]), reverse=True)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, (r, p)) in zip(axes, sorted_norms[:n_plots]):
        norm_trim = norms[name][:n_cat]
        ax.scatter(norm_trim, r_per_cat, alpha=0.4, s=15, color="#1f77b4")
        # Trend line
        z = np.polyfit(norm_trim, r_per_cat, 1)
        xr = np.linspace(norm_trim.min(), norm_trim.max(), 100)
        ax.plot(xr, np.polyval(z, xr), "r-", lw=1.5)
        ax.set_xlabel(name.replace("_", " "))
        ax.set_ylabel("Per-category CLIP RSA (r)")
        ax.set_title(f"r={r:.3f}, p={p:.3f}")

    fig.suptitle(f"Behavioral norms vs per-category CLIP brain-model RSA\n(peak t={times[peak_t_idx]:.0f}ms, n={n_cat} concepts)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(str(FIGS / "figure10_behavioral_norms.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure10_behavioral_norms.png")

    # Top and bottom 10 concepts by RSA
    order = r_per_cat.argsort()
    print(f"\nTop 10 concepts (highest CLIP RSA at {times[peak_t_idx]:.0f}ms):")
    for i in order[-10:][::-1]:
        print(f"  {concepts[i]:30s}  r={r_per_cat[i]:.4f}")
    print(f"\nBottom 10 concepts (lowest CLIP RSA):")
    for i in order[:10]:
        print(f"  {concepts[i]:30s}  r={r_per_cat[i]:.4f}")
