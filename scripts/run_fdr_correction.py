"""FDR correction for brain-model RSA time courses (vectorized).

Uses pre-ranked brain/model vectors + matrix multiply for permutation null.
All 1000 permutations × 180 timepoints computed in seconds via numpy.

Output: results/rsa_fdr.npz
        figures/figure4_rsa_fdr.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
N_PERM  = 1000
RNG     = np.random.default_rng(42)


def rank_norm_matrix(M: np.ndarray) -> np.ndarray:
    """Rank-normalize each row of M (n_times, n_pairs) → same shape."""
    out = np.zeros_like(M, dtype=np.float32)
    for i in range(M.shape[0]):
        r = rankdata(M[i]).astype(np.float32)
        r -= r.mean(); r /= (r.std() + 1e-12)
        out[i] = r
    return out


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    threshold = (np.arange(1, n+1) / n) * alpha
    sig = np.zeros(n, dtype=bool)
    last = -1
    for i in range(n):
        if pvals[order[i]] <= threshold[i]:
            last = i
    if last >= 0:
        sig[order[:last+1]] = True
    return sig


def upper_tri(m): return m[np.triu_indices(m.shape[0], k=1)]


def load_brain_rdms():
    rdms, times = [], None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists(): continue
        d = np.load(str(f))
        rdm = d["rdms"].astype(np.float32)
        rdms.append(rdm)
        if times is None: times = d["times_ms"]
    n_cat_min = min(r.shape[1] for r in rdms)
    return [r[:, :n_cat_min, :n_cat_min] for r in rdms], np.array(times)


if __name__ == "__main__":
    print("Loading brain RDMs and computing mean per timepoint...")
    brain_rdms, times = load_brain_rdms()
    n_times = len(times)
    n_cat   = brain_rdms[0].shape[1]
    n_pairs = n_cat * (n_cat - 1) // 2

    # Mean brain RDM upper-triangle across subjects: (n_times, n_pairs)
    brain_mat = np.zeros((n_times, n_pairs), dtype=np.float32)
    for r in brain_rdms:
        for t in range(n_times):
            brain_mat[t] += upper_tri(r[t]) / len(brain_rdms)

    # Pre-rank brain matrix rows: (n_times, n_pairs)
    print("Rank-normalizing brain matrix...")
    brain_ranked = rank_norm_matrix(brain_mat)  # (n_times, n_pairs)
    del brain_mat

    # Load model vectors and pre-rank
    model_files = {
        "CLIP-B/32": DERIV / "clip_image_rdm.npz",
        "DINOv2":    DERIV / "dinov2_rdm.npz",
        "ResNet-50": DERIV / "resnet50_rdm.npz",
        "CLIP-L/14": DERIV / "clip_large_rdm.npz",
    }
    model_ranked = {}
    for name, fpath in model_files.items():
        if not fpath.exists(): continue
        vec = upper_tri(np.load(str(fpath))["rdm"][:n_cat,:n_cat].astype(np.float32))
        r = rankdata(vec).astype(np.float32)
        r -= r.mean(); r /= (r.std() + 1e-12)
        model_ranked[name] = r
        print(f"  Loaded {name}")

    # Pre-computed mean RSA from saved results (for observed values)
    full = np.load(str(RESULTS / "brain_model_rsa_full1854.npz"))
    obs_lookup = {
        "CLIP-B/32": np.array(full["CLIP_image"]).mean(0),
        "DINOv2":    np.array(full["DINOv2"]).mean(0),
        "ResNet-50": np.array(full["ResNet50"]).mean(0),
        "CLIP-L/14": np.array(full["CLIP_L14"]).mean(0),
    }

    # SPOSE: analytical p-value only (no full RDM on disk)
    spose_obs = np.array(full["SPOSE"]).mean(0)
    n_eff = n_pairs
    from scipy.stats import norm as scipy_norm
    z = np.arctanh(np.clip(spose_obs, -0.999, 0.999))
    spose_pvals = np.clip(1 - scipy_norm.cdf(z * np.sqrt(n_eff - 3)), 1e-6, 1.0)
    spose_sig_fdr = bh_fdr(spose_pvals)
    print(f"  SPOSE (analytical): FDR sig {spose_sig_fdr.sum()} timepoints, "
          f"peak r={spose_obs.max():.4f} at {times[spose_obs.argmax()]:.0f}ms")

    print(f"Running {N_PERM} permutations (vectorized)...")
    save_dict = {"times": times,
                 "rsa_SPOSE": spose_obs, "pvals_SPOSE": spose_pvals,
                 "sig_fdr_SPOSE": spose_sig_fdr,
                 "sig_bonf_SPOSE": spose_pvals < 0.05/n_times}

    for name, model_r in model_ranked.items():
        print(f"  {name}...", end=" ", flush=True)
        obs = obs_lookup[name]  # (n_times,) observed Spearman r

        # Vectorized permutation: brain_ranked @ perm(model_r) for N_PERM perms
        # brain_ranked: (n_times, n_pairs), model_r: (n_pairs,)
        # For each perm, shuffle model_r indices and dot with all timepoints at once
        null_max = np.zeros(N_PERM)
        null_dist = np.zeros((N_PERM, n_times), dtype=np.float32)
        for p in range(N_PERM):
            perm_r = model_r[RNG.permutation(n_pairs)]
            # Spearman r = dot(ranked_brain[t], perm_r) / n_pairs (already normalized)
            null_dist[p] = brain_ranked @ perm_r / n_pairs

        pvals = np.maximum(np.mean(null_dist >= obs[None, :], axis=0), 1/N_PERM)
        sig_fdr  = bh_fdr(pvals)
        sig_bonf = pvals < (0.05 / n_times)

        key = name.replace("/","_").replace("-","_").replace(" ","_")
        save_dict[f"rsa_{key}"]      = obs
        save_dict[f"pvals_{key}"]    = pvals
        save_dict[f"sig_fdr_{key}"]  = sig_fdr
        save_dict[f"sig_bonf_{key}"] = sig_bonf
        print(f"FDR sig: {sig_fdr.sum()} pts | Bonf: {sig_bonf.sum()} pts | peak r={obs.max():.4f}")

    np.savez(str(RESULTS / "rsa_fdr.npz"), **save_dict)
    print("Saved results/rsa_fdr.npz")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"SPOSE":"#2ca02c","CLIP-B/32":"#1f77b4","DINOv2":"#ff7f0e",
              "ResNet-50":"#9467bd","CLIP-L/14":"#8c564b"}
    all_models = list(colors.keys())

    fig, ax = plt.subplots(figsize=(10, 4))
    y0 = -0.006
    for ni, name in enumerate(all_models):
        key = name.replace("/","_").replace("-","_").replace(" ","_")
        obs_k  = f"rsa_{key}"
        sig_k  = f"sig_fdr_{key}"
        if obs_k not in save_dict: continue
        obs = save_dict[obs_k]
        sig = save_dict[sig_k]
        ax.plot(times, obs, color=colors[name], lw=2, label=name)
        if sig.any():
            ax.fill_between(times, y0 - ni*0.0008, y0 - ni*0.0008 + 0.0006,
                            where=sig, color=colors[name], alpha=0.85)

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Spearman r")
    ax.set_title("Brain-model RSA with BH-FDR significance (colored bars below)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path("figures/figure4_rsa_fdr.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
