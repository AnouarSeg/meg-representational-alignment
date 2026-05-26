"""Apply FDR correction (Benjamini-Hochberg) to all RSA time courses.

Computes p-values via permutation (shuffle model RDM labels 1000×),
then applies BH-FDR across timepoints for each model.

Output: results/rsa_fdr.npz — adds fdr_sig_{model} boolean arrays
        figures/figure4_rsa_fdr.png — replot with FDR significance bars
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
N_PERM  = 1000
RNG     = np.random.default_rng(42)


def upper_tri(m):
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def spearman_fast(x: np.ndarray, y: np.ndarray) -> float:
    """Vectorized Spearman r — faster than scipy for repeated calls."""
    rx = rankdata(x); ry = rankdata(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float(np.dot(rx, ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns boolean significance mask."""
    n = len(pvals)
    order = np.argsort(pvals)
    threshold = (np.arange(1, n+1) / n) * alpha
    sig = np.zeros(n, dtype=bool)
    last = -1
    for i, idx in enumerate(order):
        if pvals[idx] <= threshold[i]:
            last = i
    if last >= 0:
        sig[order[:last+1]] = True
    return sig


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
    print("Loading brain RDMs...")
    brain_rdms, times = load_brain_rdms()
    n_times = len(times)
    n_cat   = brain_rdms[0].shape[1]

    # Load pre-computed RSA time courses (skip re-running slow Spearman)
    # and use them directly with permutation-derived p-values
    full = np.load(str(RESULTS / "brain_model_rsa_full1854.npz"))
    precomputed = {
        "SPOSE":     full["SPOSE"],
        "CLIP-B/32": full["CLIP_image"],
        "DINOv2":    full["DINOv2"],
        "ResNet-50": full["ResNet50"],
        "CLIP-L/14": full["CLIP_L14"],
    }
    rsa_times = full["times_ms"]
    n_times   = len(rsa_times)
    times     = rsa_times

    model_files = {
        "CLIP-B/32":  DERIV / "clip_image_rdm.npz",
        "DINOv2":     DERIV / "dinov2_rdm.npz",
        "ResNet-50":  DERIV / "resnet50_rdm.npz",
        "CLIP-L/14":  DERIV / "clip_large_rdm.npz",
    }

    # Load model vectors (for permutation null)
    model_vecs = {}
    for name, fpath in model_files.items():
        if not fpath.exists():
            print(f"  Skipping {name}")
            continue
        d = np.load(str(fpath))
        model_vecs[name] = upper_tri(d["rdm"][:n_cat, :n_cat].astype(np.float32))
        print(f"  Loaded {name}")

    # SPOSE: load from derivatives if available, else skip permutation for it
    spose_path = DERIV / "spose_rdm.npz"
    if spose_path.exists():
        model_vecs["SPOSE"] = upper_tri(np.load(str(spose_path))["rdm"][:n_cat,:n_cat].astype(np.float32))
    else:
        # No full SPOSE RDM on disk — use analytical normal approx for p-values
        model_vecs["SPOSE"] = None

    # Mean brain vector per timepoint
    print("Computing mean brain RDM per timepoint...")
    brain_mean_vecs = np.zeros((n_times, len(upper_tri(brain_rdms[0][0]))))
    for t in range(n_times):
        brain_mean_vecs[t] = np.mean([upper_tri(r[t]) for r in brain_rdms], axis=0)

    print(f"Computing {N_PERM} permutation null per model...")
    save_dict = {"times": times}

    for name in precomputed:
        print(f"  {name}...")
        obs       = precomputed[name]
        model_vec = model_vecs.get(name)

        if model_vec is None:
            # Analytical p-value: Fisher z-transform, n_eff = n_pairs
            n_eff = n_cat * (n_cat - 1) // 2
            from scipy.stats import norm as scipy_norm
            z = np.arctanh(np.clip(obs, -0.999, 0.999))
            pvals = 1 - scipy_norm.cdf(z * np.sqrt(n_eff - 3))
        else:
            null = np.zeros((N_PERM, n_times))
            perm_len = len(model_vec)
            for p in range(N_PERM):
                perm_vec = model_vec[RNG.permutation(perm_len)]
                null[p]  = [spearman_fast(brain_mean_vecs[t], perm_vec) for t in range(n_times)]
            pvals = np.maximum(np.mean(null >= obs[None, :], axis=0), 1/N_PERM)

        # Permutation null: shuffle model vec indices
        null = np.zeros((N_PERM, n_times))
        for p in range(N_PERM):
            perm_vec = model_vec[RNG.permutation(len(model_vec))]
            null[p] = [spearman_fast(brain_mean_vecs[t], perm_vec) for t in range(n_times)]

        # P-values: fraction of permutations >= observed at each timepoint
        pvals = np.mean(null >= obs[None, :], axis=0)
        pvals = np.maximum(pvals, 1 / N_PERM)  # min p = 1/N_PERM

        sig_fdr  = bh_fdr(pvals, alpha=0.05)
        sig_bonf = pvals < (0.05 / n_times)

        n_sig_fdr  = sig_fdr.sum()
        n_sig_bonf = sig_bonf.sum()
        peak_ms    = times[np.argmax(obs)]
        print(f"    peak r={obs.max():.4f} at {peak_ms:.0f}ms  |  "
              f"FDR sig: {n_sig_fdr} timepoints  |  Bonf sig: {n_sig_bonf}")

        key = name.replace("/", "_").replace("-", "_").replace(" ", "_")
        save_dict[f"rsa_{key}"]      = obs
        save_dict[f"pvals_{key}"]    = pvals
        save_dict[f"sig_fdr_{key}"]  = sig_fdr
        save_dict[f"sig_bonf_{key}"] = sig_bonf

    np.savez(str(RESULTS / "rsa_fdr.npz"), **save_dict)
    print("Saved results/rsa_fdr.npz")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "SPOSE":     "#2ca02c",
        "CLIP-B/32": "#1f77b4",
        "DINOv2":    "#ff7f0e",
        "ResNet-50": "#9467bd",
        "CLIP-L/14": "#8c564b",
    }

    fig, ax = plt.subplots(figsize=(10, 4))
    y_bar = -0.003
    bar_height = 0.0005

    for ni, name in enumerate(model_vecs.keys()):
        key = name.replace("/", "_").replace("-", "_").replace(" ", "_")
        obs     = save_dict[f"rsa_{key}"]
        sig_fdr = save_dict[f"sig_fdr_{key}"]
        color   = colors.get(name, "gray")
        ax.plot(times, obs, color=color, lw=2, label=name)
        if sig_fdr.any():
            ax.fill_between(times, y_bar - ni*bar_height*1.5,
                            y_bar - ni*bar_height*1.5 + bar_height,
                            where=sig_fdr, color=color, alpha=0.8)

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r (brain–model RSA)")
    ax.set_title("Brain-model RSA with FDR correction (colored bars = p<0.05 FDR)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path("figures/figure4_rsa_fdr.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
