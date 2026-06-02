"""Partial RSA: regress brain RDM on all model RDMs simultaneously (OLS).

Each model's partial r reflects its unique contribution after removing
variance shared with other models. Also computes a super-subject brain RDM
(mean across all 4 MEG subjects) for maximum SNR.

Outputs:
  results/partial_rsa.npz   — partial beta + unique r per model per timepoint
  figures/figure4d_partial_rsa.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

RESULTS = Path("results")
DERIV   = Path("data/derivatives")


def upper_tri(m):
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def rank_norm(v: np.ndarray) -> np.ndarray:
    """Rank-normalize vector (for Spearman-style regression)."""
    r = rankdata(v).astype(np.float32)
    r -= r.mean()
    r /= r.std()
    return r


def partial_rsa(brain_vec: np.ndarray, model_vecs: dict) -> dict:
    """OLS regression of ranked brain RDM on ranked model RDMs.

    Returns dict: model_name → partial beta coefficient (= partial r analog).
    """
    names = list(model_vecs.keys())
    X = np.column_stack([rank_norm(model_vecs[n]) for n in names])  # (n_pairs, n_models)
    y = rank_norm(brain_vec)

    # OLS: beta = (X'X)^{-1} X'y
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return {names[i]: float(beta[i]) for i in range(len(names))}


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

    # Super-subject: mean across 4 subjects (maximises SNR)
    print("Computing super-subject RDM...")
    super_brain = np.mean(brain_rdms, axis=0)  # (n_times, n_cat, n_cat)

    print("Loading model RDMs...")
    model_rdm_data = {
        "CLIP-B/32":  np.load(str(DERIV/"clip_image_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32),
        "DINOv2":     np.load(str(DERIV/"dinov2_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32),
        "ResNet-50":  np.load(str(DERIV/"resnet50_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32),
        "CLIP-L/14":  np.load(str(DERIV/"clip_large_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32),
    }
    model_vecs = {n: upper_tri(m) for n, m in model_rdm_data.items()}

    # Also load pre-computed standard RSA for comparison
    full = np.load(str(RESULTS/"brain_model_rsa_full1854.npz"))

    print("Computing partial RSA at each timepoint...")
    partial_results = {n: np.zeros(n_times) for n in model_vecs}

    for t in range(n_times):
        brain_vec = upper_tri(super_brain[t])
        betas = partial_rsa(brain_vec, model_vecs)
        for n, b in betas.items():
            partial_results[n][t] = b
        if t % 30 == 0:
            print(f"  t={t}/{n_times} ({times[t]:.0f} ms)")

    print("\n=== Partial RSA peaks ===")
    for n in model_vecs:
        obs = partial_results[n]
        peak_t = times[np.argmax(obs)]
        print(f"  {n:12s}  peak beta={obs.max():.4f} at {peak_t:.0f} ms  "
              f"(standard r peak={full[n.replace('-','_').replace('/','_').replace('B_32','image').replace('L_14','L14').replace('ResNet_50','ResNet50')].max():.4f})")

    np.savez(str(RESULTS/"partial_rsa.npz"),
             times=times,
             **{f"partial_{n.replace('-','_').replace('/','_').replace(' ','_')}": v
                for n, v in partial_results.items()})
    print("Saved results/partial_rsa.npz")

    # Plot: partial vs standard RSA
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CLIP-B/32":"#1f77b4","DINOv2":"#ff7f0e","ResNet-50":"#9467bd","CLIP-L/14":"#8c564b"}
    std_keys = {"CLIP-B/32":"CLIP_image","DINOv2":"DINOv2","ResNet-50":"ResNet50","CLIP-L/14":"CLIP_L14"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for n, color in colors.items():
        ax.plot(times, partial_results[n], color=color, lw=2, label=n)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Partial beta (unique contribution)")
    ax.set_title("Partial RSA — unique model contributions\n(super-subject brain RDM)")
    ax.legend(fontsize=9)

    ax = axes[1]
    for n, color in colors.items():
        std = np.array(full[std_keys[n]])
        if std.ndim > 1:
            std = std.mean(0)
        part = partial_results[n]
        ax.plot(times, std, color=color, lw=1.5, ls="--", alpha=0.6, label=f"{n} standard")
        ax.plot(times, part, color=color, lw=2, label=f"{n} partial")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("r / partial beta")
    ax.set_title("Standard (dashed) vs Partial (solid) RSA")
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    out = Path("figures/figure4d_partial_rsa.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
