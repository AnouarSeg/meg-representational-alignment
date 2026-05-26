"""Add random-weights CLIP ViT-B/32 to brain-model RSA comparison.

Loads existing full-1854 results and appends the random baseline.
Output: results/brain_model_rsa_random.npz
        figures/figure4c_rsa_with_random.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RESULTS = Path("results")
DERIV   = Path("data/derivatives")


def upper_tri(m):
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def load_brain_rdms():
    rdms, times = [], None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists():
            continue
        d = np.load(str(f))
        rdm = d["rdms"].astype(np.float32)
        rdms.append(rdm)
        if times is None:
            times = d["times_ms"]
    n_cat_min = min(r.shape[1] for r in rdms)
    return [r[:, :n_cat_min, :n_cat_min] for r in rdms], np.array(times)


if __name__ == "__main__":
    print("Loading brain RDMs...")
    brain_rdms, times = load_brain_rdms()
    n_times = len(times)
    n_cat = brain_rdms[0].shape[1]
    print(f"  {len(brain_rdms)} subjects, {n_cat} categories, {n_times} timepoints")

    print("Loading model RDMs...")
    random_d = np.load(str(DERIV / "clip_random_rdm.npz"))
    random_rdm = random_d["rdm"][:n_cat, :n_cat].astype(np.float32)
    random_vec = upper_tri(random_rdm)

    # Also load trained models for comparison panel
    models = {
        "CLIP-B/32": upper_tri(np.load(str(DERIV/"clip_image_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32)),
        "DINOv2":    upper_tri(np.load(str(DERIV/"dinov2_rdm.npz"))["rdm"][:n_cat,:n_cat].astype(np.float32)),
        "CLIP-B/32 (random)": random_vec,
    }

    # Compute Spearman r per timepoint per subject
    rsa_results = {name: np.zeros((len(brain_rdms), n_times)) for name in models}

    for si, brain_rdm in enumerate(brain_rdms):
        print(f"  Subject {si+1}/{len(brain_rdms)}")
        for t in range(n_times):
            brain_vec = upper_tri(brain_rdm[t])
            for name, model_vec in models.items():
                r, _ = spearmanr(brain_vec, model_vec)
                rsa_results[name][si, t] = r

    # Mean and SEM across subjects
    rsa_mean = {n: v.mean(0) for n, v in rsa_results.items()}
    rsa_sem  = {n: v.std(0) / np.sqrt(len(brain_rdms)) for n, v in rsa_results.items()}

    print("\n=== Random-weights baseline ===")
    for name in models:
        peak = rsa_mean[name].max()
        t_peak = times[np.argmax(rsa_mean[name])]
        print(f"  {name:25s}  peak r={peak:.4f} at {t_peak:.0f} ms")

    np.savez(
        str(RESULTS / "brain_model_rsa_random.npz"),
        times=times,
        **{f"{n}_mean": rsa_mean[n] for n in models},
        **{f"{n}_sem":  rsa_sem[n]  for n in models},
    )
    print("Saved results/brain_model_rsa_random.npz")

    # Plot
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"CLIP-B/32": "#1f77b4", "DINOv2": "#ff7f0e", "CLIP-B/32 (random)": "#aaaaaa"}
    styles = {"CLIP-B/32": "-", "DINOv2": "-", "CLIP-B/32 (random)": "--"}

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in models:
        m = rsa_mean[name]
        s = rsa_sem[name]
        ax.plot(times, m, color=colors[name], ls=styles[name], lw=2, label=name)
        ax.fill_between(times, m-s, m+s, color=colors[name], alpha=0.15)

    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r (brain–model RSA)")
    ax.set_title("Brain-model RSA: trained vs random-weights CLIP")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = Path("figures/figure4c_rsa_with_random.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
