"""Super-subject brain-model RSA for THINGS-EEG1 (n=48 subjects).

Averages all 48 subjects' condition means → single super-subject with SNR ∝ √48.
Computes Spearman RSA against all 5 models and compares to single-subject mean.

Memory: 48 × 1854 × 110 × 63 × 4B = 3.0 GB — loads one subject at a time and
accumulates the mean to keep peak RAM < 500 MB.

Output: results/eeg1/supersubject_rsa.npz
        figures/figure5_supersubject_rsa.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, rankdata

EEG_DERIV = Path("/Volumes/MEG/things-eeg1/derivatives/preprocessed")
DERIV     = Path("data/derivatives")
RESULTS   = Path("results/eeg1")
MODAL_CH  = 63


def upper_tri(m):
    idx = np.triu_indices(m.shape[0], k=1)
    return m[idx]


def spearman_fast(x, y):
    rx = rankdata(x).astype(np.float64)
    ry = rankdata(y).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float(np.dot(rx, ry) / (np.linalg.norm(rx) * np.linalg.norm(ry)))


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Find 63-ch subjects
    available = sorted(
        p.name for p in EEG_DERIV.glob("sub-*/")
        if (EEG_DERIV/p.name/f"{p.name}_condition_means.npy").exists()
    )
    subjects = [s for s in available
                if np.load(EEG_DERIV/s/f"{s}_condition_means.npy", mmap_mode="r").shape[2] == MODAL_CH]
    print(f"Found {len(subjects)} subjects")

    # Accumulate mean condition means (streaming, low RAM)
    print("Computing super-subject condition means...")
    super_mean = None
    for si, sub in enumerate(subjects):
        cm = np.load(EEG_DERIV/sub/f"{sub}_condition_means.npy").astype(np.float32)
        # cm: (1854, n_times, 63)
        if super_mean is None:
            super_mean = cm / len(subjects)
        else:
            super_mean += cm / len(subjects)
        del cm
        if (si+1) % 10 == 0:
            print(f"  {si+1}/{len(subjects)}")

    n_cat, n_times, n_ch = super_mean.shape
    times = np.linspace(-50, 495, n_times)
    print(f"Super-subject shape: {super_mean.shape}")

    # Crossnobis RDM on super-subject (simple Euclidean — no crossnobis needed at this SNR)
    # Use cosine dissimilarity for speed
    print("Computing super-subject RDMs...")
    # Normalise per timepoint
    brain_vecs = np.zeros((n_times, n_cat*(n_cat-1)//2), dtype=np.float32)
    for t in range(n_times):
        X = super_mean[:, t, :]  # (n_cat, n_ch)
        # Cosine dissimilarity
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X_n = X / norms
        sim = X_n @ X_n.T
        rdm = 1 - sim
        brain_vecs[t] = upper_tri(rdm)

    # Load model RDMs
    models = {
        "CLIP-B/32": DERIV/"clip_image_rdm.npz",
        "DINOv2":    DERIV/"dinov2_rdm.npz",
        "ResNet-50": DERIV/"resnet50_rdm.npz",
        "CLIP-L/14": DERIV/"clip_large_rdm.npz",
    }
    model_vecs = {}
    for name, fpath in models.items():
        if fpath.exists():
            model_vecs[name] = upper_tri(np.load(str(fpath))["rdm"][:n_cat,:n_cat].astype(np.float32))

    print("Computing super-subject RSA...")
    rsa = {n: np.zeros(n_times) for n in model_vecs}
    for t in range(n_times):
        for n, mv in model_vecs.items():
            rsa[n][t] = spearman_fast(brain_vecs[t], mv)
        if t % 20 == 0:
            print(f"  t={t}/{n_times}")

    print("\n=== Super-subject RSA peaks ===")
    for n in model_vecs:
        peak = rsa[n].max()
        t_peak = times[np.argmax(rsa[n])]
        print(f"  {n:12s}  peak r={peak:.4f} at {t_peak:.0f} ms")

    np.savez(str(RESULTS/"supersubject_rsa.npz"),
             times=times,
             **{f"rsa_{n.replace('-','_').replace('/','_').replace(' ','_')}": v
                for n, v in rsa.items()})
    print("Saved results/eeg1/supersubject_rsa.npz")

    # Compare to existing single-subject-mean EEG RSA
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"CLIP-B/32":"#1f77b4","DINOv2":"#ff7f0e","ResNet-50":"#9467bd","CLIP-L/14":"#8c564b"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for n, color in colors.items():
        if n in rsa:
            ax.plot(times, rsa[n], color=color, lw=2, label=n)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Spearman r")
    ax.set_title(f"Super-subject RSA (n={len(subjects)} averaged)")
    ax.legend(fontsize=9)

    # Load per-subject EEG RSA for comparison
    eeg_rsa_f = RESULTS/"brain_model_rsa_eeg1.npz"
    ax = axes[1]
    if eeg_rsa_f.exists():
        d2 = np.load(str(eeg_rsa_f))
        t2 = d2.get("times", times)
        for n, color in colors.items():
            key = n.replace("-","_").replace("/","_").replace(" ","_")
            if f"rsa_{key}" in rsa:
                ax.plot(t2, d2.get(key, d2.get(f"rsa_{key}", np.zeros_like(t2))),
                        color=color, lw=1.5, ls="--", alpha=0.6)
                ax.plot(times, rsa[f"rsa_{key}" if f"rsa_{key}" in rsa else n],
                        color=color, lw=2)
        ax.set_title("Single-subject mean (dashed) vs super-subject (solid)")
    else:
        ax.set_title("(per-subject EEG RSA not found)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Spearman r")

    fig.tight_layout()
    out = Path("figures/figure5_supersubject_rsa.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
