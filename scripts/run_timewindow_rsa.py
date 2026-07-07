"""Time-window averaged brain-model RSA.

Instead of correlating model RDMs with brain RDMs at single timepoints,
average the brain RDM over a sliding window of width W ms before computing
Spearman r. This increases SNR without overfitting (no free parameters —
window width chosen a priori from decoding peak latency literature).

Windows tested: 25, 50, 100 ms (centred, causal-padding at edges).
Primary window: 50 ms (standard in MEG RSA literature).

Output:
  results/timewindow_rsa.npz
  figures/figure8_timewindow_rsa.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
FIGS    = Path("figures")

WINDOW_MS  = 50   # primary window width
MODELS     = ["SPOSE", "CLIP-B/32", "CLIP-L/14", "ResNet-50", "DINOv2"]
COLORS     = {
    "SPOSE":     "#2ecc71",
    "CLIP-B/32": "#1f77b4",
    "CLIP-L/14": "#aec7e8",
    "ResNet-50": "#9467bd",
    "DINOv2":    "#ff7f0e",
}


def lower_tri(rdm):
    n = rdm.shape[-1]
    idx = np.tril_indices(n, k=-1)
    return rdm[..., idx[0], idx[1]]


def window_average_rdm(brain_tri, times, width_ms):
    """Apply centred sliding average of `width_ms` to each pair's time series."""
    dt = np.median(np.diff(times))           # ms per sample
    half = int(round(width_ms / 2 / dt))
    n_times, n_pairs = brain_tri.shape
    smoothed = np.empty_like(brain_tri)
    for t in range(n_times):
        lo = max(0, t - half)
        hi = min(n_times, t + half + 1)
        smoothed[t] = brain_tri[lo:hi].mean(axis=0)
    return smoothed


def ranked(v):
    from scipy.stats import rankdata
    return rankdata(v).astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Load per-subject brain RDMs (1852-cat, n_times × n_cat × n_cat) ──────
    sub_data = {}
    times = None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists():
            print(f"  Skipping {sub} — file not found")
            continue
        d = np.load(str(f))
        rdm = d["rdms"]
        if rdm.shape[0] != 180:
            rdm = rdm.transpose(2, 0, 1)
        sub_data[sub] = {"rdm": rdm, "valid": d["valid_categories"]}
        if times is None:
            times = d["times_ms"].astype(float)

    # Intersect valid categories across subjects
    valid_sets = [set(v["valid"].tolist()) for v in sub_data.values()]
    shared_cats = sorted(set.intersection(*valid_sets))
    print(f"Shared categories across subjects: {len(shared_cats)}")

    brain_tri_list = []
    for sub, info in sub_data.items():
        valid = info["valid"]
        pos   = np.array([np.where(valid == c)[0][0] for c in shared_cats])
        rdm_s = info["rdm"][:, pos[:, None], pos[None, :]]
        brain_tri_list.append(lower_tri(rdm_s))

    times = np.array(times)
    n_times = len(times)
    brain_tri_mean = np.mean(brain_tri_list, axis=0).astype(np.float32)
    print(f"Brain RDM: {brain_tri_mean.shape[1]} pairs, {n_times} timepoints")

    # ── Load model RDMs ───────────────────────────────────────────────────────
    model_tri = {}

    # SPOSE: from brain_model_rsa_full1854.npz (pre-averaged across subjects)
    d_full = np.load(str(RESULTS / "brain_model_rsa_full1854.npz"), allow_pickle=True)
    # We need raw model RDM, not RSA curve. Load from derivatives.
    for fname, key in [
        ("clip_image_rdm.npz",  "CLIP-B/32"),
        ("clip_large_rdm.npz",  "CLIP-L/14"),
        ("dinov2_rdm.npz",      "DINOv2"),
        ("resnet50_rdm.npz",    "ResNet-50"),
    ]:
        p = DERIV / fname
        if not p.exists():
            print(f"  Missing {fname}, skipping {key}")
            continue
        rdm = np.load(str(p))["rdm"].astype(np.float32)
        model_tri[key] = lower_tri(rdm)
        print(f"  {key}: {rdm.shape} → {model_tri[key].shape[0]} pairs")

    # SPOSE: load from model_rdms_full.npz or spose_full_rdm.npz
    for candidate in ["spose_full_rdm.npz", "model_rdms.npz"]:
        p = DERIV / candidate
        if p.exists():
            d = np.load(str(p), allow_pickle=True)
            if "rdm" in d:
                rdm = d["rdm"].astype(np.float32)
            elif "spose_rdm" in d:
                rdm = d["spose_rdm"].astype(np.float32)
            else:
                continue
            model_tri["SPOSE"] = lower_tri(rdm)
            print(f"  SPOSE: {rdm.shape} → {model_tri['SPOSE'].shape[0]} pairs")
            break
    if "SPOSE" not in model_tri:
        print("  SPOSE full RDM not found, skipping")

    if not model_tri:
        raise FileNotFoundError("No model RDMs found in data/derivatives/")

    # ── Trim to shared category count ─────────────────────────────────────────
    n_pairs_brain  = brain_tri_mean.shape[1]
    # n from lower_tri: n*(n-1)/2 pairs → n = (1+sqrt(1+8p))/2
    n_brain_cat = int(round((1 + (1 + 8 * n_pairs_brain) ** 0.5) / 2))
    print(f"Brain categories: {n_brain_cat}")

    # Trim model RDMs to match brain category count if necessary
    trimmed = {}
    for name, tri in model_tri.items():
        n_model_cat = int(round((1 + (1 + 8 * len(tri)) ** 0.5) / 2))
        if n_model_cat > n_brain_cat:
            # Rebuild square, slice, re-flatten
            sq = np.zeros((n_model_cat, n_model_cat), dtype=np.float32)
            idx = np.tril_indices(n_model_cat, k=-1)
            sq[idx] = tri; sq = sq + sq.T
            sq = sq[:n_brain_cat, :n_brain_cat]
            tri = lower_tri(sq)
        trimmed[name] = ranked(tri)
    model_tri = trimmed

    # ── Compute RSA for three window widths ───────────────────────────────────
    window_widths = [0, 25, 50, 100]   # 0 = single timepoint
    results = {w: {name: np.zeros(n_times) for name in model_tri} for w in window_widths}

    for w in window_widths:
        label = f"{w}ms" if w > 0 else "single-tp"
        print(f"\nWindow = {label}")
        if w == 0:
            smoothed = brain_tri_mean
        else:
            smoothed = window_average_rdm(brain_tri_mean, times, w)

        brain_ranked = np.array([ranked(smoothed[t]) for t in range(n_times)])
        bm = brain_ranked.mean(axis=1, keepdims=True)
        bs = brain_ranked.std(axis=1, keepdims=True) + 1e-12

        for name, mr in model_tri.items():
            mm = mr.mean(); ms = mr.std() + 1e-12
            r_ts = ((brain_ranked - bm) * (mr - mm)).mean(axis=1) / (bs.squeeze() * ms)
            results[w][name] = r_ts.astype(np.float32)
            post = times > 0
            print(f"  {name:12s}: peak r={r_ts[post].max():.4f} at {times[post][r_ts[post].argmax()]:.0f}ms")

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dict = {"times": times}
    for w in window_widths:
        for name, ts in results[w].items():
            key = f"w{w}_{name.replace('-','_').replace('/','_').replace(' ','_')}"
            save_dict[key] = ts
    np.savez(str(RESULTS / "timewindow_rsa.npz"), **save_dict)
    print(f"\nSaved results/timewindow_rsa.npz")

    # ── Plot: single-tp vs 50ms window for all models ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    post = times > 0

    for ax, w, title in zip(axes, [0, WINDOW_MS],
                             ["Single timepoint", f"{WINDOW_MS}-ms window average"]):
        for name in model_tri:
            ts = results[w][name]
            col = COLORS.get(name, "gray")
            pk  = ts[post].max()
            pt  = times[post][ts[post].argmax()]
            ax.plot(times, ts, color=col, lw=2,
                    label=f"{name}  r={pk:.3f} @ {pt:.0f}ms")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.axvline(0, color="k",    ls=":", lw=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Spearman r")
        ax.set_title(title)
        ax.legend(fontsize=8)

    fig.suptitle("Brain-model RSA: single-timepoint vs time-window averaging", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(FIGS / "figure8_timewindow_rsa.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure8_timewindow_rsa.png")

    # ── Plot: peak r vs window width per model ────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in model_tri:
        peaks = [results[w][name][post].max() for w in window_widths]
        col   = COLORS.get(name, "gray")
        ax.plot(window_widths, peaks, "o-", color=col, lw=2, ms=7, label=name)
    ax.set_xlabel("Window width (ms)  [0 = single timepoint]")
    ax.set_ylabel("Peak Spearman r (post-stimulus)")
    ax.set_title("Effect of temporal averaging on brain-model RSA")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(str(FIGS / "figure8b_window_vs_peak_r.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/figure8b_window_vs_peak_r.png")
