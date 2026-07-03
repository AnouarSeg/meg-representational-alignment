"""Layer-wise RSA: brain RDM vs each model layer at each timepoint.

Loads:
  data/derivatives/layerwise_resnet_rdms.npz  (layer1–4)
  data/derivatives/layerwise_clip_rdms.npz    (block0,3,6,9,11)
  results/brain_model_rsa_full1854.npz        (precomputed brain RDMs, 100-cat)

For each layer × timepoint: Spearman r between lower-triangular brain RDM
and layer RDM, using the same 100 MEG test categories.

Output:
  results/layerwise_rsa.npz
  figures/figure7a_layerwise_resnet.png   — line plot per layer + heatmap
  figures/figure7b_layerwise_clip.png     — line plot per layer + heatmap
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
FIGS    = Path("figures")

RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
CLIP_BLOCKS   = ["block0", "block3", "block6", "block9", "block11"]

RESNET_COLORS = ["#d4e6f1", "#7fb3d3", "#2980b9", "#1a5276"]   # light→dark blue
CLIP_COLORS   = ["#fdebd0", "#f0b27a", "#e67e22", "#ca6f1e", "#7d3c00"]


def lower_tri(rdm):
    n = rdm.shape[0]
    idx = np.tril_indices(n, k=-1)
    return rdm[idx]


def spearman_r(a, b):
    return spearmanr(a, b).statistic


def rank_vector(v):
    from scipy.stats import rankdata
    return rankdata(v).astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Load brain RDMs (per-subject, 100-cat crossnobis)
    # Shape from brain_model_rsa_full1854.npz: brain_rdms key is (n_sub, n_times, n_pairs)
    # or we can rebuild from condition_means — but easier to use the precomputed RSA results
    # which store brain RDMs per timepoint.
    # Actually the simplest: load the per-subject brain RDMs directly from rdms_full_*.npz

    brain_rdm_list = []
    times = None
    for sub in ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]:
        f = RESULTS / f"rdms_full_{sub}.npz"
        if not f.exists():
            print(f"  Missing {f}, skipping")
            continue
        d = np.load(str(f))
        # rdms: (n_cat, n_cat, n_times) or (n_times, n_cat, n_cat)
        rdm = d["rdms"]   # expected (n_times, n_cat, n_cat)
        if rdm.shape[0] != 180:
            rdm = rdm.transpose(2, 0, 1)  # → (n_times, n_cat, n_cat)
        brain_rdm_list.append(rdm)
        if times is None:
            times = d["times_ms"]

    # Identify the 100 test category indices within the 1852-cat RDMs
    cat_meta = np.load(str(RESULTS / "model_rdms.npz"), allow_pickle=True)
    test_cat_indices = cat_meta["category_numbers"]  # (100,) indices into 1852-cat space

    if not brain_rdm_list:
        raise FileNotFoundError("No rdms_full_*.npz files found in results/")

    # Slice each subject's RDM to the 100 test categories
    # rdms shape: (n_times, 1852, 1852) → slice to (n_times, 100, 100)
    sliced = []
    for rdm_full in brain_rdm_list:
        # rdm_full: (n_times, n_valid, n_valid); valid_categories maps indices
        # We need to find where test_cat_indices appear in valid_categories
        d0 = np.load(str(RESULTS / f"rdms_full_BIGMEG1.npz"))
        valid = d0["valid_categories"]
        pos = np.array([np.where(valid == idx)[0][0] for idx in test_cat_indices
                        if idx in valid])
        rdm_100 = rdm_full[:, pos[:, None], pos[None, :]]  # (n_times, 100, 100)
        sliced.append(rdm_100)

    n_times = sliced[0].shape[0]
    n_cat_brain = sliced[0].shape[1]
    brain_rdm_mean = np.mean(sliced, axis=0)  # (n_times, 100, 100)
    brain_tri_mean = np.array([lower_tri(brain_rdm_mean[t]) for t in range(n_times)])
    brain_ranked = np.array([rank_vector(brain_tri_mean[t]) for t in range(n_times)])
    print(f"Brain RDMs: {n_cat_brain} categories, {n_times} timepoints")

    # Load layer RDMs
    resnet_data = np.load(str(DERIV / "layerwise_resnet_rdms.npz"))
    clip_data   = np.load(str(DERIV / "layerwise_clip_rdms.npz"))

    # Trim to n_cat_brain (first n_cat rows/cols)
    def get_layer_tri(data, key, n):
        rdm = data[key][:n, :n]
        return lower_tri(rdm)

    def run_rsa(layer_names, layer_data, label):
        results = {}
        for name in layer_names:
            model_tri = get_layer_tri(layer_data, name, n_cat_brain)
            model_ranked = rank_vector(model_tri)
            r_ts = np.array([
                np.dot(brain_ranked[t], model_ranked) / len(model_ranked) -
                np.mean(brain_ranked[t]) * np.mean(model_ranked) /
                (np.std(brain_ranked[t]) * np.std(model_ranked) + 1e-12)
                for t in range(n_times)
            ])
            # Use proper Spearman: Pearson on ranks
            bm = brain_ranked.mean(axis=1, keepdims=True)
            bs = brain_ranked.std(axis=1, keepdims=True) + 1e-12
            mm = model_ranked.mean()
            ms = model_ranked.std() + 1e-12
            r_ts = ((brain_ranked - bm) * (model_ranked - mm)).mean(axis=1) / (bs.squeeze() * ms)
            results[name] = r_ts.astype(np.float32)
            post = times > 0
            print(f"  {label} {name}: peak r={r_ts[post].max():.4f} at {times[post][r_ts[post].argmax()]:.0f}ms")
        return results

    print("\nResNet-50 layer RSA:")
    resnet_rsa = run_rsa(RESNET_LAYERS, resnet_data, "ResNet")
    print("\nCLIP ViT-B/32 block RSA:")
    clip_rsa   = run_rsa(CLIP_BLOCKS,   clip_data,   "CLIP")

    # Save
    np.savez(str(RESULTS / "layerwise_rsa.npz"),
             times=times,
             **{f"resnet_{k}": v for k, v in resnet_rsa.items()},
             **{f"clip_{k}":   v for k, v in clip_rsa.items()})
    print(f"\nSaved results/layerwise_rsa.npz")

    # ── Plot ──────────────────────────────────────────────────────────────────
    def plot_layers(layer_names, rsa_dict, colors, model_label, out_path, layer_labels=None):
        if layer_labels is None:
            layer_labels = layer_names
        fig = plt.figure(figsize=(12, 8))
        gs  = GridSpec(2, 1, figure=fig, height_ratios=[1.6, 1], hspace=0.35)

        # Top: line plot
        ax1 = fig.add_subplot(gs[0])
        post = times > 0
        for name, lab, col in zip(layer_names, layer_labels, colors):
            ts = rsa_dict[name]
            peak_r = ts[post].max()
            peak_t = times[post][ts[post].argmax()]
            ax1.plot(times, ts, color=col, lw=2,
                     label=f"{lab}  (peak r={peak_r:.3f} @ {peak_t:.0f}ms)")
        ax1.axhline(0, color="gray", ls=":", lw=0.8)
        ax1.axvline(0, color="k",    ls=":", lw=0.8)
        ax1.set_xlabel("Time (ms)")
        ax1.set_ylabel("Spearman r (brain vs layer RDM)")
        ax1.set_title(f"{model_label} — layer-wise brain RSA")
        ax1.legend(fontsize=9, loc="upper left")

        # Bottom: heatmap (layer × time)
        ax2 = fig.add_subplot(gs[1])
        mat = np.stack([rsa_dict[n] for n in layer_names])
        im  = ax2.imshow(mat, aspect="auto", origin="lower",
                         extent=[times[0], times[-1], -0.5, len(layer_names)-0.5],
                         cmap="RdBu_r", vmin=-0.02, vmax=mat.max())
        ax2.set_yticks(range(len(layer_names)))
        ax2.set_yticklabels(layer_labels, fontsize=9)
        ax2.axvline(0, color="k", ls=":", lw=0.8)
        ax2.set_xlabel("Time (ms)")
        ax2.set_title(f"{model_label} — RSA heatmap (layer × time)")
        plt.colorbar(im, ax=ax2, label="Spearman r", fraction=0.03, pad=0.02)

        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    resnet_labels = ["ResNet layer1 (64ch)", "ResNet layer2 (128ch)",
                     "ResNet layer3 (256ch)", "ResNet layer4 (512ch)"]
    clip_labels   = [f"CLIP block {i}" for i in [0, 3, 6, 9, 11]]
    clip_labels[-1] += " (final)"

    plot_layers(RESNET_LAYERS, resnet_rsa, RESNET_COLORS, "ResNet-50",
                FIGS / "figure7a_layerwise_resnet.png", resnet_labels)
    plot_layers(CLIP_BLOCKS,   clip_rsa,   CLIP_COLORS,   "CLIP ViT-B/32",
                FIGS / "figure7b_layerwise_clip.png",   clip_labels)

    # Combined figure: peak RSA r per layer (depth vs r, both models)
    fig, ax = plt.subplots(figsize=(8, 4))
    post = times > 0
    resnet_peaks = [resnet_rsa[n][post].max() for n in RESNET_LAYERS]
    clip_peaks   = [clip_rsa[n][post].max()   for n in CLIP_BLOCKS]

    ax.plot(range(1, 5),  resnet_peaks, "o-", color="#2980b9", lw=2, ms=7,
            label="ResNet-50 (layers 1–4)")
    ax.plot([0, 3, 6, 9, 11], clip_peaks, "s--", color="#e67e22", lw=2, ms=7,
            label="CLIP ViT-B/32 (blocks 0–11)")
    ax.set_xlabel("Layer depth (ResNet: 1–4, CLIP: block 0–11)")
    ax.set_ylabel("Peak Spearman r (brain–model RSA)")
    ax.set_title("Layer depth vs brain alignment — model hierarchy")
    ax.legend(fontsize=10)
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(str(FIGS / "figure7c_layer_depth_vs_brain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figures/figure7c_layer_depth_vs_brain.png")
