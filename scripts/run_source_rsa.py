"""Source-space brain-model RSA: where × when does each model predict brain activity?

Loads source means per subject, parcellates using Desikan-Killiany atlas (aparc),
computes Spearman RSA between parcellated source RDMs and model RDMs at each
timepoint. Produces:
  - Parcel × time RSA heatmap per model
  - Peak-r cortical surface map at best timepoint
  - Comparison of top parcels for CLIP vs DINOv2

Output:
  results/source_rsa.npz
  figures/figure9a_source_rsa_heatmap.png
  figures/figure9b_source_cortical_map.png

Run with: /opt/anaconda3/envs/things-meg/bin/python3 scripts/run_source_rsa.py
"""
from __future__ import annotations
import sys, gc
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")

RESULTS  = Path("results")
SRC_DIR  = Path("results/source")
DERIV    = Path("data/derivatives")
FIGS     = Path("figures")
FS_DIR   = str(Path.home() / "mne_data" / "MNE-fsaverage-data")
FS_SUB   = "fsaverage"

MODELS   = ["CLIP-B/32", "CLIP-L/14", "ResNet-50", "DINOv2"]
COLORS   = {"CLIP-B/32": "#1f77b4", "CLIP-L/14": "#aec7e8",
            "ResNet-50": "#9467bd", "DINOv2":    "#ff7f0e"}


def lower_tri(rdm):
    n = rdm.shape[-1]
    idx = np.tril_indices(n, k=-1)
    return rdm[..., idx[0], idx[1]]


def ranked(v):
    from scipy.stats import rankdata
    return rankdata(v).astype(np.float32)


def spearman_r_fast(brain_ranked, model_ranked):
    """Vectorised Pearson-on-ranks: brain_ranked (n_times, n_pairs), model_ranked (n_pairs,)."""
    bm = brain_ranked.mean(1, keepdims=True)
    bs = brain_ranked.std(1, keepdims=True) + 1e-12
    mm, ms = model_ranked.mean(), model_ranked.std() + 1e-12
    return ((brain_ranked - bm) * (model_ranked - mm)).mean(1) / (bs.squeeze() * ms)


if __name__ == "__main__":
    import mne
    mne.set_log_level("WARNING")
    FIGS.mkdir(parents=True, exist_ok=True)

    # ── Load source means and parcellate ─────────────────────────────────────
    subjects = ["BIGMEG1", "BIGMEG2", "BIGMEG3", "BIGMEG4"]
    times    = np.load(str(SRC_DIR / "times_ms.npy"))
    n_times  = len(times)

    # Load fsaverage labels (Desikan-Killiany)
    print("Loading parcellation labels...")
    labels = mne.read_labels_from_annot(FS_SUB, parc="aparc",
                                         subjects_dir=FS_DIR, verbose=False)
    labels = [l for l in labels if "unknown" not in l.name.lower()]
    print(f"  {len(labels)} parcels")

    # Build parcel → source index mapping (from first subject's vertices)
    verts0   = np.load(str(SRC_DIR / f"sub-{subjects[0]}_vertices.npz"))
    n_lh     = len(verts0["lh"])
    lh_verts = verts0["lh"]
    rh_verts = verts0["rh"]

    parcel_src_idx = []   # list of arrays, one per label
    for label in labels:
        if label.hemi == "lh":
            common = np.intersect1d(label.vertices, lh_verts)
            idx    = np.searchsorted(lh_verts, common)
        else:
            common = np.intersect1d(label.vertices, rh_verts)
            idx    = np.searchsorted(rh_verts, common) + n_lh
        parcel_src_idx.append(idx)

    n_parcels = len(labels)
    valid_parcels = [i for i, idx in enumerate(parcel_src_idx) if len(idx) >= 3]
    print(f"Valid parcels (≥3 sources): {len(valid_parcels)}/{n_parcels}")

    # Accumulate super-subject source means across subjects
    super_src = None
    n_loaded  = 0
    for sub in subjects:
        src_file = SRC_DIR / f"sub-{sub}_source_means.npy"
        if not src_file.exists():
            print(f"  Skipping {sub}")
            continue
        print(f"  Loading {sub}...")
        sm = np.load(str(src_file)).astype(np.float32)  # (n_cat, n_times, n_src)
        if super_src is None:
            super_src = sm
        else:
            super_src = super_src + sm
        n_loaded += 1
        del sm; gc.collect()

    if super_src is None:
        print("No source means found.")
        sys.exit(1)
    super_src /= n_loaded
    n_cat, _, n_src = super_src.shape
    print(f"Super-subject source means: {super_src.shape}")

    # ── Load model RDMs ──────────────────────────────────────────────────────
    model_tri = {}
    for fname, key in [("clip_image_rdm.npz", "CLIP-B/32"),
                       ("clip_large_rdm.npz",  "CLIP-L/14"),
                       ("resnet50_rdm.npz",    "ResNet-50"),
                       ("dinov2_rdm.npz",      "DINOv2")]:
        p = DERIV / fname
        if not p.exists():
            print(f"  Missing {fname}")
            continue
        rdm = np.load(str(p))["rdm"].astype(np.float32)
        rdm = rdm[:n_cat, :n_cat]
        model_tri[key] = ranked(lower_tri(rdm))
        print(f"  {key}: {rdm.shape}")

    # ── RSA per parcel × timepoint ────────────────────────────────────────────
    print("\nComputing parcel × time RSA...")
    rsa_results = {name: np.full((n_parcels, n_times), np.nan) for name in model_tri}

    for pi in valid_parcels:
        idx  = parcel_src_idx[pi]
        # parcel features per timepoint: (n_cat, n_src_in_parcel, n_times)
        pf   = super_src[:, :, idx]   # (n_cat, n_times, n_src_in_parcel)

        # Build correlation-distance RDM per timepoint
        brain_rdm = np.zeros((n_times, n_cat * (n_cat - 1) // 2), dtype=np.float32)
        for t in range(n_times):
            feat = pf[:, t, :]   # (n_cat, n_src_in_parcel)
            # correlation distance = 1 - pearson_r between conditions
            cc = np.corrcoef(feat)   # (n_cat, n_cat)
            cc = np.nan_to_num(cc, nan=0.0)
            brain_rdm[t] = lower_tri(1.0 - cc)

        brain_ranked = np.array([ranked(brain_rdm[t]) for t in range(n_times)])

        for name, mr in model_tri.items():
            rsa_results[name][pi] = spearman_r_fast(brain_ranked, mr)

        if pi % 10 == 0:
            print(f"  Parcel {pi}/{n_parcels}: {labels[pi].name}")

    # Save
    np.savez(str(RESULTS / "source_rsa.npz"),
             times=times,
             parcel_names=np.array([l.name for l in labels]),
             **{f"rsa_{k.replace('-','_').replace('/','_')}": v
                for k, v in rsa_results.items()})
    print("Saved results/source_rsa.npz")

    # ── Figures ───────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    parcel_names = [l.name for l in labels]
    post = times > 0

    # Figure 9a: heatmap (parcel × time) for CLIP-B/32
    for model_key in ["CLIP-B/32", "DINOv2"]:
        if model_key not in rsa_results:
            continue
        mat = rsa_results[model_key]   # (n_parcels, n_times)
        # Only include valid parcels; fill NaN → 0 for display
        mat_display = np.nan_to_num(mat, nan=0.0)
        # Sort by peak post-stimulus r (valid parcels only)
        peak_r = np.where(np.isnan(mat[:, post].max(1)), -999,
                          np.nan_to_num(mat[:, post].max(1)))
        order  = peak_r.argsort()[::-1]
        mat    = mat_display   # use nan-filled for plotting
        top_n  = min(20, n_parcels)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(mat[order[:top_n]][::-1],
                       aspect="auto", origin="lower",
                       extent=[times[0], times[-1], -0.5, top_n-0.5],
                       cmap="RdBu_r", vmin=-0.05, vmax=0.15)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([parcel_names[i] for i in order[:top_n]][::-1], fontsize=7)
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_title(f"{model_key} — source-space RSA by parcel (top {top_n})")
        plt.colorbar(im, ax=ax, label="Spearman r", fraction=0.02)
        fig.tight_layout()
        slug = model_key.replace("/","_").replace("-","_")
        out = FIGS / f"figure9a_source_rsa_{slug}.png"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    # Figure 9b: line plot — top 5 parcels for CLIP-B/32
    if "CLIP-B/32" in rsa_results:
        mat = rsa_results["CLIP-B/32"]
        top5 = mat[:, post].max(1).argsort()[::-1][:5]
        fig, ax = plt.subplots(figsize=(10, 4))
        cmap = plt.cm.viridis(np.linspace(0, 1, 5))
        for rank, pi in enumerate(top5):
            peak_r = mat[pi, post].max()
            peak_t = times[post][mat[pi, post].argmax()]
            ax.plot(times, mat[pi], color=cmap[rank], lw=2,
                    label=f"{parcel_names[pi]}  r={peak_r:.3f}@{peak_t:.0f}ms")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.axvline(0, color="k",    ls=":", lw=0.8)
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Spearman r")
        ax.set_title("CLIP-B/32 brain-model RSA — top 5 source parcels")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(FIGS / "figure9b_source_top_parcels.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved figures/figure9b_source_top_parcels.png")

    # Summary: peak r per parcel, both models
    print("\n=== Top 10 parcels (CLIP-B/32) ===")
    if "CLIP-B/32" in rsa_results:
        mat = rsa_results["CLIP-B/32"]
        top = mat[:, post].max(1).argsort()[::-1][:10]
        for pi in top:
            peak_r = mat[pi, post].max()
            peak_t = times[post][mat[pi, post].argmax()]
            print(f"  {parcel_names[pi]:35s}  r={peak_r:.4f}  @{peak_t:.0f}ms")
