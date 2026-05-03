"""Brain-to-model RSA: DINOv2 ViT-B/14 image embeddings vs official 200-cat brain RDMs.

Requires:
  - data/derivatives/dinov2_rdm.npz
  - data/derivatives/official_rdm200.mat
  - data/derivatives/test_200_category_nrs.txt

Peak RAM: < 200 MB.
"""

from __future__ import annotations
import gc, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

cfg         = load_config()
deriv_dir   = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])

N_PERM   = 1000
N_SUBJ   = 4
rng      = np.random.default_rng(42)
times_ms = np.arange(-100, 1305, 5, dtype=float)

def rank_norm(v):
    r = rankdata(v).astype(np.float64)
    r -= r.mean()
    return r / (np.linalg.norm(r) + 1e-12)

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="same")

# ── Brain RDMs ────────────────────────────────────────────────────────────────
print("Loading 200-cat brain RDMs...")
with h5py.File(str(deriv_dir / "official_rdm200.mat"), "r") as f:
    rdm200 = np.array(f["mat"])
if rdm200.shape[0] == 200:
    rdm200 = rdm200.transpose(3, 2, 0, 1)
idx_upper = np.triu_indices(200, k=1)
n_pairs   = len(idx_upper[0])

brain_rns = np.zeros((N_SUBJ, 281, n_pairs), dtype=np.float32)
for s in range(N_SUBJ):
    for t in range(281):
        rdm = rdm200[s, t].copy().astype(np.float64)
        np.fill_diagonal(rdm, 0.0)
        brain_rns[s, t] = rank_norm(rdm[idx_upper]).astype(np.float32)
del rdm200; gc.collect()

# ── DINOv2 RDM ───────────────────────────────────────────────────────────────
print("Loading DINOv2 RDM...")
dino_data = np.load(str(deriv_dir / "dinov2_rdm.npz"), allow_pickle=True)
dino_rdm_full = dino_data["rdm"].astype(np.float64)

test_cat_nrs = np.loadtxt(str(deriv_dir / "test_200_category_nrs.txt"), dtype=int)
idx_200 = test_cat_nrs - 1
dino_200 = dino_rdm_full[np.ix_(idx_200, idx_200)]
np.fill_diagonal(dino_200, 0.0)
dino_rn = rank_norm(dino_200[idx_upper]).astype(np.float32)
del dino_rdm_full, dino_200; gc.collect()

# ── Compute RSA ───────────────────────────────────────────────────────────────
print("Computing DINOv2 brain-model RSA...")
dino_corrs = np.zeros((N_SUBJ, 281))
for s in range(N_SUBJ):
    for t in range(281):
        dino_corrs[s, t] = float(brain_rns[s, t] @ dino_rn)
    pk = times_ms[np.argmax(dino_corrs[s])]
    print(f"  P{s+1}: peak r={dino_corrs[s].max():.4f} at {pk:.0f}ms")

# ── Null ──────────────────────────────────────────────────────────────────────
print(f"Computing {N_PERM}-perm shuffle null...")
null = np.zeros((N_PERM, N_SUBJ, 281))
for p in range(N_PERM):
    perm_rn = dino_rn[rng.permutation(n_pairs)]
    for s in range(N_SUBJ):
        for t in range(281):
            null[p, s, t] = float(brain_rns[s, t] @ perm_rn)
null95 = np.percentile(null, 95, axis=(0, 1))
sig = dino_corrs.mean(0) > null95
print(f"  Significant: {sig.sum()}/281 timepoints")
mean_r = dino_corrs.mean(0)
pk = np.argmax(mean_r)
print(f"  DINOv2 peak r={mean_r[pk]:.4f} at {times_ms[pk]:.0f}ms")

# ── Load existing results for combined figure ─────────────────────────────────
bm = np.load(str(results_dir / "brain_model_rsa_official.npz"), allow_pickle=True)
spose_corrs   = bm["SPOSE"]
nc_upper      = bm["nc_upper"]
nc_lower      = bm["nc_lower"]
clip_text     = bm.get("CLIP_text", None)

clip_img_npz  = results_dir / "brain_model_rsa_clip_image.npz"
clip_img      = np.load(str(clip_img_npz))["CLIP_image"] if clip_img_npz.exists() else None

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(str(results_dir / "brain_model_rsa_dinov2.npz"),
         times_ms=times_ms, DINOv2=dino_corrs,
         sig_dinov2=sig, nc_upper=nc_upper, nc_lower=nc_lower)
print("Saved results/brain_model_rsa_dinov2.npz")

# ── Combined model comparison figure ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5)
ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.10, color="gray", label="Noise ceiling")

for label, tc, color, lw, ls in [
    ("SPOSE human similarity",    spose_corrs,                 "steelblue",   2.0, "-"),
    ("CLIP text (ViT-B/32)",      clip_text if clip_text is not None else None,
                                                               "darkorange",  1.8, "--"),
    ("CLIP image (ViT-B/32)",     clip_img,                    "forestgreen", 2.0, "-"),
    ("DINOv2 image (ViT-B/14)",   dino_corrs,                  "crimson",     2.2, "-"),
]:
    if tc is None:
        continue
    m = tc.mean(0) if tc.ndim == 2 else tc
    ax.plot(times_ms, smooth(m), color=color, lw=lw, ls=ls, label=label)

ax.set_xlabel("Time from stimulus onset (ms)")
ax.set_ylabel("Spearman r (mean 4 subjects)")
ax.set_title("Brain-to-model RSA: all models (200 test categories, official THINGS-MEG RDMs)")
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
fig.savefig(str(figures_dir / "figure4c_brain_model_all.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved figures/figure4c_brain_model_all.png")
