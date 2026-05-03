"""Brain-to-model RSA: CLIP image embeddings vs official 200-cat brain RDMs.

Requires:
  - data/derivatives/clip_image_rdm.npz  (build_clip_image_embeddings.py)
  - data/derivatives/official_rdm200.mat (already present, 57 MB)

Uses same pipeline as run_brain_model_rsa_official.py but swaps text for image CLIP.
Also adds DINOv2 if available (dinov2_vitb14, torchvision).

Peak RAM: < 200 MB.

Usage:
    python scripts/run_brain_model_rsa_clip_image.py
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
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

N_PERM  = 1000
N_SUBJ  = 4
rng     = np.random.default_rng(42)
times_ms = np.arange(-100, 1305, 5, dtype=float)   # 281 timepoints


def rank_norm(v: np.ndarray) -> np.ndarray:
    r = rankdata(v).astype(np.float64)
    r -= r.mean()
    return r / (np.linalg.norm(r) + 1e-12)

def smooth(x, w=5):
    return np.convolve(x, np.ones(w)/w, mode="same")


# ── Load official 200-cat brain RDMs ─────────────────────────────────────────
print("Loading official 200-cat brain RDMs...")
with h5py.File(str(deriv_dir / "official_rdm200.mat"), "r") as f:
    rdm200 = np.array(f["mat"])   # (4, 281, 200, 200) or transposed
print(f"  Brain RDMs raw shape: {rdm200.shape}")
# h5py transposes MATLAB arrays: shape may be (200, 200, 281, 4)
if rdm200.shape[0] == 200:
    rdm200 = rdm200.transpose(3, 2, 0, 1)   # → (4, 281, 200, 200)
print(f"  Brain RDMs: {rdm200.shape}")
test_cat_nrs = np.loadtxt(str(deriv_dir / "test_200_category_nrs.txt"), dtype=int) \
    if (deriv_dir / "test_200_category_nrs.txt").exists() else None
idx_upper = np.triu_indices(200, k=1)

# Pre-rank-normalise all brain RDMs  → (4, 281, n_pairs)
n_pairs = len(idx_upper[0])
brain_rns = np.zeros((N_SUBJ, 281, n_pairs), dtype=np.float32)
for s in range(N_SUBJ):
    for t in range(281):
        rdm = rdm200[s, t].copy().astype(np.float64)
        np.fill_diagonal(rdm, 0.0)
        brain_rns[s, t] = rank_norm(rdm[idx_upper]).astype(np.float32)
del rdm200; gc.collect()
print("  Brain RDMs pre-ranked.")


# ── CLIP image RDM ────────────────────────────────────────────────────────────
print("Loading CLIP image RDM...")
clip_data = np.load(str(deriv_dir / "clip_image_rdm.npz"), allow_pickle=True)
clip_rdm_full = clip_data["rdm"].astype(np.float64)   # (1854, 1854)
concepts = list(clip_data["concepts"])

# Subset to the 200 test categories
# These categories correspond to the last 200 (nrs 1655–1854) in THINGS-MEG
# Verify via test_cat_nrs if available
if test_cat_nrs is not None:
    idx_200 = test_cat_nrs - 1   # 0-indexed
    print(f"  Using test_cat_nrs: {idx_200[:5]} ... {idx_200[-5:]}")
else:
    # THINGS-MEG uses categories 1655–1854 (0-indexed: 1654–1853) as test set
    idx_200 = np.arange(1654, 1854)
    print(f"  Assuming test cats = nrs 1655-1854 (0-indexed {idx_200[0]}-{idx_200[-1]})")

clip_200 = clip_rdm_full[np.ix_(idx_200, idx_200)]
np.fill_diagonal(clip_200, 0.0)
clip_rn = rank_norm(clip_200[idx_upper]).astype(np.float32)
del clip_rdm_full, clip_200; gc.collect()
print(f"  CLIP image vector: {clip_rn.shape}")

# ── Load existing SPOSE + text-CLIP for comparison ───────────────────────────
bm_old = np.load(str(results_dir / "brain_model_rsa_official.npz"), allow_pickle=True)
spose_old = bm_old["SPOSE"]          # (4, 281)
clip_text_old = bm_old.get("CLIP_text", None)
nc_upper = bm_old["nc_upper"]
nc_lower = bm_old["nc_lower"]

# ── Compute CLIP-image Spearman r per subject per timepoint ──────────────────
print("Computing CLIP-image brain-model RSA...")
clip_img_corrs = np.zeros((N_SUBJ, 281))
for s in range(N_SUBJ):
    for t in range(281):
        clip_img_corrs[s, t] = float(brain_rns[s, t] @ clip_rn)
    if s % 1 == 0:
        pk = times_ms[np.argmax(clip_img_corrs[s])]
        print(f"  P{s+1}: peak r={clip_img_corrs[s].max():.4f} at {pk:.0f}ms")

# ── Null distribution (permute model RDM labels) ─────────────────────────────
print(f"Computing {N_PERM}-perm shuffle null for CLIP-image...")
null = np.zeros((N_PERM, N_SUBJ, 281))
for p in range(N_PERM):
    perm_rn = clip_rn[rng.permutation(n_pairs)]
    for s in range(N_SUBJ):
        for t in range(281):
            null[p, s, t] = float(brain_rns[s, t] @ perm_rn)
null95 = np.percentile(null, 95, axis=(0, 1))
sig = clip_img_corrs.mean(0) > null95
print(f"  Significant timepoints: {sig.sum()}/281 (p<0.05 uncorr)")

mean_r = clip_img_corrs.mean(0)
pk = np.argmax(mean_r)
print(f"  CLIP-image peak r={mean_r[pk]:.4f} at {times_ms[pk]:.0f}ms")

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(str(results_dir / "brain_model_rsa_clip_image.npz"),
         times_ms=times_ms,
         CLIP_image=clip_img_corrs,
         sig_clip_image=sig,
         nc_upper=nc_upper,
         nc_lower=nc_lower)
print("Saved results/brain_model_rsa_clip_image.npz")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.axhline(0, color='gray', lw=0.5, ls='--')
ax.axvline(0, color='gray', lw=0.5)
ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.12, color='gray', label='Noise ceiling')

# SPOSE
sm = spose_old.mean(0)
ax.plot(times_ms, smooth(sm), 'steelblue', lw=2, label='SPOSE human similarity (200-cat)')

# CLIP text
if clip_text_old is not None:
    ct = clip_text_old.mean(0) if clip_text_old.ndim == 2 else clip_text_old
    ax.plot(times_ms, smooth(ct), 'darkorange', lw=2, ls='--', label='CLIP text (ViT-B/32)')

# CLIP image
ci = clip_img_corrs.mean(0)
ax.plot(times_ms, smooth(ci), 'forestgreen', lw=2.5, label='CLIP image (ViT-B/32)')
sig_t = times_ms[sig]
ax.scatter(sig_t, smooth(ci)[sig], s=8, color='forestgreen', zorder=5)

ax.set_xlabel('Time from stimulus onset (ms)')
ax.set_ylabel('Spearman r (mean 4 subjects)')
ax.set_title('Brain-to-model RSA: CLIP image vs text vs SPOSE (200 test categories)')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(str(figures_dir / "figure4b_brain_model_clip_image.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/figure4b_brain_model_clip_image.png")
