"""Brain-to-model RSA using full 1854-category official RDMs.

Memory budget: ~200 MB peak.
- official_rdm1854.mat is 30.9 GB uncompressed; loaded ONE slice at a time via h5py.
- Null distribution computed inside the main loop (no brain vectors stored across
  timepoints).

Models:
  SPOSE 1854×1854, CLIP-text 1854, animacy, size, fMRI V1, fMRI FFA.

Usage:
    python scripts/run_brain_model_rsa_1854.py
"""

from __future__ import annotations
import gc, sys
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py, scipy.io as sio
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

cfg = load_config()
deriv_dir   = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

N_PERM  = 200   # permutations for shuffle null (computed per-timepoint in-loop)
N_SUBJ  = 4
SUBJ_LABELS = ["P1", "P2", "P3", "P4"]

times_ms = np.arange(-100, 1305, 5, dtype=float)   # 281 timepoints
n_times  = len(times_ms)
idx_upper = np.triu_indices(1854, k=1)              # 1,715,481 pairs

rng = np.random.default_rng(42)


# ── helpers ───────────────────────────────────────────────────────────────────
def rank_norm(v: np.ndarray) -> np.ndarray:
    """Rank-normalise a vector so that dot(rank_norm(a), rank_norm(b)) = Spearman r."""
    r = rankdata(v).astype(np.float64)
    r -= r.mean()
    norm = np.linalg.norm(r)
    return r / (norm + 1e-12)

def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


# ── 1. SPOSE model vector (27 MB, kept in RAM) ───────────────────────────────
print("Loading SPOSE...")
spose_raw = sio.loadmat(str(deriv_dir / "spose_similarity.mat"))["spose_sim"]
spose_mat = spose_raw.astype(np.float64)
np.fill_diagonal(spose_mat, 0.0)
spose_vec = (1.0 - spose_mat[idx_upper])     # dissimilarity, (1_715_481,)
spose_rn  = rank_norm(spose_vec)             # pre-ranked, stays in RAM (13.7 MB)
del spose_mat, spose_raw; gc.collect()
print(f"  spose_vec: {spose_vec.shape}  range {spose_vec.min():.3f}–{spose_vec.max():.3f}")


# ── 2. CLIP-text model vector ─────────────────────────────────────────────────
print("Building CLIP-text RDM...")
clip_rn = None
try:
    import open_clip, torch

    attrs = pd.read_csv("/Volumes/MEG/things-meg/sourcedata/sample_attributes_P1.csv")
    attrs["concept"] = attrs["image_path"].str.extract(r"images_meg/([^/]+)/")
    by_nr = attrs.dropna(subset=["concept"]).groupby(
        "things_category_nr")["concept"].first()
    concepts = [by_nr.get(nr, f"object_{nr}").replace("_", " ")
                for nr in range(1, 1855)]

    mn  = cfg.raw["models"]["clip"]
    pre = cfg.raw["models"]["clip_pretrained"]
    model, _, _ = open_clip.create_model_and_transforms(mn, pretrained=pre)
    model.eval()
    tok = open_clip.get_tokenizer(mn)
    with torch.no_grad():
        emb = model.encode_text(tok(concepts)).numpy().astype(np.float64)
    del model, tok; gc.collect()

    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb /= norms
    sim = emb @ emb.T
    clip_vec = (1.0 - sim[idx_upper])
    clip_rn  = rank_norm(clip_vec)
    del emb, sim; gc.collect()
    print(f"  clip_vec: {clip_vec.shape}  range {clip_vec.min():.3f}–{clip_vec.max():.3f}")
except Exception as e:
    print(f"  CLIP skipped: {e}")
    clip_vec = None


# ── 3. Validation time courses (animacy, size, fMRI) ─────────────────────────
def load_tc(prefix: str) -> np.ndarray | None:
    """Load per-subject CSVs → (N_SUBJ, n_times) mean over sessions."""
    rows = []
    for s in SUBJ_LABELS:
        p = deriv_dir / f"validation_{prefix}_{s}.csv"
        if not p.exists():
            return None
        rows.append(pd.read_csv(p, index_col=0).values.mean(axis=1))
    return np.stack(rows)

print("Loading validation time courses...")
animacy_tc = load_tc("animacy")
size_tc    = load_tc("size")
v1_tc      = load_tc("fmri_meg_v1")
ffa_tc     = load_tc("fmri_meg_ffa")
for name, tc in [("animacy", animacy_tc), ("size", size_tc),
                  ("V1", v1_tc), ("FFA", ffa_tc)]:
    if tc is not None:
        pk = times_ms[tc.mean(0).argmax()]
        print(f"  {name}: peak at {pk:.0f} ms")


# ── 4. Main loop: lazy h5py slice-per-timepoint ───────────────────────────────
# Memory per iteration:
#   brain_slice: 4 × 1854² × 8 = 110 MB  (released each iter)
#   brain_rns:   4 × 1.7M × 8  =  55 MB  (released each iter)
#   one perm_rn: 1.7M × 8      =  14 MB  (transient per perm)
# Peak: ~180 MB
print(f"\nMain loop: {n_times} timepoints × {N_SUBJ} subjects × {N_PERM} null perms…")
print("Peak RAM ≈ 180 MB per iteration")

spose_corrs = np.zeros((N_SUBJ, n_times))
clip_corrs  = np.zeros((N_SUBJ, n_times))
# null: (N_PERM, N_SUBJ, n_times) — only 200 × 4 × 281 × 8 = 1.8 MB
null_spose  = np.zeros((N_PERM, N_SUBJ, n_times))
null_clip   = np.zeros((N_PERM, N_SUBJ, n_times)) if clip_rn is not None else None

with h5py.File(str(deriv_dir / "official_rdm1854.mat"), "r") as f:
    mat = f["mat"]   # (4, 281, 1854, 1854) — lazy, not loaded

    for t in range(n_times):
        # Read one timepoint for all subjects: 110 MB
        brain_slice = np.array(mat[:, t, :, :])   # force load into RAM

        # Per-subject rank-normalised brain vectors
        brain_rns = []
        for s in range(N_SUBJ):
            rdm = brain_slice[s].copy()
            np.fill_diagonal(rdm, 0.0)
            bv  = rdm[idx_upper]
            brain_rns.append(rank_norm(bv))

        del brain_slice; gc.collect()

        # Real correlations
        for s in range(N_SUBJ):
            spose_corrs[s, t] = brain_rns[s] @ spose_rn
            if clip_rn is not None:
                clip_corrs[s, t] = brain_rns[s] @ clip_rn

        # Null: permute pre-ranked vectors — equivalent null, no re-ranking needed
        # spose_rn is already rank-normalised and has unit norm, so permuting
        # it gives a valid null without calling rankdata again (100× faster).
        for p in range(N_PERM):
            perm_rn = spose_rn[rng.permutation(len(spose_rn))]
            for s in range(N_SUBJ):
                null_spose[p, s, t] = brain_rns[s] @ perm_rn

            if clip_rn is not None:
                perm_rn_c = clip_rn[rng.permutation(len(clip_rn))]
                for s in range(N_SUBJ):
                    null_clip[p, s, t] = brain_rns[s] @ perm_rn_c

        del brain_rns; gc.collect()

        if t % 40 == 0:
            print(f"  t={t}/{n_times}  ({times_ms[t]:.0f} ms)  "
                  f"spose_mean_r={spose_corrs[:, t].mean():.4f}")

print("Loop done.")


# ── 5. Significance & noise ceiling ──────────────────────────────────────────
# Null 95th percentile averaged across subjects → one threshold per timepoint
null95_spose = np.percentile(null_spose, 95, axis=(0, 1))   # (n_times,)
sig_spose    = spose_corrs.mean(0) > null95_spose
print(f"SPOSE: {sig_spose.sum()}/{n_times} timepoints above shuffle null p<0.05")

sig_clip = np.zeros(n_times, dtype=bool)
if clip_rn is not None:
    null95_clip = np.percentile(null_clip, 95, axis=(0, 1))
    sig_clip    = clip_corrs.mean(0) > null95_clip
    print(f"CLIP-text: {sig_clip.sum()}/{n_times} timepoints above null")

# Noise ceiling: inter-subject consistency of brain-model RSA time courses.
# Upper: mean Pearson r between each subject's time course and the overall mean.
# Lower: mean Pearson r between each subject's time course and the LOO mean
#        (mean of the other N-1 subjects).
nc_upper_vals, nc_lower_vals = [], []
mean_tc = spose_corrs.mean(0)       # (n_times,) overall mean
for s in range(N_SUBJ):
    nc_upper_vals.append(np.corrcoef(spose_corrs[s], mean_tc)[0, 1])
    loo_mean = np.delete(spose_corrs, s, axis=0).mean(0)
    nc_lower_vals.append(np.corrcoef(spose_corrs[s], loo_mean)[0, 1])
nc_upper_scalar = float(np.mean(nc_upper_vals))
nc_lower_scalar = float(np.mean(nc_lower_vals))
# Broadcast to time array for figure compatibility
nc_upper = np.full(n_times, nc_upper_scalar)
nc_lower = np.full(n_times, nc_lower_scalar)
print(f"Inter-subject NC: upper={nc_upper_scalar:.3f}, lower={nc_lower_scalar:.3f}")


# ── 6. Save ───────────────────────────────────────────────────────────────────
save = dict(times_ms=times_ms, SPOSE=spose_corrs, sig_spose=sig_spose,
            nc_upper=nc_upper, nc_lower=nc_lower)
if clip_rn is not None:
    save.update(CLIP_text=clip_corrs, sig_clip=sig_clip)
for name, tc in [("animacy", animacy_tc), ("size", size_tc),
                  ("V1_fmri", v1_tc), ("FFA_fmri", ffa_tc)]:
    if tc is not None:
        save[name] = tc
np.savez(str(results_dir / "brain_model_rsa_1854.npz"), **save)
print("Saved results/brain_model_rsa_1854.npz")


# ── 7. Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5)
ax.fill_between(times_ms, nc_lower, nc_upper, alpha=0.15, color="gray",
                label="Noise ceiling")
for label, corrs, sig, color in [
    ("SPOSE 1854-cat", spose_corrs, sig_spose, "steelblue"),
    ("CLIP-text 1854-cat", clip_corrs if clip_rn is not None else None,
     sig_clip, "darkorange"),
]:
    if corrs is None:
        continue
    mr = smooth(corrs.mean(0))
    ax.plot(times_ms, mr, color=color, lw=2, label=label)
    if sig.any():
        ax.scatter(times_ms[sig], mr[sig], s=8, color=color, zorder=5)
ax.set_ylabel("Spearman r (mean 4 subjects)")
ax.set_title("Brain-model RSA — 1854 categories (official pairwise decoding RDMs)")
ax.legend(fontsize=9)

ax2 = axes[1]
ax2.axhline(0, color="gray", lw=0.5, ls="--")
ax2.axvline(0, color="gray", lw=0.5)
for label, tc, color in [
    ("V1 fMRI→MEG", v1_tc, "forestgreen"),
    ("FFA fMRI→MEG", ffa_tc, "crimson"),
    ("Animacy", animacy_tc, "purple"),
    ("Size", size_tc, "saddlebrown"),
]:
    if tc is None:
        continue
    ax2.plot(times_ms, smooth(tc.mean(0)), color=color, lw=2, label=label)
ax2.set_xlabel("Time from stimulus onset (ms)")
ax2.set_ylabel("Score (mean 4 subjects)")
ax2.set_title("Hierarchy markers: fMRI ROIs and stimulus dimensions")
ax2.legend(fontsize=9)

plt.tight_layout()
out = str(figures_dir / "figure4_brain_model_1854.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")

spose_mean = spose_corrs.mean(0)
pk = spose_mean.argmax()
print(f"\nSPOSE 1854-cat: peak r={spose_mean[pk]:.4f} at {times_ms[pk]:.0f} ms")
if clip_rn is not None:
    cm = clip_corrs.mean(0)
    pk2 = cm.argmax()
    print(f"CLIP-text 1854-cat: peak r={cm[pk2]:.4f} at {times_ms[pk2]:.0f} ms")
