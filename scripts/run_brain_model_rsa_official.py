"""Optimised brain-to-model RSA using official THINGS-MEG RDMs (Week 6 upgrade).

Uses the official pairwise-decoding RDMs (validation-pairwise_decoding_RDM200.mat)
from OpenNeuro ds004212 instead of our noisy 100-category crossnobis RDMs.
These are high-quality RDMs computed from many repetitions of the 200 test images.

Model RDMs:
  - SPOSE: human similarity judgements (Muttenthaler et al. 2023), 1854×1854
            subsetted to the 200 test categories
  - CLIP-text: ViT-B/32 text encoder embeddings of category names (open_clip)
  - CLIP-image: ViT-B/32 image encoder if THINGS images are available (skipped if not)

Statistics: Spearman r, noise ceiling (Nili 2014), shuffle null (1000 perms), bootstrap CIs.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_brain_model_rsa_official.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config
from thingsmeg.stats import bootstrap_ci, shuffle_null_rsa, noise_ceiling_over_time

cfg = load_config()
deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# ── 1. Load official brain RDMs ───────────────────────────────────────────────
print("Loading official RDMs...")
with h5py.File(str(deriv_dir / "official_rdm200.mat"), "r") as f:
    brain_rdms = f["mat"][:]  # (4, 281, 200, 200) float64

n_subj, n_times, n_cond, _ = brain_rdms.shape
# Time axis: -100 to 1300ms at 5ms steps
times_ms = np.arange(n_times) * 5.0 - 100.0
print(f"  Shape: {brain_rdms.shape}, time: {times_ms[0]:.0f}–{times_ms[-1]:.0f} ms")

# Fill diagonal with 0 (NaN on diagonal = self-decoding not applicable)
for s in range(n_subj):
    for t in range(n_times):
        np.fill_diagonal(brain_rdms[s, t], 0.0)

# Upper triangle indices
idx_upper = np.triu_indices(n_cond, k=1)
brain_vecs = np.stack([
    brain_rdms[s][:, idx_upper[0], idx_upper[1]]
    for s in range(n_subj)
])  # (4, 281, n_pairs)

# ── 2. Identify the 200 test categories ──────────────────────────────────────
# Extract from events files (test trial_type, repeated 12×)
print("Identifying 200 test categories from events...")
test_cats = set()
raw_dir = Path(cfg.raw["paths"]["raw"])
for evf in sorted(raw_dir.rglob("*events.tsv")):
    try:
        with open(str(evf), "r", encoding="utf-8", errors="replace") as fh:
            for line in fh.readlines()[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 4 and "test" in parts[2]:
                    try:
                        test_cats.add(int(float(parts[3])))
                    except ValueError:
                        pass
        if len(test_cats) == 200:
            break
    except Exception:
        pass

test_cat_nrs = np.array(sorted(test_cats))  # 0-indexed into THINGS (1-based)
print(f"  Found {len(test_cat_nrs)} test categories: {test_cat_nrs[:5]}…")
assert len(test_cat_nrs) == 200, f"Expected 200, got {len(test_cat_nrs)}"

# ── 3. SPOSE model RDM (subset to 200 test categories) ───────────────────────
print("Building SPOSE model RDM...")
spose = sio.loadmat(str(deriv_dir / "spose_similarity.mat"))["spose_sim"]  # (1854,1854)
# things_category_nr is 1-based; convert to 0-based index
test_idx = test_cat_nrs - 1   # 0-based
spose_sub = spose[np.ix_(test_idx, test_idx)]  # (200, 200)
spose_rdm = 1.0 - spose_sub   # dissimilarity
np.fill_diagonal(spose_rdm, 0.0)
spose_vec = spose_rdm[idx_upper]
print(f"  SPOSE RDM range: {spose_vec.min():.3f} – {spose_vec.max():.3f}")

# ── 4. CLIP-text model RDM ───────────────────────────────────────────────────
print("Building CLIP-text model RDM...")
try:
    import open_clip
    import torch

    model_name = cfg.raw["models"]["clip"]               # e.g. "ViT-B-32"
    pretrained = cfg.raw["models"]["clip_pretrained"]    # e.g. "openai"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    # Load concept names extracted from sample_attributes_P1.csv (cached locally)
    import json, csv, re
    _attr_path = Path(cfg.raw["paths"]["raw"]).parent / "sourcedata" / "sample_attributes_P1.csv"
    cat_names: dict[int, str] = {}
    if _attr_path.exists():
        with open(str(_attr_path)) as _f:
            for _row in csv.DictReader(_f):
                try:
                    _nr = int(float(_row["things_category_nr"]))
                    _m = re.search(r"images_meg/([^/]+)/", _row.get("image_path", ""))
                    if _m:
                        cat_names[_nr] = _m.group(1).replace("_", " ")
                except (ValueError, KeyError):
                    pass
    else:
        # Fallback: download sample_attributes_P1.csv from S3 once
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config as BotoConfig
        import io
        s3 = boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))
        content = s3.get_object(
            Bucket="openneuro.org", Key="ds004212/sourcedata/sample_attributes_P1.csv"
        )["Body"].read().decode("utf-8", errors="replace")
        for _row in csv.DictReader(io.StringIO(content)):
            try:
                _nr = int(float(_row["things_category_nr"]))
                _m = re.search(r"images_meg/([^/]+)/", _row.get("image_path", ""))
                if _m:
                    cat_names[_nr] = _m.group(1).replace("_", " ")
            except (ValueError, KeyError):
                pass

    names_200 = [cat_names.get(int(c), f"category_{c}") for c in test_cat_nrs]
    texts = tokenizer(names_200)
    with torch.no_grad():
        feats = model.encode_text(texts)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    feats = feats.float().numpy()
    cosine_sim = feats @ feats.T
    clip_text_rdm = 1.0 - cosine_sim
    np.fill_diagonal(clip_text_rdm, 0.0)
    clip_text_vec = clip_text_rdm[idx_upper]
    clip_text_ok = True
    print(f"  CLIP-text RDM range: {clip_text_vec.min():.3f} – {clip_text_vec.max():.3f}")
except Exception as e:
    print(f"  CLIP-text failed: {e} — skipping")
    clip_text_ok = False
    clip_text_vec = None

# ── 5. Spearman r: brain vs model at each timepoint ─────────────────────────
print("Computing brain-model Spearman r over time...")
n_pairs = len(idx_upper[0])

models = {"SPOSE": spose_vec}
if clip_text_ok:
    models["CLIP_text"] = clip_text_vec

corrs_by_model = {}
for model_name, model_vec in models.items():
    corrs = np.zeros((n_subj, n_times))
    for s in range(n_subj):
        for t in range(n_times):
            corrs[s, t] = spearmanr(brain_vecs[s, t], model_vec).statistic
    corrs_by_model[model_name] = corrs
    print(f"  {model_name}: peak r={corrs.mean(axis=0).max():.4f} "
          f"at {times_ms[corrs.mean(axis=0).argmax()]:.0f} ms")

# ── 6. Noise ceiling ─────────────────────────────────────────────────────────
print("Computing noise ceiling over time...")
subjects_fake = {str(s): brain_rdms[s] for s in range(n_subj)}
nc_upper, nc_lower = noise_ceiling_over_time(subjects_fake)
print(f"  Peak upper={nc_upper.max():.3f}, lower={nc_lower.max():.3f}")

# ── 7. Bootstrap CIs + shuffle null ──────────────────────────────────────────
print("Bootstrap CIs and shuffle null...")
brain_vecs_mean = brain_vecs.mean(axis=0)  # (n_times, n_pairs)

ci_dict = {}
null_dict = {}
for model_name, model_vec in models.items():
    ci_lo, ci_hi = bootstrap_ci(
        corrs_by_model[model_name],
        n_boot=int(cfg.raw["stats"]["n_bootstrap"]),
        seed=int(cfg.raw["stats"]["random_seed"])
    )
    ci_dict[model_name] = (ci_lo, ci_hi)

    null = shuffle_null_rsa(
        brain_vecs_mean, model_vec,
        n_perm=int(cfg.raw["stats"]["n_permutations"]),
        seed=int(cfg.raw["stats"]["random_seed"])
    )
    null_dict[model_name] = null
    null_95 = np.percentile(null, 95, axis=0)
    mean_r = corrs_by_model[model_name].mean(axis=0)
    sig = (mean_r > null_95).sum()
    print(f"  {model_name}: {sig}/{n_times} timepoints above shuffle null p<0.05")

# ── 8. Save results ───────────────────────────────────────────────────────────
save_dict = {
    "times_ms": times_ms,
    "nc_upper": nc_upper,
    "nc_lower": nc_lower,
    "test_cat_nrs": test_cat_nrs,
}
for model_name, corrs in corrs_by_model.items():
    save_dict[model_name] = corrs
np.savez(str(results_dir / "brain_model_rsa_official.npz"), **save_dict)
print(f"\nSaved results/brain_model_rsa_official.npz")

# ── 9. Figure ─────────────────────────────────────────────────────────────────
def smooth(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same")

color_map = {"SPOSE": "steelblue", "CLIP_text": "darkorange"}
fig, ax = plt.subplots(figsize=(12, 5))

for model_name, corrs in corrs_by_model.items():
    mean_r = corrs.mean(axis=0)
    ci_lo, ci_hi = ci_dict[model_name]
    null = null_dict[model_name]
    null_95 = np.percentile(null, 95, axis=0)
    sig = mean_r > null_95
    color = color_map.get(model_name, "gray")

    ax.plot(times_ms, smooth(mean_r), linewidth=2, label=model_name, color=color)
    ax.fill_between(times_ms, smooth(ci_lo), smooth(ci_hi), alpha=0.2, color=color)
    if sig.any():
        ax.scatter(times_ms[sig], smooth(mean_r)[sig],
                   s=8, color=color, alpha=0.7, zorder=5)

# Noise ceiling band
ax.fill_between(times_ms, smooth(nc_lower), smooth(nc_upper),
                color="gray", alpha=0.12, label="Noise ceiling")

ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Spearman r (brain vs. model RDM)")
ax.set_title("Brain-to-model RSA — official THINGS-MEG RDMs (200 test categories)\n"
             "Dots = above shuffle null p<0.05; band = noise ceiling")
ax.legend(fontsize=9)
ax.set_xlim(times_ms[0], times_ms[-1])
fig.tight_layout()
fig.savefig(str(figures_dir / "figure4_official_brain_model_rsa.png"), dpi=150)
print(f"Saved figures/figure4_official_brain_model_rsa.png")
