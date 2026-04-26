"""Brain-to-model RSA: CLIP text embeddings + SPOSE similarity vs. MEG RDMs (Week 6, Figure 4).

Models:
  1. SPOSE: human-rated similarity (1854×1854 matrix from sourcedata)
  2. CLIP-text: ViT-B/32 text encoder embeddings of category names
  3. DINOv2-text: using CLIP text as proxy (DINOv2 is vision-only; text via CLIP)

Method: at each timepoint, correlate (Spearman) the upper triangle of the
brain RDM with each model RDM, restricted to our top-100 categories.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_brain_model_rsa.py

Outputs:
    results/model_rdms.npz         — SPOSE and CLIP RDMs (100×100)
    results/brain_model_rsa.npz    — brain-model RSA correlations (n_models, n_subjects, n_times)
    figures/figure4_brain_model.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.config import load_config

cfg = load_config()
subjects = cfg.raw["dataset"]["subjects"]
deriv_dir = Path(cfg.path_derivatives)
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
results_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

N_CATS = 100

# ── Identify top-100 category numbers and names ───────────────────────────────
print("Identifying top-100 categories…")
labels = np.load(str(deriv_dir / "sub-BIGMEG1" / "labels.npy"))
cats, counts = np.unique(labels, return_counts=True)
top_cats = np.sort(cats[np.argsort(counts)[-N_CATS:]])  # 1-indexed

# Get category names from sample_attributes (stream first subject)
import subprocess, io, csv

print("Fetching category names from sample_attributes…")
result = subprocess.run(
    ["/opt/anaconda3/envs/things-meg/bin/aws", "s3", "cp", "--no-sign-request",
     "s3://openneuro.org/ds004212/sourcedata/sample_attributes_P1.csv", "-"],
    capture_output=True, text=True
)
attr_df = pd.read_csv(io.StringIO(result.stdout))
# Extract category name from image_path: images_meg/categoryname/file.jpg
attr_df["category_name"] = attr_df["image_path"].str.split("/").str[1]
# Map things_category_nr -> category_name (take first occurrence)
cat_name_map = attr_df.groupby("things_category_nr")["category_name"].first().to_dict()
cat_names = [cat_name_map.get(int(c), f"category_{c}") for c in top_cats]
print(f"  Example names: {cat_names[:5]}")

# ── SPOSE model RDM ───────────────────────────────────────────────────────────
print("Building SPOSE model RDM…")
spose_sim = loadmat(str(deriv_dir / "spose_similarity.mat"))["spose_sim"]  # (1854, 1854)
# Subset to top-100 categories (0-indexed: category_nr - 1)
idx = (top_cats - 1).astype(int)
spose_sub = spose_sim[np.ix_(idx, idx)]
# Convert similarity to dissimilarity
spose_rdm = 1.0 - spose_sub
np.fill_diagonal(spose_rdm, 0.0)

# ── CLIP text embedding RDM ───────────────────────────────────────────────────
print("Extracting CLIP text embeddings…")
import open_clip

clip_model_name = cfg.raw["models"]["clip"]
clip_pretrained = cfg.raw["models"]["clip_pretrained"]
model, _, _ = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_pretrained)
model.eval()
tokenizer = open_clip.get_tokenizer(clip_model_name)

with torch.no_grad():
    # Replace underscores with spaces for natural text
    texts = [name.replace("_", " ") for name in cat_names]
    tokens = tokenizer(texts)
    clip_embeddings = model.encode_text(tokens).numpy()  # (N_CATS, embed_dim)

# Cosine dissimilarity RDM
clip_norm = clip_embeddings / (np.linalg.norm(clip_embeddings, axis=1, keepdims=True) + 1e-10)
clip_rdm = 1.0 - clip_norm @ clip_norm.T
np.fill_diagonal(clip_rdm, 0.0)
print(f"  CLIP embedding shape: {clip_embeddings.shape}")

# ── Save model RDMs ───────────────────────────────────────────────────────────
np.savez(
    results_dir / "model_rdms.npz",
    spose_rdm=spose_rdm,
    clip_rdm=clip_rdm,
    category_numbers=top_cats,
    category_names=cat_names,
)
print("Saved results/model_rdms.npz")

# ── Brain-model RSA ───────────────────────────────────────────────────────────
model_rdms = {"SPOSE": spose_rdm, "CLIP-text": clip_rdm}
model_names = list(model_rdms.keys())
upper_idx = np.triu_indices(N_CATS, k=1)

brain_model_corr = {}  # model -> (n_subjects, n_times)
times_ms = None

for subject in subjects:
    rdm_path = results_dir / f"rdms_{subject}.npz"
    if not rdm_path.exists():
        print(f"[SKIP] {subject} — no RDMs found, run run_rsa.py first")
        continue

    print(f"[{subject}] Computing brain-model RSA…")
    d = np.load(str(rdm_path))
    brain_rdms = d["rdms"]   # (n_windows, 100, 100)
    if times_ms is None:
        times_ms = d["times_ms"]

    for model_name, model_rdm in model_rdms.items():
        model_vec = model_rdm[upper_idx]
        corrs = np.array([
            spearmanr(brain_rdms[t][upper_idx], model_vec).statistic
            for t in range(len(brain_rdms))
        ])
        if model_name not in brain_model_corr:
            brain_model_corr[model_name] = []
        brain_model_corr[model_name].append(corrs)

# Stack into arrays
for k in brain_model_corr:
    brain_model_corr[k] = np.array(brain_model_corr[k])  # (n_subjects, n_times)

np.savez(
    results_dir / "brain_model_rsa.npz",
    times_ms=times_ms,
    model_names=model_names,
    **{k.replace("-", "_"): v for k, v in brain_model_corr.items()},
)
print("Saved results/brain_model_rsa.npz")

# ── Figure 4 ──────────────────────────────────────────────────────────────────
def smooth(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same")

colors = {"SPOSE": "steelblue", "CLIP-text": "darkorange"}
fig, ax = plt.subplots(figsize=(11, 5))

for model_name, corrs in brain_model_corr.items():
    mean_r = corrs.mean(axis=0)
    sem_r = corrs.std(axis=0) / np.sqrt(len(corrs))
    ax.plot(times_ms, smooth(mean_r), linewidth=2, label=model_name, color=colors[model_name])
    ax.fill_between(times_ms,
                    smooth(mean_r - sem_r),
                    smooth(mean_r + sem_r),
                    alpha=0.2, color=colors[model_name])

ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Time relative to stimulus onset (ms)")
ax.set_ylabel("Spearman r (brain RDM vs. model RDM)")
ax.set_title("Figure 4: Brain-to-model RSA over time")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "figure4_brain_model.png", dpi=150)
print("Saved figures/figure4_brain_model.png")
