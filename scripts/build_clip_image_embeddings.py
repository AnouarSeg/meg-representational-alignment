"""Build CLIP ViT-B/32 IMAGE embeddings for all 1854 THINGS concepts.

Saves: data/derivatives/clip_image_rdm.npz
  - 'rdm': (1854, 1854) cosine dissimilarity
  - 'concepts': list of 1854 concept names (sorted alphabetically = THINGS nr order)

Peak RAM: ~1 GB (model + batch of images).
Runtime: ~3 min on CPU.

Usage:
    python scripts/build_clip_image_embeddings.py
"""

from __future__ import annotations
import gc, os, sys
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

cfg = load_config()
IMG_DIR = Path("/Volumes/MEG/things-meg/object_images_CC0")
deriv   = Path(cfg.path_derivatives)
OUT     = deriv / "clip_image_rdm.npz"

BATCH   = 64   # images per forward pass
MODEL   = "ViT-B-32"
PRETRAIN = "openai"

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL} ({PRETRAIN})...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAIN)
model.eval()
device = "cpu"   # safe for 16 GB machine

# ── Get sorted concept list (= THINGS category order by nr) ──────────────────
imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") and not f.startswith("._")])
concepts = [Path(f).stem for f in imgs]   # strip .jpg
print(f"{len(concepts)} concepts, first: {concepts[0]}, last: {concepts[-1]}")

# ── Extract embeddings in batches ─────────────────────────────────────────────
all_embs = []
with torch.no_grad():
    for i in range(0, len(imgs), BATCH):
        batch_files = imgs[i:i+BATCH]
        tensors = []
        for fn in batch_files:
            img = Image.open(str(IMG_DIR / fn)).convert("RGB")
            tensors.append(preprocess(img))
        batch = torch.stack(tensors).to(device)
        emb = model.encode_image(batch).float().numpy()
        all_embs.append(emb)
        if (i // BATCH) % 5 == 0:
            print(f"  {i+len(batch_files)}/{len(imgs)} images processed")
        del tensors, batch, emb; gc.collect()

embs = np.concatenate(all_embs, axis=0)   # (1854, 512)
print(f"Embeddings shape: {embs.shape}")

# Normalize → cosine similarity → dissimilarity
norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
embs /= norms
sim = embs @ embs.T
rdm = (1.0 - sim).astype(np.float32)
np.fill_diagonal(rdm, 0.0)

np.savez(str(OUT), rdm=rdm, concepts=np.array(concepts))
print(f"\nSaved {OUT}")
print(f"RDM range: {rdm.min():.4f} – {rdm.max():.4f}")
print(f"RDM mean (upper tri): {rdm[np.triu_indices(1854, k=1)].mean():.4f}")
