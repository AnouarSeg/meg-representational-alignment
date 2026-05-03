"""Build DINOv2 ViT-B/14 image embeddings for all 1854 THINGS concepts.

Saves: data/derivatives/dinov2_rdm.npz
  - 'rdm': (1854, 1854) cosine dissimilarity float32
  - 'concepts': sorted concept names (= THINGS category order)

Peak RAM: ~2 GB (model 330 MB + image batches).
Runtime: ~5 min on CPU.
"""

from __future__ import annotations
import gc, os, sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from thingsmeg.config import load_config

cfg     = load_config()
IMG_DIR = Path("/Volumes/MEG/things-meg/object_images_CC0")
deriv   = Path(cfg.path_derivatives)
OUT     = deriv / "dinov2_rdm.npz"
BATCH   = 32

print("Loading DINOv2 ViT-B/14 (cached)...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg") and not f.startswith("._")])
concepts = [Path(f).stem for f in imgs]
print(f"{len(concepts)} concepts")

all_embs = []
with torch.no_grad():
    for i in range(0, len(imgs), BATCH):
        batch_files = imgs[i:i+BATCH]
        tensors = [transform(Image.open(str(IMG_DIR / fn)).convert("RGB")) for fn in batch_files]
        batch = torch.stack(tensors)
        emb = model(batch).float().numpy()
        all_embs.append(emb)
        if (i // BATCH) % 5 == 0:
            print(f"  {i+len(batch_files)}/{len(imgs)}")
        del tensors, batch, emb; gc.collect()

embs = np.concatenate(all_embs, axis=0)
print(f"Embeddings shape: {embs.shape}")

norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
embs /= norms
sim = embs @ embs.T
rdm = (1.0 - sim).astype(np.float32)
np.fill_diagonal(rdm, 0.0)

np.savez(str(OUT), rdm=rdm, concepts=np.array(concepts))
print(f"Saved {OUT}  range {rdm.min():.4f}–{rdm.max():.4f}")
