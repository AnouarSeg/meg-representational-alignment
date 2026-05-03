"""Build CLIP ViT-L/14 image embeddings for THINGS CC0 images.

Larger model than ViT-B/32 — 307M params vs 86M, 768-dim vs 512-dim features.
Output: data/derivatives/clip_large_rdm.npz  (1854×1854 cosine dissimilarity)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

IMG_DIR  = Path("/Volumes/MEG/things-meg/object_images_CC0")
OUT_PATH = Path("data/derivatives/clip_large_rdm.npz")
BATCH    = 32   # L/14 is larger


def get_image_paths(img_dir: Path) -> list[Path]:
    paths = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("._"))
    if not paths:
        paths = sorted(p for p in img_dir.glob("*.JPEG") if not p.name.startswith("._"))
    assert len(paths) == 1854, f"Expected 1854 images, got {len(paths)}"
    return paths


if __name__ == "__main__":
    print("Loading CLIP ViT-L/14...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    model.eval()

    paths = get_image_paths(IMG_DIR)
    print(f"Found {len(paths)} images")

    embeddings = []
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i:i+BATCH]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths])
        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.numpy().astype(np.float32))
        print(f"  {i+len(batch_paths)}/{len(paths)}", end="\r")

    E = np.concatenate(embeddings, axis=0)   # (1854, 768)
    print(f"\nEmbeddings: {E.shape}")

    rdm = (1 - E @ E.T).astype(np.float32)
    np.fill_diagonal(rdm, 0)

    np.savez(str(OUT_PATH), rdm=rdm)
    print(f"Saved {OUT_PATH}  range [{rdm.min():.3f}, {rdm.max():.3f}]")
