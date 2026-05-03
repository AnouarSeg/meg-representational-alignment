"""Build ResNet-50 (supervised ImageNet) image embeddings for THINGS CC0 images.

Output: data/derivatives/resnet50_rdm.npz  (1854×1854 cosine dissimilarity)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

IMG_DIR  = Path("/Volumes/MEG/things-meg/object_images_CC0")
OUT_PATH = Path("data/derivatives/resnet50_rdm.npz")
BATCH    = 64

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_image_paths(img_dir: Path) -> list[Path]:
    paths = sorted(p for p in img_dir.glob("*.jpg") if not p.name.startswith("._"))
    if not paths:
        paths = sorted(p for p in img_dir.glob("*.JPEG") if not p.name.startswith("._"))
    assert len(paths) == 1854, f"Expected 1854 images, got {len(paths)}"
    return paths


if __name__ == "__main__":
    print("Loading ResNet-50 (pretrained ImageNet)...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove final classifier — use avgpool output (2048-dim)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    paths = get_image_paths(IMG_DIR)
    print(f"Found {len(paths)} images")

    embeddings = []
    for i in range(0, len(paths), BATCH):
        batch_paths = paths[i:i+BATCH]
        imgs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
        with torch.no_grad():
            feats = model(imgs).squeeze(-1).squeeze(-1)   # (B, 2048)
        embeddings.append(feats.numpy())
        print(f"  {i+len(batch_paths)}/{len(paths)}", end="\r")

    E = np.concatenate(embeddings, axis=0).astype(np.float32)   # (1854, 2048)
    print(f"\nEmbeddings: {E.shape}")

    # Cosine dissimilarity
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    E_n   = E / norms
    rdm   = 1 - (E_n @ E_n.T)
    rdm   = rdm.astype(np.float32)
    np.fill_diagonal(rdm, 0)

    np.savez(str(OUT_PATH), rdm=rdm)
    print(f"Saved {OUT_PATH}  range [{rdm.min():.3f}, {rdm.max():.3f}]")
