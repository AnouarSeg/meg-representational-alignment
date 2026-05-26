"""Build RDM for untrained (random-weights) CLIP ViT-B/32.

Baseline: same architecture as CLIP-B/32 but randomly initialised weights.
If brain-model RSA with random weights ≈ trained CLIP, then the architecture
(not the training) drives alignment. If random ≈ 0, training is necessary.

Output: data/derivatives/clip_random_rdm.npz
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

IMG_DIR   = Path("/Volumes/MEG/things-meg/object_images_CC0")
OUT       = Path("data/derivatives/clip_random_rdm.npz")
BATCH     = 64

if __name__ == "__main__":
    import torch, open_clip
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load architecture without pretrained weights
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None   # random init
    )
    model.eval().to(device)

    # Load concept list from trained CLIP RDM (same order)
    ref = np.load("data/derivatives/clip_image_rdm.npz")
    concepts = [str(c) for c in ref["concepts"]]
    print(f"Encoding {len(concepts)} concepts with random ViT-B/32...")

    embeddings = []
    for i in range(0, len(concepts), BATCH):
        batch_concepts = concepts[i:i+BATCH]
        imgs = []
        for c in batch_concepts:
            # Try common image extensions
            for ext in [".jpg", ".JPEG", ".jpeg", ".png"]:
                p = IMG_DIR / f"{c}{ext}"
                if p.exists():
                    imgs.append(preprocess(Image.open(str(p)).convert("RGB")))
                    break
            else:
                imgs.append(torch.zeros(3, 224, 224))
        batch_tensor = torch.stack(imgs).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch_tensor)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().float().numpy())
        if (i // BATCH) % 10 == 0:
            print(f"  {i+len(batch_concepts)}/{len(concepts)}")

    E = np.concatenate(embeddings, axis=0)
    print(f"Embeddings shape: {E.shape}")

    # Cosine dissimilarity RDM
    rdm = 1 - (E @ E.T)
    np.fill_diagonal(rdm, 0)
    rdm = rdm.astype(np.float32)

    np.savez(str(OUT), rdm=rdm, concepts=np.array(concepts))
    print(f"Saved {OUT}  shape={rdm.shape}  mean_dissim={rdm[np.triu_indices(len(rdm),1)].mean():.4f}")
