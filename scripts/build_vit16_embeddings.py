"""ViT-B/16 supervised (ImageNet) embeddings — architecture-controlled CLIP comparison.

Isolates architecture (ViT-B vs ResNet) from training objective (supervised classification
vs vision-language). If ViT-B/16 supervised ≈ ResNet-50, architecture doesn't matter.
If ViT-B/16 supervised >> ResNet-50 but < CLIP-B/32, language training adds value.

Output: data/derivatives/vit16_rdm.npz
"""
from pathlib import Path
import numpy as np, torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image

IMG_DIR = Path("/Volumes/MEG/things-meg/object_images_CC0")
OUT     = Path("data/derivatives/vit16_rdm.npz")
BATCH   = 64

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_V1)
    # Remove classification head — use CLS token features
    model.heads = torch.nn.Identity()
    model.eval().to(device)

    preprocess = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    ref      = np.load("data/derivatives/clip_image_rdm.npz")
    concepts = [str(c) for c in ref["concepts"]]
    print(f"Encoding {len(concepts)} images with ViT-B/16 (ImageNet supervised)...")

    embeddings = []
    for i in range(0, len(concepts), BATCH):
        batch = concepts[i:i+BATCH]
        imgs  = []
        for c in batch:
            for ext in [".jpg", ".JPEG", ".jpeg", ".png"]:
                p = IMG_DIR / f"{c}{ext}"
                if p.exists():
                    imgs.append(preprocess(Image.open(str(p)).convert("RGB")))
                    break
            else:
                imgs.append(torch.zeros(3, 224, 224))
        with torch.no_grad():
            feats = model(torch.stack(imgs).to(device))
            feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-12)
        embeddings.append(feats.cpu().float().numpy())
        if i % 512 == 0:
            print(f"  {i+len(batch)}/{len(concepts)}")

    E   = np.concatenate(embeddings, axis=0)
    rdm = (1 - E @ E.T).astype(np.float32)
    np.fill_diagonal(rdm, 0)
    np.savez(str(OUT), rdm=rdm, concepts=np.array(concepts))
    triu = rdm[np.triu_indices(len(rdm), 1)]
    print(f"Saved {OUT}  shape={rdm.shape}  mean={triu.mean():.4f}  std={triu.std():.4f}")
