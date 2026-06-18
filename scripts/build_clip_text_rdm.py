"""CLIP text-encoder RDM: embed concept names with CLIP ViT-B/32 text tower.

Tests whether brain-model alignment is driven by visual features (image encoder)
or linguistic/semantic features (text encoder on concept names alone).
No images needed — just the 1854 THINGS concept names.

Output: data/derivatives/clip_text_rdm.npz
"""
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image

OUT   = Path("data/derivatives/clip_text_rdm.npz")
BATCH = 128

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer   = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)

    ref      = np.load("data/derivatives/clip_image_rdm.npz")
    concepts = [str(c) for c in ref["concepts"]]
    print(f"Encoding {len(concepts)} concept names via CLIP text encoder...")

    embeddings = []
    for i in range(0, len(concepts), BATCH):
        batch = concepts[i:i+BATCH]
        # Format: "a photo of a {concept}" (standard CLIP zero-shot prompt)
        prompts = [f"a photo of a {c.replace('_', ' ')}" for c in batch]
        tokens  = tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings.append(feats.cpu().float().numpy())

    E   = np.concatenate(embeddings, axis=0)
    rdm = (1 - E @ E.T).astype(np.float32)
    np.fill_diagonal(rdm, 0)
    np.savez(str(OUT), rdm=rdm, concepts=np.array(concepts))
    triu = rdm[np.triu_indices(len(rdm), 1)]
    print(f"Saved {OUT}  shape={rdm.shape}  mean={triu.mean():.4f}  std={triu.std():.4f}")
