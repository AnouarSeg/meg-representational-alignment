"""Extract intermediate-layer features from ResNet-50 and CLIP ViT-B/32.

ResNet-50 layers: stem, layer1, layer2, layer3, layer4
CLIP ViT-B/32 transformer blocks: 0, 3, 6, 9, 11

Uses the same 100 test concepts as the main RSA pipeline.
Global average pooling applied to spatial activations.

Output:
  data/derivatives/layerwise_resnet_rdms.npz  — {layer_name: (n_cat, n_cat) RDM}
  data/derivatives/layerwise_clip_rdms.npz    — {layer_name: (n_cat, n_cat) RDM}

Memory: tiny — 100 concepts × max 2048 dims per layer.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

IMG_DIR = Path("/Volumes/MEG/things-meg/object_images_CC0")
DERIV   = Path("data/derivatives")
RESULTS = Path("results")

CLIP_BLOCKS = [0, 3, 6, 9, 11]


def load_concepts():
    d = np.load(str(RESULTS / "model_rdms.npz"), allow_pickle=True)
    return d["category_names"]


def make_rdm(feats: np.ndarray) -> np.ndarray:
    """Cosine distance RDM from (n, d) feature matrix. Zero rows (missing images) → mean."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    missing = (norms.squeeze() == 0)
    if missing.any():
        mean_feat = feats[~missing].mean(axis=0)
        feats = feats.copy()
        feats[missing] = mean_feat
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    return cdist(feats, feats, metric="cosine").astype(np.float32)


def get_image_paths(concepts):
    paths = []
    for c in concepts:
        name = str(c).replace(".jpg", "").split("/")[-1]
        found = None
        for ext in [".jpg", ".JPEG", ".jpeg", ".png"]:
            p = IMG_DIR / f"{name}{ext}"
            if p.exists():
                found = p; break
        paths.append(found)
    n_missing = sum(p is None for p in paths)
    if n_missing:
        print(f"  WARNING: {n_missing}/{len(concepts)} images not found")
    return paths


def build_resnet_rdms(concepts, img_paths):
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image

    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.eval()

    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Register hooks for each layer
    layer_feats = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # Global average pool spatial dims if needed
            x = out.detach()
            if x.dim() == 4:
                x = x.mean(dim=[2, 3])  # GAP: (1, C)
            layer_feats[name] = x.squeeze(0).float().cpu().numpy()
        return hook

    hooks.append(model.layer1.register_forward_hook(make_hook("layer1")))
    hooks.append(model.layer2.register_forward_hook(make_hook("layer2")))
    hooks.append(model.layer3.register_forward_hook(make_hook("layer3")))
    hooks.append(model.layer4.register_forward_hook(make_hook("layer4")))

    layer_names = ["layer1", "layer2", "layer3", "layer4"]
    all_feats = {n: [] for n in layer_names}

    print("  ResNet-50: extracting layers", layer_names)
    for i, (c, p) in enumerate(zip(concepts, img_paths)):
        if p is None:
            for n in layer_names:
                dims = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
                all_feats[n].append(np.zeros(dims[n], dtype=np.float32))
            continue
        img = tf(Image.open(str(p)).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            model(img)
        for n in layer_names:
            all_feats[n].append(layer_feats[n].copy())

    for h in hooks:
        h.remove()

    rdms = {}
    for n in layer_names:
        F = np.stack(all_feats[n])
        rdms[n] = make_rdm(F)
        print(f"    {n}: features {F.shape} → RDM {rdms[n].shape}")

    return rdms


def build_clip_rdms(concepts, img_paths):
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval()

    # Hooks on transformer blocks
    block_feats = {}
    hooks = []

    def make_hook(idx):
        def hook(module, inp, out):
            # out is (seq_len, batch, dim) or (batch, seq_len, dim) depending on version
            x = out
            if isinstance(x, tuple):
                x = x[0]
            x = x.detach().float()
            # CLIP ViT: (seq_len, batch, dim) → take CLS token (index 0)
            if x.dim() == 3:
                if x.shape[1] == 1:    # (seq, batch=1, dim)
                    x = x[0, 0]
                else:                   # (batch=1, seq, dim)
                    x = x[0, 0]
            block_feats[idx] = x.cpu().numpy()
        return hook

    transformer = model.visual.transformer
    for idx in CLIP_BLOCKS:
        hooks.append(transformer.resblocks[idx].register_forward_hook(make_hook(idx)))

    block_names = [f"block{idx}" for idx in CLIP_BLOCKS]
    all_feats = {n: [] for n in block_names}

    print(f"  CLIP ViT-B/32: extracting blocks {CLIP_BLOCKS}")
    for i, (c, p) in enumerate(zip(concepts, img_paths)):
        if p is None:
            for n in block_names:
                all_feats[n].append(np.zeros(768, dtype=np.float32))
            continue
        img = preprocess(Image.open(str(p)).convert("RGB")).unsqueeze(0)
        block_feats.clear()
        with torch.no_grad():
            model.encode_image(img)
        for idx, n in zip(CLIP_BLOCKS, block_names):
            v = block_feats.get(idx, np.zeros(768, dtype=np.float32))
            all_feats[n].append(v if isinstance(v, np.ndarray) else v.numpy())

    for h in hooks:
        h.remove()

    rdms = {}
    for n in block_names:
        F = np.stack(all_feats[n])
        rdms[n] = make_rdm(F)
        print(f"    {n}: features {F.shape} → RDM {rdms[n].shape}")

    return rdms


if __name__ == "__main__":
    DERIV.mkdir(parents=True, exist_ok=True)
    print("=== Layer-wise embedding extraction ===\n")

    concepts = load_concepts()
    n_cat = len(concepts)
    print(f"Concepts: {n_cat}")

    img_paths = get_image_paths(concepts)

    print("\nBuilding ResNet-50 layer RDMs...")
    resnet_rdms = build_resnet_rdms(concepts, img_paths)
    np.savez(str(DERIV / "layerwise_resnet_rdms.npz"), **resnet_rdms)
    print(f"Saved {DERIV}/layerwise_resnet_rdms.npz")

    print("\nBuilding CLIP ViT-B/32 block RDMs...")
    clip_rdms = build_clip_rdms(concepts, img_paths)
    np.savez(str(DERIV / "layerwise_clip_rdms.npz"), **clip_rdms)
    print(f"Saved {DERIV}/layerwise_clip_rdms.npz")
