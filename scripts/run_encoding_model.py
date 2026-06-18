"""Ridge encoding model: model image features → MEG brain activity.

Uses 100 MEG test categories with actual model embeddings (not RDM rows).
Extracts CLIP-B/32, DINOv2, ResNet-50, CLIP-text image/text features for the
100 test concepts, then fits cross-validated ridge regression predicting
each of 272 MEG sensors at each timepoint.

This directly measures how much variance in brain responses is explained
by each model's feature space — more sensitive than RSA.

Memory: (100, 512+768+2048) features × (100, 272) brain × 180 timepoints ≈ tiny.

Output: results/encoding_model.npz
        figures/figure6_encoding_model.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

RESULTS  = Path("results")
DERIV    = Path("data/derivatives")
IMG_DIR  = Path("/Volumes/MEG/things-meg/object_images_CC0")
ALPHAS   = np.logspace(-2, 6, 20)
CV_FOLDS = 5


def get_clip_image_features(concepts, device="cpu"):
    import torch, open_clip
    from PIL import Image
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model.eval().to(device)
    feats = []
    for c in concepts:
        name = str(c).replace(".jpg","").split("/")[-1]
        found = False
        for ext in [".jpg",".JPEG",".jpeg",".png"]:
            p = IMG_DIR / f"{name}{ext}"
            if p.exists():
                img = preprocess(Image.open(str(p)).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model.encode_image(img)
                    f = f / f.norm(dim=-1, keepdim=True)
                feats.append(f.cpu().float().numpy()[0])
                found = True; break
        if not found:
            feats.append(np.zeros(512))
    return np.array(feats)


def get_clip_text_features(concepts, device="cpu"):
    import torch, open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer   = open_clip.get_tokenizer("ViT-B-32")
    model.eval().to(device)
    prompts = [f"a photo of a {str(c).replace('_',' ').replace('.jpg','')}" for c in concepts]
    tokens  = tokenizer(prompts).to(device)
    with torch.no_grad():
        f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
    return f.cpu().float().numpy()


def get_dinov2_features(concepts, device="cpu"):
    import torch
    from PIL import Image
    import torchvision.transforms as T
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
    model.eval().to(device)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    feats = []
    for c in concepts:
        name = str(c).replace(".jpg","").split("/")[-1]
        found = False
        for ext in [".jpg",".JPEG",".jpeg",".png"]:
            p = IMG_DIR / f"{name}{ext}"
            if p.exists():
                img = tf(Image.open(str(p)).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model(img)
                    f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
                feats.append(f.cpu().float().numpy()[0])
                found = True; break
        if not found:
            feats.append(np.zeros(768))
    return np.array(feats)


def get_resnet_features(concepts, device="cpu"):
    import torch, torchvision.models as tvm
    import torchvision.transforms as T
    from PIL import Image
    model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    feats = []
    for c in concepts:
        name = str(c).replace(".jpg","").split("/")[-1]
        found = False
        for ext in [".jpg",".JPEG",".jpeg",".png"]:
            p = IMG_DIR / f"{name}{ext}"
            if p.exists():
                img = tf(Image.open(str(p)).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    f = model(img)
                    f = f / (f.norm(dim=-1, keepdim=True) + 1e-12)
                feats.append(f.cpu().float().numpy()[0])
                found = True; break
        if not found:
            feats.append(np.zeros(2048))
    return np.array(feats)


if __name__ == "__main__":
    # Load brain data
    d       = np.load(str(RESULTS/"condition_means.npz"), allow_pickle=True)
    times   = d["times_ms"]
    n_times = len(times)
    concepts = np.load(str(RESULTS/"model_rdms.npz"), allow_pickle=True)["category_names"]
    n_cat   = len(concepts)

    # Super-subject: mean across 4 subjects
    Y = np.mean([d[sub].transpose(0,2,1) for sub in ["BIGMEG1","BIGMEG2","BIGMEG3","BIGMEG4"]], axis=0)
    # Y: (n_cat, n_times, n_sensors)
    n_sensors = Y.shape[2]
    print(f"Brain data: {n_cat} categories, {n_times} timepoints, {n_sensors} sensors")

    # Extract features
    print("Extracting model features...")
    clip_f = get_clip_image_features(concepts);  print(f"  CLIP-image: {clip_f.shape}")
    text_f = get_clip_text_features(concepts);   print(f"  CLIP-text:  {text_f.shape}")
    dino_f = get_dinov2_features(concepts);      print(f"  DINOv2:     {dino_f.shape}")
    resn_f = get_resnet_features(concepts);      print(f"  ResNet-50:  {resn_f.shape}")

    model_features = {
        "CLIP-image": clip_f,
        "CLIP-text":  text_f,
        "DINOv2":     dino_f,
        "ResNet-50":  resn_f,
    }

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    r2_results = {name: np.zeros(n_times) for name in model_features}
    r2_joint   = np.zeros(n_times)

    # Joint features
    X_joint = np.concatenate([StandardScaler().fit_transform(f)
                               for f in model_features.values()], axis=1)
    print(f"Joint feature matrix: {X_joint.shape}")

    print(f"Fitting ridge encoding at {n_times} timepoints...")
    for t in range(n_times):
        Y_t = Y[:, t, :]  # (n_cat, n_sensors)

        # Joint
        r2_folds = []
        for tr, te in kf.split(X_joint):
            reg = RidgeCV(alphas=ALPHAS, cv=3, scoring="r2")
            reg.fit(X_joint[tr], Y_t[tr])
            r2_folds.append(r2_score(Y_t[te], reg.predict(X_joint[te]),
                                     multioutput="variance_weighted"))
        r2_joint[t] = np.mean(r2_folds)

        # Per-model
        for name, feats in model_features.items():
            X_g = StandardScaler().fit_transform(feats)
            r2_folds = []
            for tr, te in kf.split(X_g):
                reg = RidgeCV(alphas=ALPHAS, cv=3, scoring="r2")
                reg.fit(X_g[tr], Y_t[tr])
                r2_folds.append(r2_score(Y_t[te], reg.predict(X_g[te]),
                                         multioutput="variance_weighted"))
            r2_results[name][t] = np.mean(r2_folds)

        if t % 30 == 0:
            best = max(r2_results[n][t] for n in r2_results)
            print(f"  t={t}/{n_times} ({times[t]:.0f}ms)  joint={r2_joint[t]:.4f}  best_model={best:.4f}")

    print("\n=== Encoding model R² peaks ===")
    post = times > 0
    print(f"  Joint:       {r2_joint[post].max():.4f} at {times[post][r2_joint[post].argmax()]:.0f}ms")
    for name, ts in r2_results.items():
        print(f"  {name:15s}: {ts[post].max():.4f} at {times[post][ts[post].argmax()]:.0f}ms")

    np.savez(str(RESULTS/"encoding_model.npz"), times=times, r2_joint=r2_joint,
             **{f"r2_{n.replace('-','_').replace('/','_').replace(' ','_')}": v
                for n, v in r2_results.items()})
    print("Saved results/encoding_model.npz")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"CLIP-image":"#1f77b4","CLIP-text":"#17becf","DINOv2":"#ff7f0e","ResNet-50":"#9467bd"}
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, r2_joint, "k-", lw=2.5, label=f"Joint R²={r2_joint[post].max():.4f}")
    for name, ts in r2_results.items():
        ax.plot(times, ts, color=colors[name], lw=2, label=f"{name} R²={ts[post].max():.4f}")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Cross-validated R² (variance-weighted)")
    ax.set_title("Ridge encoding model: model features → MEG sensors (100 categories, n=4 super-subject)")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig("figures/figure6_encoding_model.png", dpi=150, bbox_inches="tight")
    print("Saved figures/figure6_encoding_model.png")
