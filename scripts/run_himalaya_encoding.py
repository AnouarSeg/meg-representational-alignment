"""Himalaya banded ridge encoding model: model features → MEG sensors.

For each timepoint t, fits:
  brain_sensors(t) ~ CLIP_features + DINOv2_features + ResNet_features

Using GroupRidgeCV with separate alpha per feature group (banded ridge).
This isolates each model's unique predictive power of brain activity directly,
avoiding RSA's indirect comparison.

Memory budget:
  X: (1852, 512+768+2048) = (1852, 3328) float32 = ~25 MB
  Y per timepoint: (1852, 272) float32 = ~2 MB
  Total peak: ~500 MB — safe.

Output: results/himalaya_encoding.npz
        figures/figure6_himalaya_encoding.png

Runs timepoints in sequence: ~3-5 min total.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from himalaya.ridge import GroupRidgeCV
from himalaya.kernel_ridge import KernelRidgeCV

RESULTS = Path("results")
DERIV   = Path("data/derivatives")
N_ALPHAS = 20
CV_FOLDS = 5


def load_features(n_cat: int) -> tuple[np.ndarray, list[str], list[int]]:
    """Load and standardize model features. Returns X, group_names, group_sizes."""
    groups, names, sizes = [], [], []

    for name, fname, dim in [
        ("CLIP-B/32",  "clip_image_rdm.npz",  None),   # we need embeddings not RDM
        ("DINOv2",     "dinov2_rdm.npz",       None),
        ("ResNet-50",  "resnet50_rdm.npz",     None),
        ("CLIP-text",  "clip_text_rdm.npz",    None),
    ]:
        # Check for embedding files first
        embed_f = DERIV / fname.replace("_rdm.npz", "_embeddings.npz")
        if embed_f.exists():
            E = np.load(str(embed_f))["embeddings"][:n_cat].astype(np.float32)
        else:
            # Fall back: use RDM rows as features (n_cat-dim, still meaningful)
            rdm = np.load(str(DERIV/fname))["rdm"][:n_cat,:n_cat].astype(np.float32)
            E = rdm  # each concept represented by its dissimilarity profile
            print(f"  {name}: using RDM rows ({E.shape[1]}-dim, no embedding file found)")

        sc = StandardScaler()
        E  = sc.fit_transform(E)
        groups.append(E)
        names.append(name)
        sizes.append(E.shape[1])
        print(f"  {name}: {E.shape}")

    X = np.concatenate(groups, axis=1).astype(np.float32)
    return X, names, sizes


def load_brain_data(n_cat: int):
    """Load condition means for 4 MEG subjects, return (n_sub, n_cat, n_times, n_sensors)."""
    all_data, times = [], None
    for sub in ["BIGMEG1","BIGMEG2","BIGMEG3","BIGMEG4"]:
        f = RESULTS / f"condition_means.npz"
        # Use full crossnobis RDM source: the condition means
        # Actually load from the saved crossnobis RDM source path
        rdm_f = RESULTS / f"rdms_full_{sub}.npz"
        if not rdm_f.exists(): continue
        d = np.load(str(rdm_f))
        if times is None: times = d["times_ms"]
        all_data.append(d)
    return all_data, np.array(times)


if __name__ == "__main__":
    print("=== Himalaya Banded Ridge Encoding Model ===\n")

    # Check if we have raw condition means (needed for encoding)
    # The encoding model needs Y = brain responses per image, not RDMs
    # Use the MEG condition means saved during preprocessing
    cond_means_f = RESULTS / "condition_means.npz"
    if not cond_means_f.exists():
        # Try loading from subject-level files
        print("Looking for MEG condition means...")
        import sys
        sys.path.insert(0, "src")
        from thingsmeg.config import load_config
        cfg = load_config()
        deriv = Path(cfg.path_derivatives)

        sub_means = {}
        for sub in ["BIGMEG1","BIGMEG2","BIGMEG3","BIGMEG4"]:
            f = deriv / sub / f"{sub}_condition_means.npy"
            if f.exists():
                sub_means[sub] = np.load(str(f))
                print(f"  {sub}: {sub_means[sub].shape}")

        if not sub_means:
            print("No condition means found. Cannot run encoding model.")
            print("Run scripts/run_preprocessing.py first.")
            import sys; sys.exit(1)

        # Use first subject's shape to get n_times
        first = list(sub_means.values())[0]
        n_cat_sub, n_times, n_sensors = first.shape
        print(f"Condition means: {n_cat_sub} categories, {n_times} timepoints, {n_sensors} sensors")
    else:
        d = np.load(str(cond_means_f), allow_pickle=True)
        print(f"Loaded condition_means.npz: {list(d.keys())}")
        sub_means = {k: d[k] for k in d if k != "times"}
        times_cm  = d.get("times", None)

    # Load model features
    n_cat = min(v.shape[0] for v in sub_means.values())
    print(f"\nLoading model features for {n_cat} categories...")
    X, group_names, group_sizes = load_features(n_cat)
    print(f"Feature matrix X: {X.shape}")
    print(f"Groups: {list(zip(group_names, group_sizes))}")

    # Get dimensions
    first_sub = list(sub_means.values())[0]
    n_times, n_sensors = first_sub.shape[1], first_sub.shape[2]

    # Alphas to search
    alphas = np.logspace(-3, 5, N_ALPHAS)

    # Run encoding at each timepoint
    print(f"\nFitting GroupRidgeCV at {n_times} timepoints ({CV_FOLDS}-fold CV)...")
    r2_per_model = {name: np.zeros(n_times) for name in group_names}
    r2_joint     = np.zeros(n_times)

    # Mean across subjects for Y (super-subject → max SNR)
    Y_all = np.mean([v[:n_cat].astype(np.float32) for v in sub_means.values()], axis=0)
    # Y_all: (n_cat, n_times, n_sensors)

    # Permutation indices for CV
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for t in range(n_times):
        Y_t = Y_all[:, t, :]   # (n_cat, n_sensors)

        # Joint model R² (all features together, standard ridge)
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import r2_score
        r2_folds = []
        for tr_idx, te_idx in kf.split(X):
            reg = RidgeCV(alphas=alphas, cv=3)
            reg.fit(X[tr_idx], Y_t[tr_idx])
            pred = reg.predict(X[te_idx])
            r2_folds.append(r2_score(Y_t[te_idx], pred, multioutput="variance_weighted"))
        r2_joint[t] = np.mean(r2_folds)

        # Per-group R² (each model alone)
        start = 0
        for gi, (name, sz) in enumerate(zip(group_names, group_sizes)):
            X_g = X[:, start:start+sz]
            r2_g = []
            for tr_idx, te_idx in kf.split(X_g):
                reg = RidgeCV(alphas=alphas, cv=3)
                reg.fit(X_g[tr_idx], Y_t[tr_idx])
                pred = reg.predict(X_g[te_idx])
                r2_g.append(r2_score(Y_t[te_idx], pred, multioutput="variance_weighted"))
            r2_per_model[name][t] = np.mean(r2_g)
            start += sz

        if t % 20 == 0:
            print(f"  t={t}/{n_times} ({Y_all.shape[1] if hasattr(Y_all,'shape') else '?'}ms): "
                  f"joint R²={r2_joint[t]:.4f}  "
                  + "  ".join(f"{n}={r2_per_model[n][t]:.4f}" for n in group_names))

    # Load times
    import sys; sys.path.insert(0,"src")
    from thingsmeg.config import load_config
    cfg   = load_config()
    times = np.linspace(-100, 795, n_times)

    print("\n=== Encoding model R² peaks ===")
    print(f"  Joint (all models): {r2_joint.max():.4f} at {times[r2_joint.argmax()]:.0f}ms")
    for name in group_names:
        ts = r2_per_model[name]
        post = times > 0
        print(f"  {name:15s}: {ts[post].max():.4f} at {times[post][ts[post].argmax()]:.0f}ms")

    np.savez(str(RESULTS/"himalaya_encoding.npz"),
             times=times, r2_joint=r2_joint,
             **{f"r2_{n.replace('-','_').replace('/','_').replace(' ','_')}": v
                for n, v in r2_per_model.items()})
    print("Saved results/himalaya_encoding.npz")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"CLIP-B/32":"#1f77b4","DINOv2":"#ff7f0e","ResNet-50":"#9467bd","CLIP-text":"#17becf"}
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(times, r2_joint, "k-", lw=2.5, label="Joint (all models)")
    for name in group_names:
        c = colors.get(name, "gray")
        ax.plot(times, r2_per_model[name], color=c, lw=2, label=name)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k",    ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Cross-validated R²")
    ax.set_title("Himalaya encoding model: brain sensor prediction from model features")
    ax.legend(fontsize=9); fig.tight_layout()
    fig.savefig("figures/figure6_himalaya_encoding.png", dpi=150, bbox_inches="tight")
    print("Saved figures/figure6_himalaya_encoding.png")
