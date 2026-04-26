"""Week 7 controls: cluster permutation tests, noise ceilings, bootstrap CIs.

Adds statistical rigour to Figures 1-4. Saves updated figures with significance
markers and noise ceiling bands. Results reported honestly — null results included.

Usage:
    /opt/anaconda3/envs/things-meg/bin/python scripts/run_controls.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.config import load_config
from thingsmeg.stats import (
    bootstrap_ci,
    cluster_permutation_1d,
    noise_ceiling,
    noise_ceiling_over_time,
    shuffle_null_rsa,
)

cfg = load_config()
subjects = cfg.raw["dataset"]["subjects"]
results_dir = Path(cfg.raw["paths"]["results"])
figures_dir = Path(cfg.raw["paths"]["figures"])
N_CATS = 100

def smooth(x, n=7):
    return np.convolve(x, np.ones(n) / n, mode="same")

# ── Figure 1 update: decoding with cluster permutation test ──────────────────
print("=== Figure 1: Decoding controls ===")
dec = np.load(str(results_dir / "decoding_scores.npz"), allow_pickle=True)
scores = dec["scores"]   # (n_subjects, n_times)
times_ms = dec["times"]
subj_list = list(dec["subjects"])
chance = 1.0 / N_CATS

t_obs, clusters, cluster_pv, _ = cluster_permutation_1d(scores, chance, cfg, tail=1)
sig_times = np.zeros(len(times_ms), dtype=bool)
for cl, pv in zip(clusters, cluster_pv):
    if pv < 0.05:
        sig_times[cl] = True

ci_lo, ci_hi = bootstrap_ci(scores, n_boot=int(cfg.raw["stats"]["n_bootstrap"]),
                              seed=int(cfg.raw["stats"]["random_seed"]))
mean_score = scores.mean(axis=0)

fig, ax = plt.subplots(figsize=(11, 5))
for i, s in enumerate(subj_list):
    ax.plot(times_ms, smooth(scores[i]), linewidth=1, alpha=0.5, label=s)
ax.plot(times_ms, smooth(mean_score), color="black", linewidth=2.5, label="Mean")
ax.fill_between(times_ms, smooth(ci_lo), smooth(ci_hi), color="black", alpha=0.15,
                label="95% bootstrap CI")
ax.axhline(chance, color="gray", linestyle="--", linewidth=1, label=f"Chance (1/{N_CATS})")
ax.axvline(0, color="black", linewidth=0.8)
if sig_times.any():
    ax.fill_between(times_ms, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=sig_times, color="gold", alpha=0.3, label="p<0.05 cluster")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Decoding accuracy")
ax.set_title("Figure 1: Time-resolved decoding with cluster permutation test")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(figures_dir / "figure1_decoding_stats.png", dpi=150)
print(f"  Significant clusters: {sum(pv < 0.05 for pv in cluster_pv)}")
print("  Saved figure1_decoding_stats.png")

# ── RSA noise ceiling over time ───────────────────────────────────────────────
print("\n=== RSA noise ceiling ===")
rdms_by_subject = {}
rsa_times_ms = None
for subject in subjects:
    rdm_path = results_dir / f"rdms_{subject}.npz"
    if rdm_path.exists():
        d = np.load(str(rdm_path))
        rdms_by_subject[subject] = d["rdms"]  # (n_times, 100, 100)
        if rsa_times_ms is None:
            rsa_times_ms = d["times_ms"]

if rdms_by_subject:
    nc_upper, nc_lower = noise_ceiling_over_time(rdms_by_subject)
    np.savez(str(results_dir / "noise_ceiling.npz"),
             upper=nc_upper, lower=nc_lower, times_ms=rsa_times_ms)
    print(f"  Peak noise ceiling: upper={nc_upper.max():.3f}, lower={nc_lower.max():.3f}")
    print("  Saved noise_ceiling.npz")

# ── Figure 4 update: brain-model RSA with noise ceiling + shuffle null ────────
print("\n=== Figure 4: Brain-model RSA controls ===")
bm = np.load(str(results_dir / "brain_model_rsa.npz"), allow_pickle=True)
bm_times = bm["times_ms"]
model_rdms = np.load(str(results_dir / "model_rdms.npz"), allow_pickle=True)
upper_idx = np.triu_indices(N_CATS, k=1)

# Build brain RDM vectors per time (averaged across subjects)
brain_vecs_by_time = None
if rdms_by_subject:
    # Stack: (n_subjects, n_times, n_pairs)
    all_rdm_vecs = np.stack([
        rdms_by_subject[s][:, upper_idx[0], upper_idx[1]]
        for s in rdms_by_subject
    ])  # (n_subj, n_times, n_pairs)
    brain_vecs_by_time = all_rdm_vecs.mean(axis=0)  # (n_times, n_pairs)

colors = {"SPOSE": "steelblue", "CLIP-text": "darkorange"}
fig, ax = plt.subplots(figsize=(11, 5))

for model_key in ["SPOSE", "CLIP_text"]:
    model_name = model_key.replace("_", "-")
    if model_key not in bm:
        continue
    corrs = bm[model_key]  # (n_subjects, n_times)
    mean_r = corrs.mean(axis=0)
    ci_lo, ci_hi = bootstrap_ci(corrs, n_boot=int(cfg.raw["stats"]["n_bootstrap"]),
                                  seed=int(cfg.raw["stats"]["random_seed"]))

    # Shuffle null
    if brain_vecs_by_time is not None:
        rdm_key = "spose_rdm" if "SPOSE" in model_name else "clip_rdm"
        model_vec = model_rdms[rdm_key][upper_idx]
        null = shuffle_null_rsa(brain_vecs_by_time, model_vec,
                                 n_perm=int(cfg.raw["stats"]["n_permutations"]),
                                 seed=int(cfg.raw["stats"]["random_seed"]))
        null_95 = np.percentile(null, 95, axis=0)
        sig = mean_r > null_95
    else:
        sig = np.zeros(len(bm_times), dtype=bool)

    color = colors.get(model_name, "gray")
    ax.plot(bm_times, smooth(mean_r), linewidth=2, label=model_name, color=color)
    ax.fill_between(bm_times, smooth(ci_lo), smooth(ci_hi), alpha=0.2, color=color)
    if sig.any():
        ax.plot(bm_times[sig], np.full(sig.sum(), mean_r[sig].mean()),
                ".", color=color, markersize=4, alpha=0.6)

# Noise ceiling band
if rdms_by_subject:
    # Interpolate noise ceiling to bm_times grid
    from scipy.interpolate import interp1d
    f_up = interp1d(rsa_times_ms, smooth(nc_upper), bounds_error=False, fill_value="extrapolate")
    f_lo = interp1d(rsa_times_ms, smooth(nc_lower), bounds_error=False, fill_value="extrapolate")
    nc_up_interp = f_up(bm_times)
    nc_lo_interp = f_lo(bm_times)
    ax.fill_between(bm_times, nc_lo_interp, nc_up_interp, color="gray", alpha=0.12,
                    label="Noise ceiling")

ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Spearman r (brain vs. model RDM)")
ax.set_title("Figure 4: Brain-to-model RSA with noise ceiling and shuffle null")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(figures_dir / "figure4_brain_model_stats.png", dpi=150)
print("  Saved figure4_brain_model_stats.png")

# ── Figure 3 update: alignment complexity with bootstrap CI ───────────────────
print("\n=== Figure 3: Complexity controls ===")
alg = np.load(str(results_dir / "alignment_complexity.npz"), allow_pickle=True)
alg_times = alg["times_ms"]
complexity = alg["complexity"]        # (n_pairs, n_times)
procrustes_acc = alg["procrustes_acc"]
ridge_acc = alg["ridge_acc"]

ci_lo_c, ci_hi_c = bootstrap_ci(complexity, n_boot=int(cfg.raw["stats"]["n_bootstrap"]),
                                  seed=int(cfg.raw["stats"]["random_seed"]))
mean_comp = complexity.mean(axis=0)

# Permutation null for complexity: shuffle timepoints
rng = np.random.default_rng(int(cfg.raw["stats"]["random_seed"]))
n_perm = int(cfg.raw["stats"]["n_permutations"])
null_comp = np.array([
    np.random.permutation(mean_comp) for _ in range(n_perm)
])
null_95_comp = np.percentile(null_comp, 95, axis=0)
sig_comp = mean_comp > null_95_comp

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(alg_times, smooth(mean_comp), color="crimson", linewidth=2.5, label="Mean complexity")
ax.fill_between(alg_times, smooth(ci_lo_c), smooth(ci_hi_c), color="crimson", alpha=0.2,
                label="95% bootstrap CI")
if sig_comp.any():
    ax.fill_between(alg_times, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=sig_comp, color="gold", alpha=0.3, label="p<0.05 (shuffle)")
ax.axvline(0, color="black", linewidth=0.8)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Ridge gain over Procrustes")
ax.set_title("Alignment complexity over time\n(+ bootstrap CI, shuffle null)")
ax.legend(fontsize=8)

ax2 = axes[1]
mean_proc = procrustes_acc.mean(axis=0)
mean_ridge = ridge_acc.mean(axis=0)
ci_lo_p, ci_hi_p = bootstrap_ci(procrustes_acc, n_boot=500, seed=42)
ci_lo_r, ci_hi_r = bootstrap_ci(ridge_acc, n_boot=500, seed=42)
ax2.plot(alg_times, smooth(mean_proc), color="steelblue", linewidth=2, label="Procrustes")
ax2.fill_between(alg_times, smooth(ci_lo_p), smooth(ci_hi_p), color="steelblue", alpha=0.2)
ax2.plot(alg_times, smooth(mean_ridge), color="darkorange", linewidth=2, label="Ridge")
ax2.fill_between(alg_times, smooth(ci_lo_r), smooth(ci_hi_r), color="darkorange", alpha=0.2)
ax2.axhline(1/N_CATS, color="gray", linestyle="--", linewidth=1, label="Chance")
ax2.axvline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Time (ms)")
ax2.set_ylabel("Transfer accuracy")
ax2.set_title("Cross-subject transfer accuracy")
ax2.legend(fontsize=8)

fig.suptitle("Figure 3: Alignment complexity with statistical controls", fontsize=13)
fig.tight_layout()
fig.savefig(figures_dir / "figure3_complexity_stats.png", dpi=150)
print("  Saved figure3_complexity_stats.png")
print("\nAll controls done.")
