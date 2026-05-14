"""Individual differences analysis (EEG1, n=48).

Questions:
  1. Does within-subject split-half reliability predict cross-subject alignment?
  2. Does decoding accuracy correlate with RSA strength?

All computation on existing .npz files — no epochs loaded, <50 MB peak.

Output: results/eeg1/individual_differences.npz
        figures/figure6_individual_differences.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import scipy.stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS     = Path("results")
RESULTS_EEG = Path("results/eeg1")


def peak_post_stim(arr: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return peak value in post-stimulus window per subject."""
    post = times > 0
    return arr[:, post].max(axis=1)


if __name__ == "__main__":
    # --- Load EEG RSA per subject ---
    rsa_d = np.load(RESULTS_EEG / "brain_model_rsa_eeg1.npz")
    times_eeg = rsa_d["times"]
    spose_sub  = rsa_d["SPOSE"]       # (48, 110)
    clip_sub   = rsa_d["CLIP_image"]  # (48, 110)

    spose_peak = peak_post_stim(spose_sub, times_eeg)   # (48,)
    clip_peak  = peak_post_stim(clip_sub,  times_eeg)

    # --- Load alignment noise ceiling (within-subject reliability) ---
    nc_d = np.load(RESULTS_EEG / "alignment_noise_ceiling.npz")
    ceiling_accs = nc_d["ceiling_accs"]   # (10, 110) — 10 subjects
    times_nc     = nc_d["times"]
    ceiling_peak = peak_post_stim(ceiling_accs, times_nc)  # (10,)

    # --- Load cross-subject alignment per subject pair (mean per subject) ---
    aln_d = np.load(RESULTS_EEG / "alignment_complexity_eeg1.npz")
    proc_all   = aln_d["procrustes_acc"]   # (1128, 110)
    pair_labels = aln_d["pair_labels"]     # (1128,) e.g. "sub-01-sub-02"
    times_aln  = np.linspace(-50, 495, proc_all.shape[1])

    # Mean cross-subject alignment per subject (average over all pairs involving that subject)
    subjects_63 = sorted(set(
        s for label in pair_labels
        for s in label.split("-", 1)[1].rsplit("-", 1)  # handles "sub-01-sub-02"
    ))
    # Parse pair labels properly
    all_subs = []
    for label in pair_labels:
        # format: "sub-XX-sub-YY"
        parts = label.split("sub-")
        sA = f"sub-{parts[1].rstrip('-')}"
        sB = f"sub-{parts[2]}"
        all_subs.append((sA, sB))

    unique_subs = sorted(set(s for pair in all_subs for s in pair))
    post_mask   = times_aln > 0
    proc_peak   = proc_all[:, post_mask].max(axis=1)   # (1128,)

    sub_mean_alignment = {}
    for si, sub in enumerate(unique_subs):
        pair_mask = np.array([sub in (a, b) for a, b in all_subs])
        sub_mean_alignment[sub] = proc_peak[pair_mask].mean()

    # --- Correlate within-subject ceiling vs cross-subject alignment ---
    # Only for the 10 subjects with ceiling computed
    nc_subjects = [f"sub-{i+1:02d}" for i in range(10)]
    cs_for_nc   = np.array([sub_mean_alignment.get(s, np.nan) for s in nc_subjects])
    valid        = ~np.isnan(cs_for_nc)

    r_nc_cs, p_nc_cs = scipy.stats.pearsonr(ceiling_peak[valid], cs_for_nc[valid])
    print(f"Within-subject ceiling vs cross-subject alignment: r={r_nc_cs:.3f}, p={p_nc_cs:.3f} (n={valid.sum()})")

    # --- Cross-subject alignment vs RSA strength (SPOSE) ---
    # Both available for all 48 subjects
    cs_all = np.array([sub_mean_alignment.get(s, np.nan) for s in unique_subs[:48]])
    valid2  = ~np.isnan(cs_all) & (cs_all > 0)
    r_cs_rsa, p_cs_rsa = scipy.stats.pearsonr(cs_all[valid2], spose_peak[valid2])
    print(f"Cross-subject alignment vs SPOSE RSA: r={r_cs_rsa:.3f}, p={p_cs_rsa:.3f} (n={valid2.sum()})")

    np.savez(str(RESULTS_EEG / "individual_differences.npz"),
             ceiling_peak=ceiling_peak,
             cs_for_nc=cs_for_nc,
             spose_peak=spose_peak,
             clip_peak=clip_peak,
             cs_all=cs_all,
             r_nc_cs=r_nc_cs, p_nc_cs=p_nc_cs,
             r_cs_rsa=r_cs_rsa, p_cs_rsa=p_cs_rsa)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.scatter(ceiling_peak[valid]*100, cs_for_nc[valid]*100,
               color="#1f77b4", s=60, zorder=3)
    nc_subjects_arr = np.array(nc_subjects)
    for i, (x, y) in enumerate(zip(ceiling_peak[valid]*100, cs_for_nc[valid]*100)):
        ax.annotate(nc_subjects_arr[valid][i], (x, y), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    m, b = np.polyfit(ceiling_peak[valid], cs_for_nc[valid], 1)
    xs = np.linspace(ceiling_peak[valid].min(), ceiling_peak[valid].max(), 50)
    ax.plot(xs*100, (m*xs+b)*100, "r--", lw=1.5)
    ax.set_xlabel("Within-subject ceiling peak (%)")
    ax.set_ylabel("Mean cross-subject alignment peak (%)")
    ax.set_title(f"Reliability → alignment\nr={r_nc_cs:.3f}, p={p_nc_cs:.3f} (n={valid.sum()})")

    ax = axes[1]
    ax.scatter(cs_all[valid2]*100, spose_peak[valid2]*1000,
               color="#e41a1c", s=40, alpha=0.7, zorder=3)
    m2, b2 = np.polyfit(cs_all[valid2], spose_peak[valid2], 1)
    xs2 = np.linspace(cs_all[valid2].min(), cs_all[valid2].max(), 50)
    ax.plot(xs2*100, (m2*xs2+b2)*1000, "k--", lw=1.5)
    ax.set_xlabel("Mean cross-subject alignment peak (%)")
    ax.set_ylabel("SPOSE RSA peak (r × 1000)")
    ax.set_title(f"Alignment → RSA\nr={r_cs_rsa:.3f}, p={p_cs_rsa:.3f} (n={valid2.sum()})")

    fig.suptitle("Individual differences: EEG1 (n=48)", fontsize=11)
    fig.tight_layout()
    out = Path("figures/figure6_individual_differences.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
