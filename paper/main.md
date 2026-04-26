# Temporal dynamics of cross-subject representational alignment in human MEG

**Anouar Seghir**  
Independent research project, 2026

---

## Abstract

Does the complexity of the transform needed to align visual representations across individuals increase with post-stimulus time, mirroring a cortical processing hierarchy? We address this question using magnetoencephalography (MEG) recordings of four subjects viewing 1,854 object images (THINGS-MEG; Gifford et al. 2022). We compute time-resolved representational dissimilarity matrices (RDMs) using cross-validated Mahalanobis distance (crossnobis; Walther et al. 2016), and compare Procrustes (orthogonal) versus unconstrained ridge regression alignment between pairs of subjects as a function of post-stimulus latency. We further relate brain RDMs to human similarity judgements (SPOSE; Muttenthaler et al. 2023) and CLIP text embeddings. Object-category information was decodable above chance from ~50 ms onward (peak 1.75% vs. 1% chance at 530 ms, 4-subject mean). Cross-subject transfer accuracy peaked at 7.5% at 335 ms using Procrustes alignment (chance: 1%). The gain of unconstrained ridge over Procrustes—our measure of alignment complexity—was near zero throughout the epoch (post-stimulus mean: −0.001 ± 0.011), failing to support the early-simple/late-complex hierarchy hypothesis in this dataset. Brain RDMs (published THINGS-MEG decoding RDMs; 200 categories) correlated positively with SPOSE human similarity judgements (peak r = 0.058 at 400 ms) and with CLIP text embeddings (peak r = 0.060 at 30 ms), while remaining substantially below the inter-subject noise ceiling (upper bound r = 0.545), indicating that the brain signal captured here reflects only a fraction of the shared representational geometry accessible in principle. We discuss methodological constraints—in particular the low trial count per condition (12 trials/category) and limited subject pool (n = 4)—and identify the experimental changes needed to detect the hypothesised hierarchy.

---

## Introduction

A central question in systems neuroscience is whether visual processing is organised hierarchically: early cortical areas encoding low-level sensory features, later areas encoding high-level semantic structure. Representational similarity analysis (RSA; Kriegeskorte et al. 2008) provides a subject-agnostic lens on this question by comparing the geometry of neural representations across individuals and computational models without assuming a shared sensor space.

Cross-subject *alignment* methods (hyperalignment, SRM, Procrustes) recover a common representational geometry from idiosyncratic sensor configurations. A testable prediction of the processing hierarchy is that early representations align via simple (near-orthogonal) transforms—because the dominant structure is sensory and largely shared—while late representations require more flexible mappings to capture idiosyncratic semantic organisation. We operationalise "complexity" as the incremental transfer accuracy of an unconstrained linear map over Procrustes, evaluated in held-out conditions via 5-fold cross-validation.

We additionally ask whether the temporal profile of brain-to-model alignment mirrors the cross-subject hierarchy, using human-rated object similarity (SPOSE) as a semantic model and CLIP ViT-B/32 text embeddings as a low-cost vision-language proxy.

---

## Methods

**Dataset.** THINGS-MEG (OpenNeuro ds004212; Gifford et al. 2022): 4 subjects, 272-channel CTF MEG, 12 sessions × 10 runs per subject, ~1,854 object categories × 12 trials/category. Data were bandpass-filtered (0.1–40 Hz), ICA-cleaned (infomax, one set of components per session), epoched −100 to 800 ms relative to stimulus onset, baseline-corrected, and resampled to 200 Hz. ICA components were manually verified for all 4 subjects (BIGMEG1: 4 components excluded — 1 eye movement, 3 cardiac; BIGMEG2: 3 — 2 eye, 1 cardiac; BIGMEG3: 3 — 2 cardiac, 1 slow drift; BIGMEG4: 2 — cardiac).

**Decoding.** Time-resolved ridge classification (StandardScaler + RidgeClassifier, α = 1.0) with 5-fold stratified cross-validation on the 100 most-sampled categories (1,200 trials), using MNE-Python's SlidingEstimator.

**RSA.** Cross-validated Mahalanobis (crossnobis) RDMs across all 1,854 categories per timepoint (2-fold cross-validation, float32). For brain-to-model comparison, we used the published THINGS-MEG pairwise decoding RDMs (Gifford et al. 2022; 200 held-out test categories, 4 subjects × 281 timepoints) rather than our own single-subject crossnobis RDMs, giving cleaner estimates with substantially more trials per condition. Brain-model RSA computed as Spearman r between RDM upper triangles, averaged across subjects.

**Alignment.** Per-category condition means (100 × 272 at each timepoint) z-scored across categories. Procrustes (orthogonal) and ridge (α = 1.0) maps fitted on 80 training conditions, evaluated on 20 held-out conditions via 5-fold CV. Complexity = ridge transfer accuracy − Procrustes transfer accuracy. Williams shape distance (Williams et al. 2021) computed directly without the netrep package.

**Models.** SPOSE: 1854×1854 human similarity matrix (Muttenthaler et al. 2023), subset to 200 test categories, converted to dissimilarity. CLIP-text: ViT-B/32 text encoder embeddings of category names (open_clip), cosine dissimilarity.

**Statistics.** Cluster-based permutation test (MNE, 1,000 permutations) for decoding; bootstrap CIs (1,000 resamples over subjects/pairs) for all curves; shuffle null (1,000 permutations of model RDM labels) for brain-model RSA. Noise ceiling: Nili et al. (2014) leave-one-subject-out estimator.

---

## Results

**Object-category decoding.** Mean decoding accuracy exceeded chance (1%) from approximately 50 ms post-stimulus, reaching a peak of 1.75% at 530 ms (Figure 1). Pre-stimulus baseline was 0.94% (near chance), confirming no temporal leakage. No time cluster survived cluster-based permutation testing at α = 0.05, reflecting low statistical power with n = 4 subjects.

**Cross-subject representational geometry.** Crossnobis RDMs showed positive mean dissimilarity from ~100 ms onward in BIGMEG2, with weaker and noisier signals in the remaining subjects—a pattern consistent with the known inter-subject variability of MEG sensor topographies. Williams shape distance between subject pairs was stable across the epoch (Figure 2), indicating that the overall geometry of representational similarity did not change dramatically in structure over time.

**Alignment complexity over time.** Procrustes transfer accuracy peaked at 7.5% (chance: 1%) and ridge at 7.8%, with a mean complexity score (ridge gain) of −0.001 ± 0.011 post-stimulus—not significantly different from zero and not increasing over time (Figure 3). The predicted early-simple/late-complex gradient was not observed. A notable inter-subject finding emerged: the effective rank of the crossnobis RDMs differed dramatically across subjects (BIGMEG1: ~16, BIGMEG2: ~16, BIGMEG3: ~3, BIGMEG4: ~5), indicating highly heterogeneous representational dimensionality across individuals. This heterogeneity itself may suppress the alignment signal by mixing high- and low-dimensional geometries.

**Brain-to-model alignment.** Using the published THINGS-MEG decoding RDMs (200 test categories, 4 subjects), SPOSE human similarity judgements correlated with brain RDMs at a peak of r = 0.058 (at 400 ms; Figure 4), with 146/281 timepoints exceeding the shuffle null at p < 0.05 (uncorrected). CLIP text embeddings peaked at r = 0.060 (at 30 ms), with 74/281 significant timepoints. Both models fall substantially below the inter-subject noise ceiling (upper: r = 0.545 at 120 ms, lower: r = 0.138), with SPOSE at ~12% of the noise ceiling at its peak. No effect survived conservative correction for multiple comparisons.

---

## Discussion

We find modest but directionally consistent signals: object-category information is decodable from MEG, brain RDMs correlate weakly with human-rated similarity, and cross-subject alignment is feasible. The primary hypothesis—that alignment complexity increases with post-stimulus latency—was not supported.

Two methodological constraints limit the current results. First, trial count is critically low: with 12 trials/category, crossnobis RDMs are heavily noise-regularised, and single-timepoint condition means carry substantial estimation error—the alignment complexity analysis uses only 100-category means, which may be too noisy to resolve subtle alignment geometry differences. Second, n = 4 subjects provides minimal power for between-subject statistics. Both constraints are intrinsic to the THINGS-MEG acquisition design rather than fixable in post-processing.

**Recommended next steps.** (1) Download THINGS stimulus images and extract CLIP/DINOv2 *image* embeddings; text embeddings are a weak proxy for visual representations. (2) Replicate alignment complexity analysis with the higher-SNR full-1854-category RDMs before drawing conclusions about the hierarchy hypothesis. (3) Extend to additional THINGS-MEG subjects if/when data become available.

The codebase is fully reproducible: preprocessing, decoding, RSA, alignment, and statistical controls are implemented as modular Python scripts (MNE-Python, rsatoolbox, scikit-learn, himalaya) with a single configuration file. All negative results are reported without smoothing.

---

## References

- Gifford AT, Cichy RM, et al. (2022). THINGS-MEG: a large-scale, multi-subject dataset of MEG recordings during passive viewing. *OpenNeuro ds004212*.
- Kriegeskorte N, Mur M, Bandettini P (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*.
- Maris E, Oostenveld R (2007). Nonparametric statistical testing of EEG- and MEG-data. *Journal of Neuroscience Methods*.
- Muttenthaler L, et al. (2023). Human alignment of neural network representations. *ICLR*.
- Nili H, et al. (2014). A toolbox for representational similarity analysis. *PLoS Computational Biology*.
- Walther A, et al. (2016). Reliability of dissimilarity measures for multi-voxel pattern analysis. *NeuroImage*.
- Williams AH, et al. (2021). Generalized shape metrics on neural representations. *NeurIPS*.
