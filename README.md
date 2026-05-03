# Temporal dynamics of cross-subject representational alignment in human MEG

**Does alignment complexity between subjects increase over post-stimulus time, mirroring the cortical hierarchy?**

This project tests whether early visual representations align across individuals via simple (near-orthogonal) transforms, while later semantic representations require more flexible mappings — and whether this gradient mirrors the supervision hierarchy of CLIP, ResNet-50, and DINOv2.

**Datasets:**
- **THINGS-MEG** (Gifford et al. 2022, OpenNeuro ds004212) — 4 subjects, 272-channel CTF MEG, 1,854 object images, 12 trials/category
- **THINGS-EEG1** (Gifford et al. 2022, OpenNeuro ds003825) — 48 subjects, 63-channel EEG, same 1,854 THINGS concepts, RSVP at 10 Hz

---

## Key results

| Analysis | Result |
|---|---|
| Object decoding (LDA, 100 categories) | Mean peak **2.7%** at 360–415 ms (range 2.2–3.3%; chance 1%) |
| Cross-subject transfer — MEG (n=4) | Peak 7.5% at 335 ms (chance 1%) |
| Cross-subject transfer — EEG (n=48, 1128 pairs) | Peak **0.335%** at 120 ms (chance 0.054%, **6.2× chance**) |
| Alignment noise ceiling (EEG) | Within-subject peak **0.411%** → cross-subject at **81.5% of ceiling** |
| Alignment complexity (ridge − Procrustes) — MEG | −0.001 ± 0.011 — null |
| Alignment complexity — EEG | **−0.013%** (95% CI: −0.020 to −0.006%); cluster permutation **p < 0.001** |
| SPOSE brain-model RSA (1852 categories) | r = 0.046 at 380 ms — **17.3% of noise ceiling** |
| CLIP ViT-B/32 brain-model RSA | r = 0.043 at 325 ms — 16.1% of NC |
| CLIP ViT-L/14 brain-model RSA | r = 0.030 at 340 ms — 11.2% of NC |
| ResNet-50 brain-model RSA | r = 0.029 at 410 ms — 10.9% of NC |
| DINOv2 ViT-B/14 brain-model RSA | r = 0.013 at 380 ms — 4.8% of NC (weakest) |
| V1 fMRI→MEG peak | **125 ms** |
| FFA fMRI→MEG peak | **450 ms** — V1→FFA gap ~325 ms confirms cortical hierarchy |

**Primary finding:** The early-simple/late-complex alignment hypothesis is **definitively rejected** across both MEG (n=4) and EEG (n=48, p < 0.001). Cross-subject EEG alignment reaches 81.5% of the within-subject noise ceiling, ruling out insufficient signal as an explanation. Ridge alignment is significantly *worse* than Procrustes throughout — consistent with EEG representations being low-dimensional (63 channels) where over-parameterised maps underfit.

**Secondary finding:** A supervision gradient is confirmed in brain-model RSA: models trained on categorical objectives (SPOSE, CLIP-B/32) align more strongly with late MEG activity than self-supervised texture models (DINOv2). Notably CLIP-L/14 underperforms CLIP-B/32, suggesting alignment benefit saturates before model scale.

---

## Figure gallery

### Figure 1: Object-category decoding (LDA)
![Decoding](figures/figure1_decoding.png)

### Figure 2 & 3: Cross-subject alignment and complexity (MEG)
![Alignment](figures/figure2_alignment.png)
![Complexity](figures/figure3_complexity.png)

### Figure 3b: Alignment complexity replication (EEG, n=48)
![EEG Alignment](figures/figure3b_alignment_complexity_eeg1.png)

### Figure 3c: Alignment noise ceiling
![Noise Ceiling](figures/figure3c_alignment_noise_ceiling.png)

### Figure 3d: Complexity null — cluster permutation test
![Permutation](figures/figure3d_alignment_complexity_permutation.png)

### Figure 4: Brain-to-model RSA (5 models, 1852 categories)
![Brain-model RSA](figures/figure4_brain_model_rsa_full1854.png)

*Supervision gradient: SPOSE ≈ CLIP-B/32 > CLIP-L/14 ≈ ResNet-50 >> DINOv2. All models peak late (325–410 ms), consistent with categorical rather than low-level representations.*

### Figure 5: Cortical hierarchy validation
![Hierarchy](figures/figure5_hierarchy_validation.png)

*Official THINGS-MEG validation time courses. V1 (green) peaks at 125 ms; FFA (red) peaks at 450 ms — a 325 ms gap confirming the ventral stream hierarchy.*

---

## Reproduction

```bash
conda env create -f environment.yml
conda activate things-meg
pip install -e .
```

**THINGS-MEG** raw data (~377 GB) to `/Volumes/MEG/things-meg/`; THINGS CC0 images to `/Volumes/MEG/things-meg/object_images_CC0/`.
**THINGS-EEG1** (~55 GB) downloaded automatically by `scripts/download_things_eeg1.py`.

Run in order:

```bash
# === THINGS-MEG pipeline ===
python scripts/run_preprocessing.py --subject BIGMEG1   # repeat for BIGMEG2–4
python scripts/run_rsa_full.py --subject BIGMEG1        # full 1854-cat crossnobis RDMs
python scripts/run_alignment.py                          # MEG alignment complexity

# Model embeddings (one-time)
python scripts/build_clip_image_embeddings.py
python scripts/build_clip_large_embeddings.py
python scripts/build_dinov2_embeddings.py
python scripts/build_resnet50_embeddings.py

# Brain-model RSA on full 1852 categories
python scripts/run_brain_model_rsa_full1854.py

# Cortical hierarchy validation
python scripts/run_hierarchy_validation.py

# === THINGS-EEG1 pipeline (n=48 replication) ===
python scripts/download_things_eeg1.py --all             # ~55 GB
bash scripts/batch_preprocess_and_align.sh               # preprocess + alignment

# Controls
python scripts/run_alignment_noise_ceiling.py
python scripts/run_alignment_complexity_permutation.py
python scripts/run_brain_model_rsa_eeg1.py
```

Or open `notebooks/01_analysis_walkthrough.ipynb` for an end-to-end walkthrough.

---

## Layout

```
config/          config.yaml — all analysis parameters
src/thingsmeg/   analysis package (preprocessing, decoding, rsa, alignment, stats)
scripts/         runnable entry points (one task per script)
notebooks/       01_analysis_walkthrough.ipynb — full pipeline walkthrough
paper/           main.md / main.html / main.pdf — research-style write-up
results/         computed .npz outputs (gitignored — regenerable)
figures/         generated figures (gitignored — regenerable)
data/            derivatives cache (gitignored — large)
```

---

## Methods summary

- **Preprocessing (MEG):** MNE-Python, bandpass 0.1–40 Hz, infomax ICA (manual verification, 4 subjects), epoch −100–800 ms, baseline, 200 Hz.
- **Preprocessing (EEG):** BrainVision → MNE, bandpass 0.1–40 Hz, average re-reference, epoch −50–495 ms, baseline, 200 Hz. 48 subjects, 1854 concepts.
- **Decoding:** LDA with Ledoit-Wolf shrinkage, 5-fold stratified CV, 100 categories.
- **RSA:** Crossnobis (Walther et al. 2016), Spearman r, Nili et al. (2014) noise ceiling.
- **Alignment:** Procrustes (orthogonal) vs. ridge (α=1.0), 5-fold CV. Complexity = ridge − Procrustes. Cluster permutation test (1,000 permutations, sign-flip).
- **Brain-model RSA:** Full 1852-category crossnobis RDMs vs. SPOSE, CLIP-B/32, CLIP-L/14, ResNet-50, DINOv2; Spearman r with bootstrap CIs.
- **Alignment noise ceiling:** Within-subject split-half reliability (10 subjects, 5-fold CV).

---

## References

- Gifford AT et al. (2022). THINGS-MEG. *OpenNeuro ds004212*.
- Gifford AT et al. (2022). THINGS-EEG1. *OpenNeuro ds003825*.
- Kriegeskorte N et al. (2008). Representational similarity analysis. *Front. Syst. Neurosci.*
- Muttenthaler L et al. (2023). Human alignment of neural network representations. *ICLR*.
- Nili H et al. (2014). A toolbox for RSA. *PLoS Comput. Biol.*
- Walther A et al. (2016). Reliability of dissimilarity measures. *NeuroImage*.
- Williams AH et al. (2021). Generalized shape metrics on neural representations. *NeurIPS*.

## License

MIT
