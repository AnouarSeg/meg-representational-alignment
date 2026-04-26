# Temporal dynamics of cross-subject representational alignment in human MEG

This project asks whether the **temporal structure** of cross-subject representational
alignment in magnetoencephalography (MEG) reflects a processing hierarchy: do early
sensory representations align quickly and rigidly across subjects, while later
semantic/categorical representations align more loosely and require richer
transformations? And does this temporal hierarchy mirror the layer hierarchy of
vision foundation models (CLIP, DINOv2)?

The work uses the **THINGS-MEG** dataset (Hebart et al.), in which subjects view
natural object images from the THINGS database. The same image set underpins much
recent brain-model alignment work, which makes the brain-to-model comparison direct.

This is not "decode images from MEG" (solved) or "align subjects" (well studied in
fMRI). It is the open question of how alignment *complexity* evolves over
post-stimulus time, and how that evolution relates to artificial vision hierarchies.

## Status

Early scaffolding. Repository structure, configuration, and dataset-inspection
tooling are in place; the analysis modules are stubs pending inspection of the raw
data.

## Approach

| Stage | Module | Output |
|-------|--------|--------|
| MEG preprocessing (filter, ICA, epoch) | `thingsmeg.preprocessing` | cleaned epochs |
| Time-resolved category decoding | `thingsmeg.decoding` | Figure 1 |
| Cross-validated RSA (crossnobis RDMs) | `thingsmeg.rsa` | per-subject RDMs |
| Cross-subject alignment (hyperalignment, SRM, CCA, Procrustes, shape metrics) | `thingsmeg.alignment` | Figure 2 |
| Alignment complexity over time (headline) | `thingsmeg.alignment` | Figure 3 |
| Brain-to-model alignment (CLIP, DINOv2) | `thingsmeg.alignment` | Figure 4 |
| Statistics (cluster permutation, noise ceiling, bootstrap) | `thingsmeg.stats` | CIs / significance |

Methods are deliberately classical and carefully validated. Cross-validated distance
estimates (crossnobis), explicit noise ceilings, permutation statistics, and
bootstrap confidence intervals are used throughout. Negative or messy results are
reported and characterized rather than smoothed over.

## Setup

```bash
conda env create -f environment.yml
conda activate things-meg
pip install -e .
```

## Data

THINGS-MEG is distributed via OpenNeuro and is large. Fetch one subject first,
inspect it, then decide on the full download:

```bash
python scripts/download_things_meg.py --subject <label>   # single-subject slice
python scripts/inspect_dataset.py                         # confirm layout before preprocessing
```

Raw data and derivatives are not tracked in git (see `.gitignore`).

## Layout

```
config/        config.yaml — all analysis parameters
src/thingsmeg/ analysis package (preprocessing, decoding, rsa, alignment, stats, viz)
scripts/       download + inspection entry points
notebooks/     numbered, reproducible analysis notebooks (01_… onward)
results/       computed outputs (gitignored)
figures/       generated figures
```

## References

- Hebart et al. — THINGS / THINGS-MEG dataset
- Walther et al. (2016) — cross-validated representational distances
- Guntupalli et al.; Chen et al. — hyperalignment / shared response model
- Williams et al. — generalized shape metrics between representations
- Nili et al. (2014) — RSA toolbox and noise-ceiling estimation

## License

MIT (see `LICENSE`).
