"""thingsmeg — temporal dynamics of cross-subject representational alignment in human MEG.

Analysis stages (mirroring the project plan):

    preprocessing  MEG cleaning + epoching (MNE-Python)
    decoding       time-resolved single-subject category decoders
    rsa            cross-validated representational dissimilarity matrices
    alignment      cross-subject alignment (hyperalignment, SRM, CCA, Procrustes, shape metrics)
    stats          permutation/cluster statistics, noise ceilings, bootstrap CIs
    viz            figure generation

Configuration lives in config/config.yaml and is loaded via `config.load_config()`.
"""

from .config import Config, load_config

__all__ = ["Config", "load_config"]
__version__ = "0.1.0"
