"""Build full 1854-category SPOSE RDM from the 49-dim embedding.

Download the embedding first:
  curl -L "https://osf.io/f5rn6/download" -o data/spose_embedding_49d_sorted.txt
  (or from https://osf.io/f5rn6 → Files → spose_embedding_49d_sorted.txt)

The embedding rows correspond to the 1854 THINGS concepts in alphabetical order.
Similarity = dot product of L2-normalised 49-dim vectors.
Dissimilarity RDM = 1 - similarity.

Output: data/derivatives/spose_full_rdm.npz
"""
from pathlib import Path
import numpy as np

EMBED_PATH = Path("data/spose_embedding_49d_sorted.txt")
OUT        = Path("data/derivatives/spose_full_rdm.npz")

if __name__ == "__main__":
    if not EMBED_PATH.exists():
        raise FileNotFoundError(
            f"{EMBED_PATH} not found.\n"
            "Download from OSF: curl -L 'https://osf.io/f5rn6/download' "
            f"-o {EMBED_PATH}"
        )

    print(f"Loading SPOSE embedding from {EMBED_PATH}...")
    E = np.loadtxt(str(EMBED_PATH), dtype=np.float32)
    print(f"Embedding shape: {E.shape}")   # expect (1854, 49)

    # L2-normalise
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_n = E / (norms + 1e-12)

    # Cosine similarity → dissimilarity
    sim = E_n @ E_n.T
    rdm = (1 - sim).astype(np.float32)
    np.fill_diagonal(rdm, 0)

    # Load concept list from CLIP RDM for consistent ordering
    clip_d   = np.load("data/derivatives/clip_image_rdm.npz")
    concepts = clip_d["concepts"]

    np.savez(str(OUT), rdm=rdm, concepts=concepts)
    triu = rdm[np.triu_indices(len(rdm), 1)]
    print(f"Saved {OUT}  shape={rdm.shape}  mean_dissim={triu.mean():.4f}  std={triu.std():.4f}")
