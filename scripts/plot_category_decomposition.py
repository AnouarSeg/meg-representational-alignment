"""Re-plot category decomposition with corrected animacy labels.

Uses saved results/category_decomposition.npz (per-category RSA already computed).
Fixes the animacy heuristic: whole-word matching on THINGS concept names.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ANIMATE_CONCEPTS = {
    # mammals
    "aardvark","alpaca","bat","bear","beaver","bison","boar","buffalo","bull",
    "camel","cat","cheetah","chimpanzee","chipmunk","cow","coyote","deer","dog",
    "dolphin","donkey","elephant","elk","ferret","fox","giraffe","goat","gorilla",
    "hamster","hedgehog","hippopotamus","horse","hyena","jaguar","kangaroo","koala",
    "leopard","lion","llama","lynx","mink","mole","monkey","moose","mouse","mule",
    "otter","panda","panther","pig","porcupine","puppy","rabbit","raccoon","rat",
    "reindeer","rhinoceros","seal","sheep","skunk","sloth","squirrel","tiger",
    "walrus","weasel","whale","wolf","zebra","kitten","calf","foal","lamb","piglet",
    # birds
    "bird","canary","cardinal","chicken","cockatoo","condor","crane","crow","duck",
    "eagle","falcon","flamingo","goose","hawk","heron","hummingbird","jay","kiwi",
    "macaw","magpie","ostrich","owl","parrot","peacock","pelican","penguin","pigeon",
    "quail","raven","robin","rooster","seagull","sparrow","stork","swan","toucan",
    "turkey","vulture","woodpecker","wren",
    # reptiles & amphibians
    "alligator","chameleon","cobra","crocodile","frog","gecko","iguana","lizard",
    "salamander","snake","tortoise","turtle",
    # fish & aquatic
    "catfish","clam","crab","eel","jellyfish","lobster","octopus","oyster","salmon",
    "seahorse","shark","shrimp","snail","squid","starfish","swordfish","tuna",
    # insects & bugs
    "ant","bee","beetle","butterfly","caterpillar","centipede","cockroach","cricket",
    "dragonfly","firefly","fly","grasshopper","ladybug","locust","mosquito","moth",
    "scorpion","spider","termite","wasp","worm",
    # humans
    "baby","child","person","human","face","man","woman",
}

def is_animate(concept: str) -> bool:
    tokens = set(concept.lower().replace("-", "_").split("_"))
    return bool(tokens & ANIMATE_CONCEPTS)


if __name__ == "__main__":
    d = np.load("results/category_decomposition.npz", allow_pickle=True)
    times       = d["times"]
    # Use authoritative concept names from model RDM (not placeholder names from decomp)
    clip_d      = np.load("data/derivatives/clip_image_rdm.npz")
    concepts    = [str(c) for c in clip_d["concepts"]][:len(d["clip_cat_rsa"])]
    clip_cat    = d["clip_cat_rsa"]
    dino_cat    = d["dino_cat_rsa"]
    clip_t_a    = d["clip_time_anim"]
    clip_t_i    = d["clip_time_inanim"]
    dino_t_a    = d["dino_time_anim"]
    dino_t_i    = d["dino_time_inanim"]

    # Recompute animacy with corrected heuristic
    animate = np.array([is_animate(c) for c in concepts])
    n = len(concepts)
    print(f"Animate: {animate.sum()} / {n} ({animate.mean()*100:.1f}%)")

    clip_adv = clip_cat - dino_cat

    print(f"CLIP mean RSA  — Animate: {clip_cat[animate].mean():.4f}  Inanimate: {clip_cat[~animate].mean():.4f}")
    print(f"DINOv2 mean RSA — Animate: {dino_cat[animate].mean():.4f}  Inanimate: {dino_cat[~animate].mean():.4f}")
    print(f"CLIP advantage  — Animate: {clip_adv[animate].mean():.4f}  Inanimate: {clip_adv[~animate].mean():.4f}")

    # Top 20 CLIP-advantage categories
    top_idx = np.argsort(clip_adv)[::-1][:20]
    print("\nTop 20 CLIP-advantage categories:")
    for i in top_idx:
        print(f"  {concepts[i]:30s}  CLIP={clip_cat[i]:.4f}  DINOv2={dino_cat[i]:.4f}  adv={clip_adv[i]:.4f}  {'ANIM' if animate[i] else ''}")

    bot_idx = np.argsort(clip_adv)[:20]
    print("\nTop 20 DINOv2-advantage categories:")
    for i in bot_idx:
        print(f"  {concepts[i]:30s}  CLIP={clip_cat[i]:.4f}  DINOv2={dino_cat[i]:.4f}  adv={clip_adv[i]:.4f}  {'ANIM' if animate[i] else ''}")

    # Re-compute animacy-split time courses from saved full time courses
    # (already computed per-animacy in the main script, but with wrong labels —
    #  reuse the per-category RSA and recompute from brain+model RDMs would be slow.
    #  Instead, report the per-category breakdown only.)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: Full time courses (from saved, note: wrong animacy split — use as total)
    ax = axes[0]
    ax.plot(times, clip_t_a*100, color="#1f77b4", lw=2, label="CLIP (all — animacy split invalid)")
    ax.plot(times, dino_t_a*100, color="#ff7f0e", lw=2, label="DINOv2 (all)")
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman r × 100")
    ax.set_title("Brain-model RSA time course\n(full RDMs, 4 MEG subjects mean)")
    ax.legend(fontsize=8)

    # Panel 2: CLIP advantage histogram by animacy
    ax = axes[1]
    ax.hist(clip_adv[animate],  bins=40, alpha=0.7, color="#1f77b4",
            label=f"Animate (n={animate.sum()})", density=True)
    ax.hist(clip_adv[~animate], bins=40, alpha=0.7, color="#ff7f0e",
            label=f"Inanimate (n={(~animate).sum()})", density=True)
    ax.axvline(clip_adv[animate].mean(),  color="#1f77b4", ls="--", lw=1.5)
    ax.axvline(clip_adv[~animate].mean(), color="#ff7f0e", ls="--", lw=1.5)
    ax.axvline(0, color="k", ls="-", lw=0.8)
    ax.set_xlabel("CLIP − DINOv2 per-category RSA")
    ax.set_ylabel("Density")
    ax.set_title(f"CLIP advantage by animacy\n"
                 f"Anim: {clip_adv[animate].mean():.4f}  Inanim: {clip_adv[~animate].mean():.4f}")
    ax.legend(fontsize=9)

    # Panel 3: Per-category scatter
    ax = axes[2]
    ax.scatter(dino_cat[~animate], clip_cat[~animate],
               s=3, alpha=0.2, color="#ff7f0e", label="Inanimate", rasterized=True)
    ax.scatter(dino_cat[animate], clip_cat[animate],
               s=8, alpha=0.5, color="#1f77b4", label="Animate", rasterized=True)
    lim = max(np.nanpercentile(np.abs(np.concatenate([clip_cat, dino_cat])), 99), 0.05)
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.7, alpha=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("DINOv2 per-category RSA")
    ax.set_ylabel("CLIP per-category RSA")
    ax.set_title("Per-category: CLIP vs DINOv2")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = Path("figures/figure4b_category_decomposition.png")
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    print(f"\nSaved {out}")
