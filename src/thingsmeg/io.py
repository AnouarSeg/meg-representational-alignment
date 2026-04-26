"""Dataset download and BIDS layout discovery for THINGS-MEG.

This module is deliberately conservative: it discovers what is actually present on
disk rather than assuming a layout. The preprocessing design decisions downstream
depend on the real file structure (CTF .ds vs. .fif, channel naming, event coding),
so `summarize_layout()` is meant to be run and read before writing the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def download_openneuro(
    openneuro_id: str,
    target_dir: str | Path,
    include: str | None = None,
    dry_run: bool = False,
) -> None:
    """Download a dataset from OpenNeuro into `target_dir`.

    Uses openneuro-py. `include` can restrict to a glob (e.g. a single subject:
    "sub-BIGMEG1/*") to fetch a small slice first. THINGS-MEG is large (tens of GB),
    so always do a single-subject pull before committing to the full download.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] would download {openneuro_id} -> {target}"
              + (f" (include={include})" if include else " (full dataset)"))
        return

    from openneuro import download  # imported lazily so the module loads without it

    kwargs: dict[str, Any] = {"dataset": openneuro_id, "target_dir": str(target)}
    if include:
        kwargs["include"] = include
    download(**kwargs)


def find_meg_files(raw_dir: str | Path) -> list[Path]:
    """Return MEG recording paths under a BIDS root, format-agnostic.

    Looks for the common MEG containers: CTF (.ds dirs), FIFF (.fif), KIT (.con/.sqd),
    and BrainVision/EEG-style (.vhdr) just in case.
    """
    root = Path(raw_dir)
    found: list[Path] = []
    # CTF: top-level *_meg.ds dirs only (exclude nested hz.ds / hz2.ds sub-recordings)
    for p in sorted(root.rglob("*_meg.ds")):
        if not p.name.startswith("._") and p.is_dir() and p.parent.name == "meg":
            found.append(p)
    # FIFF / KIT / BrainVision fallbacks
    for pat in ("*meg.fif", "*.con", "*.sqd", "*.vhdr"):
        found.extend(sorted(
            p for p in root.rglob(pat) if not p.name.startswith("._")
        ))
    return found


def list_subjects(raw_dir: str | Path) -> list[str]:
    """Return BIDS subject labels (sub-XXXX -> XXXX) found under the dataset root."""
    root = Path(raw_dir)
    subs = sorted({p.name[len("sub-"):] for p in root.glob("sub-*") if p.is_dir()})
    return subs


def read_participants(raw_dir: str | Path) -> str | None:
    """Return the contents of participants.tsv if present, else None."""
    p = Path(raw_dir) / "participants.tsv"
    return p.read_text() if p.exists() else None


def summarize_layout(raw_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-serialisable summary of the on-disk dataset.

    Intended to be printed/saved and eyeballed before any preprocessing is written.
    """
    root = Path(raw_dir)
    meg_files = find_meg_files(root)
    dataset_description = None
    dd = root / "dataset_description.json"
    if dd.exists():
        dataset_description = json.loads(dd.read_text())

    return {
        "root": str(root),
        "exists": root.exists(),
        "dataset_description": dataset_description,
        "subjects": list_subjects(root),
        "n_meg_files": len(meg_files),
        "meg_file_examples": [str(p.relative_to(root)) for p in meg_files[:10]],
        "meg_formats": sorted({p.suffix or p.name.split(".")[-1] for p in meg_files}),
        "has_participants_tsv": (root / "participants.tsv").exists(),
    }
