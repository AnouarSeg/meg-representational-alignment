#!/usr/bin/env python
"""Download THINGS-MEG from OpenNeuro.

THINGS-MEG is large (tens of GB). Pull ONE subject first, inspect it, then fetch
the rest. Examples:

    # see what would happen, no download
    python scripts/download_things_meg.py --dry-run

    # fetch dataset metadata + a single subject (recommended first step)
    python scripts/download_things_meg.py --subject BIGMEG1

    # fetch everything (only after you've confirmed the layout)
    python scripts/download_things_meg.py --all

Requires openneuro-py (in environment.yml). Alternative if it misbehaves:
    aws s3 sync --no-sign-request s3://openneuro.org/<accession> data/raw/
"""

from __future__ import annotations

import argparse

from thingsmeg.config import load_config
from thingsmeg.io import download_openneuro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", help="BIDS subject label to fetch (e.g. BIGMEG1)")
    ap.add_argument("--all", action="store_true", help="fetch the full dataset")
    ap.add_argument("--dry-run", action="store_true", help="print actions without downloading")
    args = ap.parse_args()

    cfg = load_config()
    cfg.ensure_dirs()

    if args.all:
        include = None
    elif args.subject:
        include = f"sub-{args.subject}/*"
    else:
        # Default: lightweight pull of top-level metadata only.
        include = "*.json"
        print("No --subject/--all given; fetching only top-level metadata. "
              "Use --subject <label> for a single-subject slice.")

    print(f"Dataset: {cfg.openneuro_id}  ->  {cfg.path_raw}")
    download_openneuro(cfg.openneuro_id, cfg.path_raw, include=include, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
