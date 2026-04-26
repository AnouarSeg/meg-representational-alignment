"""Re-run preprocessing for specified subjects with updated ICA exclusions.

Usage:
    python scripts/run_preprocessing.py BIGMEG2
    python scripts/run_preprocessing.py BIGMEG3
    python scripts/run_preprocessing.py BIGMEG4
    python scripts/run_preprocessing.py          # runs all subjects in config
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from thingsmeg.config import load_config
from thingsmeg.preprocessing import preprocess_subject

cfg = load_config()

subjects = sys.argv[1:] if len(sys.argv) > 1 else cfg.raw["dataset"]["subjects"]

for subject in subjects:
    print(f"\n{'='*60}")
    print(f"Preprocessing {subject}")
    print(f"{'='*60}")
    result = preprocess_subject(subject, cfg)
    print(f"Done: {result}")
