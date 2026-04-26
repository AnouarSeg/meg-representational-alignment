"""Figure generation (Week 8).

Figure register should match the target literature (King, Dehaene, Bethge, Macke):
time on the x-axis in ms with stimulus onset at 0, chance/noise-ceiling references
drawn explicitly, shaded bootstrap CIs, significance bars from cluster tests. Keep
a single style module so all figures are visually consistent.

Planned figures:
  Figure 1  time-resolved decoding accuracy per subject (+ cluster sig bars)
  Figure 2  cross-subject decoding accuracy over time, by alignment method
  Figure 3  alignment complexity over time (headline)
  Figure 4  overlay of brain-model and cross-subject alignment time courses
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def set_style() -> None:
    """Apply a consistent, publication-leaning matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
    })


def savefig(fig, name: str, figures_dir: str | Path) -> Path:
    """Save a figure as both PDF (vector, for the paper) and PNG (for the README)."""
    figdir = Path(figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    pdf = figdir / f"{name}.pdf"
    png = figdir / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    return pdf
