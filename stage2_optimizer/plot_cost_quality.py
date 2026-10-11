"""Cost / performance scatter plot of the stage-2 optimiser configurations.

    uv run python stage2_optimizer/plot_cost_quality.py

Reads the per-device result CSVs of ``stage2_optimizer/results`` (test seeds 1000-1059,
written by ``optimize_vg.py --out``) and saves ``stage2_optimizer/figures/cost_quality.png``.
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
# (label, file, pixel column, quality column, colour), in the order of the compression steps
CONFIGS = [
    ("starter optimiser", "test_seeds_1000_1059.csv", "baseline_pixels", "baseline", "gray"),
    ("scene at every grid point", "test_seeds_1000_1059.csv", "n_pixels", "quality", "tab:blue"),
    (
        "patches after 30 scenes",
        "test_seeds_1000_1059_patches.csv",
        "n_pixels",
        "quality",
        "tab:orange",
    ),
    (
        "compressed (no scene)",
        "test_seeds_1000_1059_compressed.csv",
        "n_pixels",
        "quality",
        "tab:green",
    ),
]


def load(results: pathlib.Path = RESULTS) -> list[tuple[str, np.ndarray, np.ndarray, str]]:
    """Per configuration: (label, pixels [M px], quality, colour); missing files are skipped."""
    out = []
    for label, name, px_col, q_col, colour in CONFIGS:
        path = results / name
        if not path.exists():
            continue
        with path.open() as f:
            rows = list(csv.DictReader(f))
        px = np.array([float(r[px_col]) for r in rows]) / 1e6
        out.append((label, px, np.array([float(r[q_col]) for r in rows]), colour))
    return out


def plot(configs: list[tuple[str, np.ndarray, np.ndarray, str]]) -> plt.Figure:
    """Scatter of quality against pixels: one dot per device, one labelled median per config.

    Args:
        configs: Output of :func:`load`.

    Returns:
        The figure (left: all configurations; right: zoom on quality >= 0.85).
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, zoom in zip(axes, (False, True)):
        shown = [c for c in configs if not (zoom and np.median(c[2]) < 0.85)]
        for label, px, q, colour in shown:
            ax.scatter(px, q, s=14, color=colour, alpha=0.35, linewidths=0)
            mx, mq = np.median(px), np.median(q)
            text = f"{label}: {mx:.2f} M px, median {mq:.3f}, mean {q.mean():.3f}"
            ax.scatter(
                mx,
                mq,
                s=110,
                color=colour,
                edgecolors="white",
                linewidths=1.5,
                zorder=3,
                label=text,
            )
        med = np.array([(np.median(px), np.median(q)) for _, px, q, _ in shown])
        ax.plot(med[:, 0], med[:, 1], color="dimgray", lw=1, ls=":", zorder=2)
        ax.set_xlabel("measured pixels per device (M px)")
        ax.set_xlim(0, None)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower left" if zoom else "center left")
    axes[0].set_ylabel("quality = contrast reached / maximal contrast")
    axes[0].set_ylim(0, 1.03)
    axes[0].set_title("Cost vs performance, 60 test devices (large dot = median)")
    axes[1].set_ylim(0.85, 1.005)
    axes[1].set_title("Zoom on quality ≥ 0.85")
    fig.tight_layout()
    return fig


def main() -> None:
    """Draw the figure and save it under ``figures/cost_quality.png``."""
    fig = plot(load())
    out = HERE / "figures" / "cost_quality.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
