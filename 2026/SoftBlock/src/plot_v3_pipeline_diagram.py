"""
Generate pipeline diagram for SoftBlock (overlap-aware coarse-to-fine recovery).

Outputs:
  paper_v3_ismb/figures/fig_v3_pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot v3 pipeline diagram.")
    default_out = Path(__file__).parent.parent / "paper_v3_ismb" / "figures" / "fig_v3_pipeline.png"
    p.add_argument("--out", type=str, default=str(default_out))
    return p.parse_args()


def box(ax, xy, w, h, text: str, facecolor: str = "white", fontsize: int = 9) -> Rectangle:
    """Draw a box with centered text."""
    r = FancyBboxPatch(xy, w, h, linewidth=1.2, edgecolor="black", facecolor=facecolor,
                       boxstyle="round,pad=0.02,rounding_size=0.1")
    ax.add_patch(r)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize,
            wrap=True)
    return r


def arrow(ax, p1, p2, color: str = "black") -> None:
    """Draw an arrow from p1 to p2."""
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color)
    ax.add_patch(a)


def orthogonal_arrow(ax, p1, p2, color: str = "black", mid_y: float | None = None) -> None:
    """Draw an orthogonal (right-angle) arrow from p1 to p2."""
    x1, y1 = p1
    x2, y2 = p2
    if mid_y is None:
        mid_y = (y1 + y2) / 2
    # Draw path segments
    ax.plot([x1, x1], [y1, mid_y], color=color, linewidth=1.2, solid_capstyle='round')
    ax.plot([x1, x2], [mid_y, mid_y], color=color, linewidth=1.2, solid_capstyle='round')
    ax.plot([x2, x2], [mid_y, y2 + 0.08], color=color, linewidth=1.2, solid_capstyle='round')
    # Arrowhead at end
    a = FancyArrowPatch((x2, y2 + 0.15), (x2, y2), arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color=color)
    ax.add_patch(a)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Vertical layout for better page utilization
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Colors
    input_color = "#e3f2fd"  # light blue
    process_color = "#fff3e0"  # light orange
    output_color = "#e8f5e9"  # light green
    transfer_color = "#f3e5f5"  # light purple

    # Title
    ax.text(5.0, 6.7, "SoftBlock: Transferable Soft Blocks",
            ha="center", va="center", fontsize=13, fontweight="bold")

    # Row 1: Reference PPI training (top)
    y1 = 5.5
    b1 = box(ax, (0.5, y1), 2.2, 0.8, "Reference PPI\n(STRING)", facecolor=input_color, fontsize=10)
    b2 = box(ax, (3.5, y1), 2.5, 0.8, "DGI + GCN Encoder", facecolor=process_color, fontsize=10)
    b3 = box(ax, (6.8, y1), 2.7, 0.8, "Soft Memberships\nR ∈ [0,1]^(N×K)", facecolor=output_color, fontsize=10)

    arrow(ax, (b1.get_x() + b1.get_width(), y1 + 0.4), (b2.get_x(), y1 + 0.4))
    arrow(ax, (b2.get_x() + b2.get_width(), y1 + 0.4), (b3.get_x(), y1 + 0.4))

    # Transfer arrow (vertical, prominent)
    ax.annotate("", xy=(5.0, 4.3), xytext=(5.0, 5.5),
                arrowprops=dict(arrowstyle="-|>", lw=2.5, color="#7b1fa2", mutation_scale=15))
    ax.text(5.8, 4.9, "Transfer\n(frozen)", ha="left", va="center", fontsize=10,
            color="#7b1fa2", fontweight="bold", style="italic")

    # Row 2: Target PPI + Block formation
    y2 = 3.6
    b4 = box(ax, (0.5, y2), 2.2, 0.8, "Target PPI\n(BioPlex, HuRI, ...)", facecolor=input_color, fontsize=10)
    b5 = box(ax, (3.5, y2), 2.8, 0.8, "Overlapping Coarse\nBlocks (top-k)", facecolor=transfer_color, fontsize=10)
    b6 = box(ax, (7.0, y2), 2.5, 0.8, "Block-local MCL\n(per block)", facecolor=process_color, fontsize=10)

    arrow(ax, (b4.get_x() + b4.get_width(), y2 + 0.4), (b5.get_x(), y2 + 0.4))
    arrow(ax, (b5.get_x() + b5.get_width(), y2 + 0.4), (b6.get_x(), y2 + 0.4))

    # Row 3: Union + Dedup + Rerank
    y3 = 2.0
    b7 = box(ax, (1.0, y3), 2.3, 0.8, "Union + Dedup\n(Jaccard merge)", facecolor=process_color, fontsize=10)
    b8 = box(ax, (4.0, y3), 2.5, 0.8, "Graph-only Rerank\n(weighted density)", facecolor=process_color, fontsize=10)
    b9 = box(ax, (7.2, y3), 2.3, 0.8, "Diversity-aware\nTop-N Selection", facecolor=process_color, fontsize=10)

    # Orthogonal arrow from b6 to b7 (right-angle bend)
    orthogonal_arrow(ax, (b6.get_x() + b6.get_width()/2, b6.get_y()),
                     (b7.get_x() + b7.get_width()/2, b7.get_y() + b7.get_height()), mid_y=3.0)
    arrow(ax, (b7.get_x() + b7.get_width(), y3 + 0.4), (b8.get_x(), y3 + 0.4))
    arrow(ax, (b8.get_x() + b8.get_width(), y3 + 0.4), (b9.get_x(), y3 + 0.4))

    # Row 4: Output
    y4 = 0.6
    b10 = box(ax, (3.5, y4), 3.0, 0.8, "Predicted Complexes\n(overlap-aware)", facecolor=output_color, fontsize=10)

    # Orthogonal arrow from b9 to b10 (right-angle bend)
    orthogonal_arrow(ax, (b9.get_x() + b9.get_width()/2, b9.get_y()),
                     (b10.get_x() + b10.get_width()/2, b10.get_y() + b10.get_height()), mid_y=1.5)

    # Legend (bottom right)
    legend_x = 7.5
    legend_y = 0.15
    ax.add_patch(Rectangle((legend_x, legend_y + 0.4), 0.25, 0.18, facecolor=input_color, edgecolor="black", linewidth=0.5))
    ax.text(legend_x + 0.35, legend_y + 0.49, "Input", fontsize=8, va="center")
    ax.add_patch(Rectangle((legend_x + 1.0, legend_y + 0.4), 0.25, 0.18, facecolor=process_color, edgecolor="black", linewidth=0.5))
    ax.text(legend_x + 1.35, legend_y + 0.49, "Process", fontsize=8, va="center")
    ax.add_patch(Rectangle((legend_x, legend_y), 0.25, 0.18, facecolor=transfer_color, edgecolor="black", linewidth=0.5))
    ax.text(legend_x + 0.35, legend_y + 0.09, "Transfer", fontsize=8, va="center")
    ax.add_patch(Rectangle((legend_x + 1.0, legend_y), 0.25, 0.18, facecolor=output_color, edgecolor="black", linewidth=0.5))
    ax.text(legend_x + 1.35, legend_y + 0.09, "Output", fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
