"""
Generate comparison bar chart for SoftBlock cross-PPI performance.

Uses multi-seed mean±std data for error bars. Creates a taller vertical layout.

Outputs:
  paper_v3_ismb/figures/fig_v3_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot v3 cross-PPI comparison bars.")
    default_out = Path(__file__).parent.parent / "paper_v3_ismb" / "figures" / "fig_v3_comparison.png"
    default_csv = Path(__file__).parent.parent / "results" / "tables" / "table2_v3_paperbench_seeds42_43_44_meanstd.csv"
    p.add_argument("--out", type=str, default=str(default_out))
    p.add_argument("--csv", type=str, default=str(default_csv))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load mean±std data
    df = pd.read_csv(args.csv)

    # Prepare data
    graphs = df["Graph"].tolist()
    f1_corum_mean = df["F1_CORUM_mean"].tolist()
    f1_corum_std = df["F1_CORUM_std"].tolist()
    f1_cp_mean = df["F1_ComplexPortal_mean"].tolist()
    f1_cp_std = df["F1_ComplexPortal_std"].tolist()

    # Create TALLER figure with stacked subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    x = np.arange(len(graphs))
    width = 0.35

    # Colors
    corum_color = "#1976d2"  # blue
    cp_color = "#43a047"  # green

    # Top plot: Best-match F1 with error bars
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, f1_corum_mean, width, yerr=f1_corum_std,
                    label="CORUM", color=corum_color, alpha=0.8, capsize=3)
    bars2 = ax1.bar(x + width/2, f1_cp_mean, width, yerr=f1_cp_std,
                    label="Complex Portal", color=cp_color, alpha=0.8, capsize=3)

    ax1.set_ylabel("Best-match F1", fontsize=11)
    ax1.set_title("(a) Uncapped Pool Performance (mean ± std, n=3 seeds)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(graphs, rotation=30, ha="right", fontsize=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.set_ylim(0, 0.7)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom plot: Operating point data with error bars
    operating_point_csv = Path(args.csv).parent / "table3_v3_paperbench_seeds42_43_44_meanstd.csv"
    if operating_point_csv.exists():
        df_op = pd.read_csv(operating_point_csv)
        acc_mean = df_op["Acc_avg_mean"].tolist()
        acc_std = df_op["Acc_avg_std"].tolist()
        mmr_mean = df_op["MMR_avg_mean"].tolist()
        mmr_std = df_op["MMR_avg_std"].tolist()

        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, acc_mean, width, yerr=acc_std,
                        label="OS Accuracy", color="#ff7043", alpha=0.8, capsize=3)
        bars4 = ax2.bar(x + width/2, mmr_mean, width, yerr=mmr_std,
                        label="Greedy MMR", color="#7e57c2", alpha=0.8, capsize=3)

        ax2.set_ylabel("Metric Value", fontsize=11)
        ax2.set_title("(b) Fixed Operating Point (N=2000, diversity-aware)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(graphs, rotation=30, ha="right", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.set_ylim(0, 0.3)
        ax2.grid(axis="y", alpha=0.3)
    else:
        # Fallback: show MMR from table2
        mmr_corum_mean = df["MMR_CORUM_mean"].tolist()
        mmr_corum_std = df["MMR_CORUM_std"].tolist()
        mmr_cp_mean = df["MMR_ComplexPortal_mean"].tolist()
        mmr_cp_std = df["MMR_ComplexPortal_std"].tolist()

        ax2 = axes[1]
        bars3 = ax2.bar(x - width/2, mmr_corum_mean, width, yerr=mmr_corum_std,
                        label="CORUM", color=corum_color, alpha=0.8, capsize=3)
        bars4 = ax2.bar(x + width/2, mmr_cp_mean, width, yerr=mmr_cp_std,
                        label="Complex Portal", color=cp_color, alpha=0.8, capsize=3)

        ax2.set_ylabel("Greedy MMR", fontsize=11)
        ax2.set_title("(b) Uncapped Pool MMR (mean ± std)", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(graphs, rotation=30, ha="right", fontsize=10)
        ax2.legend(loc="upper right", fontsize=10)
        ax2.set_ylim(0, 0.4)
        ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("SoftBlock: Cross-PPI Performance under Frozen Protocol",
                 fontsize=13, fontweight="bold", y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
