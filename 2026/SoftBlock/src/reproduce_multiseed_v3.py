"""
Multi-seed reproduction helper for SoftBlock (reconstruction).

Runs `reproduce_paper_v3.py` across multiple seeds and writes mean±std summaries
for Table 2/3 style outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


PAPER_V3_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PAPER_V3_DIR / "src"
RESULTS_DIR = PAPER_V3_DIR / "results"

REPRODUCE = SRC_DIR / "reproduce_paper_v3.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper v3 reproduction across multiple seeds and summarize mean±std")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--graphs",
        type=str,
        nargs="+",
        default=["STRING", "BioPlex", "HuRI", "BioGRID", "ComPPI", "IntAct", "hu.MAP2"],
    )
    p.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 16])
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--ref-dir", type=str, default=None)
    p.add_argument(
        "--rerank-mode",
        choices=["graph_only", "learned"],
        default="graph_only",
        help="graph_only matches the v3 PDF headline tables; learned uses a STRING-trained reranker.",
    )
    p.add_argument("--rerank-score", type=str, default="weighted_density")
    p.add_argument(
        "--gold-snapshot",
        choices=["current", "paper_v3"],
        default="current",
    )
    p.add_argument("--corum-gmt", type=str, default=None)
    p.add_argument("--complexportal-gmt", type=str, default=None)
    p.add_argument("--tag", type=str, default=None, help="Stable tag shared across seeds (recommended).")
    p.add_argument("--out-tag", type=str, default=None, help="Tag used for summary filenames (defaults to --tag).")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _infer_tag(*, rerank_mode: str, rerank_score: str, ref_dir: str | None) -> str:
    tag = "v3" if rerank_mode == "graph_only" else "v3learned"
    if rerank_mode == "graph_only" and rerank_score != "weighted_density":
        tag = f"{tag}_{rerank_score}"
    if ref_dir is not None:
        tag = f"{tag}_customref"
    return tag


def _summarize(path_by_seed: dict[int, Path], *, key_cols: list[str]) -> pd.DataFrame:
    frames = []
    for seed, path in sorted(path_by_seed.items()):
        df = pd.read_csv(path)
        df["seed"] = int(seed)
        frames.append(df)
    all_df = pd.concat(frames, axis=0, ignore_index=True)
    metric_cols = [c for c in all_df.columns if c not in set(key_cols + ["seed"])]
    for c in metric_cols:
        all_df[c] = pd.to_numeric(all_df[c], errors="coerce")
    grouped = all_df.groupby(key_cols, dropna=False)[metric_cols].agg(["mean", "std"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds]
    graphs = [str(g) for g in args.graphs]
    Ks = [int(k) for k in args.Ks]
    rerank_mode = str(args.rerank_mode)
    rerank_score = str(args.rerank_score)
    gold_snapshot = str(args.gold_snapshot)
    ref_dir = str(args.ref_dir) if args.ref_dir else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tag = str(args.tag) if args.tag else _infer_tag(rerank_mode=rerank_mode, rerank_score=rerank_score, ref_dir=ref_dir)
    out_tag = str(args.out_tag) if args.out_tag else tag

    table_dir = RESULTS_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table2_paths: dict[int, Path] = {}
    table3_paths: dict[int, Path] = {}

    for seed in seeds:
        cmd = [
            sys.executable,
            str(REPRODUCE),
            "--seed",
            str(int(seed)),
            "--graphs",
            *graphs,
            "--Ks",
            *[str(k) for k in Ks],
            "--rerank-mode",
            rerank_mode,
            "--gold-snapshot",
            gold_snapshot,
            "--tag",
            tag,
        ]
        if rerank_mode == "graph_only":
            cmd.extend(["--rerank-score", rerank_score])
        if ref_dir is not None:
            cmd.extend(["--ref-dir", ref_dir])
        if args.corum_gmt:
            cmd.extend(["--corum-gmt", str(args.corum_gmt)])
        if args.complexportal_gmt:
            cmd.extend(["--complexportal-gmt", str(args.complexportal_gmt)])
        if args.skip_training:
            cmd.append("--skip-training")
        if args.force:
            cmd.append("--force")

        _run(cmd)

        table2_paths[int(seed)] = table_dir / f"table2_{tag}_seed{int(seed)}.csv"
        table3_paths[int(seed)] = table_dir / f"table3_{tag}_seed{int(seed)}.csv"

    t2 = _summarize(table2_paths, key_cols=["Graph"])
    t3 = _summarize(table3_paths, key_cols=["Graph"])

    seeds_str = "_".join(str(s) for s in seeds)
    t2_path = table_dir / f"table2_{out_tag}_seeds{seeds_str}_meanstd.csv"
    t3_path = table_dir / f"table3_{out_tag}_seeds{seeds_str}_meanstd.csv"
    t2.to_csv(t2_path, index=False)
    t3.to_csv(t3_path, index=False)
    print(f"\nWrote: {t2_path}")
    print(f"Wrote: {t3_path}")


if __name__ == "__main__":
    main()

