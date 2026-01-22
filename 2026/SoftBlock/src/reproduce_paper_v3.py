"""
Paper v3 reproducibility runner (reconstruction).

This runner is intentionally "boring":
  - trains reference memberships (STRING) if missing
  - runs the v3 frozen protocol per graph
  - emits CSVs matching the main Table 2 / Table 3 layouts

Examples:
  conda run -n rapids python ./src/reproduce_paper_v3.py --graphs STRING BioPlex
  conda run -n rapids python ./src/reproduce_paper_v3.py --seed 42
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

TRAIN_REF = SRC_DIR / "train_reference_memberships.py"
TRAIN_RERANK = SRC_DIR / "train_reranker.py"
RUN_FROZEN = SRC_DIR / "frozen_protocol.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce SoftBlock (reconstruction)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--graphs",
        type=str,
        nargs="+",
        default=["STRING", "BioPlex", "HuRI", "BioGRID", "ComPPI", "IntAct", "hu.MAP2"],
    )
    p.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 16])
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--ref-dir", type=str, default=None, help="Directory holding reference memberships (and cache outputs).")
    p.add_argument(
        "--rerank-mode",
        choices=["graph_only", "learned"],
        default="graph_only",
        help="graph_only matches the v3 PDF headline tables; learned uses a STRING-trained linear reranker.",
    )
    p.add_argument(
        "--rerank-score",
        type=str,
        default="weighted_density",
        help="Passed through to frozen_protocol.py when --rerank-mode graph_only (e.g., weighted_density, auto).",
    )
    p.add_argument("--rerank-model", type=str, default=None, help="Optional path to reranker JSON to use.")
    p.add_argument("--tag", type=str, default=None, help="Output tag (defaults to a derived tag).")
    p.add_argument(
        "--gold-snapshot",
        choices=["current", "paper_v3"],
        default="current",
        help="Passed through to frozen_protocol.py (controls CORUM/ComplexPortal snapshot selection).",
    )
    p.add_argument("--corum-gmt", type=str, default=None, help="Optional override path to a CORUM GMT file.")
    p.add_argument(
        "--complexportal-gmt",
        type=str,
        default=None,
        help="Optional override path to a ComplexPortal GMT file.",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _ensure(path: Path, cmd: list[str], force: bool) -> None:
    if path.exists() and not force:
        print(f"✓ exists: {path}")
        return
    _run(cmd)
    if not path.exists():
        raise RuntimeError(f"Expected output not found: {path}")


def _expected_eval_path(graph: str, seed: int, tag: str) -> Path:
    # Must match frozen_protocol.py naming.
    out_dir = RESULTS_DIR / "frozen_protocol" / graph.lower()
    return out_dir / f"{tag}_{graph.lower()}_seed{seed}.eval.csv"


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    graphs = [str(g) for g in args.graphs]
    Ks = [int(k) for k in args.Ks]
    rerank_mode = str(args.rerank_mode)
    ref_dir = Path(args.ref_dir) if args.ref_dir else None
    rerank_score = str(args.rerank_score)
    gold_snapshot = str(args.gold_snapshot)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.tag:
        tag = str(args.tag)
    else:
        tag = "v3" if rerank_mode == "graph_only" else "v3learned"
        if rerank_mode == "graph_only" and rerank_score != "weighted_density":
            tag = f"{tag}_{rerank_score}"
        if ref_dir is not None:
            tag = f"{tag}_customref"

    if not args.skip_training:
        cmd = [sys.executable, str(TRAIN_REF), "--seed", str(seed), "--Ks", *[str(k) for k in Ks]]
        if ref_dir is not None:
            cmd.extend(["--out-dir", str(ref_dir)])
        _run(cmd)
        if rerank_mode == "learned":
            cmd = [sys.executable, str(TRAIN_RERANK), "--seed", str(seed), "--Ks", *[str(k) for k in Ks]]
            if ref_dir is not None:
                cmd.extend(["--ref-dir", str(ref_dir)])
            if args.rerank_model:
                cmd.extend(["--out", str(args.rerank_model)])
            _run(cmd)

    # Run frozen protocol for each graph.
    for g in graphs:
        eval_path = _expected_eval_path(g, seed=seed, tag=tag)
        _ensure(
            eval_path,
            (
                [
                sys.executable,
                str(RUN_FROZEN),
                "--graph",
                g,
                "--seed",
                str(seed),
                "--Ks",
                *[str(k) for k in Ks],
                "--tag",
                tag,
                "--rerank-mode",
                rerank_mode,
                ]
                + (["--rerank-score", rerank_score] if rerank_mode == "graph_only" else [])
                + (["--gold-snapshot", gold_snapshot] if gold_snapshot else [])
                + (["--ref-dir", str(ref_dir)] if ref_dir is not None else [])
                + (["--rerank-model", str(args.rerank_model)] if args.rerank_model else [])
                + (["--corum-gmt", str(args.corum_gmt)] if args.corum_gmt else [])
                + (["--complexportal-gmt", str(args.complexportal_gmt)] if args.complexportal_gmt else [])
            ),
            force=args.force,
        )

    # Build Table 2 / Table 3 style summaries from eval CSVs.
    table_dir = RESULTS_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table2_rows = []
    table3_rows = []
    for g in graphs:
        eval_path = _expected_eval_path(g, seed=seed, tag=tag)
        df = pd.read_csv(eval_path)

        # Pool (Table 2).
        pool = df[df["mode"] == "pool"].copy()
        if not pool.empty:
            n_pred = int(pool["n_pred"].iloc[0])
            f1_corum = float(pool[pool["gold"] == "CORUM"]["bestmatch_f1"].iloc[0])
            f1_cp = float(pool[pool["gold"] == "ComplexPortal"]["bestmatch_f1"].iloc[0])
            mmr_corum = float(pool[pool["gold"] == "CORUM"]["mmr"].iloc[0])
            mmr_cp = float(pool[pool["gold"] == "ComplexPortal"]["mmr"].iloc[0])
            table2_rows.append(
                {
                    "Graph": g,
                    "#PC": n_pred,
                    "F1_CORUM": f1_corum,
                    "F1_ComplexPortal": f1_cp,
                    "MMR_CORUM": mmr_corum,
                    "MMR_ComplexPortal": mmr_cp,
                }
            )

        # Operating point (Table 3).
        op = df[df["mode"] == "operating_point"].copy()
        if not op.empty:
            n_pred = int(op["n_pred"].iloc[0])
            bestmatch = float(op["bestmatch_f1"].astype(float).mean())
            acc = float(op["acc"].astype(float).mean())
            mmr = float(op["mmr"].astype(float).mean())
            table3_rows.append(
                {
                    "Graph": g,
                    "#PC": n_pred,
                    "Acc_avg": acc,
                    "MMR_avg": mmr,
                    "BestMatch_avg": bestmatch,
                }
            )

    t2 = pd.DataFrame(table2_rows)
    t3 = pd.DataFrame(table3_rows)
    t2_path = table_dir / f"table2_{tag}_seed{seed}.csv"
    t3_path = table_dir / f"table3_{tag}_seed{seed}.csv"
    t2.to_csv(t2_path, index=False)
    t3.to_csv(t3_path, index=False)
    print(f"\nWrote: {t2_path}")
    print(f"Wrote: {t3_path}")

    # Back-compat table names used elsewhere in the repo.
    if tag == "v3" and rerank_mode == "graph_only" and rerank_score == "weighted_density" and ref_dir is None:
        compat_t2 = table_dir / f"table2_graph_only_seed{seed}.csv"
        compat_t3 = table_dir / f"table3_graph_only_seed{seed}.csv"
        t2.to_csv(compat_t2, index=False)
        t3.to_csv(compat_t3, index=False)
        print(f"Wrote: {compat_t2}")
        print(f"Wrote: {compat_t3}")
    if tag == "v3learned" and rerank_mode == "learned" and args.rerank_model is None and ref_dir is None:
        compat_t2 = table_dir / f"table2_learned_seed{seed}.csv"
        compat_t3 = table_dir / f"table3_learned_seed{seed}.csv"
        t2.to_csv(compat_t2, index=False)
        t3.to_csv(compat_t3, index=False)
        print(f"Wrote: {compat_t2}")
        print(f"Wrote: {compat_t3}")


if __name__ == "__main__":
    main()
