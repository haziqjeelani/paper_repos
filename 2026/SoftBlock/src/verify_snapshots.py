"""
Verify data + gold snapshots for SoftBlock reproduction.

This script exists for reviewer-defensible reproducibility:
  - checks graph node/edge counts against the PDF's Table 1 targets
  - computes evaluable gold-complex counts (|G∩V| >= min_size) under a chosen gold snapshot
  - optionally prints SHA256 hashes for the exact files used
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


PAPER_V3_DIR = Path(__file__).resolve().parent.parent
CODESEG_DIR = PAPER_V3_DIR.parent


@dataclass(frozen=True)
class Targets:
    nodes: int
    edges: int
    corum: int
    complexportal: int


TABLE1_TARGETS: dict[str, Targets] = {
    "STRING": Targets(nodes=15882, edges=236712, corum=1317, complexportal=1198),
    "BioPlex": Targets(nodes=13923, edges=118144, corum=1164, complexportal=961),
    "HuRI": Targets(nodes=8109, edges=51686, corum=537, complexportal=616),
    "BioGRID": Targets(nodes=27590, edges=1002631, corum=1355, complexportal=1157),
    "ComPPI": Targets(nodes=15277, edges=170728, corum=1310, complexportal=1158),
    "IntAct": Targets(nodes=17733, edges=527860, corum=1315, complexportal=1211),
    "hu.MAP2": Targets(nodes=7824, edges=19631, corum=848, complexportal=730),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify v3 graph + gold snapshot counts (Table 1 parity checks)")
    p.add_argument(
        "--graphs",
        type=str,
        nargs="+",
        default=list(TABLE1_TARGETS.keys()),
        help="Graphs to check (default: all Table 1 graphs).",
    )
    p.add_argument(
        "--gold-snapshot",
        choices=["current", "paper_v3"],
        default="paper_v3",
        help="Gold snapshot mode to evaluate (affects CORUM/ComplexPortal mapping).",
    )
    p.add_argument("--min-gold-size", type=int, default=3, help="Minimum gold complex size after intersection.")
    p.add_argument("--corum-gmt", type=str, default=None, help="Optional override path to a CORUM GMT file.")
    p.add_argument(
        "--complexportal-gmt",
        type=str,
        default=None,
        help="Optional override path to a ComplexPortal GMT file (gene-symbol GMT).",
    )
    p.add_argument("--hashes", action="store_true", help="Print SHA256 hashes for graph + gold files.")
    p.add_argument("--out-json", type=str, default=None, help="Optional JSON path to write all computed counts/hashes.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any graph or gold count differs from the PDF Table 1 targets.",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _read_nodes(path: Path) -> list[str]:
    nodes = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()]
    return nodes


def main() -> None:
    args = parse_args()
    graphs = [str(g) for g in args.graphs]
    gold_snapshot = str(args.gold_snapshot)
    min_gold_size = int(args.min_gold_size)

    sys.path.insert(0, str((PAPER_V3_DIR / "src").resolve()))
    from frozen_protocol import _graph_spec, _load_gold_complexes, _resolve_gold_paths  # type: ignore

    results: dict[str, dict[str, object]] = {"args": vars(args), "graphs": {}}
    any_mismatch = False

    header = [
        "Graph",
        "Nodes",
        "Edges",
        "CORUM(n>=3)",
        "CP(n>=3)",
        "TargetNodes",
        "TargetEdges",
        "TargetCORUM",
        "TargetCP",
        "OK?",
    ]
    print("\t".join(header))

    for graph in graphs:
        if graph not in TABLE1_TARGETS:
            raise ValueError(f"Unknown graph (no Table 1 targets): {graph}")

        spec = _graph_spec(graph)
        nodes_path = Path(spec.nodes_path)
        edges_path = Path(spec.edgelist_path)
        nodes = _read_nodes(nodes_path)
        n_nodes = int(len(nodes))
        n_edges = int(_count_lines(edges_path))

        node_set = set(nodes)
        corum_path, cp_path = _resolve_gold_paths(
            gold_snapshot=gold_snapshot,
            corum_gmt=str(args.corum_gmt) if args.corum_gmt else None,
            complexportal_gmt=str(args.complexportal_gmt) if args.complexportal_gmt else None,
        )
        gold = _load_gold_complexes(node_set, corum_path=corum_path, complexportal_gmt=cp_path)
        corum_n = sum(1 for genes in gold["CORUM"].values() if len(genes) >= min_gold_size)
        cp_n = sum(1 for genes in gold["ComplexPortal"].values() if len(genes) >= min_gold_size)

        tgt = TABLE1_TARGETS[graph]
        ok = (n_nodes == tgt.nodes) and (n_edges == tgt.edges) and (corum_n == tgt.corum) and (cp_n == tgt.complexportal)
        any_mismatch = any_mismatch or (not ok)

        print(
            "\t".join(
                [
                    graph,
                    str(n_nodes),
                    str(n_edges),
                    str(corum_n),
                    str(cp_n),
                    str(tgt.nodes),
                    str(tgt.edges),
                    str(tgt.corum),
                    str(tgt.complexportal),
                    "OK" if ok else "MISMATCH",
                ]
            )
        )

        rec: dict[str, object] = {
            "nodes": n_nodes,
            "edges": n_edges,
            "gold_snapshot": gold_snapshot,
            "min_gold_size": min_gold_size,
            "corum_complexes": int(corum_n),
            "complexportal_complexes": int(cp_n),
            "targets": {
                "nodes": int(tgt.nodes),
                "edges": int(tgt.edges),
                "corum": int(tgt.corum),
                "complexportal": int(tgt.complexportal),
            },
            "paths": {
                "nodes": str(nodes_path),
                "edges": str(edges_path),
                "corum_gmt": str(corum_path),
                "complexportal_gmt": str(cp_path) if cp_path is not None else "",
            },
        }
        if args.hashes:
            rec["sha256"] = {
                "nodes": _sha256(nodes_path),
                "edges": _sha256(edges_path),
                "corum_gmt": _sha256(corum_path) if corum_path.exists() else "",
                "complexportal_gmt": _sha256(cp_path) if (cp_path is not None and cp_path.exists()) else "",
            }

        results["graphs"][graph] = rec

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nWrote: {out_path}")

    if args.strict and any_mismatch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

