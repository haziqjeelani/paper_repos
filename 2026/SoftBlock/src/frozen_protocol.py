"""
SoftBlock — frozen protocol runner (reconstruction).

Implements the v3 pipeline described in `./paper_v3.pdf`:
  - train/reference soft memberships R on STRING (handled by a separate script)
  - transfer R to a target PPI by node overlap (V ∩ Vref)
  - top-k block assignment (k=4, pmin=0) with membership calibration a=1.5
  - local solving inside blocks (MCL inflation=4.0)
  - multi-K union (K ∈ {6, 8, 16})
  - (optional) hybrid_auto augmentation via "link communities" candidates
  - global dedup by Jaccard (0.85)
  - graph-only reranking by (weighted) density
  - operating-point control: keep top-N with NMS overlap suppression (max Jaccard < 0.5)

This file is intentionally self-contained for reproducibility.
Run with the RAPIDS env:
  conda run -n rapids python ./src/frozen_protocol.py --graph BioPlex --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PAPER_V3_DIR = Path(__file__).resolve().parent.parent
CODESEG_DIR = PAPER_V3_DIR.parent
sys.path.insert(0, str(CODESEG_DIR / "src"))

from config import NETWORK_DIR, DATA_DIR  # noqa: E402
from complex_eval import load_corum_complexes  # noqa: E402
from complexportal_eval import (  # noqa: E402
    build_uniprot_to_symbol_from_goa,
    complexes_to_gene_sets,
    load_complexportal_uniprots,
    COMPLEXPORTAL_TSV,
    GOA_HUMAN_GAF_GZ,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _seed_everything(seed: int) -> None:
    np.random.seed(int(seed))
    try:
        import random

        random.seed(int(seed))
    except Exception:
        pass
    try:
        import cupy as cp

        cp.random.seed(int(seed))
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SoftBlock frozen protocol on a graph")
    p.add_argument(
        "--graph",
        type=str,
        required=True,
        choices=["STRING", "BioPlex", "HuRI", "BioGRID", "IntAct", "ComPPI", "hu.MAP2"],
    )
    p.add_argument("--seed", type=int, default=42)

    # Frozen protocol (defaults from the v3 PDF)
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--pmin", type=float, default=0.0)
    p.add_argument("--membership-power", type=float, default=1.5)
    p.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 16])
    p.add_argument("--mcl-inflation", type=float, default=4.0)
    p.add_argument("--global-dedup-jaccard", type=float, default=0.85)
    p.add_argument(
        "--global-dedup-mode",
        choices=["nms", "union"],
        default="union",
        help="Global deduplication mode at Jaccard threshold (paper describes a Jaccard-merge/union).",
    )
    p.add_argument(
        "--rerank-score",
        choices=[
            "weighted_density",
            "weighted_density_n2",
            "cohesiveness",
            "density_x_cohesiveness",
            "density_n2_x_cohesiveness",
            "auto",
        ],
        default="weighted_density",
        help=(
            "Graph-only reranking score used before the pool cap and operating-point selection. "
            "`weighted_density` is edge-density Win(C)/(|C| choose 2); "
            "`weighted_density_n2` is Win(C)/|C|^2 (as written in the v3 PDF text); "
            "`density_n2_x_cohesiveness` multiplies weighted_density_n2 by cohesiveness; "
            "`auto` uses weighted_density on weighted graphs and density_x_cohesiveness on unweighted graphs."
        ),
    )
    p.add_argument(
        "--rerank-mode",
        choices=["graph_only", "learned", "oracle"],
        default="graph_only",
        help="Use graph-only scoring (paper default), a STRING-trained linear reranker (optional), or an oracle ranker.",
    )
    p.add_argument(
        "--oracle-target",
        choices=["avg", "CORUM", "ComplexPortal"],
        default="avg",
        help="When --rerank-mode oracle: which gold(s) to use for oracle scoring.",
    )
    p.add_argument(
        "--rerank-model",
        type=str,
        default=None,
        help=(
            "Path to a learned reranker model JSON (see train_reranker.py). "
            "If omitted, uses paper_v3/results/cache/reranker_string_seed{seed}.json."
        ),
    )
    p.add_argument("--pool-cap", type=int, default=8000)
    p.add_argument("--op-cap", type=int, default=2000)
    p.add_argument("--op-max-jaccard", type=float, default=0.5)
    p.add_argument(
        "--op-selection",
        choices=["nms", "rerank_only"],
        default="nms",
        help="Operating point selector: paper default is nms; rerank_only reproduces Table-16 'no NMS' ablation.",
    )

    # MCL specifics (kept close to markov_clustering defaults)
    p.add_argument(
        "--mcl-backend",
        choices=["gpu", "cpu", "mcl"],
        default="gpu",
        help="MCL backend: gpu uses a CuPyX sparse implementation; cpu uses markov_clustering; mcl uses the MCL binary.",
    )
    p.add_argument("--mcl-max-iter", type=int, default=100)
    p.add_argument("--mcl-prune-threshold", type=float, default=1e-3)
    p.add_argument("--mcl-convergence-tol", type=float, default=1e-5)

    # Hybrid policy
    p.add_argument(
        "--candidate-source",
        choices=["hybrid_auto", "codeseg_only", "linkcomm_only"],
        default="hybrid_auto",
        help="Which candidate sources to use before reranking/selection.",
    )
    p.add_argument(
        "--hybrid-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="auto_confidence_weighted gate from the v3 PDF, or force on/off",
    )
    p.add_argument(
        "--linkcomm-method",
        choices=["plain_linegraph", "jaccard_linegraph", "ahn_single_linkage"],
        default="jaccard_linegraph",
        help="Reconstructed link-communities implementation family.",
    )
    p.add_argument(
        "--linkcomm-resolution",
        type=float,
        default=5.0,
        help="Line-graph Louvain resolution (tuned on STRING).",
    )
    p.add_argument(
        "--linkcomm-jaccard-threshold",
        type=float,
        default=0.2,
        help="When using jaccard_linegraph: keep edge-pairs with Jaccard >= this threshold.",
    )
    p.add_argument(
        "--linkcomm-batch-pairs",
        type=int,
        default=2_000_000,
        help="When using jaccard_linegraph: max neighbor-pairs per GPU Jaccard batch.",
    )
    p.add_argument(
        "--linkcomm-split-components",
        action="store_true",
        help="Split each edge-community into connected components (increases candidate count).",
    )
    p.add_argument(
        "--linkcomm-inclusive-neighborhoods",
        action="store_true",
        help=(
            "Use Ahn et al. 'inclusive' neighbor sets for edge-similarity Jaccard by adding self-loops "
            "before computing cugraph.jaccard (affects jaccard_linegraph/ahn_single_linkage)."
        ),
    )
    p.add_argument(
        "--linkcomm-min-size-unweighted",
        type=int,
        default=4,
        help="Frozen filter used by hybrid_auto on unweighted graphs (v3 PDF Table 17).",
    )
    p.add_argument("--min-cluster-size", type=int, default=3)

    # Gold standards
    p.add_argument(
        "--gold-snapshot",
        choices=["current", "paper_v3"],
        default="current",
        help=(
            "Which gold-standard snapshot to use for evaluation. "
            "`current` uses codeseg/data/pathways (CORUM GMT + ComplexPortal TSV+GOA mapping); "
            "`paper_v3` uses pinned GMT files under ./data/gold/."
        ),
    )
    p.add_argument("--corum-gmt", type=str, default=None, help="Optional override path to a CORUM GMT file.")
    p.add_argument(
        "--complexportal-gmt",
        type=str,
        default=None,
        help="Optional override path to a ComplexPortal GMT file (gene symbols).",
    )

    # Paths
    p.add_argument("--ref-dir", type=str, default=None, help="Directory holding reference soft memberships")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory under paper_v3/results/")
    p.add_argument("--tag", type=str, default=None, help="Optional tag added to output filenames")

    # Diagnostics
    p.add_argument("--write-intermediate", action="store_true")
    return p.parse_args()


@dataclass(frozen=True)
class GraphSpec:
    name: str
    nodes_path: Path
    edgelist_path: Path
    expected_weighted: bool


def _graph_spec(graph: str) -> GraphSpec:
    graph = str(graph)
    if graph == "STRING":
        return GraphSpec(
            name="STRING",
            nodes_path=NETWORK_DIR / "ppi_network_nodes.txt",
            edgelist_path=NETWORK_DIR / "ppi_network_weighted.edgelist",
            expected_weighted=True,
        )
    if graph == "BioPlex":
        return GraphSpec(
            name="BioPlex",
            nodes_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "bioplex" / "bioplex_nodes.txt",
            edgelist_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "bioplex" / "bioplex.edgelist",
            expected_weighted=False,
        )
    if graph == "HuRI":
        # v3 uses HuRI restricted to the largest connected component (Table 19).
        v3_nodes = PAPER_V3_DIR / "data" / "processed" / "huri" / "huri_nodes.txt"
        v3_edges = PAPER_V3_DIR / "data" / "processed" / "huri" / "huri.edgelist"
        if v3_nodes.exists() and v3_edges.exists():
            return GraphSpec(
                name="HuRI",
                nodes_path=v3_nodes,
                edgelist_path=v3_edges,
                expected_weighted=False,
            )
        return GraphSpec(
            name="HuRI",
            nodes_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "huri" / "huri_nodes.txt",
            edgelist_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "huri" / "huri.edgelist",
            expected_weighted=False,
        )
    if graph == "BioGRID":
        # v3 uses a weighted BioGRID (weights not in [0,1]) to disable linkcomm in hybrid_auto (Table 17).
        # If a v3-specific weighted file exists, prefer it; otherwise fall back to the v2 unweighted edgelist.
        v3_weighted = PAPER_V3_DIR / "data" / "processed" / "biogrid" / "biogrid_weighted.edgelist"
        v3_nodes = PAPER_V3_DIR / "data" / "processed" / "biogrid" / "biogrid_nodes.txt"
        if v3_weighted.exists() and v3_nodes.exists():
            return GraphSpec(
                name="BioGRID",
                nodes_path=v3_nodes,
                edgelist_path=v3_weighted,
                expected_weighted=True,
            )
        return GraphSpec(
            name="BioGRID",
            nodes_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "biogrid" / "biogrid_nodes.txt",
            edgelist_path=CODESEG_DIR / "paper_v2" / "data" / "processed" / "biogrid" / "biogrid.edgelist",
            expected_weighted=False,
        )
    if graph == "IntAct":
        return GraphSpec(
            name="IntAct",
            nodes_path=PAPER_V3_DIR / "data" / "processed" / "intact" / "intact_nodes.txt",
            edgelist_path=PAPER_V3_DIR / "data" / "processed" / "intact" / "intact_weighted.edgelist",
            expected_weighted=True,
        )
    if graph == "ComPPI":
        return GraphSpec(
            name="ComPPI",
            nodes_path=PAPER_V3_DIR / "data" / "processed" / "comppi" / "comppi_nodes.txt",
            edgelist_path=PAPER_V3_DIR / "data" / "processed" / "comppi" / "comppi_weighted.edgelist",
            expected_weighted=True,
        )
    if graph == "hu.MAP2":
        return GraphSpec(
            name="hu.MAP2",
            nodes_path=PAPER_V3_DIR / "data" / "processed" / "humap2" / "humap2_nodes.txt",
            edgelist_path=PAPER_V3_DIR / "data" / "processed" / "humap2" / "humap2_weighted.edgelist",
            expected_weighted=True,
        )
    raise ValueError(f"Unknown graph: {graph}")


@dataclass(frozen=True)
class LoadedGraph:
    name: str
    nodes: list[str]
    node_to_idx: dict[str, int]
    # Symmetric adjacency, float32 CSR (CPU, scipy)
    adj: object
    weighted: bool
    w_min: float
    w_max: float
    weights_in_01: bool


def _read_nodes(path: Path) -> list[str]:
    with open(path) as f:
        nodes = [ln.strip() for ln in f if ln.strip()]
    if not nodes:
        raise ValueError(f"Empty nodes file: {path}")
    return nodes


def _load_graph(spec: GraphSpec) -> LoadedGraph:
    from scipy import sparse

    if not spec.nodes_path.exists():
        raise FileNotFoundError(f"Missing nodes file: {spec.nodes_path}")
    if not spec.edgelist_path.exists():
        raise FileNotFoundError(f"Missing edgelist: {spec.edgelist_path}")

    nodes = _read_nodes(spec.nodes_path)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    src: list[int] = []
    dst: list[int] = []
    wts: list[float] = []
    saw_weight_col = False

    with open(spec.edgelist_path) as f:
        for ln in f:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            u = parts[0].strip()
            v = parts[1].strip()
            if not u or not v or u == v:
                continue
            if u not in node_to_idx or v not in node_to_idx:
                continue
            w = 1.0
            if len(parts) >= 3:
                saw_weight_col = True
                try:
                    w = float(parts[2])
                except Exception:
                    w = 1.0
            src.append(int(node_to_idx[u]))
            dst.append(int(node_to_idx[v]))
            wts.append(float(w))

    if not src:
        raise ValueError(f"No edges loaded from {spec.edgelist_path}")

    src_arr = np.asarray(src, dtype=np.int32)
    dst_arr = np.asarray(dst, dtype=np.int32)
    w_arr = np.asarray(wts, dtype=np.float32)

    # Determine weight semantics.
    weighted = bool(saw_weight_col or spec.expected_weighted)
    w_min = float(np.min(w_arr)) if w_arr.size else 0.0
    w_max = float(np.max(w_arr)) if w_arr.size else 0.0
    weights_in_01 = bool(weighted and w_min >= 0.0 and w_max <= 1.0 + 1e-12)

    # Build symmetric adjacency (CSR).
    n = len(nodes)
    src2 = np.concatenate([src_arr, dst_arr], axis=0)
    dst2 = np.concatenate([dst_arr, src_arr], axis=0)
    w2 = np.concatenate([w_arr, w_arr], axis=0).astype(np.float32, copy=False)

    adj = sparse.coo_matrix((w2, (src2, dst2)), shape=(n, n), dtype=np.float32).tocsr()
    adj.sum_duplicates()

    return LoadedGraph(
        name=spec.name,
        nodes=nodes,
        node_to_idx=node_to_idx,
        adj=adj,
        weighted=weighted,
        w_min=w_min,
        w_max=w_max,
        weights_in_01=weights_in_01,
    )


def _load_reference_membership(ref_dir: Path, seed: int, K: int) -> tuple[list[str], np.ndarray]:
    path = ref_dir / f"codeseg_soft_membership_string_seed{int(seed)}_K{int(K)}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing reference membership: {path}\n"
            f"Generate it via: conda run -n rapids python ./src/train_reference_memberships.py --seed {seed}"
        )
    data = np.load(path, allow_pickle=True)
    if "nodes" not in data or "P" not in data:
        raise ValueError(f"Expected keys {{'nodes','P'}} in {path}")
    nodes = [str(x) for x in data["nodes"].tolist()]
    P = np.asarray(data["P"], dtype=np.float32)
    if P.shape[0] != len(nodes) or P.shape[1] != int(K):
        raise ValueError(f"Bad membership shape in {path}: P={P.shape}, nodes={len(nodes)}, K={K}")
    return nodes, P


def _calibrate_membership(P: np.ndarray, power: float) -> np.ndarray:
    power = float(power)
    if not np.isfinite(power) or power <= 0:
        raise ValueError("--membership-power must be > 0")
    if abs(power - 1.0) < 1e-12:
        return P
    Pp = np.power(P, power, dtype=np.float32)
    denom = Pp.sum(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return (Pp / denom).astype(np.float32, copy=False)


def _topk_blocks(
    target_nodes: list[str],
    ref_nodes: list[str],
    P_ref: np.ndarray,
    topk: int,
    pmin: float,
    membership_power: float,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Return:
      - blocks: list length K, each an int32 array of target node indices assigned to that block
      - overlap_mask: boolean mask (len(target_nodes)) where True for nodes in V ∩ Vref
    """
    ref_index = {n: i for i, n in enumerate(ref_nodes)}
    overlap_idx: list[int] = []
    ref_rows: list[int] = []
    for i, n in enumerate(target_nodes):
        j = ref_index.get(n)
        if j is None:
            continue
        overlap_idx.append(i)
        ref_rows.append(int(j))

    overlap_mask = np.zeros((len(target_nodes),), dtype=bool)
    if overlap_idx:
        overlap_mask[np.asarray(overlap_idx, dtype=np.int64)] = True

    if not overlap_idx:
        raise ValueError("No node overlap between target graph and reference memberships")

    P = _calibrate_membership(P_ref[np.asarray(ref_rows, dtype=np.int64)], power=membership_power)
    N, K = P.shape
    topk = int(topk)
    if topk <= 0 or topk > K:
        raise ValueError(f"--topk must be in [1, K]; got topk={topk}, K={K}")
    pmin = float(pmin)

    # Assign each overlapped target node to its top-k membership columns.
    # Use argpartition for speed; K is small.
    top_idx = np.argpartition(P, kth=K - topk, axis=1)[:, -topk:]  # (N, topk), unsorted
    blocks: list[list[int]] = [[] for _ in range(K)]
    for row_i in range(N):
        tgt_i = int(overlap_idx[row_i])
        cols = top_idx[row_i]
        for c in cols.tolist():
            if pmin > 0.0 and float(P[row_i, int(c)]) < pmin:
                continue
            blocks[int(c)].append(tgt_i)

    blocks_arr = [np.asarray(sorted(b), dtype=np.int32) for b in blocks if b]
    # Some empty blocks can occur if pmin > 0; keep K-length structure for determinism.
    if len(blocks_arr) != K:
        # Rebuild with empty arrays kept.
        blocks_full: list[np.ndarray] = []
        for b in blocks:
            blocks_full.append(np.asarray(sorted(b), dtype=np.int32))
        blocks_arr = blocks_full

    return blocks_arr, overlap_mask


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def _merge_union_dedup(clusters: list[set[str]], thr: float, min_size: int) -> list[set[str]]:
    thr = float(thr)
    min_size = int(min_size)
    # Remove exact duplicates first (cheap).
    uniq = {tuple(sorted(c)) for c in clusters if len(c) >= min_size}
    clusters2 = [set(t) for t in sorted(uniq)]

    merged: list[set[str]] = []
    for c in clusters2:
        best_i = -1
        best_j = 0.0
        for i, ex in enumerate(merged):
            j = _jaccard(c, ex)
            if j > best_j:
                best_j = j
                best_i = i
        if best_j >= thr and best_i >= 0:
            merged[best_i] = merged[best_i] | c
        else:
            merged.append(c)
    return [c for c in merged if len(c) >= min_size]


def _dedup_nms_fast(clusters: list[set[str]], thr: float, min_size: int) -> list[set[str]]:
    """
    Near-duplicate removal by Jaccard threshold (NMS-style): keep one representative cluster.

    Uses an inverted index to avoid O(n^2) scans. Deterministic ordering: larger clusters first,
    then lexicographic member order.
    """
    thr = float(thr)
    min_size = int(min_size)

    uniq = {tuple(sorted(c)) for c in clusters if len(c) >= min_size}
    tuples = sorted(uniq, key=lambda t: (-len(t), t))
    clusters2 = [set(t) for t in tuples]

    kept: list[set[str]] = []
    inv: dict[str, list[int]] = {}

    for c in clusters2:
        cand_counts: dict[int, int] = {}
        for x in c:
            for ki in inv.get(x, ()):
                cand_counts[ki] = int(cand_counts.get(ki, 0)) + 1

        is_dup = False
        m = int(len(c))
        for ki, inter in cand_counts.items():
            k = kept[int(ki)]
            n = int(len(k))
            max_size = m if m >= n else n
            min_size_pair = n if m >= n else m
            # Necessary conditions for Jaccard >= thr.
            if float(min_size_pair) < thr * float(max_size):
                continue
            if float(inter) < thr * float(max_size):
                continue
            union = m + n - int(inter)
            if union <= 0:
                continue
            if (float(inter) / float(union)) >= thr:
                is_dup = True
                break

        if is_dup:
            continue

        idx = int(len(kept))
        kept.append(c)
        for x in c:
            inv.setdefault(x, []).append(idx)

    return kept


def _score_weighted_density(adj_csr, node_to_idx: dict[str, int], cluster: set[str]) -> float:
    from scipy import sparse

    if len(cluster) < 2:
        return 0.0
    idx = [node_to_idx[n] for n in cluster if n in node_to_idx]
    if len(idx) < 2:
        return 0.0
    sub = adj_csr[idx, :][:, idx]
    if not sparse.isspmatrix(sub):
        sub = sparse.csr_matrix(sub)
    win = float(sub.sum()) / 2.0
    n = float(len(idx))
    denom = n * (n - 1.0) / 2.0
    if denom <= 0:
        return 0.0
    return float(win / denom)


def _score_weighted_density_n2(adj_csr, node_to_idx: dict[str, int], cluster: set[str]) -> float:
    from scipy import sparse

    if len(cluster) < 2:
        return 0.0
    idx = [node_to_idx[n] for n in cluster if n in node_to_idx]
    if len(idx) < 2:
        return 0.0
    sub = adj_csr[idx, :][:, idx]
    if not sparse.isspmatrix(sub):
        sub = sparse.csr_matrix(sub)
    win = float(sub.sum()) / 2.0
    n = float(len(idx))
    denom = n * n
    if denom <= 0:
        return 0.0
    return float(win / denom)


def _score_cohesiveness(adj_csr, node_to_idx: dict[str, int], cluster: set[str]) -> float:
    """
    Boundary-aware cohesiveness from the v3 PDF:
      2Win(C) / (2Win(C) + Wcut(C))
    For a symmetric adjacency, this equals (sum of internal weights *2) / (sum of degrees within C).
    """
    from scipy import sparse

    if len(cluster) < 2:
        return 0.0
    idx = [node_to_idx[n] for n in cluster if n in node_to_idx]
    if len(idx) < 2:
        return 0.0
    sub = adj_csr[idx, :][:, idx]
    if not sparse.isspmatrix(sub):
        sub = sparse.csr_matrix(sub)
    two_win = float(sub.sum())
    if two_win <= 0.0:
        return 0.0
    deg_sum = float(adj_csr[idx, :].sum())
    if deg_sum <= 0.0:
        return 0.0
    return float(two_win / deg_sum)


def _rerank_by_density(
    clusters: list[set[str]],
    adj_csr,
    node_to_idx: dict[str, int],
    score: str,
) -> list[tuple[float, set[str]]]:
    scored: list[tuple[float, set[str]]] = []
    for c in clusters:
        if score == "weighted_density_n2":
            s = _score_weighted_density_n2(adj_csr, node_to_idx=node_to_idx, cluster=c)
        elif score == "cohesiveness":
            s = _score_cohesiveness(adj_csr, node_to_idx=node_to_idx, cluster=c)
        elif score == "density_x_cohesiveness":
            d = _score_weighted_density(adj_csr, node_to_idx=node_to_idx, cluster=c)
            coh = _score_cohesiveness(adj_csr, node_to_idx=node_to_idx, cluster=c)
            s = float(d * coh)
        elif score == "density_n2_x_cohesiveness":
            d = _score_weighted_density_n2(adj_csr, node_to_idx=node_to_idx, cluster=c)
            coh = _score_cohesiveness(adj_csr, node_to_idx=node_to_idx, cluster=c)
            s = float(d * coh)
        else:
            s = _score_weighted_density(adj_csr, node_to_idx=node_to_idx, cluster=c)
        scored.append((float(s), c))
    # Sort: higher density first, then larger, then lexicographic for determinism.
    scored.sort(key=lambda x: (-x[0], -len(x[1]), sorted(x[1])[:5]))
    return scored


def _rerank_by_learned(
    clusters: list[set[str]],
    *,
    graph: LoadedGraph,
    model_path: Path,
) -> list[tuple[float, set[str]]]:
    from reranker import extract_features, load_reranker

    model = load_reranker(model_path)
    degrees = np.asarray(graph.adj.sum(axis=1)).ravel().astype(np.float32, copy=False)
    X, _names = extract_features(
        clusters,
        adj_csr=graph.adj,
        node_to_idx=graph.node_to_idx,
        degrees=degrees,
        feature_names=model.feature_names,
    )
    scores = model.score(X)
    scored = [(float(scores[i]), clusters[i]) for i in range(len(clusters))]
    scored.sort(key=lambda x: (-x[0], -len(x[1]), sorted(x[1])[:5]))
    return scored


def _rerank_by_oracle(
    clusters: list[set[str]],
    *,
    gold: dict[str, dict[str, set[str]]],
    target: str,
) -> list[tuple[float, set[str]]]:
    """
    Oracle reranking for debugging:
    score(cluster) = max OS(cluster, gold_complex) for the chosen gold standard, or the average
    across CORUM/ComplexPortal.
    """
    from reranker import max_overlap_score_per_cluster

    target = str(target)
    if target == "CORUM":
        scores = max_overlap_score_per_cluster(gold["CORUM"], clusters)
    elif target == "ComplexPortal":
        scores = max_overlap_score_per_cluster(gold["ComplexPortal"], clusters)
    else:
        s1 = max_overlap_score_per_cluster(gold["CORUM"], clusters)
        s2 = max_overlap_score_per_cluster(gold["ComplexPortal"], clusters)
        scores = 0.5 * (s1 + s2)

    scored = [(float(scores[i]), clusters[i]) for i in range(len(clusters))]
    scored.sort(key=lambda x: (-x[0], -len(x[1]), sorted(x[1])[:5]))
    return scored


def _select_top_nms(
    scored: list[tuple[float, set[str]]],
    cap: int,
    max_jaccard: float,
) -> list[set[str]]:
    cap = int(cap)
    max_jaccard = float(max_jaccard)
    kept: list[set[str]] = []
    for _score, c in scored:
        if cap > 0 and len(kept) >= cap:
            break
        if not kept:
            kept.append(c)
            continue
        ok = True
        for k in kept:
            if _jaccard(c, k) >= max_jaccard:
                ok = False
                break
        if ok:
            kept.append(c)
    return kept


def _save_clusters_tsv(clusters: list[set[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, c in enumerate(clusters):
            members = "\t".join(sorted(c))
            f.write(f"cluster_{i}\t{len(c)}\t{members}\n")


def _resolve_gold_paths(
    gold_snapshot: str,
    corum_gmt: str | None,
    complexportal_gmt: str | None,
) -> tuple[Path, Path | None]:
    snapshot = str(gold_snapshot)

    if corum_gmt:
        corum_path = Path(corum_gmt)
    elif snapshot == "paper_v3":
        corum_path = PAPER_V3_DIR / "data" / "gold" / "corum.gmt"
    else:
        corum_path = DATA_DIR / "pathways" / "corum" / "corum_complexes.gmt"

    if complexportal_gmt:
        cp_path: Path | None = Path(complexportal_gmt)
    elif snapshot == "paper_v3":
        cp_path = PAPER_V3_DIR / "data" / "gold" / "complexportal.gmt"
    else:
        cp_path = None

    return corum_path, cp_path


def _load_gold_complexes(
    nodes_universe: set[str],
    *,
    corum_path: Path,
    complexportal_gmt: Path | None,
) -> dict[str, dict[str, set[str]]]:
    gold: dict[str, dict[str, set[str]]] = {}

    if not corum_path.exists():
        raise FileNotFoundError(
            f"Missing CORUM GMT: {corum_path}\n"
            "If you intended to reproduce the PDF snapshot, generate it via:\n"
            "  conda run -n rapids python ./src/fetch_gold_standards.py"
        )
    corum = load_corum_complexes(corum_path)
    gold["CORUM"] = {cid: (genes & nodes_universe) for cid, genes in corum.items()}

    if complexportal_gmt is not None:
        if not complexportal_gmt.exists():
            raise FileNotFoundError(
                f"Missing ComplexPortal GMT: {complexportal_gmt}\n"
                "If you intended to reproduce the PDF snapshot, generate it via:\n"
                "  conda run -n rapids python ./src/fetch_gold_standards.py"
            )
        cp = load_corum_complexes(complexportal_gmt)
        gold["ComplexPortal"] = {cid: (genes & nodes_universe) for cid, genes in cp.items()}
    else:
        if not COMPLEXPORTAL_TSV.exists() or not GOA_HUMAN_GAF_GZ.exists():
            raise FileNotFoundError(f"Missing ComplexPortal files: {COMPLEXPORTAL_TSV} or {GOA_HUMAN_GAF_GZ}")
        cp_uniprots = load_complexportal_uniprots(COMPLEXPORTAL_TSV)
        wanted = {u for _, (_name, unis) in cp_uniprots.items() for u in unis}
        uniprot_to_symbol = build_uniprot_to_symbol_from_goa(GOA_HUMAN_GAF_GZ, wanted=wanted)
        cp_genes, _stats = complexes_to_gene_sets(cp_uniprots, uniprot_to_symbol=uniprot_to_symbol)
        gold["ComplexPortal"] = {cid: (genes & nodes_universe) for cid, genes in cp_genes.items()}

    return gold


def _best_match_f1(
    gold_complexes: dict[str, set[str]],
    predicted: list[set[str]],
    min_complex_size: int = 3,
    min_overlap: int = 2,
) -> float:
    pred = [c for c in predicted if len(c) >= 2]
    inv: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(pred):
        for g in c:
            inv[g].append(j)

    f1s: list[float] = []
    for _cid, genes in gold_complexes.items():
        if len(genes) < int(min_complex_size):
            continue
        cand_idx: set[int] = set()
        for g in genes:
            cand_idx.update(inv.get(g, ()))
        best = 0.0
        for j in cand_idx:
            c = pred[j]
            inter = len(genes & c)
            if inter < int(min_overlap):
                continue
            p = inter / len(c)
            r = inter / len(genes)
            denom = p + r
            f1 = (2.0 * p * r / denom) if denom > 0 else 0.0
            if f1 > best:
                best = f1
        f1s.append(best)
    return float(np.mean(f1s)) if f1s else 0.0


def _os_score(a: set[str], b: set[str]) -> float:
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return float((inter * inter) / (len(a) * len(b)))


def _os_metrics(
    gold_complexes: dict[str, set[str]],
    predicted: list[set[str]],
    min_complex_size: int = 3,
    min_overlap: int = 1,
) -> tuple[float, float, float]:
    gold = [(cid, g) for cid, g in gold_complexes.items() if len(g) >= int(min_complex_size)]
    pred = [c for c in predicted if len(c) >= 2]

    inv_pred: dict[str, list[int]] = defaultdict(list)
    for j, c in enumerate(pred):
        for x in c:
            inv_pred[x].append(j)

    inv_gold: dict[str, list[int]] = defaultdict(list)
    for i, (_cid, g) in enumerate(gold):
        for x in g:
            inv_gold[x].append(i)

    # Sn: mean over gold complexes of best OS.
    best_gold: list[float] = []
    for _cid, g in gold:
        cand: set[int] = set()
        for x in g:
            cand.update(inv_pred.get(x, ()))
        best = 0.0
        for j in cand:
            c = pred[j]
            inter = len(g & c)
            if inter < int(min_overlap):
                continue
            os = float((inter * inter) / (len(g) * len(c)))
            if os > best:
                best = os
        best_gold.append(best)
    sn = float(np.mean(best_gold)) if best_gold else 0.0

    # PPV: mean over predicted clusters of best OS.
    best_pred: list[float] = []
    for c in pred:
        cand: set[int] = set()
        for x in c:
            cand.update(inv_gold.get(x, ()))
        best = 0.0
        for i in cand:
            g = gold[i][1]
            inter = len(g & c)
            if inter < int(min_overlap):
                continue
            os = float((inter * inter) / (len(g) * len(c)))
            if os > best:
                best = os
        best_pred.append(best)
    ppv = float(np.mean(best_pred)) if best_pred else 0.0

    acc = float(math.sqrt(max(0.0, sn * ppv)))
    return sn, ppv, acc


def _greedy_mmr(
    gold_complexes: dict[str, set[str]],
    predicted: list[set[str]],
    min_complex_size: int = 3,
    min_overlap: int = 1,
) -> float:
    gold = [(cid, g) for cid, g in gold_complexes.items() if len(g) >= int(min_complex_size)]
    pred = [c for c in predicted if len(c) >= 2]
    if not gold or not pred:
        return 0.0

    inv_gold: dict[str, list[int]] = defaultdict(list)
    for i, (_cid, g) in enumerate(gold):
        for x in g:
            inv_gold[x].append(i)

    edges: list[tuple[float, int, int]] = []
    for j, c in enumerate(pred):
        overlap_counts: dict[int, int] = defaultdict(int)
        for x in c:
            for gi in inv_gold.get(x, ()):
                overlap_counts[int(gi)] += 1
        for gi, inter in overlap_counts.items():
            if inter < int(min_overlap):
                continue
            g = gold[int(gi)][1]
            os = float((inter * inter) / (len(g) * len(c)))
            if os > 0.0:
                edges.append((os, int(gi), int(j)))

    if not edges:
        return 0.0
    edges.sort(key=lambda t: (-t[0], t[1], t[2]))

    used_gold: set[int] = set()
    used_pred: set[int] = set()
    total = 0.0
    for w, gi, pj in edges:
        if gi in used_gold or pj in used_pred:
            continue
        used_gold.add(gi)
        used_pred.add(pj)
        total += float(w)
        if len(used_gold) >= len(gold):
            break

    return float(total / len(gold))


def _gpu_mcl_sparse(
    adj_csr_cpu,
    inflation: float,
    max_iter: int,
    prune_threshold: float,
    tol: float,
) -> list[set[int]]:
    """
    Sparse MCL on GPU (CuPy/CuPyX).

    Notes:
    - Works best when `adj_csr_cpu` is reasonably sparse and moderate-sized.
    - This is an approximation of common MCL implementations (pruning + column-stochastic normalization).
    """
    import cupy as cp
    from cupyx.scipy import sparse as cpx_sparse

    n = int(adj_csr_cpu.shape[0])
    if n <= 1:
        return []

    # Move to GPU and ensure CSR.
    M = cpx_sparse.csr_matrix(adj_csr_cpu, dtype=cp.float32)

    # Add self loops.
    eye = cpx_sparse.identity(n, dtype=cp.float32, format="csr")
    M = (M + eye).tocsr()
    M.sum_duplicates()

    def normalize_cols(X):
        # Normalize columns to sum to 1 (column-stochastic).
        col_sum = cp.asarray(X.sum(axis=0)).ravel()
        col_sum = cp.where(col_sum == 0.0, 1.0, col_sum)
        inv = 1.0 / col_sum
        # X * diag(inv)
        return (X @ cpx_sparse.diags(inv.astype(cp.float32), format="csr")).tocsr()

    M = normalize_cols(M)

    inflation = float(inflation)
    max_iter = int(max_iter)
    prune_threshold = float(prune_threshold)
    tol = float(tol)

    for _ in range(max_iter):
        M_prev = M

        # Expansion (power 2).
        M = (M @ M).tocsr()
        M.sum_duplicates()

        # Inflation.
        M.data = cp.power(M.data, inflation)
        M = normalize_cols(M)

        # Prune.
        if prune_threshold > 0.0:
            # Robust prune: rebuild from COO (keeps code simple/deterministic).
            M = M.tocoo()
            keep = M.data >= prune_threshold
            M = cpx_sparse.csr_matrix(
                (M.data[keep], (M.row[keep], M.col[keep])),
                shape=(n, n),
                dtype=cp.float32,
            )
            M.sum_duplicates()

        # Convergence check (cheap heuristic): max absolute delta on shared nonzeros.
        try:
            diff = (M - M_prev).tocsr()
            if diff.nnz == 0:
                break
            if float(cp.max(cp.abs(diff.data))) < tol:
                break
        except Exception:
            # If subtraction fails due to structural issues, keep iterating.
            pass

    # Extract clusters (markov_clustering-style): attractors are non-zero diagonal.
    diag = M.diagonal()
    attractors = cp.where(diag > 0.0)[0]
    if attractors.size == 0:
        return []

    uniq: set[tuple[int, ...]] = set()
    for a in cp.asnumpy(attractors).tolist():
        row = M.getrow(int(a))
        cols = cp.asnumpy(row.indices).tolist()
        cols = sorted({int(x) for x in cols})
        if cols:
            uniq.add(tuple(cols))

    return [set(t) for t in sorted(uniq)]


def _cpu_mcl_markov(
    adj_csr_cpu,
    inflation: float,
    max_iter: int,
    prune_threshold: float,
    tol: float,
) -> list[set[int]]:
    import markov_clustering as mcl

    # markov_clustering uses scipy sparse on CPU.
    _ = tol
    result = mcl.run_mcl(
        adj_csr_cpu,
        expansion=2,
        inflation=float(inflation),
        loop_value=1.0,
        iterations=int(max_iter),
        pruning_threshold=float(prune_threshold),
        pruning_frequency=1,
        convergence_check_frequency=1,
    )
    clusters = mcl.get_clusters(result)
    return [set(map(int, c)) for c in clusters]


def _cpu_mcl_binary_mcl(
    adj_csr_cpu,
    inflation: float,
) -> list[set[int]]:
    """
    Use the reference `mcl` binary on an ABC edge list.

    This is slower than the GPU backend but closer to the paper's likely implementation.
    """
    from pathlib import Path
    import subprocess
    import tempfile

    adj = adj_csr_cpu.tocsr()
    n = int(adj.shape[0])
    if n <= 1:
        return []

    coo = adj.tocoo(copy=False)
    # Keep upper triangle to represent each undirected edge once.
    mask = coo.row < coo.col
    rows = coo.row[mask]
    cols = coo.col[mask]
    data = coo.data[mask]

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / "graph.abc"
        out_path = td_path / "out.txt"

        with open(in_path, "w") as f:
            for i, j, w in zip(rows.tolist(), cols.tolist(), data.tolist()):
                f.write(f"{int(i)}\t{int(j)}\t{float(w)}\n")
            # Add loops to match the other backends.
            for i in range(n):
                f.write(f"{int(i)}\t{int(i)}\t1.0\n")

        cmd = [
            "mcl",
            str(in_path),
            "--abc",
            "-I",
            str(float(inflation)),
            "-o",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)

        clusters: list[set[int]] = []
        with open(out_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                clusters.append({int(x) for x in parts})
        return clusters


def _codeseg_blocks_mcl(
    graph: LoadedGraph,
    ref_dir: Path,
    seed: int,
    Ks: list[int],
    topk: int,
    pmin: float,
    membership_power: float,
    inflation: float,
    min_cluster_size: int,
    mcl_max_iter: int,
    mcl_prune_threshold: float,
    mcl_tol: float,
    mcl_backend: str,
) -> tuple[list[set[str]], dict]:
    t0 = _now_ms()
    all_clusters: list[set[str]] = []
    diagnostics: dict = {"Ks": list(map(int, Ks)), "blocks": {}, "runtime_ms": {}}

    for K in Ks:
        k0 = _now_ms()
        ref_nodes, P_ref = _load_reference_membership(ref_dir=ref_dir, seed=seed, K=K)
        blocks, overlap_mask = _topk_blocks(
            target_nodes=graph.nodes,
            ref_nodes=ref_nodes,
            P_ref=P_ref,
            topk=topk,
            pmin=pmin,
            membership_power=membership_power,
        )
        diagnostics["blocks"][str(K)] = {"n_blocks": int(len(blocks)), "overlap_nodes": int(overlap_mask.sum())}

        # Run MCL inside each block.
        for bi, block_idx in enumerate(blocks):
            if block_idx.size < 2:
                continue
            sub = graph.adj[block_idx, :][:, block_idx]
            if mcl_backend == "cpu":
                clusters_local = _cpu_mcl_markov(
                    sub,
                    inflation=inflation,
                    max_iter=mcl_max_iter,
                    prune_threshold=mcl_prune_threshold,
                    tol=mcl_tol,
                )
            elif mcl_backend == "mcl":
                _ = mcl_max_iter, mcl_prune_threshold, mcl_tol
                clusters_local = _cpu_mcl_binary_mcl(
                    sub,
                    inflation=inflation,
                )
            else:
                clusters_local = _gpu_mcl_sparse(
                    sub,
                    inflation=inflation,
                    max_iter=mcl_max_iter,
                    prune_threshold=mcl_prune_threshold,
                    tol=mcl_tol,
                )
            for c in clusters_local:
                if len(c) < int(min_cluster_size):
                    continue
                genes = {graph.nodes[int(block_idx[i])] for i in c if int(i) < int(block_idx.size)}
                if len(genes) >= int(min_cluster_size):
                    all_clusters.append(genes)

        diagnostics["runtime_ms"][str(K)] = int(_now_ms() - k0)

    diagnostics["runtime_ms"]["total"] = int(_now_ms() - t0)
    return all_clusters, diagnostics


def _linkcomm_candidates_linegraph_louvain(
    graph: LoadedGraph,
    min_cluster_size: int,
    seed: int,
    resolution: float,
    split_components: bool,
) -> list[set[str]]:
    """
    A GPU-friendly "link communities" approximation:
      - build the line graph (edge-as-node) adjacency
      - run GPU Louvain on the line graph
      - map edge-communities back to node sets
    """
    import cudf
    import cugraph

    # Build an undirected edge list with stable edge ids.
    coo = graph.adj.tocoo(copy=False)
    # Keep only upper triangle to represent each undirected edge once.
    mask = coo.row < coo.col
    src = coo.row[mask].astype(np.int32, copy=False)
    dst = coo.col[mask].astype(np.int32, copy=False)
    if src.size == 0:
        return []
    e = int(src.size)
    edge_id = np.arange(e, dtype=np.int32)

    # Incidence: for each endpoint, list incident edge ids.
    node_ids = np.concatenate([src, dst], axis=0)
    edge_ids = np.concatenate([edge_id, edge_id], axis=0)
    order = np.argsort(node_ids, kind="mergesort")
    node_ids = node_ids[order]
    edge_ids = edge_ids[order]

    # Count degrees.
    uniq_nodes, counts = np.unique(node_ids, return_counts=True)
    # Total line-graph edges = sum choose2(deg(u)).
    total_pairs = int(np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2))
    if total_pairs <= 0:
        return []

    lg_src = np.empty((total_pairs,), dtype=np.int32)
    lg_dst = np.empty((total_pairs,), dtype=np.int32)

    pos = 0
    start = 0
    for deg in counts.tolist():
        deg = int(deg)
        if deg >= 2:
            ids = edge_ids[start : start + deg]
            ii, jj = np.triu_indices(deg, k=1)
            m = int(ii.size)
            lg_src[pos : pos + m] = ids[ii]
            lg_dst[pos : pos + m] = ids[jj]
            pos += m
        start += deg
    if pos != total_pairs:
        lg_src = lg_src[:pos]
        lg_dst = lg_dst[:pos]

    df = cudf.DataFrame({"src": lg_src, "dst": lg_dst})
    lg = cugraph.Graph(directed=False)
    lg.from_cudf_edgelist(df, source="src", destination="dst", renumber=False)

    parts, _ = cugraph.louvain(lg, resolution=float(resolution), threshold=1e-7)
    parts = parts.sort_values("vertex")
    # Note: isolated edge-nodes may be absent from the line graph and therefore absent from `parts`.
    vertices = parts["vertex"].to_numpy()
    part = parts["partition"].to_numpy()
    comm_to_edges: dict[int, list[int]] = defaultdict(list)
    for eid, cid in zip(vertices.tolist(), part.tolist()):
        comm_to_edges[int(cid)].append(int(eid))

    out: list[set[str]] = []
    if not split_components:
        for eids in comm_to_edges.values():
            nodes_set: set[str] = set()
            for eid in eids:
                nodes_set.add(graph.nodes[int(src[eid])])
                nodes_set.add(graph.nodes[int(dst[eid])])
            if len(nodes_set) >= int(min_cluster_size):
                out.append(nodes_set)
    else:
        # Split each edge-community into connected components (union-find over the community's edges).
        for eids in comm_to_edges.values():
            local_nodes: dict[int, int] = {}
            edges_local: list[tuple[int, int]] = []
            for eid in eids:
                u = int(src[eid])
                v = int(dst[eid])
                if u not in local_nodes:
                    local_nodes[u] = len(local_nodes)
                if v not in local_nodes:
                    local_nodes[v] = len(local_nodes)
                edges_local.append((local_nodes[u], local_nodes[v]))

            m = len(local_nodes)
            if m < int(min_cluster_size):
                continue

            parent = list(range(m))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra = find(a)
                rb = find(b)
                if ra != rb:
                    parent[rb] = ra

            for a, b in edges_local:
                union(int(a), int(b))

            comp: dict[int, list[int]] = defaultdict(list)
            inv = {li: ni for ni, li in local_nodes.items()}
            for li in range(m):
                comp[find(li)].append(inv[li])

            for nodes_idx in comp.values():
                if len(nodes_idx) < int(min_cluster_size):
                    continue
                out.append({graph.nodes[int(n)] for n in nodes_idx})

    # Deterministic order.
    out.sort(key=lambda c: (-len(c), sorted(c)[:5]))
    return out


def _linkcomm_candidates_jaccard_linegraph(
    graph: LoadedGraph,
    min_cluster_size: int,
    seed: int,
    resolution: float,
    jaccard_threshold: float,
    batch_pairs: int,
    split_components: bool,
    inclusive_neighborhoods: bool,
) -> list[set[str]]:
    """
    Link-communities-style candidates via a weighted line graph:
      - create edge-nodes for each original edge
      - connect two edge-nodes if the corresponding edges share a node
      - weight that connection by Jaccard similarity between the *other endpoints* (cugraph.jaccard)
      - cluster the weighted line graph via Louvain
      - map edge-communities back to node sets (optionally split into components)
    """
    import cudf
    import cugraph

    _ = seed  # cugraph.louvain currently has no explicit random_state
    jaccard_threshold = float(jaccard_threshold)
    batch_pairs = int(batch_pairs)
    if batch_pairs <= 0:
        raise ValueError("--linkcomm-batch-pairs must be positive")

    # Undirected edges with stable ids.
    coo = graph.adj.tocoo(copy=False)
    mask = coo.row < coo.col
    src = coo.row[mask].astype(np.int32, copy=False)
    dst = coo.col[mask].astype(np.int32, copy=False)
    e = int(src.size)
    if e == 0:
        return []
    edge_id = np.arange(e, dtype=np.int32)

    # Original graph for Jaccard queries.
    base_edges = cudf.DataFrame({"src": src, "dst": dst})
    if inclusive_neighborhoods:
        loops = np.arange(int(graph.adj.shape[0]), dtype=np.int32)
        base_edges = cudf.concat([base_edges, cudf.DataFrame({"src": loops, "dst": loops})], ignore_index=True)
    base_g = cugraph.Graph(directed=False)
    base_g.from_cudf_edgelist(base_edges, source="src", destination="dst", renumber=False)

    # Incidence lists: for each node, its neighbor nodes and the corresponding edge id.
    node_ids = np.concatenate([src, dst], axis=0)
    neigh_ids = np.concatenate([dst, src], axis=0)
    edge_ids = np.concatenate([edge_id, edge_id], axis=0)
    order = np.argsort(node_ids, kind="mergesort")
    node_ids = node_ids[order]
    neigh_ids = neigh_ids[order]
    edge_ids = edge_ids[order]
    _, counts = np.unique(node_ids, return_counts=True)

    lg_src_chunks: list[np.ndarray] = []
    lg_dst_chunks: list[np.ndarray] = []
    lg_w_chunks: list[np.ndarray] = []

    batch_first: list[np.ndarray] = []
    batch_second: list[np.ndarray] = []
    batch_e1: list[np.ndarray] = []
    batch_e2: list[np.ndarray] = []
    batch_n = 0

    def flush_batch() -> None:
        nonlocal batch_first, batch_second, batch_e1, batch_e2, batch_n
        if batch_n <= 0:
            return
        first = np.concatenate(batch_first, axis=0)
        second = np.concatenate(batch_second, axis=0)
        e1 = np.concatenate(batch_e1, axis=0)
        e2 = np.concatenate(batch_e2, axis=0)

        pairs = cudf.DataFrame({"first": first, "second": second})
        res = cugraph.jaccard(base_g, pairs)
        coeff = res["jaccard_coeff"].to_numpy()
        keep = coeff >= jaccard_threshold
        if np.any(keep):
            lg_src_chunks.append(e1[keep].astype(np.int32, copy=False))
            lg_dst_chunks.append(e2[keep].astype(np.int32, copy=False))
            lg_w_chunks.append(coeff[keep].astype(np.float32, copy=False))

        batch_first = []
        batch_second = []
        batch_e1 = []
        batch_e2 = []
        batch_n = 0

    start = 0
    for deg in counts.tolist():
        deg = int(deg)
        if deg >= 2:
            nb = neigh_ids[start : start + deg]
            ei = edge_ids[start : start + deg]
            ii, jj = np.triu_indices(deg, k=1)
            first = nb[ii]
            second = nb[jj]
            e1 = ei[ii]
            e2 = ei[jj]

            batch_first.append(first.astype(np.int32, copy=False))
            batch_second.append(second.astype(np.int32, copy=False))
            batch_e1.append(e1.astype(np.int32, copy=False))
            batch_e2.append(e2.astype(np.int32, copy=False))
            batch_n += int(first.size)

            if batch_n >= batch_pairs:
                flush_batch()
        start += deg
    flush_batch()

    if not lg_src_chunks:
        return []

    lg_src = np.concatenate(lg_src_chunks, axis=0)
    lg_dst = np.concatenate(lg_dst_chunks, axis=0)
    lg_w = np.concatenate(lg_w_chunks, axis=0)
    if lg_src.size == 0:
        return []

    lg_df = cudf.DataFrame({"src": lg_src, "dst": lg_dst, "w": lg_w})
    lg = cugraph.Graph(directed=False)
    lg.from_cudf_edgelist(lg_df, source="src", destination="dst", edge_attr="w", renumber=False)

    parts, _ = cugraph.louvain(lg, resolution=float(resolution), threshold=1e-7)
    parts = parts.sort_values("vertex")
    vertices = parts["vertex"].to_numpy()
    part = parts["partition"].to_numpy()

    comm_to_edges: dict[int, list[int]] = defaultdict(list)
    for eid, cid in zip(vertices.tolist(), part.tolist()):
        comm_to_edges[int(cid)].append(int(eid))

    out: list[set[str]] = []
    if not split_components:
        for eids in comm_to_edges.values():
            nodes_set: set[str] = set()
            for eid in eids:
                nodes_set.add(graph.nodes[int(src[eid])])
                nodes_set.add(graph.nodes[int(dst[eid])])
            if len(nodes_set) >= int(min_cluster_size):
                out.append(nodes_set)
    else:
        for eids in comm_to_edges.values():
            local_nodes: dict[int, int] = {}
            edges_local: list[tuple[int, int]] = []
            for eid in eids:
                u = int(src[eid])
                v = int(dst[eid])
                if u not in local_nodes:
                    local_nodes[u] = len(local_nodes)
                if v not in local_nodes:
                    local_nodes[v] = len(local_nodes)
                edges_local.append((local_nodes[u], local_nodes[v]))

            m = len(local_nodes)
            if m < int(min_cluster_size):
                continue

            parent = list(range(m))

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a: int, b: int) -> None:
                ra = find(a)
                rb = find(b)
                if ra != rb:
                    parent[rb] = ra

            for a, b in edges_local:
                union(int(a), int(b))

            comp: dict[int, list[int]] = defaultdict(list)
            inv = {li: ni for ni, li in local_nodes.items()}
            for li in range(m):
                comp[find(li)].append(inv[li])

            for nodes_idx in comp.values():
                if len(nodes_idx) < int(min_cluster_size):
                    continue
                out.append({graph.nodes[int(n)] for n in nodes_idx})

    out.sort(key=lambda c: (-len(c), sorted(c)[:5]))
    return out


def _linkcomm_candidates_ahn_single_linkage(
    graph: LoadedGraph,
    min_cluster_size: int,
    seed: int,
    jaccard_threshold: float,
    batch_pairs: int,
    split_components: bool,
    inclusive_neighborhoods: bool,
) -> list[set[str]]:
    """
    A closer reconstruction of classic "link communities" (Ahn et al. 2010):
      - compute Jaccard similarities between incident edges (same as jaccard_linegraph weights)
      - run single-linkage clustering on edge-nodes
      - choose a cut by maximizing partition density
      - map edge-communities back to node sets (optionally split into components)

    Implementation note: we still use GPU (cugraph.jaccard) for similarity computation, but the
    single-linkage + partition-density scan is done on CPU.
    """
    import cudf
    import cugraph

    _ = seed  # deterministic path (no explicit RNG in cugraph.jaccard)
    jaccard_threshold = float(jaccard_threshold)
    batch_pairs = int(batch_pairs)
    if batch_pairs <= 0:
        raise ValueError("--linkcomm-batch-pairs must be positive")

    # Undirected edges with stable ids.
    coo = graph.adj.tocoo(copy=False)
    mask = coo.row < coo.col
    src = coo.row[mask].astype(np.int32, copy=False)
    dst = coo.col[mask].astype(np.int32, copy=False)
    e = int(src.size)
    if e == 0:
        return []
    edge_id = np.arange(e, dtype=np.int32)

    # Base graph for Jaccard queries.
    base_edges = cudf.DataFrame({"src": src, "dst": dst})
    if inclusive_neighborhoods:
        loops = np.arange(int(graph.adj.shape[0]), dtype=np.int32)
        base_edges = cudf.concat([base_edges, cudf.DataFrame({"src": loops, "dst": loops})], ignore_index=True)
    base_g = cugraph.Graph(directed=False)
    base_g.from_cudf_edgelist(base_edges, source="src", destination="dst", renumber=False)

    # Incidence lists: for each node, its neighbor nodes and the corresponding edge id.
    node_ids = np.concatenate([src, dst], axis=0)
    neigh_ids = np.concatenate([dst, src], axis=0)
    edge_ids = np.concatenate([edge_id, edge_id], axis=0)
    order = np.argsort(node_ids, kind="mergesort")
    node_ids = node_ids[order]
    neigh_ids = neigh_ids[order]
    edge_ids = edge_ids[order]
    _, counts = np.unique(node_ids, return_counts=True)

    lg_src_chunks: list[np.ndarray] = []
    lg_dst_chunks: list[np.ndarray] = []
    lg_w_chunks: list[np.ndarray] = []

    batch_first: list[np.ndarray] = []
    batch_second: list[np.ndarray] = []
    batch_e1: list[np.ndarray] = []
    batch_e2: list[np.ndarray] = []
    batch_n = 0

    def flush_batch() -> None:
        nonlocal batch_first, batch_second, batch_e1, batch_e2, batch_n
        if batch_n <= 0:
            return
        first = np.concatenate(batch_first, axis=0)
        second = np.concatenate(batch_second, axis=0)
        e1 = np.concatenate(batch_e1, axis=0)
        e2 = np.concatenate(batch_e2, axis=0)

        pairs = cudf.DataFrame({"first": first, "second": second})
        res = cugraph.jaccard(base_g, pairs)
        coeff = res["jaccard_coeff"].to_numpy()
        keep = coeff >= jaccard_threshold
        if np.any(keep):
            lg_src_chunks.append(e1[keep].astype(np.int32, copy=False))
            lg_dst_chunks.append(e2[keep].astype(np.int32, copy=False))
            lg_w_chunks.append(coeff[keep].astype(np.float32, copy=False))

        batch_first = []
        batch_second = []
        batch_e1 = []
        batch_e2 = []
        batch_n = 0

    start = 0
    for deg in counts.tolist():
        deg = int(deg)
        if deg >= 2:
            nb = neigh_ids[start : start + deg]
            ei = edge_ids[start : start + deg]
            ii, jj = np.triu_indices(deg, k=1)
            first = nb[ii]
            second = nb[jj]
            e1 = ei[ii]
            e2 = ei[jj]

            batch_first.append(first.astype(np.int32, copy=False))
            batch_second.append(second.astype(np.int32, copy=False))
            batch_e1.append(e1.astype(np.int32, copy=False))
            batch_e2.append(e2.astype(np.int32, copy=False))
            batch_n += int(first.size)
            if batch_n >= batch_pairs:
                flush_batch()
        start += deg
    flush_batch()

    if not lg_src_chunks:
        return []

    lg_src = np.concatenate(lg_src_chunks, axis=0)
    lg_dst = np.concatenate(lg_dst_chunks, axis=0)
    lg_w = np.concatenate(lg_w_chunks, axis=0)
    if lg_src.size == 0:
        return []

    # Sort similarity edges by descending weight.
    order = np.argsort(-lg_w, kind="mergesort")
    lg_src = lg_src[order]
    lg_dst = lg_dst[order]
    lg_w = lg_w[order]

    def contrib(m: int, n: int) -> float:
        if n <= 2:
            return 0.0
        denom = float((n - 1) * (n - 2) / 2.0)
        if denom <= 0.0:
            return 0.0
        extra = int(m - (n - 1))
        if extra <= 0:
            return 0.0
        return float(m * extra / denom)

    def find(parent: np.ndarray, x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    # Pass 1: scan for best cut (max partition density).
    parent = np.arange(e, dtype=np.int32)
    m_edges = np.ones((e,), dtype=np.int32)
    contrib_v = np.zeros((e,), dtype=np.float32)
    node_sets: list[set[int] | None] = [None] * e
    total_contrib = 0.0

    def nodeset_for_root(r: int) -> set[int]:
        s = node_sets[r]
        if s is None:
            return {int(src[r]), int(dst[r])}
        return s

    def union(a: int, b: int) -> None:
        nonlocal total_contrib
        ra = find(parent, int(a))
        rb = find(parent, int(b))
        if ra == rb:
            return
        # Subtract old contributions.
        total_contrib -= float(contrib_v[ra]) + float(contrib_v[rb])

        sa = nodeset_for_root(ra)
        sb = nodeset_for_root(rb)
        # Merge smaller into larger.
        if len(sa) < len(sb):
            ra, rb = rb, ra
            sa, sb = sb, sa

        parent[rb] = int(ra)
        if node_sets[ra] is None:
            node_sets[ra] = sa
        node_sets[ra].update(sb)
        node_sets[rb] = None
        m_edges[ra] = int(m_edges[ra] + m_edges[rb])
        m_edges[rb] = 0

        c_new = float(contrib(int(m_edges[ra]), int(len(node_sets[ra]))))
        contrib_v[ra] = np.float32(c_new)
        contrib_v[rb] = np.float32(0.0)
        total_contrib += c_new

    best_w = float(lg_w[0])
    best_d = -1.0
    i = 0
    while i < int(lg_w.size):
        w = float(lg_w[i])
        while i < int(lg_w.size) and float(lg_w[i]) == w:
            union(int(lg_src[i]), int(lg_dst[i]))
            i += 1
        d = float(total_contrib / float(e)) if e > 0 else 0.0
        if d > best_d:
            best_d = d
            best_w = float(w)

    # Pass 2: rebuild clustering at threshold best_w (similarities >= best_w).
    parent2 = np.arange(e, dtype=np.int32)
    m2 = np.ones((e,), dtype=np.int32)

    def find2(x: int) -> int:
        while parent2[x] != x:
            parent2[x] = parent2[parent2[x]]
            x = int(parent2[x])
        return int(x)

    def union2(a: int, b: int) -> None:
        ra = find2(int(a))
        rb = find2(int(b))
        if ra == rb:
            return
        if int(m2[ra]) < int(m2[rb]):
            ra, rb = rb, ra
        parent2[rb] = int(ra)
        m2[ra] = int(m2[ra] + m2[rb])
        m2[rb] = 0

    for s, t, w in zip(lg_src.tolist(), lg_dst.tolist(), lg_w.tolist()):
        if float(w) < float(best_w):
            break
        union2(int(s), int(t))

    comm_to_edges: dict[int, list[int]] = defaultdict(list)
    for eid in range(e):
        r = find2(int(eid))
        if int(m2[r]) >= 2:
            comm_to_edges[int(r)].append(int(eid))

    out: list[set[str]] = []
    if not split_components:
        for eids in comm_to_edges.values():
            nodes_set: set[str] = set()
            for eid in eids:
                nodes_set.add(graph.nodes[int(src[eid])])
                nodes_set.add(graph.nodes[int(dst[eid])])
            if len(nodes_set) >= int(min_cluster_size):
                out.append(nodes_set)
    else:
        for eids in comm_to_edges.values():
            local_nodes: dict[int, int] = {}
            edges_local: list[tuple[int, int]] = []
            for eid in eids:
                u = int(src[eid])
                v = int(dst[eid])
                if u not in local_nodes:
                    local_nodes[u] = len(local_nodes)
                if v not in local_nodes:
                    local_nodes[v] = len(local_nodes)
                edges_local.append((local_nodes[u], local_nodes[v]))

            m = len(local_nodes)
            if m < int(min_cluster_size):
                continue

            parent_u = list(range(m))

            def fu(x: int) -> int:
                while parent_u[x] != x:
                    parent_u[x] = parent_u[parent_u[x]]
                    x = parent_u[x]
                return x

            def uu(a: int, b: int) -> None:
                ra = fu(a)
                rb = fu(b)
                if ra != rb:
                    parent_u[rb] = ra

            for a, b in edges_local:
                uu(int(a), int(b))

            comp: dict[int, list[int]] = defaultdict(list)
            inv = {li: ni for ni, li in local_nodes.items()}
            for li in range(m):
                comp[fu(li)].append(inv[li])

            for nodes_idx in comp.values():
                if len(nodes_idx) < int(min_cluster_size):
                    continue
                out.append({graph.nodes[int(n)] for n in nodes_idx})

    out.sort(key=lambda c: (-len(c), sorted(c)[:5]))
    return out


def _hybrid_gate_auto(graph: LoadedGraph, linkcomm_min_size_unweighted: int) -> tuple[bool, int, str]:
    """
    Table 17 policy: auto_confidence_weighted.
      - unweighted graphs: use linkcomm with min|C|>=4 filter
      - confidence-like weighted graphs (weights in [0,1]): use linkcomm
      - otherwise: do not use linkcomm
    """
    if not graph.weighted:
        return True, int(linkcomm_min_size_unweighted), "unweighted"
    if graph.weights_in_01:
        return True, 3, "confidence-like weights in [0,1]"
    return False, 0, "weights not in [0,1]"


@dataclass(frozen=True)
class EvalRow:
    graph: str
    mode: str  # pool or operating_point
    seed: int
    tag: str
    n_pred: int
    gold: str
    bestmatch_f1: float
    sn: float
    ppv: float
    acc: float
    mmr: float
    n_gold: int


def _evaluate_all(
    graph_name: str,
    predicted: list[set[str]],
    gold: dict[str, dict[str, set[str]]],
    seed: int,
    mode: str,
    tag: str,
) -> list[EvalRow]:
    rows: list[EvalRow] = []
    for gold_name, complexes in gold.items():
        complexes_f = {cid: g for cid, g in complexes.items() if len(g) >= 3}
        best_f1 = _best_match_f1(complexes_f, predicted)
        sn, ppv, acc = _os_metrics(complexes_f, predicted)
        mmr = _greedy_mmr(complexes_f, predicted)
        rows.append(
            EvalRow(
                graph=graph_name,
                mode=mode,
                seed=int(seed),
                tag=str(tag),
                n_pred=int(len(predicted)),
                gold=str(gold_name),
                bestmatch_f1=float(best_f1),
                sn=float(sn),
                ppv=float(ppv),
                acc=float(acc),
                mmr=float(mmr),
                n_gold=int(len(complexes_f)),
            )
        )
    return rows


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)

    spec = _graph_spec(args.graph)
    graph = _load_graph(spec)

    ref_dir = Path(args.ref_dir) if args.ref_dir else (PAPER_V3_DIR / "results" / "cache")
    out_dir = Path(args.out_dir) if args.out_dir else (PAPER_V3_DIR / "results" / "frozen_protocol")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or "v3"
    tag = f"{tag}_{args.graph.lower()}_seed{args.seed}"

    # Gold complexes (intersected to node universe).
    node_universe = set(graph.nodes)
    corum_path, complexportal_gmt = _resolve_gold_paths(
        gold_snapshot=str(args.gold_snapshot),
        corum_gmt=args.corum_gmt,
        complexportal_gmt=args.complexportal_gmt,
    )
    gold = _load_gold_complexes(node_universe, corum_path=corum_path, complexportal_gmt=complexportal_gmt)

    # ------------------------
    # Candidate generation
    # ------------------------
    t0 = _now_ms()
    Ks = [int(k) for k in args.Ks]
    codeseg_clusters: list[set[str]] = []
    codeseg_diag: dict = {"runtime_ms": {}}
    if args.candidate_source in {"hybrid_auto", "codeseg_only"}:
        codeseg_clusters, codeseg_diag = _codeseg_blocks_mcl(
            graph=graph,
            ref_dir=ref_dir,
            seed=int(args.seed),
            Ks=Ks,
            topk=int(args.topk),
            pmin=float(args.pmin),
            membership_power=float(args.membership_power),
            inflation=float(args.mcl_inflation),
            min_cluster_size=int(args.min_cluster_size),
            mcl_max_iter=int(args.mcl_max_iter),
            mcl_prune_threshold=float(args.mcl_prune_threshold),
            mcl_tol=float(args.mcl_convergence_tol),
            mcl_backend=str(args.mcl_backend),
        )

    # Hybrid candidate augmentation.
    linkcomm_used = False
    linkcomm_reason = ""
    linkcomm_min_size = 0
    linkcomm_clusters: list[set[str]] = []

    if args.candidate_source == "codeseg_only":
        linkcomm_used = False
        linkcomm_reason = "codeseg_only"
    elif args.candidate_source == "linkcomm_only":
        linkcomm_used = True
        linkcomm_reason = "linkcomm_only"
        linkcomm_min_size = 3
    else:
        if args.hybrid_mode == "on":
            linkcomm_used = True
            linkcomm_reason = "forced"
            linkcomm_min_size = 3
        elif args.hybrid_mode == "off":
            linkcomm_used = False
            linkcomm_reason = "forced off"
        else:
            linkcomm_used, linkcomm_min_size, linkcomm_reason = _hybrid_gate_auto(
                graph, linkcomm_min_size_unweighted=int(args.linkcomm_min_size_unweighted)
            )

    if linkcomm_used:
        if args.linkcomm_method == "plain_linegraph":
            linkcomm_clusters = _linkcomm_candidates_linegraph_louvain(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=int(args.seed),
                resolution=float(args.linkcomm_resolution),
                split_components=bool(args.linkcomm_split_components),
            )
        elif args.linkcomm_method == "ahn_single_linkage":
            linkcomm_clusters = _linkcomm_candidates_ahn_single_linkage(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=int(args.seed),
                jaccard_threshold=float(args.linkcomm_jaccard_threshold),
                batch_pairs=int(args.linkcomm_batch_pairs),
                split_components=bool(args.linkcomm_split_components),
                inclusive_neighborhoods=bool(args.linkcomm_inclusive_neighborhoods),
            )
        else:
            linkcomm_clusters = _linkcomm_candidates_jaccard_linegraph(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=int(args.seed),
                resolution=float(args.linkcomm_resolution),
                jaccard_threshold=float(args.linkcomm_jaccard_threshold),
                batch_pairs=int(args.linkcomm_batch_pairs),
                split_components=bool(args.linkcomm_split_components),
                inclusive_neighborhoods=bool(args.linkcomm_inclusive_neighborhoods),
            )

    all_candidates = list(codeseg_clusters) + list(linkcomm_clusters)
    all_candidates = [c for c in all_candidates if len(c) >= int(args.min_cluster_size)]

    # Global dedup.
    if str(args.global_dedup_mode) == "union":
        deduped = _merge_union_dedup(
            all_candidates,
            thr=float(args.global_dedup_jaccard),
            min_size=int(args.min_cluster_size),
        )
    else:
        deduped = _dedup_nms_fast(
            all_candidates,
            thr=float(args.global_dedup_jaccard),
            min_size=int(args.min_cluster_size),
        )

    # Rerank.
    rerank_mode = str(args.rerank_mode)
    rerank_model_path: Path | None = None
    rerank_score_eff: str | None = None
    if rerank_mode == "learned":
        if args.rerank_model:
            rerank_model_path = Path(args.rerank_model)
        else:
            rerank_model_path = PAPER_V3_DIR / "results" / "cache" / f"reranker_string_seed{int(args.seed)}.json"
        if not rerank_model_path.exists():
            raise FileNotFoundError(
                f"Missing learned reranker: {rerank_model_path}\n"
                f"Generate it via: conda run -n rapids python ./src/train_reranker.py --seed {int(args.seed)}"
            )
        scored = _rerank_by_learned(deduped, graph=graph, model_path=rerank_model_path)
    elif rerank_mode == "oracle":
        rerank_score_eff = "oracle"
        scored = _rerank_by_oracle(deduped, gold=gold, target=str(args.oracle_target))
    else:
        rerank_score_eff = str(args.rerank_score)
        if rerank_score_eff == "auto":
            rerank_score_eff = "weighted_density" if graph.weighted else "density_x_cohesiveness"
        scored = _rerank_by_density(
            deduped,
            adj_csr=graph.adj,
            node_to_idx=graph.node_to_idx,
            score=rerank_score_eff,
        )

    # Large pool (uncapped except pool_cap).
    pool_cap = int(args.pool_cap)
    scored_pool = scored[:pool_cap] if pool_cap > 0 else scored
    pool_clusters = [c for _s, c in scored_pool]

    # Operating point (cap N with overlap suppression).
    if str(args.op_selection) == "rerank_only":
        op_clusters = [c for _s, c in scored[: int(args.op_cap)]]
    else:
        op_clusters = _select_top_nms(
            scored=scored,
            cap=int(args.op_cap),
            max_jaccard=float(args.op_max_jaccard),
        )

    runtime_ms = int(_now_ms() - t0)

    # ------------------------
    # Outputs
    # ------------------------
    base = out_dir / args.graph.lower()
    base.mkdir(parents=True, exist_ok=True)

    pool_path = base / f"{tag}_pool.tsv"
    op_path = base / f"{tag}_op.tsv"
    _save_clusters_tsv(pool_clusters, pool_path)
    _save_clusters_tsv(op_clusters, op_path)

    meta = {
        "graph": graph.name,
        "seed": int(args.seed),
        "tag": str(tag),
        "gold": {
            "snapshot": str(args.gold_snapshot),
            "corum_gmt": str(corum_path),
            "complexportal_gmt": str(complexportal_gmt) if complexportal_gmt is not None else None,
            "complexportal_source": (
                None if complexportal_gmt is not None else f"{COMPLEXPORTAL_TSV} + {GOA_HUMAN_GAF_GZ}"
            ),
        },
        "frozen": {
            "topk": int(args.topk),
            "pmin": float(args.pmin),
            "membership_power": float(args.membership_power),
            "Ks": Ks,
            "mcl_inflation": float(args.mcl_inflation),
            "mcl_backend": str(args.mcl_backend),
            "global_dedup_jaccard": float(args.global_dedup_jaccard),
            "global_dedup_mode": str(args.global_dedup_mode),
            "rerank_mode": str(args.rerank_mode),
            "rerank_score": str(args.rerank_score),
            "rerank_score_effective": str(rerank_score_eff) if rerank_mode in {"graph_only", "oracle"} else None,
            "rerank_model": str(rerank_model_path) if rerank_model_path else None,
            "pool_cap": int(args.pool_cap),
            "op_cap": int(args.op_cap),
            "op_max_jaccard": float(args.op_max_jaccard),
            "op_selection": str(args.op_selection),
            "oracle_target": str(args.oracle_target) if rerank_mode == "oracle" else None,
        },
        "graph_weights": {
            "weighted": bool(graph.weighted),
            "w_min": float(graph.w_min),
            "w_max": float(graph.w_max),
            "weights_in_01": bool(graph.weights_in_01),
        },
        "hybrid": {
            "mode": str(args.hybrid_mode),
            "linkcomm_used": bool(linkcomm_used),
            "linkcomm_reason": str(linkcomm_reason),
            "linkcomm_min_size": int(linkcomm_min_size),
            "n_linkcomm_clusters": int(len(linkcomm_clusters)),
        },
        "counts": {
            "n_codeseg_clusters": int(len(codeseg_clusters)),
            "n_candidates_pre_dedup": int(len(all_candidates)),
            "n_candidates_post_dedup": int(len(deduped)),
            "n_pool": int(len(pool_clusters)),
            "n_op": int(len(op_clusters)),
        },
        "runtime_ms": int(runtime_ms),
        "codeseg_diag": codeseg_diag,
    }
    meta_path = base / f"{tag}.meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # Evaluate + write CSV.
    eval_rows = []
    eval_rows.extend(_evaluate_all(graph.name, pool_clusters, gold=gold, seed=args.seed, mode="pool", tag=tag))
    eval_rows.extend(_evaluate_all(graph.name, op_clusters, gold=gold, seed=args.seed, mode="operating_point", tag=tag))

    import pandas as pd

    df = pd.DataFrame([asdict(r) for r in eval_rows])
    eval_path = base / f"{tag}.eval.csv"
    df.to_csv(eval_path, index=False)

    # Print a compact summary for quick iteration.
    def _avg(metric: str, mode: str) -> float:
        x = df[(df["mode"] == mode)][metric].astype(float).to_numpy()
        return float(np.mean(x)) if x.size else 0.0

    print(f"Graph={graph.name}  seed={args.seed}  linkcomm={linkcomm_used} ({linkcomm_reason})")
    print(f"Pool: #PC={len(pool_clusters)}  BestMatch(avg)={_avg('bestmatch_f1','pool'):.3f}  Acc(avg)={_avg('acc','pool'):.3f}  MMR(avg)={_avg('mmr','pool'):.3f}")
    print(f" OP : #PC={len(op_clusters)}  BestMatch(avg)={_avg('bestmatch_f1','operating_point'):.3f}  Acc(avg)={_avg('acc','operating_point'):.3f}  MMR(avg)={_avg('mmr','operating_point'):.3f}")
    print(f"Wrote: {pool_path}")
    print(f"Wrote: {op_path}")
    print(f"Wrote: {eval_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
