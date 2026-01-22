"""
Train a lightweight linear reranker on STRING (optional v3 component).

The v3 PDF describes an optional linear reranker trained on STRING using only:
  - graph-derived cluster features
  - gold supervision

This script trains on the same frozen protocol candidate pool (pre-rerank) and fits a simple
ridge-regression model to predict the average max overlap-score (OS) to CORUM and
ComplexPortal complexes. The resulting model can then be used as a replacement ranking
score for fixed-cap operating-point selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from frozen_protocol import (  # noqa: E402
    PAPER_V3_DIR,
    _codeseg_blocks_mcl,
    _graph_spec,
    _hybrid_gate_auto,
    _linkcomm_candidates_ahn_single_linkage,
    _linkcomm_candidates_jaccard_linegraph,
    _linkcomm_candidates_linegraph_louvain,
    _load_gold_complexes,
    _load_graph,
    _merge_union_dedup,
)
from reranker import (  # noqa: E402
    LinearReranker,
    extract_features,
    max_overlap_score_per_cluster,
    save_reranker,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SoftBlock optional linear reranker on STRING")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 16])
    p.add_argument("--topk", type=int, default=4)
    p.add_argument("--pmin", type=float, default=0.0)
    p.add_argument("--membership-power", type=float, default=1.5)
    p.add_argument("--mcl-inflation", type=float, default=4.0)
    p.add_argument("--global-dedup-jaccard", type=float, default=0.85)
    p.add_argument("--min-cluster-size", type=int, default=3)
    p.add_argument(
        "--linkcomm-method",
        choices=["plain_linegraph", "jaccard_linegraph", "ahn_single_linkage"],
        default="jaccard_linegraph",
    )
    p.add_argument("--linkcomm-resolution", type=float, default=5.0)
    p.add_argument("--linkcomm-jaccard-threshold", type=float, default=0.2)
    p.add_argument("--linkcomm-batch-pairs", type=int, default=2_000_000)
    p.add_argument("--linkcomm-split-components", action="store_true")
    p.add_argument("--linkcomm-inclusive-neighborhoods", action="store_true")
    p.add_argument("--linkcomm-min-size-unweighted", type=int, default=4)
    p.add_argument("--ref-dir", type=str, default=None)
    p.add_argument("--out", type=str, default=None, help="Output path (.json). Defaults to results/cache/")
    p.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    ref_dir = Path(args.ref_dir) if args.ref_dir else (PAPER_V3_DIR / "results" / "cache")
    out_path = Path(args.out) if args.out else (PAPER_V3_DIR / "results" / "cache" / f"reranker_string_seed{seed}.json")

    spec = _graph_spec("STRING")
    graph = _load_graph(spec)
    gold = _load_gold_complexes(set(graph.nodes))

    codeseg_clusters, _diag = _codeseg_blocks_mcl(
        graph=graph,
        ref_dir=ref_dir,
        seed=seed,
        Ks=[int(k) for k in args.Ks],
        topk=int(args.topk),
        pmin=float(args.pmin),
        membership_power=float(args.membership_power),
        inflation=float(args.mcl_inflation),
        min_cluster_size=int(args.min_cluster_size),
        mcl_max_iter=100,
        mcl_prune_threshold=1e-3,
        mcl_tol=1e-5,
        mcl_backend="gpu",
    )

    linkcomm_used, linkcomm_min_size, _reason = _hybrid_gate_auto(
        graph, linkcomm_min_size_unweighted=int(args.linkcomm_min_size_unweighted)
    )
    linkcomm_clusters: list[set[str]] = []
    if linkcomm_used:
        if args.linkcomm_method == "plain_linegraph":
            linkcomm_clusters = _linkcomm_candidates_linegraph_louvain(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=seed,
                resolution=float(args.linkcomm_resolution),
                split_components=bool(args.linkcomm_split_components),
            )
        elif args.linkcomm_method == "ahn_single_linkage":
            linkcomm_clusters = _linkcomm_candidates_ahn_single_linkage(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=seed,
                jaccard_threshold=float(args.linkcomm_jaccard_threshold),
                batch_pairs=int(args.linkcomm_batch_pairs),
                split_components=bool(args.linkcomm_split_components),
                inclusive_neighborhoods=bool(args.linkcomm_inclusive_neighborhoods),
            )
        else:
            linkcomm_clusters = _linkcomm_candidates_jaccard_linegraph(
                graph,
                min_cluster_size=int(linkcomm_min_size),
                seed=seed,
                resolution=float(args.linkcomm_resolution),
                jaccard_threshold=float(args.linkcomm_jaccard_threshold),
                batch_pairs=int(args.linkcomm_batch_pairs),
                split_components=bool(args.linkcomm_split_components),
                inclusive_neighborhoods=bool(args.linkcomm_inclusive_neighborhoods),
            )

    all_candidates = list(codeseg_clusters) + list(linkcomm_clusters)
    all_candidates = [c for c in all_candidates if len(c) >= int(args.min_cluster_size)]
    deduped = _merge_union_dedup(
        all_candidates,
        thr=float(args.global_dedup_jaccard),
        min_size=int(args.min_cluster_size),
    )

    degrees = np.asarray(graph.adj.sum(axis=1)).ravel().astype(np.float32, copy=False)
    X, feature_names = extract_features(
        deduped,
        adj_csr=graph.adj,
        node_to_idx=graph.node_to_idx,
        degrees=degrees,
    )

    y_corum = max_overlap_score_per_cluster(gold["CORUM"], deduped)
    y_cp = max_overlap_score_per_cluster(gold["ComplexPortal"], deduped)
    y = 0.5 * (y_corum + y_cp)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    model = Ridge(alpha=float(args.alpha), fit_intercept=True)
    model.fit(Xs, y)

    out_model = LinearReranker(
        feature_names=feature_names,
        mean_=np.asarray(scaler.mean_, dtype=np.float32),
        scale_=np.asarray(scaler.scale_, dtype=np.float32),
        coef_=np.asarray(model.coef_, dtype=np.float32),
        intercept_=float(model.intercept_),
        meta={
            "graph": "STRING",
            "seed": seed,
            "Ks": [int(k) for k in args.Ks],
            "topk": int(args.topk),
            "pmin": float(args.pmin),
            "membership_power": float(args.membership_power),
            "mcl_inflation": float(args.mcl_inflation),
            "global_dedup_jaccard": float(args.global_dedup_jaccard),
            "min_cluster_size": int(args.min_cluster_size),
            "linkcomm_method": str(args.linkcomm_method),
            "linkcomm_resolution": float(args.linkcomm_resolution),
            "linkcomm_jaccard_threshold": float(args.linkcomm_jaccard_threshold),
            "linkcomm_split_components": bool(args.linkcomm_split_components),
            "alpha": float(args.alpha),
            "n_train": int(X.shape[0]),
        },
    )
    save_reranker(out_model, out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
