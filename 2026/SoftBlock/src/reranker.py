from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LinearReranker:
    feature_names: list[str]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: float
    meta: dict

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        mean_ = np.asarray(self.mean_, dtype=np.float32)
        scale_ = np.asarray(self.scale_, dtype=np.float32)
        coef_ = np.asarray(self.coef_, dtype=np.float32)
        z = (X - mean_) / np.where(scale_ == 0.0, 1.0, scale_)
        return (z @ coef_) + float(self.intercept_)


def save_reranker(model: LinearReranker, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": list(model.feature_names),
        "mean": np.asarray(model.mean_, dtype=np.float32).tolist(),
        "scale": np.asarray(model.scale_, dtype=np.float32).tolist(),
        "coef": np.asarray(model.coef_, dtype=np.float32).tolist(),
        "intercept": float(model.intercept_),
        "meta": dict(model.meta),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_reranker(path: Path) -> LinearReranker:
    payload = json.loads(path.read_text())
    feature_names = [str(x) for x in payload["feature_names"]]
    mean_ = np.asarray(payload["mean"], dtype=np.float32)
    scale_ = np.asarray(payload["scale"], dtype=np.float32)
    coef_ = np.asarray(payload["coef"], dtype=np.float32)
    intercept_ = float(payload["intercept"])
    meta = dict(payload.get("meta", {}))
    if mean_.shape != coef_.shape or mean_.shape != scale_.shape:
        raise ValueError(f"Bad reranker arrays in {path}: mean={mean_.shape}, scale={scale_.shape}, coef={coef_.shape}")
    if len(feature_names) != int(mean_.shape[0]):
        raise ValueError(f"Bad feature_names length in {path}: {len(feature_names)} != {mean_.shape[0]}")
    return LinearReranker(
        feature_names=feature_names,
        mean_=mean_,
        scale_=scale_,
        coef_=coef_,
        intercept_=intercept_,
        meta=meta,
    )


def default_feature_names() -> list[str]:
    return [
        "n",
        "log_n",
        "weighted_density",
        "cohesiveness",
        "avg_internal_degree",
        "avg_degree",
        "cut_fraction",
        "internal_edges",
    ]


def _safe_log(x: float) -> float:
    return float(math.log(max(1.0, x)))


def extract_features(
    clusters: Iterable[set[str]],
    *,
    adj_csr,
    node_to_idx: dict[str, int],
    degrees: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    from scipy import sparse

    if feature_names is None:
        feature_names = default_feature_names()
    feature_names = [str(n) for n in feature_names]

    if degrees is None:
        degrees = np.asarray(adj_csr.sum(axis=1)).ravel().astype(np.float32, copy=False)
    else:
        degrees = np.asarray(degrees, dtype=np.float32)

    feats: list[list[float]] = []
    for c in clusters:
        idx = [node_to_idx[n] for n in c if n in node_to_idx]
        n = int(len(idx))
        if n < 2:
            row = {k: 0.0 for k in feature_names}
            row["n"] = float(n)
            row["log_n"] = _safe_log(float(n))
            feats.append([float(row[k]) for k in feature_names])
            continue

        sub = adj_csr[idx, :][:, idx]
        if not sparse.isspmatrix(sub):
            sub = sparse.csr_matrix(sub)

        two_win = float(sub.sum())
        win = two_win / 2.0
        denom = float(n * (n - 1) / 2.0)
        weighted_density = float(win / denom) if denom > 0.0 else 0.0

        deg_sum = float(np.sum(degrees[idx], dtype=np.float64))
        cohesiveness = float(two_win / deg_sum) if deg_sum > 0.0 else 0.0

        cut = max(0.0, deg_sum - two_win)
        cut_fraction = float(cut / deg_sum) if deg_sum > 0.0 else 0.0

        internal_edges = float(sub.nnz) / 2.0
        avg_internal_degree = float(two_win / n) if n > 0 else 0.0
        avg_degree = float(deg_sum / n) if n > 0 else 0.0

        row = {
            "n": float(n),
            "log_n": _safe_log(float(n)),
            "weighted_density": float(weighted_density),
            "cohesiveness": float(cohesiveness),
            "avg_internal_degree": float(avg_internal_degree),
            "avg_degree": float(avg_degree),
            "cut_fraction": float(cut_fraction),
            "internal_edges": float(internal_edges),
        }
        feats.append([float(row[k]) for k in feature_names])

    X = np.asarray(feats, dtype=np.float32)
    return X, feature_names


def max_overlap_score_per_cluster(
    gold_complexes: dict[str, set[str]],
    predicted: list[set[str]],
    *,
    min_complex_size: int = 3,
    min_overlap: int = 2,
) -> np.ndarray:
    gold = [g for g in gold_complexes.values() if len(g) >= int(min_complex_size)]
    if not gold:
        return np.zeros((len(predicted),), dtype=np.float32)

    inv_gold: dict[str, list[int]] = {}
    for i, g in enumerate(gold):
        for x in g:
            inv_gold.setdefault(x, []).append(int(i))

    out = np.zeros((len(predicted),), dtype=np.float32)
    for j, c in enumerate(predicted):
        overlap_counts: dict[int, int] = {}
        for x in c:
            for gi in inv_gold.get(x, ()):
                overlap_counts[gi] = int(overlap_counts.get(gi, 0)) + 1
        best = 0.0
        for gi, inter in overlap_counts.items():
            if inter < int(min_overlap):
                continue
            g = gold[int(gi)]
            os = float((inter * inter) / (len(g) * len(c)))
            if os > best:
                best = os
        out[int(j)] = float(best)
    return out

