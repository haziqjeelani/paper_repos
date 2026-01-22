"""
Preprocess hu.MAP2 PPIs into a weighted PPI network for paper v3.

The hu.MAP2 download page provides a "pairsWprob.gz" file containing gene-name pairs
and an SVM probability score in [0,1]. We:
  - download the genename file from Zenodo (if missing)
  - keep undirected edges with prob >= --min-prob
  - (optional) restrict to proteins present in the STRING node list (for transferability)
  - deduplicate by keeping max prob per pair
  - restrict to the largest connected component (default)
  - write nodes + weighted edgelist under ./data/processed/humap2/
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


HUMAP2_ZENODO = "https://zenodo.org/records/15293715/files/humap2_ppis_genename_20200821.pairsWprob.gz?download=1"


@dataclass(frozen=True)
class Meta:
    source_url: str
    out_raw: str
    min_prob: float
    restrict_to_string: bool
    edges_raw_rows: int
    edges_kept_rows: int
    edges_uniq: int
    nodes_uniq: int
    lcc_nodes: int
    lcc_edges: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess hu.MAP2 PPIs (paper v3)")
    p.add_argument("--download", action="store_true", help="Download the Zenodo file if missing.")
    p.add_argument("--raw", type=str, default=None, help="Path to humap2_ppis_genename_*.pairsWprob.gz")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--min-prob", type=float, default=0.1)
    p.add_argument("--restrict-to-string", action="store_true", help="Keep only proteins present in STRING nodes.")
    p.add_argument(
        "--lcc",
        action="store_true",
        help="Restrict to the largest connected component (paper v3 Table 1 does not apply this for hu.MAP2).",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Debug: stop after N rows.")
    return p.parse_args()


def _paper_v3_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _codeseg_dir() -> Path:
    return _paper_v3_dir().parent


def _default_raw_path() -> Path:
    return _paper_v3_dir() / "data" / "raw" / "humap2" / "humap2_ppis_genename_20200821.pairsWprob.gz"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _load_string_nodes() -> set[str]:
    nodes_path = _codeseg_dir() / "data" / "processed" / "network" / "ppi_network_nodes.txt"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Missing STRING nodes file: {nodes_path}")
    with open(nodes_path) as f:
        return {ln.strip() for ln in f if ln.strip()}


def _lcc(nodes: list[str], edges: list[tuple[int, int, float]]) -> tuple[list[str], list[tuple[int, int, float]]]:
    # DSU over int nodes.
    parent = list(range(len(nodes)))
    size = [1] * len(nodes)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for u, v, _w in edges:
        union(int(u), int(v))

    # Largest root.
    root = max(range(len(nodes)), key=lambda i: size[find(i)])
    root = find(root)
    keep = {i for i in range(len(nodes)) if find(i) == root}
    remap = {old: new for new, old in enumerate(sorted(keep))}
    nodes2 = [nodes[old] for old in sorted(keep)]
    edges2 = []
    for u, v, w in edges:
        if u in keep and v in keep:
            edges2.append((remap[int(u)], remap[int(v)], float(w)))
    return nodes2, edges2


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw) if args.raw else _default_raw_path()
    if not raw_path.exists():
        if not args.download:
            raise FileNotFoundError(f"Missing hu.MAP2 raw file: {raw_path} (pass --download to fetch it)")
        print(f"Downloading hu.MAP2 -> {raw_path}")
        _download(HUMAP2_ZENODO, raw_path)

    out_dir = Path(args.out_dir) if args.out_dir else (_paper_v3_dir() / "data" / "processed" / "humap2")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_prob = float(args.min_prob)
    max_rows = int(args.max_rows) if args.max_rows is not None else None

    string_nodes = _load_string_nodes() if args.restrict_to_string else None

    # Parse + dedup by max probability per undirected pair.
    edges: dict[tuple[str, str], float] = {}
    rows = 0
    kept_rows = 0
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            rows += 1
            if max_rows is not None and rows > max_rows:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            a = parts[0].strip()
            b = parts[1].strip()
            if not a or not b or a == b:
                continue
            try:
                p = float(parts[2])
            except Exception:
                continue
            if p < min_prob:
                continue
            if string_nodes is not None and (a not in string_nodes or b not in string_nodes):
                continue
            u, v = (a, b) if a < b else (b, a)
            prev = edges.get((u, v))
            if prev is None or p > prev:
                edges[(u, v)] = float(p)
            kept_rows += 1

    nodes = sorted({x for (u, v) in edges.keys() for x in (u, v)})
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges_int = [(node_to_idx[u], node_to_idx[v], w) for (u, v), w in edges.items()]

    lcc_nodes = nodes
    lcc_edges_int = edges_int
    if bool(args.lcc) and nodes and edges_int:
        lcc_nodes, lcc_edges_int = _lcc(nodes, edges_int)

    nodes_path = out_dir / "humap2_nodes.txt"
    edges_path = out_dir / "humap2_weighted.edgelist"
    meta_path = out_dir / "humap2_weighted.meta.json"

    with open(nodes_path, "w") as f:
        for n in lcc_nodes:
            f.write(f"{n}\n")
    with open(edges_path, "w") as f:
        for u, v, w in sorted(lcc_edges_int):
            f.write(f"{lcc_nodes[int(u)]}\t{lcc_nodes[int(v)]}\t{float(w):.6f}\n")

    meta = Meta(
        source_url=HUMAP2_ZENODO,
        out_raw=str(raw_path),
        min_prob=min_prob,
        restrict_to_string=bool(args.restrict_to_string),
        edges_raw_rows=int(rows),
        edges_kept_rows=int(kept_rows),
        edges_uniq=int(len(edges)),
        nodes_uniq=int(len(nodes)),
        lcc_nodes=int(len(lcc_nodes)),
        lcc_edges=int(len(lcc_edges_int)),
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    print(f"Wrote: {nodes_path} ({len(lcc_nodes):,} nodes)")
    print(f"Wrote: {edges_path} ({len(lcc_edges_int):,} edges)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
