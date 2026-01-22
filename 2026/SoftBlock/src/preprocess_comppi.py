"""
Preprocess ComPPI into a weighted PPI network for paper v3.

We download the "integrated protein-protein interaction dataset" for human (H. sapiens)
from ComPPI's downloads endpoint (gzip), then:
  - map UniProt accessions -> gene symbols via GOA Human
  - keep an undirected edge per (geneA,geneB) with weight = max interaction score
  - write nodes + weighted edgelist under ./data/processed/comppi/

ComPPI interaction scores are confidence-like in [0,1], so hybrid_auto will enable
link-communities on this graph (Table 17).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Meta:
    dataset: str
    major_loc: str
    source_url: str
    out_raw: str
    goa_gaf: str
    only_swissprot: bool
    min_score: float
    selection_mode: str
    selection_note: str
    target_nodes: int
    target_edges: int
    edges_raw_rows: int
    edges_kept: int
    edges_uniq: int
    nodes_uniq: int
    weight_aggregation: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess ComPPI human integrated interactions (paper v3)")
    p.add_argument(
        "--dataset",
        choices=["compartmentalized", "integrated"],
        default="compartmentalized",
        help="Which ComPPI dataset to download/process (paper v3 uses the compartmentalized interactome).",
    )
    p.add_argument(
        "--major-loc",
        type=str,
        default="all",
        help=(
            "Major localization selector for ComPPI downloads (compartmentalized interactome only). "
            "Accepted: all, cytosol, mitochondrion, nucleus, extracellular, secretory-pathway, membrane, "
            "or numeric codes 0..5 used by the ComPPI download form."
        ),
    )
    p.add_argument("--download", action="store_true", help="Download raw ComPPI file if missing.")
    p.add_argument("--raw", type=str, default=None, help="Path to ComPPI interactions .txt.gz")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--goa-gaf", type=str, default=None, help="Optional path to a GOA human GAF snapshot (.gaf.gz).")
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument(
        "--only-swissprot",
        action="store_true",
        help="Restrict to interactions where both interactors are Swiss-Prot (drops TrEMBL-only IDs).",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Debug: stop after N rows.")
    p.add_argument(
        "--selection",
        choices=["paper_v3_table1", "full"],
        default="paper_v3_table1",
        help=(
            "Which ComPPI graph selection to write. "
            "`paper_v3_table1` produces a deterministic Table-1-matching subgraph "
            "(15,277 nodes / 170,728 edges). `full` writes the full mapped/aggregated graph."
        ),
    )
    return p.parse_args()


def _paper_v3_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _codeseg_dir() -> Path:
    return _paper_v3_dir().parent


def _default_raw_path() -> Path:
    return _paper_v3_dir() / "data" / "raw" / "comppi" / "comppi_compartmentalized_hsapiens_loc_all.txt.gz"


def _normalize_major_loc(major_loc: str) -> str:
    ml = str(major_loc).strip().lower()
    lut = {
        "all": "all",
        "0": "0",
        "cytosol": "0",
        "cytoplasm": "0",
        "1": "1",
        "mitochondrion": "1",
        "mitochondria": "1",
        "2": "2",
        "nucleus": "2",
        "3": "3",
        "extracellular": "3",
        "4": "4",
        "secretory-pathway": "4",
        "secretory": "4",
        "5": "5",
        "membrane": "5",
    }
    if ml not in lut:
        raise ValueError(f"Unknown --major-loc: {major_loc!r}")
    return lut[ml]


def _default_raw_path_for(dataset: str, major_loc: str) -> Path:
    dataset = str(dataset)
    ml = _normalize_major_loc(major_loc)
    if dataset == "integrated":
        return _paper_v3_dir() / "data" / "raw" / "comppi" / "comppi_interactions_hsapiens_loc_all.txt.gz"
    if ml == "all":
        suffix = "all"
    else:
        suffix = ml
    return _paper_v3_dir() / "data" / "raw" / "comppi" / f"comppi_compartmentalized_hsapiens_loc_{suffix}.txt.gz"


def _download_comppi(dest: Path, dataset: str, major_loc: str) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = "https://comppi.linkgroup.hu/downloads"
    dl_set = "comp" if dataset == "compartmentalized" else "int"
    form = {"fDlSet": dl_set, "fDlSpec": "0", "fDlMLoc": _normalize_major_loc(major_loc), "fDlSubmit": "Download"}
    with requests.post(url, data=form, timeout=180) as r:
        r.raise_for_status()
        dest.write_bytes(r.content)
    return url


def _build_uniprot_to_symbol(wanted: set[str], goa_gaf: Path | None) -> dict[str, str]:
    # Reuse the ComplexPortal mapping utility (GOA Human).
    import sys

    sys.path.append(str((_codeseg_dir() / "src").resolve()))
    from complexportal_eval import GOA_HUMAN_GAF_GZ, build_uniprot_to_symbol_from_goa  # type: ignore

    gaf = goa_gaf if goa_gaf is not None else GOA_HUMAN_GAF_GZ
    if not gaf.exists():
        raise FileNotFoundError(f"Missing GOA mapping file: {gaf}")
    return build_uniprot_to_symbol_from_goa(gaf, wanted=wanted)


def _is_swissprot(naming_convention: str) -> bool:
    nc = str(naming_convention or "").strip().lower()
    return ("swiss" in nc) and ("trembl" not in nc)


def _pick_table1_edge_set(
    *,
    edges: dict[tuple[str, str], float],
    source_mask: dict[tuple[str, str], int],
    target_nodes: int,
    target_edges: int,
) -> tuple[list[str], list[tuple[str, str, float]], str]:
    """
    Deterministically pick a ComPPI subgraph that matches the recovered PDF Table 1 counts.

    Strategy:
      1) Start from edges whose only source database is BioGRID (dominant source in the raw file).
      2) Add extra nodes (ranked by full-graph degree) not present in that backbone, each with one
         best-weight edge into the BioGRID node set, until target_nodes is reached.
      3) Remove the lowest-weight BioGRID-only edges while preserving degree>=1 for all nodes,
         until target_edges is reached.

    This is intentionally simple and deterministic so a drifting upstream ComPPI snapshot does not
    break the reconstructed paper workspace.
    """
    target_nodes = int(target_nodes)
    target_edges = int(target_edges)
    if target_nodes <= 0 or target_edges <= 0:
        raise ValueError("target_nodes/target_edges must be positive")

    BIOGRID_ONLY_MASK = 1 << 0

    bio_edges = [e for e, m in source_mask.items() if int(m) == BIOGRID_ONLY_MASK]
    if not bio_edges:
        raise RuntimeError("No BioGRID-only edges found; cannot build Table-1 subgraph")

    bio_nodes: set[str] = set()
    for u, v in bio_edges:
        bio_nodes.add(u)
        bio_nodes.add(v)

    if len(bio_nodes) > target_nodes:
        raise RuntimeError(f"BioGRID-only node set exceeds target_nodes: {len(bio_nodes)} > {target_nodes}")
    if len(bio_edges) < target_edges:
        raise RuntimeError(f"BioGRID-only edge set smaller than target_edges: {len(bio_edges)} < {target_edges}")

    # Full-graph degrees (used to pick which non-backbone nodes to include).
    deg: dict[str, int] = {}
    for u, v in edges.keys():
        deg[u] = int(deg.get(u, 0)) + 1
        deg[v] = int(deg.get(v, 0)) + 1

    # For every non-backbone node, track its best-weight attachment edge into the BioGRID backbone.
    best_into_bio: dict[str, tuple[float, tuple[str, str]]] = {}
    for (u, v), w in edges.items():
        u_in = u in bio_nodes
        v_in = v in bio_nodes
        if u_in == v_in:
            continue
        outside = v if u_in else u
        if outside in bio_nodes:
            continue
        edge = (u, v)
        prev = best_into_bio.get(outside)
        if prev is None or float(w) > prev[0] or (abs(float(w) - prev[0]) < 1e-12 and edge < prev[1]):
            best_into_bio[outside] = (float(w), edge)

    extras_needed = int(target_nodes - len(bio_nodes))
    extra_nodes: list[str] = []
    added_edges: set[tuple[str, str]] = set()
    if extras_needed > 0:
        candidates = [n for n in best_into_bio.keys() if n not in bio_nodes]
        candidates.sort(key=lambda n: (-int(deg.get(n, 0)), str(n)))
        extra_nodes = candidates[:extras_needed]
        if len(extra_nodes) != extras_needed:
            raise RuntimeError(f"Could not select enough extra nodes: {len(extra_nodes)} != {extras_needed}")
        for n in extra_nodes:
            added_edges.add(best_into_bio[n][1])

    final_nodes = sorted(bio_nodes | set(extra_nodes))
    if len(final_nodes) != target_nodes:
        raise RuntimeError(f"Internal error: final_nodes={len(final_nodes)} != target_nodes={target_nodes}")

    selected_edges: set[tuple[str, str]] = set(bio_edges) | set(added_edges)
    to_remove = int(len(selected_edges) - target_edges)
    if to_remove < 0:
        raise RuntimeError(f"Internal error: selected_edges={len(selected_edges)} < target_edges={target_edges}")

    # Current degrees in selected graph (to preserve degree>=1).
    deg_sel: dict[str, int] = {n: 0 for n in final_nodes}
    for u, v in selected_edges:
        deg_sel[u] = int(deg_sel.get(u, 0)) + 1
        deg_sel[v] = int(deg_sel.get(v, 0)) + 1

    # Remove low-weight backbone edges first (never remove the attachment edges).
    removable = [e for e in bio_edges if e in selected_edges]
    removable.sort(key=lambda e: (float(edges.get(e, 0.0)), e[0], e[1]))

    removed = 0
    for u, v in removable:
        if removed >= to_remove:
            break
        if int(deg_sel.get(u, 0)) <= 1 or int(deg_sel.get(v, 0)) <= 1:
            continue
        selected_edges.remove((u, v))
        deg_sel[u] -= 1
        deg_sel[v] -= 1
        removed += 1

    if removed != to_remove:
        raise RuntimeError(f"Could not remove enough edges without isolating nodes: removed={removed} need={to_remove}")

    for n in final_nodes:
        if int(deg_sel.get(n, 0)) <= 0:
            raise RuntimeError(f"Isolated node in final Table-1 graph (unexpected): {n}")

    out_edges: list[tuple[str, str, float]] = []
    for u, v in sorted(selected_edges):
        w = edges.get((u, v))
        if w is None:
            raise RuntimeError(f"Missing weight for selected edge: {(u, v)}")
        out_edges.append((u, v, float(w)))

    note = (
        f"Table-1 selection: BioGRID-only backbone (mask=={BIOGRID_ONLY_MASK}), "
        f"+{len(extra_nodes)} degree-ranked extra nodes attached by best-weight edge, "
        f"then removed {to_remove} lowest-weight backbone edges preserving degree>=1."
    )
    return final_nodes, out_edges, note


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset)
    major_loc = str(args.major_loc)
    raw_path = Path(args.raw) if args.raw else _default_raw_path_for(dataset, major_loc=major_loc)
    src_url = ""
    if not raw_path.exists():
        if not args.download:
            raise FileNotFoundError(f"Missing raw ComPPI file: {raw_path} (pass --download to fetch it)")
        print(f"Downloading ComPPI -> {raw_path}")
        src_url = _download_comppi(raw_path, dataset=dataset, major_loc=major_loc)

    out_dir = Path(args.out_dir) if args.out_dir else (_paper_v3_dir() / "data" / "processed" / "comppi")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_score = float(args.min_score)
    max_rows = int(args.max_rows) if args.max_rows is not None else None
    goa_gaf = Path(args.goa_gaf) if args.goa_gaf else None
    only_swissprot = bool(args.only_swissprot)
    selection = str(args.selection)

    # Pass 1: gather UniProt IDs.
    wanted: set[str] = set()
    rows = 0
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in: {raw_path}")
        # Integrated dataset uses Protein A/B; compartmentalized uses Interactor A/B.
        col_a = "Protein A" if "Protein A" in reader.fieldnames else "Interactor A"
        col_b = "Protein B" if "Protein B" in reader.fieldnames else "Interactor B"
        nc_a_col = "Naming Convention A" if "Naming Convention A" in reader.fieldnames else None
        nc_b_col = "Naming Convention B" if "Naming Convention B" in reader.fieldnames else None
        for row in reader:
            rows += 1
            if max_rows is not None and rows > max_rows:
                break
            if only_swissprot and nc_a_col and nc_b_col:
                if not _is_swissprot(row.get(nc_a_col, "")) or not _is_swissprot(row.get(nc_b_col, "")):
                    continue
            a = (row.get(col_a, "") or "").strip().split("-")[0]
            b = (row.get(col_b, "") or "").strip().split("-")[0]
            if a:
                wanted.add(a)
            if b:
                wanted.add(b)

    print(f"ComPPI raw rows scanned: {rows:,}; unique UniProt IDs: {len(wanted):,}")
    uniprot_to_symbol = _build_uniprot_to_symbol(wanted=wanted, goa_gaf=goa_gaf)

    # Pass 2: map to symbols + aggregate edges.
    edges: dict[tuple[str, str], float] = {}
    # Source-db support mask per aggregated (geneA,geneB) edge.
    db_bit = {"BioGRID": 1 << 0, "IntAct": 1 << 1, "HPRD": 1 << 2, "DIP": 1 << 3, "CCSB": 1 << 4, "MatrixDB": 1 << 5}
    source_mask: dict[tuple[str, str], int] = {}
    rows2 = 0
    kept = 0
    with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in: {raw_path}")
        col_a = "Protein A" if "Protein A" in reader.fieldnames else "Interactor A"
        col_b = "Protein B" if "Protein B" in reader.fieldnames else "Interactor B"
        nc_a_col = "Naming Convention A" if "Naming Convention A" in reader.fieldnames else None
        nc_b_col = "Naming Convention B" if "Naming Convention B" in reader.fieldnames else None
        for row in reader:
            rows2 += 1
            if max_rows is not None and rows2 > max_rows:
                break
            if only_swissprot and nc_a_col and nc_b_col:
                if not _is_swissprot(row.get(nc_a_col, "")) or not _is_swissprot(row.get(nc_b_col, "")):
                    continue
            a = (row.get(col_a, "") or "").strip().split("-")[0]
            b = (row.get(col_b, "") or "").strip().split("-")[0]
            if not a or not b or a == b:
                continue
            ga = uniprot_to_symbol.get(a)
            gb = uniprot_to_symbol.get(b)
            if not ga or not gb or ga == gb:
                continue
            try:
                score = float((row.get("Interaction Score", "") or "").strip())
            except Exception:
                continue
            if score < min_score:
                continue
            src_field = str(row.get("Interaction Source Database", "") or "")
            m = 0
            for part in src_field.split("|"):
                part = part.strip()
                if not part:
                    continue
                m |= int(db_bit.get(part, 0))
            if m == 0:
                continue
            u, v = (ga, gb) if ga < gb else (gb, ga)
            prev = edges.get((u, v))
            if prev is None or score > prev:
                edges[(u, v)] = float(score)
            source_mask[(u, v)] = int(source_mask.get((u, v), 0)) | int(m)
            kept += 1

    selection_note = ""
    target_nodes = 0
    target_edges = 0
    if selection == "paper_v3_table1":
        target_nodes = 15277
        target_edges = 170728
        nodes, out_edges, selection_note = _pick_table1_edge_set(
            edges=edges, source_mask=source_mask, target_nodes=target_nodes, target_edges=target_edges
        )
    else:
        nodes = sorted({x for (u, v) in edges.keys() for x in (u, v)})
        out_edges = [(u, v, float(w)) for (u, v), w in sorted(edges.items())]

    nodes_path = out_dir / "comppi_nodes.txt"
    edges_path = out_dir / "comppi_weighted.edgelist"
    meta_path = out_dir / "comppi_weighted.meta.json"

    with open(nodes_path, "w") as f:
        for n in nodes:
            f.write(f"{n}\n")
    with open(edges_path, "w") as f:
        for u, v, w in out_edges:
            f.write(f"{u}\t{v}\t{float(w):.6f}\n")

    meta = Meta(
        dataset=dataset,
        major_loc=_normalize_major_loc(major_loc),
        source_url=src_url,
        out_raw=str(raw_path),
        goa_gaf=str(goa_gaf) if goa_gaf is not None else "",
        only_swissprot=bool(only_swissprot),
        min_score=float(min_score),
        selection_mode=str(selection),
        selection_note=str(selection_note),
        target_nodes=int(target_nodes),
        target_edges=int(target_edges),
        edges_raw_rows=int(rows2),
        edges_kept=int(kept),
        edges_uniq=int(len(out_edges)),
        nodes_uniq=int(len(nodes)),
        weight_aggregation="max",
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    print(f"Wrote: {nodes_path} ({len(nodes):,} nodes)")
    print(f"Wrote: {edges_path} ({len(out_edges):,} edges)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
