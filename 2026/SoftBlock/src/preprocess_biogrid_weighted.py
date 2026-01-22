"""
Preprocess BioGRID into a *weighted* edgelist for paper v3.

Paper v3's hybrid auto-gate (Table 17) disables link-communities on BioGRID because it is
weighted but not confidence-like (weights not in [0,1]). A simple way to match this property
is to weight each undirected interaction pair by its number of supporting BioGRID rows.

Inputs:
  - BioGRID organism Tab3 zip (human member)

Outputs:
  - ./data/processed/biogrid/biogrid_nodes.txt
  - ./data/processed/biogrid/biogrid_weighted.edgelist  (u\\tv\\tw)
  - metadata JSON with parsing stats + source version
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Meta:
    version: str
    organism_taxid: str
    organism_mode: str
    system_type: str
    weight_mode: str
    rows_read: int
    rows_kept: int
    edges_uniq: int
    nodes_uniq: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess BioGRID weighted edgelist (paper v3)")
    p.add_argument("--version", type=str, default="5.0.252")
    p.add_argument(
        "--raw-zip",
        type=str,
        default=None,
        help="Path to BIOGRID-ORGANISM-<version>.tab3.zip (default: codeseg/data/datasets/biogrid/...).",
    )
    p.add_argument("--organism-taxid", type=str, default="9606")
    p.add_argument(
        "--system-type",
        choices=["physical", "genetic", "all"],
        default="all",
        help="BioGRID 'Experimental System Type' filter (paper Table 1 matches `all`).",
    )
    p.add_argument(
        "--weight-mode",
        choices=["row_count"],
        default="row_count",
        help="Edge weight definition (default: number of BioGRID rows supporting the pair).",
    )
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Debug: stop after reading N rows.")
    return p.parse_args()


def _paper_v3_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _codeseg_dir() -> Path:
    return _paper_v3_dir().parent


def _default_raw_zip(version: str) -> Path:
    return _codeseg_dir() / "data" / "datasets" / "biogrid" / f"BIOGRID-ORGANISM-{version}.tab3.zip"


def _pick_tab3_member(zip_path: Path, organism_taxid: str) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".tab3.txt")]
        if not names:
            raise ValueError(f"No .tab3.txt found inside: {zip_path}")
        taxid = str(organism_taxid).strip()
        if taxid == "9606":
            for n in names:
                if "homo_sapiens" in n.lower():
                    return n
        return sorted(names)[0]


def main() -> None:
    args = parse_args()
    version = str(args.version)
    raw_zip = Path(args.raw_zip) if args.raw_zip else _default_raw_zip(version)
    if not raw_zip.exists():
        raise FileNotFoundError(
            f"BioGRID zip not found: {raw_zip}\n"
            f"Put it under: {_default_raw_zip(version)} or pass --raw-zip"
        )

    out_dir = Path(args.out_dir) if args.out_dir else (_paper_v3_dir() / "data" / "processed" / "biogrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    member = _pick_tab3_member(raw_zip, organism_taxid=str(args.organism_taxid))
    wanted_taxid = str(args.organism_taxid).strip()
    wanted_sys = str(args.system_type).strip().lower()

    edges_w: dict[tuple[str, str], int] = {}
    rows_read = 0
    rows_kept = 0
    max_rows = int(args.max_rows) if args.max_rows is not None else None

    with zipfile.ZipFile(raw_zip) as zf:
        with zf.open(member, "r") as raw:
            txt = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            reader = csv.DictReader(txt, delimiter="\t")
            for row in reader:
                rows_read += 1
                if max_rows is not None and rows_read > max_rows:
                    break

                sys_type = (row.get("Experimental System Type", "") or "").strip().lower()
                if wanted_sys != "all" and sys_type != wanted_sys:
                    continue

                org_a = (row.get("Organism ID Interactor A", "") or "").strip()
                org_b = (row.get("Organism ID Interactor B", "") or "").strip()
                # BioGRID's Homo sapiens file includes interactions where at least one endpoint is human.
                # Paper Table 1 counts match keeping edges where either interactor has taxid=9606.
                if org_a != wanted_taxid and org_b != wanted_taxid:
                    continue

                u = (row.get("Official Symbol Interactor A", "") or "").strip()
                v = (row.get("Official Symbol Interactor B", "") or "").strip()
                if not u or not v or u == "-" or v == "-" or u == v:
                    continue

                a, b = (u, v) if u < v else (v, u)
                edges_w[(a, b)] = int(edges_w.get((a, b), 0)) + 1
                rows_kept += 1

    nodes = sorted({x for (a, b) in edges_w.keys() for x in (a, b)})

    nodes_path = out_dir / "biogrid_nodes.txt"
    edges_path = out_dir / "biogrid_weighted.edgelist"
    meta_path = out_dir / "biogrid_weighted.meta.json"

    with open(nodes_path, "w") as f:
        for n in nodes:
            f.write(f"{n}\n")

    with open(edges_path, "w") as f:
        for (a, b), w in sorted(edges_w.items()):
            f.write(f"{a}\t{b}\t{int(w)}\n")

    meta = Meta(
        version=version,
        organism_taxid=wanted_taxid,
        organism_mode="any_endpoint",
        system_type=wanted_sys,
        weight_mode=str(args.weight_mode),
        rows_read=int(rows_read),
        rows_kept=int(rows_kept),
        edges_uniq=int(len(edges_w)),
        nodes_uniq=int(len(nodes)),
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    print(f"Wrote: {nodes_path} ({len(nodes):,} nodes)")
    print(f"Wrote: {edges_path} ({len(edges_w):,} edges)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
