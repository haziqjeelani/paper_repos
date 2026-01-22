"""
Search CORUM archived releases for Table 1 parity (paper v3).

This is a reverse-engineering aid: CORUM releases drift, and the PDF reports specific
evaluable-complex counts per graph (Table 1). We sweep archived CORUM versions/kinds
and report which snapshot best matches those counts under the current graph node sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


PAPER_V3_DIR = Path(__file__).resolve().parent.parent


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
    p = argparse.ArgumentParser(description="Search CORUM archived releases for Table 1 parity (v3)")
    p.add_argument(
        "--versions",
        type=str,
        nargs="+",
        default=None,
        help="CORUM archived versions to test (default: query FastAPI /public/releases/archived).",
    )
    p.add_argument("--kinds", type=str, nargs="+", default=["all", "core"], choices=["all", "core", "human"])
    p.add_argument("--goa-id", type=str, default="220", help="GOA Human snapshot id used for UniProt->symbol when needed.")
    p.add_argument(
        "--out-csv",
        type=str,
        default=str(PAPER_V3_DIR / "results" / "snapshots" / "corum_search_table1.csv"),
    )
    p.add_argument("--force", action="store_true", help="Re-download/rebuild even if cached.")
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, force: bool) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=180)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.SSLError:
        r = requests.get(url, stream=True, timeout=180, verify=False)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _corum_archive_url(version: str) -> str:
    return f"https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_archived_file?version={version}"


def _goa_old_url(goa_id: str) -> str:
    return f"https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN/goa_human.gaf.{goa_id}.gz"


def _fetch_archived_versions() -> list[str]:
    url = "https://mips.helmholtz-muenchen.de/fastapi-corum/public/releases/archived"
    r = requests.get(url, timeout=60, verify=False)
    r.raise_for_status()
    rels = r.json()
    return [str(x["version"]) for x in rels]


def _load_gmt_gene_sets(path: Path) -> list[set[str]]:
    genesets: list[set[str]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            genes = {g for g in parts[2:] if g}
            if len(genes) >= 2:
                genesets.append(genes)
    return genesets


def _count_evaluable(genesets: list[set[str]], nodes: set[str], *, min_size: int = 3) -> int:
    return int(sum(1 for g in genesets if len(g & nodes) >= int(min_size)))


def main() -> None:
    args = parse_args()
    versions = [str(v) for v in (args.versions or _fetch_archived_versions())]
    kinds = [str(k) for k in args.kinds]
    goa_id = str(args.goa_id)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Local caches.
    cache_dir = PAPER_V3_DIR / "data" / "gold_search_cache"
    raw_dir = cache_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    goa_path = raw_dir / f"goa_human.gaf.{goa_id}.gz"
    _download(_goa_old_url(goa_id), goa_path, force=bool(args.force))

    # Import converter.
    import sys

    sys.path.insert(0, str((PAPER_V3_DIR / "src").resolve()))
    from fetch_gold_standards import _convert_corum_release_to_gmt  # type: ignore
    from frozen_protocol import _graph_spec  # type: ignore

    graph_nodes: dict[str, set[str]] = {}
    for graph in TABLE1_TARGETS.keys():
        spec = _graph_spec(graph)
        nodes = {ln.strip() for ln in Path(spec.nodes_path).read_text(encoding="utf-8", errors="replace").splitlines() if ln.strip()}
        graph_nodes[graph] = nodes

    rows: list[dict[str, object]] = []
    for version in versions:
        corum_zip = raw_dir / f"corum_release_{version}.zip"
        _download(_corum_archive_url(version), corum_zip, force=bool(args.force))
        zip_hash = _sha256(corum_zip)

        for kind in kinds:
            out_dir = cache_dir / f"corum_v{version}_kind{kind}_goa{goa_id}"
            out_dir.mkdir(parents=True, exist_ok=True)
            corum_gmt = out_dir / "corum.gmt"
            if not corum_gmt.exists() or args.force:
                try:
                    _convert_corum_release_to_gmt(corum_zip, kind=kind, out_gmt=corum_gmt, goa_gaf_gz=goa_path)
                except FileNotFoundError:
                    # Some releases don't contain every variant (e.g., coreComplexes).
                    continue

            genesets = _load_gmt_gene_sets(corum_gmt)
            rec: dict[str, object] = {
                "corum_version": version,
                "corum_kind": kind,
                "goa_id": goa_id,
                "n_complexes_total": int(len(genesets)),
                "corum_zip_sha256": zip_hash,
                "corum_gmt_sha256": _sha256(corum_gmt),
            }

            mismatch = 0
            for graph, tgt in TABLE1_TARGETS.items():
                n = _count_evaluable(genesets, graph_nodes[graph], min_size=3)
                rec[f"{graph}_corum"] = int(n)
                diff = int(n) - int(tgt.corum)
                rec[f"{graph}_diff"] = int(diff)
                mismatch += abs(diff)
            rec["absdiff_sum"] = int(mismatch)
            rows.append(rec)

    # Write CSV.
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda x: int(x["absdiff_sum"])):  # type: ignore[arg-type]
            w.writerow(r)

    best = sorted(rows, key=lambda x: int(x["absdiff_sum"]))[0] if rows else None
    print(f"Wrote: {out_csv}")
    if best is not None:
        print("Best (lowest absdiff_sum):")
        print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
