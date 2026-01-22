"""
Preprocess IntAct (human) into a weighted PPI network for paper v3.

We use IntAct's species file (human.zip) and build an undirected gene-symbol network with
unit edge weights (weights=1.0), matching the paper's "confidence-like (all weights = 1)"
gating behavior (Table 17).

This script is streaming-friendly (reads the zip member line-by-line).
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import requests


INTACT_HUMAN_ZIP = "https://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/species/human.zip"
UNIPROT_RE = re.compile(r"uniprotkb:([A-Z0-9]+)(?:-[0-9]+)?", re.IGNORECASE)
TAXID_HUMAN = "9606"
GENE_NAME_RE = re.compile(r"uniprotkb:([A-Za-z0-9-]+)\(gene name\)", re.IGNORECASE)


@dataclass(frozen=True)
class Meta:
    source_url: str
    out_raw: str
    member: str
    rows_read: int
    rows_kept: int
    edges_uniq: int
    nodes_uniq: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess IntAct human PPIs (paper v3)")
    p.add_argument("--download", action="store_true", help="Download IntAct human.zip if missing.")
    p.add_argument("--raw-zip", type=str, default=None, help="Path to IntAct human.zip")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--max-rows", type=int, default=None, help="Debug: stop after N rows.")
    return p.parse_args()


def _paper_v3_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _codeseg_dir() -> Path:
    return _paper_v3_dir().parent


def _default_raw_zip() -> Path:
    return _paper_v3_dir() / "data" / "raw" / "intact" / "human.zip"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _build_uniprot_to_symbol(wanted: set[str]) -> dict[str, str]:
    import sys

    sys.path.append(str((_codeseg_dir() / "src").resolve()))
    from complexportal_eval import GOA_HUMAN_GAF_GZ, build_uniprot_to_symbol_from_goa  # type: ignore

    if not GOA_HUMAN_GAF_GZ.exists():
        raise FileNotFoundError(f"Missing GOA mapping file: {GOA_HUMAN_GAF_GZ}")
    return build_uniprot_to_symbol_from_goa(GOA_HUMAN_GAF_GZ, wanted=wanted)


def _extract_uniprot(field: str) -> str | None:
    if not field:
        return None
    # MITAB fields can include multiple IDs separated by '|'
    for part in str(field).split("|"):
        m = UNIPROT_RE.search(part.strip())
        if m:
            return m.group(1).upper()
    return None


def _is_human_taxid(field: str) -> bool:
    if not field:
        return False
    return f"taxid:{TAXID_HUMAN}" in str(field).lower()


def _extract_gene_name_from_alias(field: str) -> str | None:
    """
    Prefer the explicit (gene name) alias in MITAB alias fields.

    Example:
      uniprotkb:CRKL(gene name) -> CRKL
    """
    if not field:
        return None
    for part in str(field).split("|"):
        m = GENE_NAME_RE.search(part.strip())
        if m:
            sym = m.group(1).strip()
            return sym.upper() if sym else None
    return None


def main() -> None:
    args = parse_args()
    raw_zip = Path(args.raw_zip) if args.raw_zip else _default_raw_zip()
    if not raw_zip.exists():
        if not args.download:
            raise FileNotFoundError(f"Missing IntAct zip: {raw_zip} (pass --download to fetch it)")
        print(f"Downloading IntAct -> {raw_zip}")
        _download(INTACT_HUMAN_ZIP, raw_zip)

    out_dir = Path(args.out_dir) if args.out_dir else (_paper_v3_dir() / "data" / "processed" / "intact")
    out_dir.mkdir(parents=True, exist_ok=True)

    max_rows = int(args.max_rows) if args.max_rows is not None else None

    with zipfile.ZipFile(raw_zip) as zf:
        txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
        if not txt_members:
            raise ValueError(f"No .txt member found in: {raw_zip}")
        member = sorted(txt_members)[0]

        # Pass 1: gather UniProt IDs for GOA mapping.
        wanted: set[str] = set()
        rows_read = 0
        with zf.open(member, "r") as raw:
            for line in raw:
                rows_read += 1
                if max_rows is not None and rows_read > max_rows:
                    break
                try:
                    s = line.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    continue
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) < 11:
                    continue
                if not (_is_human_taxid(parts[9]) and _is_human_taxid(parts[10])):
                    continue
                a = _extract_uniprot(parts[0])
                b = _extract_uniprot(parts[1])
                if a:
                    wanted.add(a)
                if b:
                    wanted.add(b)

    print(f"IntAct rows scanned: {rows_read:,}; unique UniProt IDs: {len(wanted):,}")
    uniprot_to_symbol = _build_uniprot_to_symbol(wanted=wanted)

    # Pass 2: build weighted edges (w=1.0) on gene symbols.
    edges: set[tuple[str, str]] = set()
    rows2 = 0
    kept = 0
    with zipfile.ZipFile(raw_zip) as zf:
        with zf.open(member, "r") as raw:
            for line in raw:
                rows2 += 1
                if max_rows is not None and rows2 > max_rows:
                    break
                try:
                    s = line.decode("utf-8", errors="replace").rstrip("\n")
                except Exception:
                    continue
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) < 11:
                    continue
                if not (_is_human_taxid(parts[9]) and _is_human_taxid(parts[10])):
                    continue

                a = _extract_uniprot(parts[0])
                b = _extract_uniprot(parts[1])
                # Primary: UniProt->symbol (GOA). Only fall back to (gene name) alias when UniProt is missing.
                if a:
                    ga = uniprot_to_symbol.get(a)
                else:
                    ga = _extract_gene_name_from_alias(parts[4]) if len(parts) > 4 else None
                if b:
                    gb = uniprot_to_symbol.get(b)
                else:
                    gb = _extract_gene_name_from_alias(parts[5]) if len(parts) > 5 else None

                if not ga or not gb or ga == gb:
                    continue
                u, v = (ga, gb) if ga < gb else (gb, ga)
                edges.add((u, v))
                kept += 1

    nodes = sorted({x for (u, v) in edges for x in (u, v)})
    nodes_path = out_dir / "intact_nodes.txt"
    edges_path = out_dir / "intact_weighted.edgelist"
    meta_path = out_dir / "intact_weighted.meta.json"

    with open(nodes_path, "w") as f:
        for n in nodes:
            f.write(f"{n}\n")
    with open(edges_path, "w") as f:
        for u, v in sorted(edges):
            f.write(f"{u}\t{v}\t1.0\n")

    meta = Meta(
        source_url=INTACT_HUMAN_ZIP,
        out_raw=str(raw_zip),
        member=str(member),
        rows_read=int(rows2),
        rows_kept=int(kept),
        edges_uniq=int(len(edges)),
        nodes_uniq=int(len(nodes)),
    )
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    print(f"Wrote: {nodes_path} ({len(nodes):,} nodes)")
    print(f"Wrote: {edges_path} ({len(edges):,} edges)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
