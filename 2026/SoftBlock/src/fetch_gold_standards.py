"""
Fetch and freeze gold standards for SoftBlock reproduction.

This script creates pinned GMT files (gene-symbol sets) under `./data/gold/`
so the v3 evaluation can be run without relying on mutable upstream snapshots.

Outputs (by default):
  - ./data/gold/corum.gmt
  - ./data/gold/complexportal.gmt
  - ./data/gold/meta.json
  - ./data/gold/raw/...

Notes:
  - CORUM is downloaded via the CORUM FastAPI release archive.
  - ComplexPortal complexes are mapped from UniProt -> gene symbols via a pinned GOA Human GAF snapshot.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import requests


PAPER_V3_DIR = Path(__file__).resolve().parent.parent
CODESEG_DIR = PAPER_V3_DIR.parent


@dataclass(frozen=True)
class GoldMeta:
    corum_version: str
    corum_kind: str
    corum_source_url: str
    corum_archive_path: str
    goa_id: str
    goa_source_url: str
    goa_path: str
    complexportal_tsv: str
    outputs: dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch pinned CORUM/ComplexPortal gold standards for paper v3")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: ./data/gold)")
    p.add_argument(
        "--corum-version",
        type=str,
        default="2.2",
        help="CORUM release version from the CORUM release archive (e.g., 2.2, 3.0, 4.1).",
    )
    p.add_argument(
        "--corum-kind",
        choices=["all", "core", "human"],
        default="all",
        help="Which CORUM file to extract from the release (allComplexes/coreComplexes/humanComplexes when available).",
    )
    p.add_argument(
        "--goa-id",
        type=str,
        default="220",
        help="GOA Human archived snapshot id (e.g., 220 for 2023-12-04).",
    )
    p.add_argument(
        "--complexportal-tsv",
        type=str,
        default=str(CODESEG_DIR / "data" / "pathways" / "complex_portal" / "homo_sapiens.tsv"),
        help="Path to ComplexPortal homo_sapiens.tsv (participants as UniProt accessions).",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return p.parse_args()


def _download(url: str, dest: Path, *, force: bool) -> None:
    if dest.exists() and not force:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.exceptions.SSLError:
        # Some environments (notably WSL/conda setups) may lack CA roots for specific hosts.
        # We conservatively fall back to verify=False so the reconstruction can proceed.
        with requests.get(url, stream=True, timeout=180, verify=False) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)


def _corum_archive_url(version: str) -> str:
    return f"https://mips.helmholtz-muenchen.de/fastapi-corum/public/file/download_archived_file?version={version}"


def _goa_old_url(goa_id: str) -> str:
    # EBI GOA archive (numbered snapshots).
    return f"https://ftp.ebi.ac.uk/pub/databases/GO/goa/old/HUMAN/goa_human.gaf.{goa_id}.gz"


def _iter_tab_rows_from_corum_release(archive_path: Path, kind: str) -> Iterable[dict[str, str]]:
    """
    Yields tab3-style CORUM rows from a CORUM release archive zip.

    Handles:
      - plain *.txt
      - plain *.csv (typically ';' delimited)
      - nested *.txt.zip inside the release archive
      - nested *.csv.zip inside the release archive
    """
    kind = str(kind)

    def pick_member(names: list[str]) -> str:
        def is_data(name: str) -> bool:
            lname = name.lower()
            return ("psimi" not in lname) and (
                lname.endswith(".txt")
                or lname.endswith(".csv")
                or lname.endswith(".txt.zip")
                or lname.endswith(".csv.zip")
                or lname.endswith(".zip")
            )

        # Prefer direct (non-nested) text/csv, then nested.
        preferred_exts = [".txt", ".csv", ".txt.zip", ".csv.zip", ".zip"]

        def pick(pred) -> str | None:
            cand = [n for n in names if is_data(n) and pred(n.lower())]
            if not cand:
                return None
            # Stable pick order: by preferred extension, then lexical.
            ordered: list[str] = []
            for ext in preferred_exts:
                ordered.extend(sorted([n for n in cand if n.lower().endswith(ext)]))
            return ordered[0] if ordered else None

        if kind == "human":
            got = pick(lambda ln: "humancomplexes" in ln)
            if got:
                return got

        if kind == "core":
            got = pick(lambda ln: ("corecomplexes" in ln) or ("allcomplexescore" in ln))
            if got:
                return got

        if kind == "all":
            got = pick(lambda ln: ("allcomplexes" in ln) and ("core" not in ln))
            if got:
                return got

        raise FileNotFoundError(f"Could not find CORUM member for kind={kind!r}; members={names}")

    with zipfile.ZipFile(archive_path) as zf:
        member = pick_member(zf.namelist())
        data = zf.read(member)

    if member.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as inner:
            data_names = [n for n in inner.namelist() if n.lower().endswith(".txt") or n.lower().endswith(".csv")]
            if not data_names:
                raise FileNotFoundError(f"No .txt/.csv found inside nested zip: {member}")
            inner_name = sorted(data_names)[0]
            with inner.open(inner_name) as raw:
                txt = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                # CORUM CSVs are often ';' delimited.
                delim = "\t"
                if inner_name.lower().endswith(".csv"):
                    head = txt.readline()
                    delim = ";" if head.count(";") >= head.count(",") else ","
                    txt.seek(0)
                reader = csv.DictReader(txt, delimiter=delim)
                if reader.fieldnames is None:
                    raise ValueError(f"Missing header in CORUM member: {member}:{inner_name}")
                for row in reader:
                    yield {k: (v if v is not None else "") for k, v in row.items()}
        return

    txt = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8", errors="replace", newline="")
    delim = "\t"
    if member.lower().endswith(".csv"):
        head = txt.readline()
        delim = ";" if head.count(";") >= head.count(",") else ","
        txt.seek(0)
    reader = csv.DictReader(txt, delimiter=delim)
    if reader.fieldnames is None:
        raise ValueError(f"Missing header in CORUM member: {member}")
    for row in reader:
        yield {k: (v if v is not None else "") for k, v in row.items()}


def _convert_corum_release_to_gmt(archive_path: Path, kind: str, out_gmt: Path, *, goa_gaf_gz: Path | None) -> int:
    out_gmt.parent.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_tab_rows_from_corum_release(archive_path, kind=kind))

    # Some older CORUM releases provide UniProt IDs but not gene names; map via GOA if needed.
    def row_get(row: dict[str, str], keys: list[str]) -> str:
        for k in keys:
            if k in row:
                return str(row.get(k) or "")
        # fall back: case-insensitive match
        lower_map = {str(k).lower(): str(k) for k in row.keys()}
        for k in keys:
            hit = lower_map.get(str(k).lower())
            if hit is not None:
                return str(row.get(hit) or "")
        return ""

    wanted_uniprots: set[str] = set()
    needs_goa = False
    for row in rows:
        org = row_get(row, ["organism", "Organism"]).strip().lower()
        if org and "human" not in org:
            continue
        gene_field = row_get(
            row,
            [
                "subunits_gene_name",
                "subunits(Gene name)",
                "Subunits(Gene name)",
                "subunits (Gene name)",
            ],
        ).strip()
        if gene_field:
            continue
        uni_field = row_get(
            row,
            [
                "subunits_uniprot_id",
                "subunits (UniProt IDs)",
                "subunits (UniProt IDs )",
                "subunits(UniProt IDs)",
                "subunits (UniProt ID)",
                "subunits_uniprot_ids",
            ],
        ).strip()
        if not uni_field:
            continue
        needs_goa = True
        for u in uni_field.replace(";", ",").split(","):
            u = u.strip()
            if u:
                wanted_uniprots.add(u)

    uniprot_to_symbol: dict[str, str] = {}
    if needs_goa:
        if goa_gaf_gz is None:
            raise ValueError("CORUM release lacks gene names; pass GOA GAF via --goa-id (or provide goa_gaf_gz)")
        import sys

        sys.path.insert(0, str((CODESEG_DIR / "src").resolve()))
        from complexportal_eval import build_uniprot_to_symbol_from_goa  # type: ignore

        uniprot_to_symbol = build_uniprot_to_symbol_from_goa(goa_gaf_gz, wanted=wanted_uniprots)

    n_written = 0
    with open(out_gmt, "w", encoding="utf-8") as out:
        for row in rows:
            # Prefer stable keys across CORUM versions.
            cid = row_get(row, ["complex_id", "ComplexID", "Complex id", "Complex id "]).strip()
            name = row_get(row, ["complex_name", "ComplexName", "Complex name"]).strip()
            org = row_get(row, ["organism", "Organism"]).strip().lower()
            if org and "human" not in org:
                continue

            gene_field = row_get(
                row,
                [
                    "subunits_gene_name",
                    "subunits(Gene name)",
                    "Subunits(Gene name)",
                    "subunits (Gene name)",
                ],
            ).strip()
            genes: list[str] = []
            if gene_field:
                genes = [g.strip() for g in str(gene_field).replace(",", ";").split(";") if g.strip()]
            else:
                uni_field = row_get(
                    row,
                    [
                        "subunits_uniprot_id",
                        "subunits (UniProt IDs)",
                        "subunits (UniProt IDs )",
                        "subunits(UniProt IDs)",
                        "subunits (UniProt ID)",
                        "subunits_uniprot_ids",
                    ],
                ).strip()
                unis = [u.strip() for u in str(uni_field).replace(";", ",").split(",") if u.strip()]
                genes = [uniprot_to_symbol.get(u, "") for u in unis]
                genes = [g for g in genes if g]

            if not cid or len(genes) < 2:
                continue
            genes = sorted(set(genes))
            out.write(f"{cid}\t{name}\t" + "\t".join(genes) + "\n")
            n_written += 1

    return int(n_written)


def _write_complexportal_gmt(complexportal_tsv: Path, goa_gaf_gz: Path, out_gmt: Path) -> int:
    import sys

    sys.path.insert(0, str((CODESEG_DIR / "src").resolve()))
    from complexportal_eval import (  # type: ignore
        build_uniprot_to_symbol_from_goa,
        complexes_to_gene_sets,
        load_complexportal_uniprots,
    )

    if not complexportal_tsv.exists():
        raise FileNotFoundError(f"Missing ComplexPortal TSV: {complexportal_tsv}")
    if not goa_gaf_gz.exists():
        raise FileNotFoundError(f"Missing GOA GAF: {goa_gaf_gz}")

    cp_uniprots = load_complexportal_uniprots(complexportal_tsv)
    wanted = {u for _, (_name, unis) in cp_uniprots.items() for u in unis}
    uniprot_to_symbol = build_uniprot_to_symbol_from_goa(goa_gaf_gz, wanted=wanted)
    cp_genes, _stats = complexes_to_gene_sets(cp_uniprots, uniprot_to_symbol=uniprot_to_symbol)

    out_gmt.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_gmt, "w", encoding="utf-8") as out:
        for cid, genes in sorted(cp_genes.items()):
            genes = sorted({g for g in genes if g})
            if len(genes) < 2:
                continue
            out.write(f"{cid}\t-\t" + "\t".join(genes) + "\n")
            n_written += 1
    return int(n_written)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (PAPER_V3_DIR / "data" / "gold")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    corum_version = str(args.corum_version)
    corum_kind = str(args.corum_kind)
    goa_id = str(args.goa_id)

    # Download CORUM release archive.
    corum_url = _corum_archive_url(corum_version)
    corum_zip = raw_dir / f"corum_release_{corum_version}.zip"
    _download(corum_url, corum_zip, force=bool(args.force))

    # Download GOA snapshot.
    goa_url = _goa_old_url(goa_id)
    goa_gz = raw_dir / f"goa_human.gaf.{goa_id}.gz"
    _download(goa_url, goa_gz, force=bool(args.force))

    # Sanity check gzip.
    with gzip.open(goa_gz, "rt", encoding="utf-8", errors="replace") as f:
        _ = f.readline()

    # Build GMTs.
    corum_gmt = out_dir / "corum.gmt"
    complexportal_gmt = out_dir / "complexportal.gmt"

    if corum_gmt.exists() and not args.force:
        corum_n = sum(1 for _ in open(corum_gmt, "r", encoding="utf-8"))
    else:
        corum_n = _convert_corum_release_to_gmt(corum_zip, kind=corum_kind, out_gmt=corum_gmt, goa_gaf_gz=goa_gz)

    if complexportal_gmt.exists() and not args.force:
        cp_n = sum(1 for _ in open(complexportal_gmt, "r", encoding="utf-8"))
    else:
        cp_n = _write_complexportal_gmt(Path(args.complexportal_tsv), goa_gaf_gz=goa_gz, out_gmt=complexportal_gmt)

    meta = GoldMeta(
        corum_version=corum_version,
        corum_kind=corum_kind,
        corum_source_url=corum_url,
        corum_archive_path=str(corum_zip),
        goa_id=goa_id,
        goa_source_url=goa_url,
        goa_path=str(goa_gz),
        complexportal_tsv=str(Path(args.complexportal_tsv)),
        outputs={"corum_gmt": str(corum_gmt), "complexportal_gmt": str(complexportal_gmt)},
    )
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(asdict(meta), indent=2, sort_keys=True))

    print(f"Wrote: {corum_gmt} ({corum_n} complexes)")
    print(f"Wrote: {complexportal_gmt} ({cp_n} complexes)")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
