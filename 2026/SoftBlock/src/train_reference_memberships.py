"""
Train SoftBlock reference soft memberships on STRING.

This is a thin wrapper that reuses the recovered DGI+GCN trainer from paper v2
(`codeseg/paper_v2/src/train_soft_membership_dgi_gcn.py`) and writes v3-named
artifacts under `./results/cache/`.

Example:
  conda run -n rapids python ./src/train_reference_memberships.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PAPER_V3_DIR = Path(__file__).resolve().parent.parent
CODESEG_DIR = PAPER_V3_DIR.parent
TRAINER = CODESEG_DIR / "paper_v2" / "src" / "train_soft_membership_dgi_gcn.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train v3 reference memberships (STRING)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--Ks", type=int, nargs="+", default=[6, 8, 16])
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--objective", choices=["dgi", "modularity", "hybrid"], default="dgi")
    p.add_argument("--dgi-weight", type=float, default=1.0)
    p.add_argument("--modularity-weight", type=float, default=0.0)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--svd-components", type=int, default=64)
    p.add_argument("--cluster-temp", type=float, default=30.0)
    p.add_argument("--num-cluster-iter", type=int, default=1)
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if not TRAINER.exists():
        raise FileNotFoundError(f"Missing trainer script: {TRAINER}")

    out_dir = Path(args.out_dir) if args.out_dir else (PAPER_V3_DIR / "results" / "cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    for K in [int(k) for k in args.Ks]:
        out_npz = out_dir / f"codeseg_soft_membership_string_seed{int(args.seed)}_K{int(K)}.npz"
        out_meta = out_npz.with_suffix(".meta.json")
        if out_npz.exists() and not args.force:
            ok = False
            if out_meta.exists():
                try:
                    meta = json.loads(out_meta.read_text(encoding="utf-8"))
                    ok = (
                        int(meta.get("seed", -1)) == int(args.seed)
                        and int(meta.get("K", -1)) == int(K)
                        and int(meta.get("epochs", -1)) == int(args.epochs)
                        and str(meta.get("objective", "")) == str(args.objective)
                        and abs(float(meta.get("dgi_weight", -999.0)) - float(args.dgi_weight)) < 1e-9
                        and abs(float(meta.get("modularity_weight", -999.0)) - float(args.modularity_weight)) < 1e-9
                        and int(meta.get("hidden_dim", -1)) == int(args.hidden_dim)
                        and int(meta.get("svd_components", -1)) == int(args.svd_components)
                        and int(meta.get("num_cluster_iter", -1)) == int(args.num_cluster_iter)
                        and abs(float(meta.get("cluster_temp", -999.0)) - float(args.cluster_temp)) < 1e-9
                    )
                except Exception:
                    ok = False
            if ok:
                print(f"✓ exists (matching meta): {out_npz}")
                continue
            print(f"↻ retrain (meta mismatch): {out_npz}")

        _run(
            [
                sys.executable,
                str(TRAINER),
                "--graph",
                "STRING",
                "--seed",
                str(int(args.seed)),
                "--K",
                str(int(K)),
                "--topk-prototypes",
                str(int(K)),
                "--prototype-algo",
                "louvain",
                "--prototype-resolution",
                "1.0",
                "--svd-components",
                str(int(args.svd_components)),
                "--hidden-dim",
                str(int(args.hidden_dim)),
                "--cluster-temp",
                str(float(args.cluster_temp)),
                "--num-cluster-iter",
                str(int(args.num_cluster_iter)),
                "--objective",
                str(args.objective),
                "--dgi-weight",
                str(float(args.dgi_weight)),
                "--modularity-weight",
                str(float(args.modularity_weight)),
                "--epochs",
                str(int(args.epochs)),
                "--out-npz",
                str(out_npz),
                "--out-meta",
                str(out_meta),
            ]
        )

        if not out_npz.exists():
            raise RuntimeError(f"Expected output missing: {out_npz}")


if __name__ == "__main__":
    main()
