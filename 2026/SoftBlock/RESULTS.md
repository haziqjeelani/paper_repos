# SoftBlock (v3) — Results (Reconstruction Status)

This workspace now contains a runnable, end-to-end reconstruction of the paper v3 pipeline:

- Frozen protocol runner: `paper_v3/src/frozen_protocol.py`
- Repro driver (Table 2/3 CSVs): `paper_v3/src/reproduce_paper_v3.py`
- Optional learned reranker (STRING-trained): `paper_v3/src/train_reranker.py`

## How To Reproduce

From repo root (RAPIDS env):

- Graph-only (paper headline style): `make -C paper_v3 operating-point SEED=42`
- Learned reranker variant: `make -C paper_v3 operating-point-learned SEED=42`
- Snapshot parity check (Table 1 targets): `conda run -n rapids python paper_v3/src/verify_snapshots.py --gold-snapshot paper_v3`

### Paper-benchmark (pinned gold snapshot)

This repo supports a pinned “gold snapshot” mode (CORUM/ComplexPortal) to avoid mutable upstream mapping:

```bash
conda run -n rapids python paper_v3/src/fetch_gold_standards.py --force
conda run -n rapids python paper_v3/src/reproduce_paper_v3.py \
  --seed 42 --skip-training \
  --rerank-mode graph_only --rerank-score weighted_density \
  --gold-snapshot paper_v3 \
  --tag v3_paperbench --force
```

Outputs:
- `paper_v3/results/tables/table2_v3_paperbench_seed42.csv`
- `paper_v3/results/tables/table3_v3_paperbench_seed42.csv`

### Paper-benchmark (graph-only, stronger density variant)

An additional graph-only reranker option is implemented for experimentation:

```bash
conda run -n rapids python paper_v3/src/reproduce_paper_v3.py \
  --seed 42 --skip-training \
  --rerank-mode graph_only --rerank-score weighted_density_n2 \
  --gold-snapshot paper_v3 \
  --tag v3_paperbench_wdn2 --force
```

Outputs:
- `paper_v3/results/tables/table2_v3_paperbench_wdn2_seed42.csv`
- `paper_v3/results/tables/table3_v3_paperbench_wdn2_seed42.csv`

### Multi-seed (42/43/44)

```bash
conda run -n rapids python paper_v3/src/reproduce_multiseed_v3.py \
  --seeds 42 43 44 \
  --gold-snapshot paper_v3 \
  --tag v3_paperbench --out-tag v3_paperbench
```

Outputs:
- `paper_v3/results/tables/table2_v3_paperbench_seeds42_43_44_meanstd.csv`
- `paper_v3/results/tables/table3_v3_paperbench_seeds42_43_44_meanstd.csv`

Artifacts are written under `paper_v3/results/`:

- Per-graph clusters: `paper_v3/results/frozen_protocol/<graph>/<tag>_<...>_{pool,op}.tsv`
- Per-graph eval: `paper_v3/results/frozen_protocol/<graph>/<tag>_<...>.eval.csv`
- Summaries: `paper_v3/results/tables/table2_<tag>_seed42.csv`, `paper_v3/results/tables/table3_<tag>_seed42.csv`

## Current Tables (Seed 42)

- Pinned-gold paperbench (closest to PDF framing): `paper_v3/results/tables/table3_v3_paperbench_seed42.csv`
- Pinned-gold paperbench (stronger density): `paper_v3/results/tables/table3_v3_paperbench_wdn2_seed42.csv`
- Pinned-gold paperbench (large-K ref + auto scorer, experimental): `paper_v3/results/tables/table3_v3_paperbench_ref_dgi_largeK_auto_seed42.csv`

## Status vs Paper v3 (High-Level)

- Under the pinned snapshot, `ComPPI` and `IntAct` meet/beat most PDF headline numbers; the largest remaining gap is HuRI OS `Acc` (low `PPV` at N=2000).
- `weighted_density_n2` improves most weighted graphs, but does not move HuRI.
- Large-K ref + `rerank-score auto` improves HuRI best-match/MMR, but can hurt ComPPI depending on the reference-membership training run.

## Notes / Known Mismatches vs PDF

- BioGRID preprocessing: paper Table 1 matches the “Homo sapiens member” when keeping edges where *either* endpoint is taxid=9606 and `system-type=all`; see `paper_v3/src/preprocess_biogrid_weighted.py`.
- Snapshot parity (Table 1 counts) is still not exact under the current pinned gold; see `paper_v3/results/snapshots/paper_v3_snapshot_check.json`.
- Some transfer graphs depend on external dataset versions/filters (ComPPI in particular); preprocessing scripts live in `paper_v3/src/`.
