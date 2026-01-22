# SoftBlock (ISMB 2026)

**SoftBlock: Transferable Soft Blocks for Overlapping Protein Complex Recovery under a Frozen Protocol**

## Repository Structure

```
├── paper/              # Manuscript source (Overleaf-ready)
├── supplementary/      # Supplementary material source
├── src/                # Reproducibility code
├── data/               # Data snapshots (gold standards, etc.)
├── results/            # Cached results for frozen-protocol evaluation
├── Makefile            # Reproduce main tables
├── requirements.txt    # Python dependencies
└── requirements-freeze.txt
```

## Quick Start

**Compile paper locally:**
```bash
cd paper && latexmk -pdf main.tex
```

**Compile supplement locally:**
```bash
cd supplementary && latexmk -pdf supplement.tex
```

**Reproduce the main operating-point tables:**
```bash
make operating-point
# Or with custom Python: make operating-point PY=python
```

## Data Note

Large raw PPI downloads are not included. The preprocessing scripts in `src/` support `--download` to fetch from original sources where applicable.

