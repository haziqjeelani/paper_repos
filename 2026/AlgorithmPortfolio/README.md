# Algorithm-Portfolio Decision Support for Network Immunization

Benchmarking pre-epidemic vaccination policies under compute constraints with holdout evaluation on real contact networks.

## Quickstart (reproducible baseline)
1. Install deps (Mesa is optional; the baseline runner uses NetworkX + NumPy):
   - `pip install -r requirements.txt`
2. Run the baseline suite:
   - `python src/run_experiments.py --config src/config_baseline.yaml`
3. Open the latest outputs:
   - `cat results/LATEST_RUN.txt`

Artifacts are written under `results/<experiment>_<timestamp>/` (tables in `tables/`, figures in `figures/`).

## Quickstart (intervention benchmark)
Benchmarks pre-epidemic vaccination policies (effectiveness + compute) on BA vs ER graphs.

1. Run the intervention suite:
   - `python src/run_intervention_benchmarks.py --config src/config_interventions.yaml`
2. Open the latest outputs:
   - `cat results/LATEST_INTERVENTIONS.txt`

## Quickstart (compute-optimal + holdout evaluation)
Includes a lightweight learned policy baseline and evaluates on a SocioPatterns network (if downloaded).

1. (Optional) Download a SocioPatterns dataset:
   - See `DATASETS.md`
2. Run:
   - `python src/run_intervention_benchmarks.py --config src/config_compute_optimal.yaml`

## Draft manuscript (LaTeX)
- Build: `cd paper && ./build.sh`
- Output: `paper/build/main.pdf`

Paper assets (figures/tables) correspond to the run recorded in `paper/figures/SOURCE_RUN.txt`.

## One-command paper reproduction (KBS)
To regenerate the core results, refresh `paper/figures/` + `paper/tables/`, and rebuild the PDF:

- `./reproduce_kbs.sh`

This runs the main cost-aware suite plus the small scaling and robustness sweeps used in the appendix. SocioPatterns networks are optional; see `DATASETS.md`.


AlgorithmPortfolio/
├── README.md                    
├── DATASETS.md                  # Dataset download instructions
├── requirements.txt             # dependencies
├── reproduce_kbs.sh            # ONE-COMMAND reproduction script
├── .gitignore                   # 
│
├── src/                         # ALL source code
│   ├── run_experiments.py
│   ├── run_intervention_benchmarks.py
│   ├── export_paper_assets.py
│   ├── scale_ranking_time.py
│   ├── robustness_sweep.py
│   ├── policies.py
│   ├── sir.py
│   ├── graphs.py
│   ├── router.py
│   ├── learned_policy.py
│   ├── plotting.py
│   ├── metrics.py
│   ├── io_utils.py
│   ├── real_networks.py
│   ├── compute_effectiveness.py
│   ├── config_baseline.yaml
│   ├── config_interventions.yaml
│   ├── config_compute_optimal.yaml
│   └── config_transfer.yaml
│
├── paper/                       # LaTeX source + compiled PDF
│   ├── main.tex
│   ├── references.bib
│   ├── build.sh
│   ├── sections/               # All .tex sections
│   ├── tables/                 # All .tex tables
│   ├── figures/                # All .png figures + SOURCE_RUN.txt
│   ├── build/
│   │   └── main.pdf           # Final compiled PDF
│   └── elsarticle.cls         # Required LaTeX class
│
├── data/
│   ├── .gitkeep
│   └── raw/                    # Empty (users download via DATASETS.md)
│
└── results/                     # OPTIONAL outputs
    └── .gitkeep