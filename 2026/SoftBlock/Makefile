# SoftBlock reproducibility (reconstruction)
#
# Default assumes the RAPIDS environment is available.
# Override, e.g.:  make operating-point PY="python"

PY ?= conda run -n rapids python
SEED ?= 42
SEEDS ?= 42 43 44

.PHONY: operating-point operating-point-sensitivity node-overlap case-studies runtime-scalability pcgan-head-to-head-ci
.PHONY: operating-point-learned
.PHONY: verify-snapshots
.PHONY: multiseed

operating-point:
	$(PY) src/reproduce_paper_v3.py --seed $(SEED)

operating-point-learned:
	$(PY) src/reproduce_paper_v3.py --seed $(SEED) --rerank-mode learned

verify-snapshots:
	$(PY) src/verify_snapshots.py --gold-snapshot paper_v3 --strict

multiseed:
	$(PY) src/reproduce_multiseed_v3.py --seeds $(SEEDS)

# The following targets are referenced by the v3 PDF but are not yet reconstructed.
operating-point-sensitivity:
	@echo "TODO: reconstruct operating-point sensitivity (Appendix Table 18 / Figure 1)."
	@exit 1

node-overlap:
	@echo "TODO: reconstruct node-overlap analysis (Table 21 / Figure 4)."
	@exit 1

case-studies:
	@echo "TODO: reconstruct GO-slim case studies (Appendix Table 20)."
	@exit 1

runtime-scalability:
	@echo "TODO: reconstruct runtime/scalability table (Appendix Table 19)."
	@exit 1

pcgan-head-to-head-ci:
	@echo "TODO: reconstruct PCGAN head-to-head CI experiments (Table 5/6)."
	@exit 1
