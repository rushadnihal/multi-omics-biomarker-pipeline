# One-command reproduction: `make reproduce` rebuilds every table and figure from the raw downloads.
# Added to the main Makefile with:  include reproduce.mk
# Long (about 1-2 hours on a laptop). caffeinate keeps macOS awake; override with CAF= to disable.
PY  ?= .venv/bin/python
CAF ?= caffeinate -i
RUN  = $(CAF) $(PY)

.PHONY: reproduce data cohort matrices baselines integration nonlinear internal-eval signature external summary

reproduce: data cohort matrices baselines integration nonlinear internal-eval signature external summary
	@echo "reproduce: all stages finished"

data:
	$(PY) scripts/01_download.py

cohort: data
	$(PY) scripts/02_cohort.py

matrices: cohort
	$(RUN) scripts/03c_build_matrices_qc.py
	$(RUN) scripts/03d_filter_meth_autosomal.py

baselines: matrices
	$(PY) scripts/04_clinical_baseline.py
	for l in rna mirna meth; do $(RUN) scripts/05_single_omics.py $$l || exit 1; done
	$(PY) scripts/06_compare_diagnostics.py

integration: baselines
	for l in rna mirna meth all; do $(RUN) scripts/07_clinical_plus_omics.py $$l || exit 1; done
	$(RUN) scripts/08_late_integration.py

nonlinear: integration
	$(RUN) scripts/09_nonlinear.py
	$(RUN) scripts/10_mofa.py

internal-eval: integration
	$(PY) scripts/11_m7a_discrimination.py
	$(PY) scripts/12_m7b_calibration.py

# regenerates the signature under a different file name and verifies it against the frozen one (never overwrites it)
signature: integration
	$(PY) scripts/13_shared_gene_pool.py
	$(RUN) scripts/14_combo_rna_shared.py
	$(PY) scripts/15_freeze_signature.py signature_v1.repro.json
	$(PY) scripts/18_verify_and_summarize.py verify

external: signature
	$(PY) scripts/16_prepare_external.py
	$(PY) scripts/17_external_validation.py --rerun

summary: external nonlinear internal-eval
	$(PY) scripts/18_verify_and_summarize.py summary
