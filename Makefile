# NetBench — reproduce every reported number from the committed evidence.
#
#   make setup       create the analysis environment from requirements.lock
#   make reproduce   regenerate all results and assets, then verify they match
#
# `make reproduce` is the claim this repository makes: given the judged
# workbooks under NetBench-LLM/outputs/ and NetBench-RAG/outputs/, every CSV,
# table, and figure behind the paper is re-derived by these scripts. It exits
# non-zero if anything comes back different.
#
# Override the interpreter if you are not using the project venv:
#   make reproduce PYTHON=python3

PYTHON ?= analysis/.venv/bin/python
OUT    := analysis/outputs
TMP    := .verify

# Absolute form, so recipes that change directory still resolve it. A bare
# command name (e.g. `python3`) is left alone for PATH lookup.
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))

.DEFAULT_GOAL := help
.PHONY: help setup results assets reproduce verify audit audit-sft clean

help:
	@echo "NetBench targets:"
	@echo "  setup      build analysis/.venv from requirements.lock"
	@echo "  results    judged workbooks -> $(OUT)/*.csv, *.md, agreement workbooks"
	@echo "  assets     results -> $(OUT)/tables/*.tex and $(OUT)/figures/*.pdf"
	@echo "  verify     assert the regenerated outputs match the committed ones"
	@echo "  reproduce  results + assets + verify   <- the one that matters"
	@echo "  audit      corpus-dependent audits; needs CORPUS=/path/to/corpus.json"
	@echo "  audit-sft  SFT-vs-benchmark overlap audit; needs SFT=/path/to/v3_run_plus_json"

setup:
	python3 -m venv analysis/.venv
	analysis/.venv/bin/pip install --upgrade pip
	analysis/.venv/bin/pip install -r requirements.lock

# Committed judged workbooks -> the aggregated results of record.
#
# The chain starts at the judged workbooks and the merged human rating CSVs.
# Everything upstream of those is model inference or human data entry, not
# analysis, and is not re-run here: regenerating answers needs the model
# weights, and re-merging ratings needs nothing but reproduces its own inputs.
# Those steps live in NetBench-LLM/evaluation/ and analysis/human_eval/.
#
# Order matters once: judge_vendor_overlap reads the agreement workbook that
# judge_agreement writes.
results:
	$(PY) analysis/aggregate_scores.py
	$(PY) analysis/supplementary_contrasts.py
	$(PY) analysis/threshold_sensitivity.py
	$(PY) analysis/difficulty_validation.py
	cd NetBench-LLM && $(PY) evaluation/judge_agreement.py \
	    --excluded ../analysis/excluded_items.csv \
	    --overall_field overall --output_dir ../$(OUT)
	cd NetBench-LLM && $(PY) evaluation/judge_agreement.py \
	    --excluded ../analysis/excluded_items.csv \
	    --overall_field overall_formula --output_dir ../$(OUT)
	$(PY) analysis/judge_vendor_overlap.py
	$(PY) analysis/human_eval/human_judge_agreement.py
# Added in v1.1.0: source-paper clustering, API equivalence tests, the scale
# trend, the practical-threshold anchor, serving cost, and the contamination
# check. The first four read scores_long.csv and so must follow
# aggregate_scores; threshold_anchor and deployment_cost read the judged
# workbooks and profiling JSONs directly.
	$(PY) analysis/clustered_uncertainty.py
	$(PY) analysis/equivalence_tests.py
	$(PY) analysis/scale_trend.py
	$(PY) analysis/threshold_anchor.py
	$(PY) analysis/deployment_cost.py
	$(PY) analysis/contamination.py
# Per-system CIs on the composite (reads scores_long.csv) and intervals for
# the two-judge agreement (reads the judged workbooks through
# judge_agreement.py's loader).
	$(PY) analysis/system_cis.py
	$(PY) analysis/judge_agreement_ci.py
# analysis/retrieval_recall.py is deliberately NOT here: it needs the RAG
# chunk cache (135 MB), which ARTIFACTS.md excludes from distribution, so it
# cannot run on a fresh clone. Its CSVs are committed like every other number
# and regenerate only where the index exists.

# The paper's tables and figures, emitted next to the CSVs they came from.
assets:
	$(PY) analysis/make_tables.py  --paper eacl --outdir $(OUT)/tables
	$(PY) analysis/make_figures.py --paper eacl --outdir $(OUT)/figures

reproduce: results assets verify

# CSV, Markdown, and LaTeX must be byte-identical. PDF and XLSX embed a
# creation timestamp, so those are compared by content (tools/compare_outputs.py).
verify:
	@echo "== exact: CSV, Markdown, LaTeX =="
	@git diff --quiet -- $(OUT)/*.csv $(OUT)/*.md $(OUT)/tables || { \
	  echo "FAIL: exactly-reproducible outputs changed"; \
	  git --no-pager diff -- $(OUT)/*.csv $(OUT)/*.md $(OUT)/tables; exit 1; }
	@echo "   unchanged"
	@rm -rf $(TMP) && mkdir -p $(TMP)
	@git archive HEAD $(OUT) | tar -x -C $(TMP)
	@echo "== by content: agreement workbooks =="
	@$(PY) tools/compare_outputs.py $(OUT) $(TMP)/$(OUT) --glob '*.xlsx'
	@echo "== by content: figures =="
	@$(PY) tools/compare_outputs.py $(OUT)/figures $(TMP)/$(OUT)/figures --glob '*.pdf'
	@rm -rf $(TMP)

# Audits that read the source corpus. The corpus is not redistributed, so these
# ship as committed CSVs and are re-runnable only where a copy is available.
audit:
	@test -n "$(CORPUS)" || { echo "usage: make audit CORPUS=/path/to/research_corpus_v3.json"; exit 2; }
	$(PY) tools/audit_evidence_quotes.py --corpus $(CORPUS)
	$(PY) tools/annotate_benchmark_release.py --dry-run

# The final SFT records are not redistributed either; the committed
# sft_overlap_audit.csv is the record, re-runnable only where a copy exists.
audit-sft:
	@test -n "$(SFT)" || { echo "usage: make audit-sft SFT=/path/to/Instruct-FTD/v3_run_plus_json"; exit 2; }
	$(PY) tools/audit_sft_overlap.py --sft $(SFT)

clean:
	rm -rf $(TMP) $(OUT)/tables $(OUT)/figures
