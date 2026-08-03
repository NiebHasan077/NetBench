# NetBench — Master Results (auto-generated)

Source: consolidated GPT-5.1 judged workbooks under `NetBench-LLM/outputs/by_model/`. Regenerate with `analysis/.venv/bin/python analysis/aggregate_scores.py`. Protocol: see `docs/STUDY_DESIGN.md` (statistical analysis protocol).

Coverage: **11 models**, **63 model-variants**, **233 questions** (judge `gpt-5.1`, HPN v5.0). Settings: **closed-book** = P1–P5, **open-book (RAG)** = P6–P8.

Metric note: this report and `significance*.csv` use the judge-reported `overall` (IEEE `paper/`). The EACL paper uses the deterministic `overall_formula` (`significance_formula*.csv`); see `docs/JUDGE_OVERALL_AUDIT.md`.


## Overall score by model × variant (P-taxonomy)

Closed-book: P1–P5 · Open-book/RAG: P6–P8 · BASE-API = frontier baselines (closed-book). Cells for the provisional variants listed above are not yet reliable.

| model             | size         |     P1 |     P2 |     P3 |     P4 |     P5 |     P6 |     P7 |     P8 |   BASE-API |
|:------------------|:-------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-----------:|
| llama-3.2-1b      | small        |   2.51 |   2.8  |   2.76 |   2.85 |   2.89 |   2.99 |   3.04 |   3.14 |     nan    |
| gemma-3-1b        | small        |   2.53 |   2.71 |   2.72 |   2.6  |   2.82 |   3.08 |   2.89 |   2.95 |     nan    |
| qwen3.5-2b        | small        |   2.71 |   3.3  |   3.36 |   3.5  |   3.41 |   3.86 |   3.82 |   3.92 |     nan    |
| llama-3.1-8b      | medium       |   3.34 |   3.58 |   3.47 |   3.59 |   3.59 |   3.89 |   4.04 |   4.04 |     nan    |
| qwen3.5-9b        | medium       |   3.69 |   3.75 |   3.74 |   3.84 |   3.83 |   4.22 |   4.21 |   4.17 |     nan    |
| gemma-3-12b       | medium       |   3.62 |   3.54 |   3.56 |   3.57 |   3.57 |   4.26 |   4.04 |   4.07 |     nan    |
| gemma-3-27b       | large        |   3.79 |   3.66 |   3.54 | nan    |   3.73 |   4.35 | nan    |   4.07 |     nan    |
| qwen3.5-27b       | large        |   4.06 |   4.04 |   4    | nan    |   4.09 |   4.38 | nan    |   4.36 |     nan    |
| gpt-4o            | api-baseline | nan    | nan    | nan    | nan    | nan    | nan    | nan    | nan    |       4.17 |
| gemini-2.5-pro    | api-baseline | nan    | nan    | nan    | nan    | nan    | nan    | nan    | nan    |       4.28 |
| claude-sonnet-4-6 | api-baseline | nan    | nan    | nan    | nan    | nan    | nan    | nan    | nan    |       4.29 |


## Completeness grid (which variants exist)

|              | P1   | P2   | P3   | P4   | P5   | P6   | P7   | P8   |
|:-------------|:-----|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
| llama-3.2-1b | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| gemma-3-1b   | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| qwen3.5-2b   | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| llama-3.1-8b | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| qwen3.5-9b   | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| gemma-3-12b  | yes  | yes  | yes  | yes  | yes  | yes  | yes  | yes  |
| gemma-3-27b  | yes  | yes  | yes  | —    | yes  | yes  | —    | yes  |
| qwen3.5-27b  | yes  | yes  | yes  | —    | yes  | yes  | —    | yes  |


## Primary significance (pre-registered family, BH-FDR pooled, q=0.05)

Positive `mean_diff` ⇒ second variant higher. `p_bh` = Benjamini–Hochberg adjusted; **`practically_sig`** requires |mean_diff| ≥ 0.25 *and* |d_z| ≥ 0.2 *and* BH-significant. Groups: closed_book / open_book (within-setting adaptation) and rag_effect (matched model ± retrieval).

| base_model   | group       | contrast   | description                             |   n |   mean_diff |   ci95_lo |   ci95_hi |    p_bh |   cohens_dz | practically_sig   | provisional   |
|:-------------|:------------|:-----------|:----------------------------------------|----:|------------:|----------:|----------:|--------:|------------:|:------------------|:--------------|
| llama-3.2-1b | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |       0.245 |     0.157 |     0.332 | 0       |       0.359 | False             | False         |
| llama-3.2-1b | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |       0.34  |     0.243 |     0.435 | 0       |       0.454 | True              | False         |
| llama-3.2-1b | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |       0.095 |     0.003 |     0.188 | 0.09824 |       0.133 | False             | False         |
| llama-3.2-1b | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.088 |     0.004 |     0.171 | 0.05575 |       0.134 | False             | False         |
| llama-3.2-1b | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.379 |     0.276 |     0.48  | 0       |       0.48  | True              | False         |
| llama-3.2-1b | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |       0.051 |    -0.07  |     0.172 | 0.31902 |       0.054 | False             | False         |
| llama-3.2-1b | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |       0.15  |     0.047 |     0.254 | 0.00791 |       0.187 | False             | False         |
| llama-3.2-1b | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.479 |     0.371 |     0.585 | 0       |       0.578 | True              | False         |
| llama-3.2-1b | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.19  |     0.062 |     0.317 | 0.00791 |       0.191 | False             | False         |
| llama-3.2-1b | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.25  |     0.137 |     0.364 | 0.00024 |       0.28  | True              | False         |
| gemma-3-1b   | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |       0.188 |     0.11  |     0.266 | 1e-05   |       0.306 | False             | False         |
| gemma-3-1b   | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |       0.076 |    -0.011 |     0.162 | 0.12114 |       0.111 | False             | False         |
| gemma-3-1b   | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |      -0.112 |    -0.204 |    -0.021 | 0.03076 |      -0.157 | False             | False         |
| gemma-3-1b   | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.119 |     0.025 |     0.213 | 0.01171 |       0.164 | False             | False         |
| gemma-3-1b   | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.297 |     0.206 |     0.388 | 0       |       0.422 | True              | False         |
| gemma-3-1b   | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |      -0.185 |    -0.298 |    -0.076 | 0.00738 |      -0.213 | False             | False         |
| gemma-3-1b   | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |      -0.127 |    -0.234 |    -0.018 | 0.05103 |      -0.15  | False             | False         |
| gemma-3-1b   | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.549 |     0.433 |     0.666 | 0       |       0.606 | True              | False         |
| gemma-3-1b   | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.289 |     0.182 |     0.4   | 1e-05   |       0.336 | True              | False         |
| gemma-3-1b   | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.126 |     0.015 |     0.239 | 0.0912  |       0.144 | False             | False         |
| qwen3.5-2b   | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |       0.651 |     0.537 |     0.762 | 0       |       0.745 | True              | False         |
| qwen3.5-2b   | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |       0.786 |     0.688 |     0.882 | 0       |       1.029 | True              | False         |
| qwen3.5-2b   | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |       0.135 |     0.036 |     0.237 | 0.0141  |       0.172 | False             | False         |
| qwen3.5-2b   | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.108 |     0.001 |     0.217 | 0.05103 |       0.129 | False             | False         |
| qwen3.5-2b   | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.693 |     0.582 |     0.802 | 0       |       0.809 | True              | False         |
| qwen3.5-2b   | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |      -0.036 |    -0.149 |     0.073 | 0.81132 |      -0.042 | False             | False         |
| qwen3.5-2b   | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |       0.065 |    -0.043 |     0.168 | 0.04899 |       0.079 | False             | False         |
| qwen3.5-2b   | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       1.144 |     1.022 |     1.263 | 0       |       1.214 | True              | False         |
| qwen3.5-2b   | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.322 |     0.186 |     0.455 | 0       |       0.307 | True              | False         |
| qwen3.5-2b   | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.516 |     0.382 |     0.645 | 0       |       0.496 | True              | False         |
| llama-3.1-8b | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |       0.132 |     0.021 |     0.237 | 0.00545 |       0.154 | False             | False         |
| llama-3.1-8b | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |       0.25  |     0.136 |     0.365 | 1e-05   |       0.282 | True              | False         |
| llama-3.1-8b | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |       0.118 |     0.01  |     0.229 | 0.02567 |       0.138 | False             | False         |
| llama-3.1-8b | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.002 |    -0.103 |     0.108 | 0.85279 |       0.003 | False             | False         |
| llama-3.1-8b | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.25  |     0.132 |     0.367 | 2e-05   |       0.275 | True              | False         |
| llama-3.1-8b | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |       0.153 |     0.073 |     0.233 | 0.00028 |       0.248 | False             | False         |
| llama-3.1-8b | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |       0.154 |     0.073 |     0.238 | 0.00022 |       0.242 | False             | False         |
| llama-3.1-8b | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.553 |     0.432 |     0.67  | 0       |       0.601 | True              | False         |
| llama-3.1-8b | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.456 |     0.323 |     0.588 | 0       |       0.447 | True              | False         |
| llama-3.1-8b | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.457 |     0.324 |     0.588 | 0       |       0.444 | True              | False         |
| qwen3.5-9b   | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |       0.05  |    -0.061 |     0.161 | 0.19272 |       0.058 | False             | False         |
| qwen3.5-9b   | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |       0.146 |     0.043 |     0.247 | 0.00162 |       0.181 | False             | False         |
| qwen3.5-9b   | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |       0.097 |     0.001 |     0.193 | 0.0912  |       0.127 | False             | False         |
| qwen3.5-9b   | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.08  |    -0.015 |     0.175 | 0.17059 |       0.108 | False             | False         |
| qwen3.5-9b   | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.134 |     0.026 |     0.241 | 0.00298 |       0.159 | False             | False         |
| qwen3.5-9b   | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |      -0.008 |    -0.081 |     0.067 | 0.74106 |      -0.013 | False             | False         |
| qwen3.5-9b   | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |      -0.047 |    -0.125 |     0.032 | 0.61334 |      -0.077 | False             | False         |
| qwen3.5-9b   | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.529 |     0.413 |     0.644 | 0       |       0.593 | True              | False         |
| qwen3.5-9b   | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.375 |     0.244 |     0.503 | 0       |       0.373 | True              | False         |
| qwen3.5-9b   | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.348 |     0.221 |     0.47  | 0       |       0.361 | True              | False         |
| gemma-3-12b  | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |      -0.058 |    -0.161 |     0.046 | 0.53771 |      -0.071 | False             | False         |
| gemma-3-12b  | closed_book | P1 vs P4   | Instruct vs full CPT+SFT                | 233 |      -0.048 |    -0.16  |     0.064 | 0.47615 |      -0.054 | False             | False         |
| gemma-3-12b  | closed_book | P3 vs P4   | full SFT vs full CPT+SFT                | 233 |       0.01  |    -0.096 |     0.118 | 0.96007 |       0.012 | False             | False         |
| gemma-3-12b  | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.034 |    -0.072 |     0.136 | 0.63845 |       0.042 | False             | False         |
| gemma-3-12b  | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |      -0.048 |    -0.161 |     0.062 | 0.74106 |      -0.055 | False             | False         |
| gemma-3-12b  | open_book   | P6 vs P7   | RAG-Instruct vs RAG-CPT+SFT             | 233 |      -0.227 |    -0.306 |    -0.147 | 0       |      -0.366 | False             | False         |
| gemma-3-12b  | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |      -0.196 |    -0.282 |    -0.112 | 1e-05   |      -0.295 | False             | False         |
| gemma-3-12b  | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.644 |     0.521 |     0.767 | 0       |       0.667 | True              | False         |
| gemma-3-12b  | rag_effect  | P4 vs P7   | add RAG to full CPT+SFT (matched model) | 233 |       0.465 |     0.337 |     0.595 | 0       |       0.465 | True              | False         |
| gemma-3-12b  | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.496 |     0.371 |     0.621 | 0       |       0.51  | True              | False         |
| gemma-3-27b  | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |      -0.25  |    -0.369 |    -0.139 | 0.00024 |      -0.281 | True              | False         |
| gemma-3-27b  | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.074 |    -0.034 |     0.178 | 0.22009 |       0.091 | False             | False         |
| gemma-3-27b  | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |      -0.053 |    -0.161 |     0.053 | 0.40425 |      -0.063 | False             | False         |
| gemma-3-27b  | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |      -0.286 |    -0.36  |    -0.213 | 0       |      -0.498 | True              | False         |
| gemma-3-27b  | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.567 |     0.439 |     0.692 | 0       |       0.577 | True              | False         |
| gemma-3-27b  | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.334 |     0.22  |     0.445 | 0       |       0.375 | True              | False         |
| qwen3.5-27b  | closed_book | P1 vs P3   | Instruct vs full SFT                    | 233 |      -0.063 |    -0.162 |     0.03  | 0.5339  |      -0.084 | False             | False         |
| qwen3.5-27b  | closed_book | P2 vs P5   | LoRA SFT vs LoRA CPT+SFT                | 233 |       0.049 |    -0.042 |     0.138 | 0.15992 |       0.069 | False             | False         |
| qwen3.5-27b  | closed_book | P1 vs P5   | Instruct vs LoRA CPT+SFT                | 233 |       0.034 |    -0.053 |     0.119 | 0.34163 |       0.051 | False             | False         |
| qwen3.5-27b  | open_book   | P6 vs P8   | RAG-Instruct vs RAG-LoRA-CPT+SFT        | 233 |      -0.017 |    -0.077 |     0.041 | 0.96007 |      -0.036 | False             | False         |
| qwen3.5-27b  | rag_effect  | P1 vs P6   | add RAG to Instruct (matched model)     | 233 |       0.316 |     0.22  |     0.412 | 0       |       0.429 | True              | False         |
| qwen3.5-27b  | rag_effect  | P5 vs P8   | add RAG to LoRA CPT+SFT (matched model) | 233 |       0.266 |     0.17  |     0.36  | 0       |       0.361 | True              | False         |


**Practically-significant primary contrasts: 30 / 72** (0 excluded as provisional/think-corrupted, not BH-tested). Many statistically-significant gaps still fail the practical bar — see the discussion of what counts as a meaningful improvement.


## Cross-setting (exploratory — information access, NOT model quality)

RAG variants retrieve source passages the closed-book models never see; these contrasts are reported only as an information-access comparison.

| base_model   | contrast   | description                                        |   mean_diff |   ci95_lo |   ci95_hi |   p_raw |
|:-------------|:-----------|:---------------------------------------------------|------------:|----------:|----------:|--------:|
| llama-3.2-1b | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.139 |     0.027 |     0.251 | 0.01832 |
| llama-3.2-1b | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.1   |    -0.008 |     0.211 | 0.1414  |
| gemma-3-1b   | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.473 |     0.361 |     0.589 | 0       |
| gemma-3-1b   | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.253 |     0.141 |     0.37  | 7e-05   |
| qwen3.5-2b   | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.358 |     0.237 |     0.477 | 0       |
| qwen3.5-2b   | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.451 |     0.327 |     0.576 | 0       |
| llama-3.1-8b | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.303 |     0.181 |     0.424 | 0       |
| llama-3.1-8b | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.303 |     0.181 |     0.42  | 0       |
| qwen3.5-9b   | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.383 |     0.255 |     0.506 | 0       |
| qwen3.5-9b   | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.395 |     0.27  |     0.514 | 0       |
| gemma-3-12b  | P4 vs P6   | best closed-adapted (CPT+SFT) vs Instruct+RAG      |       0.691 |     0.564 |     0.819 | 0       |
| gemma-3-12b  | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.692 |     0.566 |     0.816 | 0       |
| gemma-3-27b  | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.62  |     0.506 |     0.735 | 0       |
| qwen3.5-27b  | P5 vs P6   | best closed-adapted (LoRA CPT+SFT) vs Instruct+RAG |       0.282 |     0.189 |     0.375 | 0       |


## Stratified highlights (exploratory)

Per-stratum tests live in `significance_stratified.csv` (720 rows; strata with n<10 are skipped). 
Sign-flip strata (effect reverses vs the full-set result), 124 total — first 12:

- `gemma-3-12b` P1 vs P3: overall -0.06 but Bottleneck Diagnosis and End-to-End Reasoning +0.07
- `gemma-3-12b` P1 vs P3: overall -0.06 but Pipelining and Small-File Optimization +0.41
- `gemma-3-12b` P1 vs P3: overall -0.06 but Transfer Parameters: Definitions and Roles +0.05
- `gemma-3-12b` P1 vs P3: overall -0.06 but hard +0.14
- `gemma-3-12b` P1 vs P4: overall -0.05 but Bottleneck Diagnosis and End-to-End Reasoning +0.19
- `gemma-3-12b` P1 vs P4: overall -0.05 but Pipelining and Small-File Optimization +0.23
- `gemma-3-12b` P1 vs P4: overall -0.05 but Transfer Parameters: Definitions and Roles +0.01
- `gemma-3-12b` P1 vs P4: overall -0.05 but hard +0.10
- `gemma-3-12b` P1 vs P5: overall -0.05 but Bottleneck Diagnosis and End-to-End Reasoning +0.19
- `gemma-3-12b` P1 vs P5: overall -0.05 but Pipelining and Small-File Optimization +0.42
- `gemma-3-12b` P1 vs P5: overall -0.05 but Practical HPN Scenarios and Design +0.04
- `gemma-3-12b` P1 vs P5: overall -0.05 but Transfer Parameters: Definitions and Roles +0.08
