# Benchmark-Generator — development guide

This module turns the 2371-paper HPN research corpus into the HPN-QA benchmark.
It collects the rules, conventions, and hard-won failure modes that are not
obvious from the code, and should be read before changing anything under `src/`.

## Read these first

- [PLAN.md](PLAN.md) — what the pipeline does (stages, schemas, prompts, splits)
- [DEV_PLAN.md](DEV_PLAN.md) — phase-by-phase build plan with status, exit criteria, decision log

DEV_PLAN.md is a **living document**. Update phase status (`[ ] [~] [x] [!] [s]`),
exit-criteria checkboxes, the decision log, and the daily log as work proceeds.
Do not "complete" a phase by rewriting the plan to match what got built — fix
the build or amend the plan with a logged reason.

## Hard rules

1. **No paid API calls for generation, validation, filtering, or calibration.**
   Ollama only. Paid APIs (GPT-4o, Gemini 2.5 Pro) appear *only* in Phase 9
   judge calibration.
2. **Deterministic validators come before LLM validators.** Phase 6 ordering
   is non-negotiable: schema → quote substring → memorization-leak → dedup →
   critic. The deterministic gates carry the load; the LLM critic is the
   smallest, last filter.
3. **Generator and judges must be different vendor families.** Generation =
   local Ollama (Qwen / Gemma). Judging = OpenAI + Google.
4. **The benchmark must never feed back into SFT or CPT data.** The corpus
   that built the model is in
   [`../NetBench-LLM/data/raw/research_corpus_v3.json`](../NetBench-LLM/data/raw/research_corpus_v3.json);
   the benchmark generator reads it but produces output that lives only under
   `benchmark/`.
5. **Every stage runner supports `--resume`.** Re-running after a crash must
   skip already-processed inputs. Append-only JSONL with `flush()` per line.

## Anti-patterns to avoid

- Don't add MCQ generation to v1.
- Don't add multi-agent rewriter loops; one rewrite attempt max in Phase 6.
- Don't manually pick papers — Phase 2's filter is fully automatic by design.
- Don't generate SFT data from this pipeline.
- Don't mix candidate questions across phases (5a/5b/5c) into one undifferentiated bucket — each split pulls from its own stream.
- Don't repair questions in the candidate-generation phases. Rejections happen in Phase 6.

## Code conventions

- One stage = one file under `src/`, named `stage_NN_name.py`.
- Each stage has a CLI: `python -m src.stage_NN_name --input X --output Y --resume`.
- All inter-stage data is JSONL under `data/`. Use `src/io_utils.py` for read/write.
- Schema lives in `src/schema.py` (pydantic). New fields are additive — do not break the v4-compatible question shape (see [PLAN.md §5](PLAN.md#5-final-question-schema-jsonl-line)).
- Prompts live in `prompts/<name>.txt`, loaded as plain strings. Track a `prompt_version` per phase (e.g. `v5.0-questions`) and stamp it on every output record.
- Logs to `data/_logs/<stage>.jsonl` with `{input_id, status, model, latency_ms, retries}`.
- No new top-level dependencies without updating `requirements.txt`.

## Useful external paths

| Purpose | Path |
|---|---|
| Source corpus | [`../NetBench-LLM/data/raw/research_corpus_v3.json`](../NetBench-LLM/data/raw/research_corpus_v3.json) |
| v4 benchmark (seed for relevance filter, format reference for generator) | [`../NetBench-LLM/data/prompts/hpn_qa_benchmark_v4_general_skills.json`](../NetBench-LLM/data/prompts/hpn_qa_benchmark_v4_general_skills.json) |
| Existing evaluator (consumes the JSONL we produce) | [`../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py`](../NetBench-LLM Direct Pipeline/evaluation/hpn_qa_benchmark.py) |
| Existing judge | [`../NetBench-LLM Direct Pipeline/evaluation/judge_responses.py`](../NetBench-LLM Direct Pipeline/evaluation/judge_responses.py) |
| OpenAI API key (Phase 9 only) | [`OPENAI_API_KEY`](OPENAI_API_KEY) |
| Gemini API key (Phase 9 only) | [`GEMINI_API_KEY`](GEMINI_API_KEY) |

## How to update DEV_PLAN.md

When finishing a phase:
1. Flip the top-of-phase box to `[x] done`.
2. Tick the exit-criteria checkboxes that actually held; if one didn't, leave it unchecked and add a note.
3. Append a one-line entry to the **Daily log** at the bottom.
4. If a decision was made (model swap, threshold change, scope cut), append to the **Decision log**.

When stuck:
- Flip the box to `[!] blocked` and write the blocker in the phase notes.
- Don't silently skip a phase. If skipping, use `[s]` and record the reason.

## Verification

- Phase 0 smoke test: `.venv/bin/python scripts/smoke_test.py` — runs 13 checks (config load, schema validation, io_utils round-trip, JSON extraction edge cases, real Ollama call to the configured `card_generator`). Run after touching anything in `src/` or `config.yaml`.
- Currently installed Ollama models on this machine: `gemma4:26b`, `qwen3.5:9b`, `llama3.2:3b`, `phi3:latest`. `qwen2.5:72b-instruct-q4_K_M` is **not** installed; flagged as optional upgrade.
- **Card generation uses `qwen3.5:9b` via `/api/chat` with `think=False`** — NOT `gemma4:26b`. gemma4:26b produces token-repetition loops (unusable). qwen3.5:9b requires the chat endpoint (generate returns empty string). See `OllamaClient.chat_json()` and `stage_03_cards.py`.
- **Stage 05a/05b use `gemma4:26b`** for question/synthesis generation — acceptable for shorter structured outputs (~2-4 KB). ~5-15% parse-failure rate on specific paper-content combinations; persistent failures are dropped (not retried infinitely). Evidence field-name hallucinations (`passage_lag_id` etc.) are normalized in `process_card()`.
- **`Literal[...]` enum fields in Pydantic schemas cause parse failures** when passed to Ollama's schema enforcement — the model emits unquoted enum values. Use `str` type + post-parse validation instead of `Literal` for optional enum fields.
- **Fields with `default` annotations in Pydantic schemas cause parse failures** — same root cause. Remove the field from the schema and hard-code the value in the validator instead.
- **gemma4:26b writes `"fieldname: "value"` (colon inside key name)** for any field whose name reads as a prose citation word (e.g. `quote`, `excerpt`). Fix: rename the field to a compound noun (e.g. `passage_text`) OR apply a pre-parse string fixup `raw.replace('"fieldname: "', '"fieldname": "')` before calling `json.loads`.
- **gemma4:26b paraphrases evidence quotes despite "verbatim" instructions.** 90% of synthesis and 44% of adversarial candidates had non-verbatim quotes after Phase 5b/5c. Fix: run `scripts/repair_evidence_quotes.py` before validation to replace paraphrased quotes with best-matching verbatim excerpts (Jaccard word-window scan). Not all quotes are repairable (11% unrepairable — fully fabricated).
- **gemma4:26b enters thinking mode silently, returning empty `message.content`.** When gemma4:26b is used for judge or answer generation, it can route all output to `message.thinking` instead of `message.content`, resulting in an empty string from `client.chat()`. This happens for complex/longer inputs. Fix: always pass `think=False` for gemma4 models. `_think_param()` in `stage_07_calibrate.py` handles this — it returns `False` for both `qwen3.*` and `gemma4.*` models.

## Ollama endpoint policy

- **Use `/api/chat` (`OllamaClient.chat`/`chat_json`) by default for thinking-class models** (qwen3.x, deepseek-r1). Always pass `think=False` to suppress hidden chain-of-thought and get cooperative structured output.
- For schema-enforced JSON, pass the Pydantic-derived schema as `format=...`. With a schema, `chat_json` parses with `json.loads` directly and skips the tolerant fallback in `_extract_json` (which can wrongly grab an inner array if the outer object is mid-generation).
- `gemma4:26b` is fine for short structured outputs (questions, critic, calibration) but unreliable for long ones (cards) — token-repetition loops appear past ~3 KB of output. Reserve it for shorter generations.

## Python environment

- Use `.venv` (created with `python3 -m venv --system-site-packages .venv`) so we inherit pydantic/pyyaml/requests/tqdm from system Python and only install net-new packages (e.g. tiktoken).
- All stage runners must be invoked as `.venv/bin/python -m src.stage_NN_<name>` from the project root.
- New deps go into `requirements.txt` AND get installed into `.venv` via `.venv/bin/pip install <pkg>`. The venv is gitignored.

## How to run stages

```
.venv/bin/python -m src.stage_01_chunk                      # full corpus chunking
.venv/bin/python -m src.stage_02_filter --device cuda:0     # relevance filter (12 min first run, seconds after via cache)
.venv/bin/python -m src.stage_03_cards --workers 2          # paper cards via qwen3.5:9b (~62 min, 300 papers)
.venv/bin/python -m src.stage_04_cluster --device cuda:0    # cluster cards (seconds with cache)
.venv/bin/python -m src.stage_05a_questions --workers 2     # per-paper Qs via gemma4:26b (~35 min, 298 cards)
.venv/bin/python -m src.stage_05b_synthesis --workers 2     # synthesis Qs via gemma4:26b (~15 min, 80 pairs)
.venv/bin/python -m src.stage_05c_adversarial --workers 2   # adversarial Qs via gemma4:26b (~6 min, 36 pairs)
.venv/bin/python scripts/repair_evidence_quotes.py           # fix paraphrased evidence quotes in 5b/5c candidates
.venv/bin/python -m src.stage_06_validate --workers 2        # full validation gauntlet (~8 min, 1253 candidates)
.venv/bin/python -m src.stage_06_validate --resume --sources 5b 5c --workers 2  # re-validate only 5b/5c
.venv/bin/python -m src.stage_07_calibrate --limit 10 --verbose  # pilot (10 Qs, ~2 min)
.venv/bin/python -m src.stage_07_calibrate --workers 2           # full run (~1.4 hr, 366 Qs)
.venv/bin/python -m src.stage_07_calibrate --resume --workers 2  # resume after crash
.venv/bin/python -m src.stage_08_split                            # write 4 split JSONL + manifest.json (~1 sec)
```

- `--resume` is supported wherever it makes sense. Append-only stages skip IDs already in output.
- For embedding/clustering stages, `data/_cache/` holds reusable NPY tensors keyed by paired ID lists; mismatch triggers recomputation.

## GPU device policy

- This box has 2× RTX A6000 (49 GB each).
- **Ollama keeps gemma4:26b loaded on `cuda:1`** (~36 GB resident). Anything else that needs a GPU MUST pin to `cuda:0`.
- Stage runners that touch a GPU expose `--device cuda:0` (the default in `src/embeddings.py:detect_device()`) and should never default to "cuda" without specifying the index.

## Pinned decisions

Acceptance criteria, thresholds, and the targets in PLAN.md are pinned: they
were set deliberately and several of them encode a validity argument for the
benchmark. Changing one is a design decision to be recorded in the DEV_PLAN
decision log, not a tuning knob.
