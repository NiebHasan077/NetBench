"""Phase 6 full candidate generation over the normalized corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk, RagPromptBundle
from .normalize import NormalizedPaper
from .ollama_client import OllamaClient
from .pilot import (
    _BUCKET_ORDER,
    _HPN_KEYWORDS,
    _LOW_QUALITY_TEXT_RE,
    _build_evidence_metadata,
    _chunk_heading_score,
    _extract_response_content,
    _is_low_quality_chunk,
    _normalize_question_text,
    _parse_generated_json,
    _prompt_length_bucket,
    _word_count,
    load_chunks,
    load_rag_bundles,
)
from .progress import append_jsonl, ensure_resume_safe, fmt_elapsed, load_checkpoint_map, log, read_jsonl, write_json
from .prompt_families import build_rag_user_turn, build_teacher_messages, get_family


@dataclass
class CandidateRequest:
    """One full-generation teacher request."""

    request_id: str
    family_name: str
    difficulty: str
    paper_id: str
    chunk_ids: list[str]
    bundle_id: str = ""


def keyword_score(paper: NormalizedPaper) -> int:
    """Compute a lightweight HPN relevance score from title and abstract."""

    haystack = f"{paper.title} {paper.abstract}".lower()
    return sum(1 for keyword in _HPN_KEYWORDS if keyword in haystack)


def select_generation_papers(
    papers: list[NormalizedPaper],
    *,
    paper_limit: int | None,
    min_parse_confidence: float,
    min_keyword_score: int,
) -> list[NormalizedPaper]:
    """Select HPN-relevant papers for full candidate generation."""

    eligible = [
        paper
        for paper in papers
        if paper.parse_method == "json_structured"
        and paper.parse_confidence >= min_parse_confidence
        and paper.title != "Unknown"
        and not _LOW_QUALITY_TEXT_RE.search(paper.abstract)
        and keyword_score(paper) >= min_keyword_score
    ]

    eligible.sort(
        key=lambda paper: (
            -keyword_score(paper),
            -paper.parse_confidence,
            -len(paper.sections),
            -paper.total_section_tokens,
            paper.paper_id,
        )
    )

    if paper_limit is None or paper_limit <= 0:
        return eligible
    return eligible[:paper_limit]


def default_family_plan(paper_index: int, include_unanswerable_every: int) -> list[str]:
    """Return the default 4-5 example plan for one paper."""

    reasoning_pool = ["hpn_diagnosis", "hpn_scenario"]
    advanced_pool = [
        "hpn_comparison",
        "hpn_limitation_analysis",
        "hpn_task",
        "hpn_calculation",
    ]

    families = [
        "hpn_fact_qa",
        "hpn_concept_explanation",
        reasoning_pool[paper_index % len(reasoning_pool)],
        advanced_pool[paper_index % len(advanced_pool)],
        "rag_grounded_qa",
    ]

    if include_unanswerable_every > 0 and (paper_index + 1) % include_unanswerable_every == 0:
        families.append("rag_unanswerable")

    return families


def _select_single_chunk(
    chunks: list[Chunk],
    family_name: str,
    used_chunk_ids: set[str],
) -> Chunk | None:
    usable = [
        chunk
        for chunk in chunks
        if chunk.chunk_profile == "short" and not _is_low_quality_chunk(chunk)
    ]
    if not usable:
        return None

    ranked = sorted(usable, key=lambda chunk: _chunk_heading_score(chunk, family_name), reverse=True)
    unused = [chunk for chunk in ranked if chunk.chunk_id not in used_chunk_ids]
    chosen = unused[0] if unused else ranked[0]

    if family_name == "hpn_calculation" and not re.search(r"\d", chosen.text):
        for chunk in ranked:
            if re.search(r"\d", chunk.text):
                return chunk
        return None
    return chosen


def _select_pair_chunks(
    chunks: list[Chunk],
    family_name: str,
    used_chunk_ids: set[str],
) -> list[Chunk] | None:
    usable = [
        chunk
        for chunk in chunks
        if chunk.chunk_profile == "short" and not _is_low_quality_chunk(chunk)
    ]
    if len(usable) < 2:
        return None

    ranked = sorted(usable, key=lambda chunk: _chunk_heading_score(chunk, family_name), reverse=True)
    preferred = [chunk for chunk in ranked if chunk.chunk_id not in used_chunk_ids]
    pool = preferred if len(preferred) >= 2 else ranked

    first = pool[0]
    second = next(
        (
            chunk
            for chunk in pool[1:]
            if chunk.section_heading != first.section_heading or chunk.is_abstract != first.is_abstract
        ),
        None,
    )
    if second is None:
        second = pool[1] if len(pool) > 1 else None

    if second is None:
        return None
    return [first, second]


def _select_rag_bundle(
    bundles: list[RagPromptBundle],
    used_bundle_ids: set[str],
) -> RagPromptBundle | None:
    if not bundles:
        return None

    ranked = sorted(
        bundles,
        key=lambda bundle: (
            bundle.bundle_type != "abstract_plus_body",
            _BUCKET_ORDER.get(bundle.prompt_length_bucket, 99),
            bundle.token_estimate,
        ),
    )
    unused = [bundle for bundle in ranked if bundle.bundle_id not in used_bundle_ids]
    return unused[0] if unused else ranked[0]


def build_candidate_requests(
    *,
    papers: list[NormalizedPaper],
    short_chunks: list[Chunk],
    rag_chunks: list[Chunk],
    rag_bundles: list[RagPromptBundle],
    families_override: list[str] | None,
    include_unanswerable_every: int,
) -> list[CandidateRequest]:
    """Plan deterministic full-generation requests across selected papers."""

    short_by_paper: dict[str, list[Chunk]] = {}
    for chunk in short_chunks:
        short_by_paper.setdefault(chunk.paper_id, []).append(chunk)

    rag_by_chunk_id = {chunk.chunk_id: chunk for chunk in rag_chunks}
    rag_bundles_by_paper: dict[str, list[RagPromptBundle]] = {}
    for bundle in rag_bundles:
        if len(bundle.paper_ids) == 1:
            rag_bundles_by_paper.setdefault(bundle.paper_ids[0], []).append(bundle)

    requests: list[CandidateRequest] = []
    difficulty_cycle = ["easy", "medium", "hard"]
    request_counter = 0

    for paper_index, paper in enumerate(papers):
        family_plan = families_override or default_family_plan(paper_index, include_unanswerable_every)
        paper_short_chunks = short_by_paper.get(paper.paper_id, [])
        paper_rag_bundles = rag_bundles_by_paper.get(paper.paper_id, [])
        used_chunk_ids: set[str] = set()
        used_bundle_ids: set[str] = set()

        for family_offset, family_name in enumerate(family_plan):
            family = get_family(family_name)
            difficulty = difficulty_cycle[(paper_index + family_offset) % len(difficulty_cycle)]
            selected_chunk_ids: list[str] | None = None
            bundle_id = ""

            if family.evidence_mode == "single_short":
                chunk = _select_single_chunk(paper_short_chunks, family_name, used_chunk_ids)
                if chunk is not None:
                    selected_chunk_ids = [chunk.chunk_id]
                    used_chunk_ids.add(chunk.chunk_id)
            elif family.evidence_mode == "pair_short":
                pair = _select_pair_chunks(paper_short_chunks, family_name, used_chunk_ids)
                if pair is not None:
                    selected_chunk_ids = [chunk.chunk_id for chunk in pair]
                    used_chunk_ids.update(selected_chunk_ids)
            elif family.evidence_mode == "rag_bundle":
                bundle = _select_rag_bundle(paper_rag_bundles, used_bundle_ids)
                if bundle is not None and all(chunk_id in rag_by_chunk_id for chunk_id in bundle.chunk_ids):
                    selected_chunk_ids = list(bundle.chunk_ids)
                    bundle_id = bundle.bundle_id
                    used_bundle_ids.add(bundle.bundle_id)

            if not selected_chunk_ids:
                continue

            requests.append(
                CandidateRequest(
                    request_id=f"req_{request_counter:06d}",
                    family_name=family_name,
                    difficulty=difficulty,
                    paper_id=paper.paper_id,
                    chunk_ids=selected_chunk_ids,
                    bundle_id=bundle_id,
                )
            )
            request_counter += 1

    return requests


def default_run_dir(project_root: Path) -> Path:
    """Return a timestamped default run directory for candidate generation."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "data" / "intermediate" / "candidates" / f"run_{timestamp}"


def run_candidate_generation(
    *,
    model: str,
    output_dir: Path,
    papers: list[NormalizedPaper],
    short_chunks: list[Chunk],
    rag_chunks: list[Chunk],
    rag_bundles: list[RagPromptBundle],
    paper_limit: int | None,
    min_parse_confidence: float,
    min_keyword_score: int,
    families_override: list[str] | None,
    include_unanswerable_every: int,
    temperature: float,
    top_p: float,
    num_predict: int,
    base_url: str,
    dry_run: bool = False,
    resume: bool = False,
) -> dict:
    """Run full candidate generation and write Phase 6 artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_resume_safe(
        output_dir,
        [
            "candidate_raw_generations.jsonl",
            "candidate_failures.jsonl",
            "candidate_hpn.jsonl",
            "candidate_rag.jsonl",
            "candidate_checkpoint.jsonl",
            "candidate_manifest.json",
        ],
        resume=resume,
    )
    selected_papers = select_generation_papers(
        papers,
        paper_limit=paper_limit,
        min_parse_confidence=min_parse_confidence,
        min_keyword_score=min_keyword_score,
    )
    requests = build_candidate_requests(
        papers=selected_papers,
        short_chunks=short_chunks,
        rag_chunks=rag_chunks,
        rag_bundles=rag_bundles,
        families_override=families_override,
        include_unanswerable_every=include_unanswerable_every,
    )

    short_by_id = {chunk.chunk_id: chunk for chunk in short_chunks}
    rag_by_id = {chunk.chunk_id: chunk for chunk in rag_chunks}
    all_by_id = {**short_by_id, **rag_by_id}

    client = OllamaClient(base_url=base_url)
    run_id = output_dir.name
    raw_path = output_dir / "candidate_raw_generations.jsonl"
    failures_path = output_dir / "candidate_failures.jsonl"
    hpn_path = output_dir / "candidate_hpn.jsonl"
    rag_path = output_dir / "candidate_rag.jsonl"
    checkpoint_path = output_dir / "candidate_checkpoint.jsonl"
    manifest_path = output_dir / "candidate_manifest.json"

    checkpoint_map = load_checkpoint_map(checkpoint_path) if resume else {}
    failures: list[dict] = read_jsonl(failures_path) if resume else []
    hpn_examples: list[dict] = read_jsonl(hpn_path) if resume else []
    rag_examples: list[dict] = read_jsonl(rag_path) if resume else []
    family_counter: Counter[str] = Counter(example.get("generation_family") for example in hpn_examples + rag_examples)
    split_counter: Counter[str] = Counter(example.get("source_split") for example in hpn_examples + rag_examples)

    skipped = sum(1 for r in requests if r.request_id in checkpoint_map)
    pending = len(requests) - skipped
    mode = "dry-run" if dry_run else f"model={model}"
    log(
        f"Phase 6 candidate generation  |  {mode}"
        f"  |  papers={len(selected_papers)}  requests={len(requests)}"
        f" ({skipped} already done, {pending} pending)"
        f"  |  run={run_id}"
    )
    if dry_run:
        log("DRY RUN — Ollama will not be contacted.")

    done_count = skipped
    t_start = __import__("time").monotonic()

    for request in requests:
        if request.request_id in checkpoint_map:
            continue

        done_count += 1
        family = get_family(request.family_name)
        chunks = [all_by_id[chunk_id] for chunk_id in request.chunk_ids]
        messages = build_teacher_messages(
            family=family,
            difficulty=request.difficulty,
            chunks=chunks,
        )

        raw_text = ""
        api_response = {}

        if not dry_run:
            try:
                api_response = client.chat_json(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    num_predict=num_predict,
                    think=False,
                )
                raw_text = _extract_response_content(api_response)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "run_id": run_id,
                        "request_id": request.request_id,
                        "model": model,
                        "family_name": request.family_name,
                        "paper_id": request.paper_id,
                        "difficulty": request.difficulty,
                        "chunk_ids": request.chunk_ids,
                        "bundle_id": request.bundle_id,
                        "raw_text": "",
                        "error": f"ollama_request_failed: {exc}",
                    }
                )
                raw_row = {
                    "run_id": run_id,
                    "request_id": request.request_id,
                    "model": model,
                    "family_name": request.family_name,
                    "paper_id": request.paper_id,
                    "difficulty": request.difficulty,
                    "chunk_ids": request.chunk_ids,
                    "bundle_id": request.bundle_id,
                    "messages": messages,
                    "raw_text": "",
                    "api_response": {},
                }
                append_jsonl(raw_path, raw_row)
                append_jsonl(failures_path, failures[-1])
                checkpoint_row = {
                    "request_id": request.request_id,
                    "status": "failure",
                    "family_name": request.family_name,
                    "paper_id": request.paper_id,
                    "source_split": family.source_split,
                }
                append_jsonl(checkpoint_path, checkpoint_row)
                checkpoint_map[request.request_id] = checkpoint_row
                log(
                    f"  [{done_count}/{len(requests)}] {request.request_id}"
                    f" | {request.family_name} | {request.paper_id} | {request.difficulty}"
                    f"  ->  FAIL (ollama error)"
                )
                continue

        raw_row = {
            "run_id": run_id,
            "request_id": request.request_id,
            "model": model,
            "family_name": request.family_name,
            "paper_id": request.paper_id,
            "difficulty": request.difficulty,
            "chunk_ids": request.chunk_ids,
            "bundle_id": request.bundle_id,
            "messages": messages,
            "raw_text": raw_text,
            "api_response": api_response,
        }
        append_jsonl(raw_path, raw_row)

        if dry_run:
            checkpoint_row = {
                "request_id": request.request_id,
                "status": "dry_run",
                "family_name": request.family_name,
                "paper_id": request.paper_id,
                "source_split": family.source_split,
            }
            append_jsonl(checkpoint_path, checkpoint_row)
            checkpoint_map[request.request_id] = checkpoint_row
            log(
                f"  [{done_count}/{len(requests)}] {request.request_id}"
                f" | {request.family_name} | {request.paper_id} | {request.difficulty}"
                f"  ->  dry-run (planned)"
            )
            continue

        try:
            parsed = _parse_generated_json(raw_text)
            question_only = _normalize_question_text(parsed["question"], family.source_split == "rag")
            response = " ".join(str(parsed["response"]).strip().split())
            final_question = (
                build_rag_user_turn(question_only, chunks)
                if family.source_split == "rag"
                else question_only
            )

            example = {
                "system": family.target_system_prompt,
                "question": final_question,
                "response": response,
                "category": family.category,
                "source_split": family.source_split,
                "task_type": family.task_type,
                "difficulty": request.difficulty,
                "paper_id": request.paper_id,
                "paper_title": chunks[0].paper_title,
                "source_pdf": chunks[0].source_pdf,
                "question_type": family.question_type,
                "prompt_length_bucket": _prompt_length_bucket(chunks, family.source_split),
                "generation_family": family.name,
                "teacher_model": model,
                "candidate_run_id": run_id,
                "request_id": request.request_id,
                "bundle_id": request.bundle_id,
                "word_count": _word_count(response),
                "evidence": _build_evidence_metadata(chunks),
            }

            if family.source_split == "rag":
                rag_examples.append(example)
                append_jsonl(rag_path, example)
            else:
                hpn_examples.append(example)
                append_jsonl(hpn_path, example)

            family_counter[family.name] += 1
            split_counter[family.source_split] += 1
            checkpoint_row = {
                "request_id": request.request_id,
                "status": "success",
                "family_name": request.family_name,
                "paper_id": request.paper_id,
                "source_split": family.source_split,
            }
            append_jsonl(checkpoint_path, checkpoint_row)
            checkpoint_map[request.request_id] = checkpoint_row
            log(
                f"  [{done_count}/{len(requests)}] {request.request_id}"
                f" | {request.family_name} | {request.paper_id} | {request.difficulty}"
                f"  ->  ok"
            )
        except Exception as exc:  # noqa: BLE001
            failure_row = {
                "run_id": run_id,
                "request_id": request.request_id,
                "model": model,
                "family_name": request.family_name,
                "paper_id": request.paper_id,
                "difficulty": request.difficulty,
                "chunk_ids": request.chunk_ids,
                "bundle_id": request.bundle_id,
                "raw_text": raw_text,
                "error": str(exc),
            }
            failures.append(failure_row)
            append_jsonl(failures_path, failure_row)
            checkpoint_row = {
                "request_id": request.request_id,
                "status": "failure",
                "family_name": request.family_name,
                "paper_id": request.paper_id,
                "source_split": family.source_split,
            }
            append_jsonl(checkpoint_path, checkpoint_row)
            checkpoint_map[request.request_id] = checkpoint_row
            log(
                f"  [{done_count}/{len(requests)}] {request.request_id}"
                f" | {request.family_name} | {request.paper_id} | {request.difficulty}"
                f"  ->  FAIL (parse error: {exc})"
            )
            checkpoint_map[request.request_id] = checkpoint_row

    success_count = len(hpn_examples) + len(rag_examples)
    log(
        f"Phase 6 candidate generation done  |  elapsed={fmt_elapsed(t_start)}"
        f"  |  success={success_count} (hpn={len(hpn_examples)}, rag={len(rag_examples)})"
        f"  failures={len(failures)}"
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "paper_limit": paper_limit if paper_limit is not None else 0,
        "selected_paper_count": len(selected_papers),
        "selected_paper_ids_preview": [paper.paper_id for paper in selected_papers[:25]],
        "request_count": len(requests),
        "completed_request_count": len(checkpoint_map),
        "success_count": success_count,
        "failure_count": len(failures),
        "family_success_counts": dict(family_counter),
        "source_split_counts": dict(split_counter),
        "families_override": families_override or [],
        "include_unanswerable_every": include_unanswerable_every,
        "dry_run": dry_run,
        "resume_mode": resume,
    }

    write_json(manifest_path, manifest)

    return manifest
