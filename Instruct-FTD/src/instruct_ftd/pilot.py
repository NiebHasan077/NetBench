"""Phase 5 pilot generation helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk, RagPromptBundle, read_normalized_papers
from .normalize import NormalizedPaper
from .ollama_client import OllamaClient
from .progress import append_jsonl, ensure_resume_safe, fmt_elapsed, load_checkpoint_map, log, read_jsonl, write_json
from .prompt_families import build_rag_user_turn, build_teacher_messages, get_family


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_LOW_QUALITY_TEXT_RE = re.compile(
    r"(table of contents|acknowledgements|chapter 1|chapter 2|organization of thesis|abstract \.\s*v)",
    re.IGNORECASE,
)
_BUCKET_ORDER = {"short": 0, "medium": 1, "long": 2, "xlong": 3}
_HPN_KEYWORDS = (
    "network",
    "networks",
    "networking",
    "tcp",
    "rdma",
    "throughput",
    "bandwidth",
    "latency",
    "congestion",
    "transfer",
    "flow",
    "routing",
    "datacenter",
    "data center",
    "hpc",
    "parallelism",
    "pipelining",
    "nic",
)


@dataclass
class PilotRequest:
    """One teacher request planned for the pilot run."""

    request_id: str
    family_name: str
    difficulty: str
    paper_id: str
    chunk_ids: list[str]
    bundle_id: str = ""


def load_chunks(path: Path) -> list[Chunk]:
    """Load chunk JSONL rows into dataclass instances."""

    rows: list[Chunk] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(Chunk(**json.loads(line)))
    return rows


def load_rag_bundles(path: Path) -> list[RagPromptBundle]:
    """Load RAG bundle JSONL rows into dataclass instances."""

    rows: list[RagPromptBundle] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(RagPromptBundle(**json.loads(line)))
    return rows


def _chunk_heading_score(chunk: Chunk, family_name: str) -> tuple[int, int, int]:
    heading = (chunk.section_heading or "").lower()
    digit_bonus = 1 if re.search(r"\d", chunk.text) else 0

    preferred = 0
    if family_name == "hpn_fact_qa":
        preferred = 3 if chunk.is_abstract else 2 if "introduction" in heading else 1
    elif family_name == "hpn_concept_explanation":
        preferred = 3 if heading in {"design", "architecture", "framework", "method"} else 2 if chunk.is_abstract else 1
    elif family_name == "hpn_limitation_analysis":
        preferred = 4 if heading in {"evaluation", "results", "analysis", "discussion", "conclusion"} else 1
    elif family_name == "hpn_calculation":
        preferred = 4 if digit_bonus else 0
    elif family_name == "hpn_diagnosis":
        preferred = 4 if heading in {"evaluation", "results", "analysis", "discussion"} else 2 if "performance" in heading else 1
    elif family_name == "hpn_scenario":
        preferred = 3 if heading in {"design", "evaluation", "discussion"} else 2 if chunk.is_abstract else 1
    else:
        preferred = 2 if not chunk.is_abstract else 1

    return (preferred, digit_bonus, -chunk.token_estimate)


def _is_low_quality_chunk(chunk: Chunk) -> bool:
    text = chunk.text.strip()
    if not text:
        return True
    if chunk.paper_title == "Unknown":
        return True
    if _LOW_QUALITY_TEXT_RE.search(text):
        return True
    if text.startswith(".") and "chapter" in text[:120].lower():
        return True
    return False


def choose_single_chunk(chunks: list[Chunk], family_name: str) -> Chunk | None:
    """Choose one short chunk for a single-evidence HPN prompt family."""

    usable = [
        chunk
        for chunk in chunks
        if chunk.chunk_profile == "short" and not _is_low_quality_chunk(chunk)
    ]
    if not usable:
        return None

    ranked = sorted(usable, key=lambda chunk: _chunk_heading_score(chunk, family_name), reverse=True)
    top = ranked[0]
    if family_name == "hpn_calculation" and not re.search(r"\d", top.text):
        return None
    return top


def choose_pair_chunks(chunks: list[Chunk], family_name: str) -> list[Chunk] | None:
    """Choose a pair of short chunks for pair-based HPN prompt families."""

    usable = [
        chunk
        for chunk in chunks
        if chunk.chunk_profile == "short" and not _is_low_quality_chunk(chunk)
    ]
    if len(usable) < 2:
        return None

    ranked = sorted(usable, key=lambda chunk: _chunk_heading_score(chunk, family_name), reverse=True)
    first = ranked[0]
    second = next(
        (
            chunk
            for chunk in ranked[1:]
            if chunk.section_heading != first.section_heading or chunk.is_abstract != first.is_abstract
        ),
        None,
    )
    if second is None:
        second = ranked[1]
    return [first, second]


def choose_rag_bundle(bundles: list[RagPromptBundle]) -> RagPromptBundle | None:
    """Choose one deterministic RAG bundle candidate for a paper."""

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
    return ranked[0]


def select_pilot_papers(papers: list[NormalizedPaper], limit: int, min_parse_confidence: float) -> list[NormalizedPaper]:
    """Select high-quality papers for a deterministic pilot run."""

    def keyword_score(paper: NormalizedPaper) -> int:
        haystack = f"{paper.title} {paper.abstract}".lower()
        return sum(1 for keyword in _HPN_KEYWORDS if keyword in haystack)

    eligible = [
        paper
        for paper in papers
        if paper.parse_method == "json_structured" and paper.parse_confidence >= min_parse_confidence
        and paper.title != "Unknown"
        and not _LOW_QUALITY_TEXT_RE.search(paper.abstract)
        and keyword_score(paper) >= 1
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
    return eligible[:limit]


def build_pilot_requests(
    *,
    papers: list[NormalizedPaper],
    short_chunks: list[Chunk],
    rag_chunks: list[Chunk],
    rag_bundles: list[RagPromptBundle],
    family_names: list[str],
    difficulty_cycle: list[str],
) -> list[PilotRequest]:
    """Plan a deterministic set of pilot teacher requests."""

    chunks_by_paper: dict[str, list[Chunk]] = {}
    for chunk in short_chunks:
        chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)

    rag_chunks_by_id = {chunk.chunk_id: chunk for chunk in rag_chunks}
    bundles_by_paper: dict[str, list[RagPromptBundle]] = {}
    for bundle in rag_bundles:
        if len(bundle.paper_ids) == 1:
            bundles_by_paper.setdefault(bundle.paper_ids[0], []).append(bundle)

    requests: list[PilotRequest] = []
    difficulty_count = len(difficulty_cycle)
    request_counter = 0

    for paper_index, paper in enumerate(papers):
        paper_chunks = chunks_by_paper.get(paper.paper_id, [])
        paper_bundles = bundles_by_paper.get(paper.paper_id, [])
        for family_index, family_name in enumerate(family_names):
            family = get_family(family_name)
            difficulty = difficulty_cycle[(paper_index + family_index) % difficulty_count]

            if family.evidence_mode == "single_short":
                chunk = choose_single_chunk(paper_chunks, family_name)
                if chunk is None:
                    continue
                requests.append(
                    PilotRequest(
                        request_id=f"pilot_req_{request_counter:06d}",
                        family_name=family_name,
                        difficulty=difficulty,
                        paper_id=paper.paper_id,
                        chunk_ids=[chunk.chunk_id],
                    )
                )
                request_counter += 1
            elif family.evidence_mode == "pair_short":
                pair = choose_pair_chunks(paper_chunks, family_name)
                if pair is None:
                    continue
                requests.append(
                    PilotRequest(
                        request_id=f"pilot_req_{request_counter:06d}",
                        family_name=family_name,
                        difficulty=difficulty,
                        paper_id=paper.paper_id,
                        chunk_ids=[chunk.chunk_id for chunk in pair],
                    )
                )
                request_counter += 1
            elif family.evidence_mode == "rag_bundle":
                bundle = choose_rag_bundle(paper_bundles)
                if bundle is None or any(chunk_id not in rag_chunks_by_id for chunk_id in bundle.chunk_ids):
                    continue
                requests.append(
                    PilotRequest(
                        request_id=f"pilot_req_{request_counter:06d}",
                        family_name=family_name,
                        difficulty=difficulty,
                        paper_id=paper.paper_id,
                        chunk_ids=list(bundle.chunk_ids),
                        bundle_id=bundle.bundle_id,
                    )
                )
                request_counter += 1

    return requests


def _extract_response_content(api_response: dict) -> str:
    message = api_response.get("message") or {}
    content = str(message.get("content") or "").strip()
    if content:
        return content
    return str(message.get("thinking") or "").strip()


def _parse_generated_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    match = _JSON_RE.search(cleaned)
    if match:
        cleaned = match.group(0)

    return json.loads(cleaned)


def _prompt_length_bucket(chunks: list[Chunk], source_split: str) -> str:
    token_estimate = sum(chunk.token_estimate for chunk in chunks)
    if source_split != "rag":
        token_estimate = max((chunk.token_estimate for chunk in chunks), default=0)
    if token_estimate < 400:
        return "short"
    if token_estimate < 900:
        return "medium"
    if token_estimate < 1600:
        return "long"
    return "xlong"


def _normalize_question_text(text: str, is_rag: bool) -> str:
    question = " ".join(str(text).strip().split())
    if is_rag:
        lowered = question.lower()
        if "question:" in lowered:
            question = question.split("Question:")[-1].strip()
    return question


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _build_evidence_metadata(chunks: list[Chunk]) -> list[dict]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "paper_title": chunk.paper_title,
            "section": f"{chunk.section_number} {chunk.section_heading}".strip() or "Body",
            "text": chunk.text,
            "role": "primary" if index == 0 else "supporting",
        }
        for index, chunk in enumerate(chunks)
    ]


def run_pilot(
    *,
    model: str,
    output_dir: Path,
    papers: list[NormalizedPaper],
    short_chunks: list[Chunk],
    rag_chunks: list[Chunk],
    rag_bundles: list[RagPromptBundle],
    family_names: list[str],
    paper_limit: int,
    min_parse_confidence: float,
    temperature: float,
    top_p: float,
    num_predict: int,
    base_url: str,
    dry_run: bool = False,
    resume: bool = False,
) -> dict:
    """Run a pilot generation pass and write raw/parsed artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_resume_safe(
        output_dir,
        [
            "pilot_raw_generations.jsonl",
            "pilot_examples.jsonl",
            "pilot_failures.jsonl",
            "pilot_checkpoint.jsonl",
            "pilot_manifest.json",
        ],
        resume=resume,
    )

    selected_papers = select_pilot_papers(papers, paper_limit, min_parse_confidence)
    requests = build_pilot_requests(
        papers=selected_papers,
        short_chunks=short_chunks,
        rag_chunks=rag_chunks,
        rag_bundles=rag_bundles,
        family_names=family_names,
        difficulty_cycle=["easy", "medium", "hard"],
    )

    short_by_id = {chunk.chunk_id: chunk for chunk in short_chunks}
    rag_by_id = {chunk.chunk_id: chunk for chunk in rag_chunks}
    all_by_id = {**short_by_id, **rag_by_id}

    client = OllamaClient(base_url=base_url)
    run_id = output_dir.name
    raw_path = output_dir / "pilot_raw_generations.jsonl"
    examples_path = output_dir / "pilot_examples.jsonl"
    failures_path = output_dir / "pilot_failures.jsonl"
    checkpoint_path = output_dir / "pilot_checkpoint.jsonl"
    manifest_path = output_dir / "pilot_manifest.json"

    checkpoint_map = load_checkpoint_map(checkpoint_path) if resume else {}
    examples: list[dict] = read_jsonl(examples_path) if resume else []
    failures: list[dict] = read_jsonl(failures_path) if resume else []
    family_counter: Counter[str] = Counter(example.get("generation_family") for example in examples)

    skipped = sum(1 for r in requests if r.request_id in checkpoint_map)
    pending = len(requests) - skipped
    mode = "dry-run" if dry_run else f"model={model}"
    log(
        f"Phase 5 pilot  |  {mode}  |  papers={len(selected_papers)}"
        f"  |  requests={len(requests)} ({skipped} already done, {pending} pending)"
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
                "pilot_run_id": run_id,
                "request_id": request.request_id,
                "bundle_id": request.bundle_id,
                "word_count": _word_count(response),
                "evidence": _build_evidence_metadata(chunks),
            }
            examples.append(example)
            family_counter[family.name] += 1
            append_jsonl(examples_path, example)
            checkpoint_row = {
                "request_id": request.request_id,
                "status": "success",
                "family_name": request.family_name,
                "paper_id": request.paper_id,
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
            }
            append_jsonl(checkpoint_path, checkpoint_row)
            checkpoint_map[request.request_id] = checkpoint_row
            log(
                f"  [{done_count}/{len(requests)}] {request.request_id}"
                f" | {request.family_name} | {request.paper_id} | {request.difficulty}"
                f"  ->  FAIL (parse error: {exc})"
            )

    log(
        f"Phase 5 pilot done  |  elapsed={fmt_elapsed(t_start)}"
        f"  |  success={len(examples)}  failures={len(failures)}"
    )

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "paper_limit": paper_limit,
        "selected_paper_ids": [paper.paper_id for paper in selected_papers],
        "family_names": family_names,
        "request_count": len(requests),
        "completed_request_count": len(checkpoint_map),
        "success_count": len(examples),
        "failure_count": len(failures),
        "family_success_counts": dict(family_counter),
        "dry_run": dry_run,
        "resume_mode": resume,
    }

    write_json(manifest_path, manifest)

    return manifest


def default_run_dir(project_root: Path) -> Path:
    """Return a timestamped default pilot output directory."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "data" / "intermediate" / "pilot" / f"run_{timestamp}"
