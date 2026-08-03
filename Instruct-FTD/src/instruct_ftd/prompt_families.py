"""Phase 4 prompt-family registry and prompt builders for synthetic data generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .chunking import Chunk


HPN_TARGET_SYSTEM_PROMPT = (
    "You are an expert in high-performance networking (HPN), "
    "HPC data transfer systems, and computer networking.\n\n"
    "Answer the question below in a focused response (100-350 words). "
    "Be technically precise: include specific numbers, thresholds, or "
    "formulas where relevant. Explain the underlying reasoning, not just "
    "the conclusion."
)

RAG_TARGET_SYSTEM_PROMPT = (
    "You are an expert in high-performance networking (HPN),\n"
    "HPC data transfer systems, and computer networking.\n\n"
    "Answer the question below in a focused response (100-350 words)\n"
    "using the provided excerpts as your primary source.\n"
    "Be technically precise: include specific numbers, thresholds, or\n"
    "formulas where relevant. Explain the underlying reasoning, not just\n"
    "the conclusion."
)

TEACHER_SYSTEM_PROMPT = (
    "You generate high-quality instruction fine-tuning examples for open-weight "
    "LLMs trained on HPN and networking papers.\n\n"
    "Rules:\n"
    "1. Use only the supplied evidence.\n"
    "2. Do not invent unsupported claims, metrics, or formulas.\n"
    "3. Return exactly one JSON object and nothing else.\n"
    "4. The JSON object must contain string fields named 'question' and 'response'.\n"
    "5. The question must be useful for instruction tuning, not a verbatim copy of the evidence.\n"
    "6. The response must sound like the final assistant answer, not like dataset annotation notes."
)


@dataclass(frozen=True)
class PromptFamily:
    """Metadata for one synthetic-generation family."""

    name: str
    source_split: str
    category: str
    task_type: str
    question_type: str
    evidence_mode: str
    target_system_prompt: str
    description: str
    question_guidance: str
    response_guidance: str

    def to_dict(self) -> dict:
        return asdict(self)


FAMILIES: dict[str, PromptFamily] = {
    "hpn_fact_qa": PromptFamily(
        name="hpn_fact_qa",
        source_split="hpn",
        category="hpn_fact_qa",
        task_type="fact_qa",
        question_type="concept",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Direct but non-trivial HPN fact or definition question from one evidence excerpt.",
        question_guidance=(
            "Ask one standalone HPN question that checks understanding of a concrete fact, "
            "mechanism, definition, or role described in the evidence. Define uncommon acronyms "
            "inline if the question would otherwise be unclear."
        ),
        response_guidance=(
            "Write a technically precise answer in 100-220 words. Keep it specific to the "
            "evidence and include key terminology from the source."
        ),
    ),
    "hpn_concept_explanation": PromptFamily(
        name="hpn_concept_explanation",
        source_split="hpn",
        category="hpn_concept_explanation",
        task_type="concept_explanation",
        question_type="reasoning",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Conceptual or mechanism explanation from one evidence excerpt.",
        question_guidance=(
            "Ask for an explanation of why a method, parameter, or design choice matters. "
            "The question should require reasoning, not rote copying."
        ),
        response_guidance=(
            "Write a focused 120-280 word explanation. Explain the mechanism, tradeoff, or "
            "reasoning behind the concept rather than just restating it."
        ),
    ),
    "hpn_comparison": PromptFamily(
        name="hpn_comparison",
        source_split="hpn",
        category="hpn_comparison",
        task_type="comparison",
        question_type="comparison",
        evidence_mode="pair_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Comparison question requiring contrast between two related excerpts.",
        question_guidance=(
            "Ask one comparison question that contrasts two methods, signals, parameters, "
            "tradeoffs, or design choices present in the evidence pair."
        ),
        response_guidance=(
            "Write a 120-280 word answer that clearly compares the two items and states the "
            "main tradeoff or deployment implication."
        ),
    ),
    "hpn_limitation_analysis": PromptFamily(
        name="hpn_limitation_analysis",
        source_split="hpn",
        category="hpn_limitation_analysis",
        task_type="limitation_analysis",
        question_type="reasoning",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Question about limitations, bottlenecks, or failure modes from one excerpt.",
        question_guidance=(
            "Ask about a limitation, bottleneck, failure mode, or downside discussed or implied "
            "by the evidence."
        ),
        response_guidance=(
            "Write a 120-250 word answer that explains both the limitation and why it appears in "
            "practice."
        ),
    ),
    "hpn_calculation": PromptFamily(
        name="hpn_calculation",
        source_split="hpn",
        category="hpn_calculation",
        task_type="numerical_reasoning",
        question_type="calculation",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Simple numerical or parameter-sizing reasoning grounded in one excerpt.",
        question_guidance=(
            "Ask for a short calculation, threshold estimate, or parameter reasoning task that is "
            "supported by the evidence. If the evidence lacks numbers, do not invent them."
        ),
        response_guidance=(
            "Write a 100-220 word answer. Show the reasoning clearly and keep the arithmetic simple "
            "and evidence-grounded."
        ),
    ),
    "hpn_diagnosis": PromptFamily(
        name="hpn_diagnosis",
        source_split="hpn",
        category="hpn_diagnosis",
        task_type="diagnosis",
        question_type="diagnosis",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Diagnosis question about a realistic HPN problem pattern or symptom.",
        question_guidance=(
            "Ask a diagnosis-style question where the user observes symptoms and wants the likely "
            "problem and corrective action based on the evidence."
        ),
        response_guidance=(
            "Write a 120-280 word answer that identifies the likely issue, explains why, and gives "
            "a practical next step."
        ),
    ),
    "hpn_scenario": PromptFamily(
        name="hpn_scenario",
        source_split="hpn",
        category="hpn_scenario",
        task_type="scenario",
        question_type="scenario",
        evidence_mode="single_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Scenario-based HPN decision or deployment question.",
        question_guidance=(
            "Ask one realistic deployment or tuning scenario question that requires applying the "
            "evidence to a practical situation."
        ),
        response_guidance=(
            "Write a 120-280 word answer with an actionable recommendation and the reasoning behind it."
        ),
    ),
    "hpn_task": PromptFamily(
        name="hpn_task",
        source_split="hpn",
        category="hpn_task",
        task_type="task",
        question_type="task",
        evidence_mode="pair_short",
        target_system_prompt=HPN_TARGET_SYSTEM_PROMPT,
        description="Task-oriented recommendation or tuning plan using two evidence excerpts.",
        question_guidance=(
            "Ask for a concrete recommendation, tuning plan, or design action based on the evidence pair."
        ),
        response_guidance=(
            "Write a 120-280 word answer with a clear recommended action and the main decision criteria."
        ),
    ),
    "rag_grounded_qa": PromptFamily(
        name="rag_grounded_qa",
        source_split="rag",
        category="rag_grounded_qa",
        task_type="rag_grounded_qa",
        question_type="scenario",
        evidence_mode="rag_bundle",
        target_system_prompt=RAG_TARGET_SYSTEM_PROMPT,
        description="Grounded multi-excerpt RAG question that should be answerable from supplied evidence.",
        question_guidance=(
            "Write one query-only question that requires the assistant to use multiple excerpts as "
            "primary evidence. The final dataset pipeline will wrap the query with the excerpts, so "
            "do not repeat or reformat the excerpts inside the question."
        ),
        response_guidance=(
            "Write a 120-300 word grounded answer. Cite excerpt ids inline like [1] or [2] for key "
            "claims. Use only supported information."
        ),
    ),
    "rag_unanswerable": PromptFamily(
        name="rag_unanswerable",
        source_split="rag",
        category="rag_unanswerable",
        task_type="rag_unanswerable",
        question_type="scenario",
        evidence_mode="rag_bundle",
        target_system_prompt=RAG_TARGET_SYSTEM_PROMPT,
        description="Grounded RAG question that should not be answerable from the supplied excerpts.",
        question_guidance=(
            "Write one query-only question that looks plausible in the topic area but cannot be fully "
            "answered from the supplied excerpts."
        ),
        response_guidance=(
            "Write a 90-220 word abstaining answer that says the provided excerpts do not supply enough "
            "evidence, avoids hallucination, and optionally states what information is missing."
        ),
    ),
}


def family_specs() -> dict[str, dict]:
    """Return prompt-family metadata as plain dicts for docs and reports."""

    return {name: family.to_dict() for name, family in FAMILIES.items()}


def get_family(name: str) -> PromptFamily:
    """Return one registered prompt family."""

    try:
        return FAMILIES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown prompt family: {name}") from exc


def format_chunk_for_teacher(chunk: Chunk) -> str:
    """Format one short evidence chunk for teacher prompting."""

    section = f"{chunk.section_number} {chunk.section_heading}".strip() or "Body"
    return (
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Paper: {chunk.paper_title}\n"
        f"Section: {section}\n"
        f"Token estimate: {chunk.token_estimate}\n"
        f"Evidence:\n{chunk.text}"
    )


def format_chunk_pair_for_teacher(chunks: list[Chunk]) -> str:
    """Format a short evidence pair for comparison/task prompting."""

    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        section = f"{chunk.section_number} {chunk.section_heading}".strip() or "Body"
        lines.append(
            f"[{index}] Chunk ID: {chunk.chunk_id}\n"
            f"Paper: {chunk.paper_title}\n"
            f"Section: {section}\n"
            f"Token estimate: {chunk.token_estimate}\n"
            f"Evidence:\n{chunk.text}"
        )
    return "\n\n".join(lines)


def format_rag_excerpts(chunks: list[Chunk]) -> str:
    """Render excerpts in the exact user-turn structure planned for RAG examples."""

    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        section = f"{chunk.section_number} {chunk.section_heading}".strip() or "Body"
        parts.append(
            f"[{index}] Paper: '{chunk.paper_title}' | Section: {section}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(parts)


def build_rag_user_turn(query: str, chunks: list[Chunk]) -> str:
    """Build the final dataset user turn for a RAG example."""

    return f"Excerpts:\n\n{format_rag_excerpts(chunks)}\n\nQuestion: {query.strip()}"


def build_teacher_messages(
    family: PromptFamily,
    *,
    difficulty: str,
    chunks: list[Chunk],
) -> list[dict[str, str]]:
    """Build an Ollama chat message list for one prompt family and evidence set."""

    if family.evidence_mode == "single_short":
        evidence_block = format_chunk_for_teacher(chunks[0])
    elif family.evidence_mode == "pair_short":
        evidence_block = format_chunk_pair_for_teacher(chunks)
    elif family.evidence_mode == "rag_bundle":
        evidence_block = format_rag_excerpts(chunks)
    else:
        raise ValueError(f"Unsupported evidence mode: {family.evidence_mode}")

    rag_note = ""
    if family.source_split == "rag":
        rag_note = (
            "\nThe final dataset pipeline will construct the user turn as:\n"
            "Excerpts:\n\n"
            "<numbered excerpts>\n\n"
            "Question: <your generated question>\n\n"
            "Your JSON 'question' field must contain only the bare query sentence, not the excerpt block."
        )

    user_prompt = (
        f"Create one synthetic instruction fine-tuning example.\n\n"
        f"Prompt family: {family.name}\n"
        f"Difficulty target: {difficulty}\n"
        f"Source split: {family.source_split}\n"
        f"Category: {family.category}\n"
        f"Task type: {family.task_type}\n"
        f"Question type: {family.question_type}\n\n"
        f"Target assistant system prompt for the final dataset:\n"
        f"{family.target_system_prompt}\n\n"
        f"Family description:\n{family.description}\n\n"
        f"Question guidance:\n{family.question_guidance}\n\n"
        f"Response guidance:\n{family.response_guidance}\n"
        f"{rag_note}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Return exactly this JSON object shape:\n"
        "{\n"
        '  "question": "<one standalone user question>",\n'
        '  "response": "<one assistant answer>"\n'
        "}\n\n"
        "Do not include markdown fences, notes, or extra keys."
    )

    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

