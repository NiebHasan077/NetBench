# Chat Guide — NetBench-RAG Interactive Query Interface

This guide covers `query.py`, the interactive command-line interface for asking questions using the RAG pipeline.

---

## Starting the REPL

```bash
source venv/bin/activate
python query.py
```

You will see a prompt:
```
NetBench-RAG  [backend: openai | chunks: 5]
>
```

Type any question and press Enter. The system retrieves relevant passages and generates a grounded answer.

---

## CLI Options

```
python query.py [OPTIONS]

Options:
  --backend {openai,local}   Generation backend to use (overrides config.yaml)
  --top-k INT                Number of final chunks passed to LLM (default: 5)
  --question TEXT            Single-shot mode: answer this question and exit
  --config PATH              Path to config.yaml (default: config.yaml)
```

---

## Usage Examples

### Basic interactive session

```bash
python query.py
```

### Force a specific backend

```bash
python query.py --backend openai    # Use GPT-4o (requires openai_api_key.txt)
python query.py --backend local     # Use fine-tuned Llama 3.1 8B (requires GPU)
```

### Reduce context (faster, uses fewer chunks)

```bash
python query.py --top-k 3
```

### Single-shot (no REPL)

```bash
python query.py --question "How does TCP slow start affect throughput in WAN?"
```

Useful for scripting or quick lookups:
```bash
python query.py --question "What is RDMA?" | tee output.txt
```

---

## Understanding the Output

A typical response looks like:

```
╔══════════════════════════════════════════════════════╗
║  Answer                                              ║
╚══════════════════════════════════════════════════════╝

BBR (Bottleneck Bandwidth and Round-Trip Propagation) controls
congestion by estimating the bottleneck bandwidth and minimum
RTT, rather than reacting to packet loss like CUBIC does.
[BBR: Congestion-Based Congestion Control, Section III.A]

It paces packets to match the estimated delivery rate, keeping
queues short even at high utilization.
[Performance Analysis of TCP Congestion Control, Section II]

──────────────────────────────────────────────────────
Retrieved chunks: 5  |  Retrieval: 2.8s  |  Generation: 1.2s
```

**Inline citations** reference the source paper and section. The model is instructed to cite only from the retrieved excerpts and explicitly flag when information is insufficient.

---

## How the Retrieval Works

For each query, the pipeline runs three stages:

1. **Dual Search** — your query is embedded and used for Qdrant vector search (top-30), while a tokenized + acronym-expanded version queries BM25 (top-30).
2. **RRF Fusion** — Reciprocal Rank Fusion merges both ranked lists into 20 candidates.
3. **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 scores each (query, chunk) pair and returns the top-5.

These top-5 chunks are formatted as numbered excerpts and passed to the LLM.

---

## HPN Domain Acronyms

The BM25 stage automatically expands these acronyms in your query to improve keyword matching:

| Acronym | Expansion |
|---------|-----------|
| BBR | bottleneck bandwidth and round-trip propagation |
| RDMA | remote direct memory access |
| RTT | round trip time |
| BDP | bandwidth delay product |
| NIC | network interface card |
| SDN | software defined networking |
| HPN | high performance networking |
| MPI | message passing interface |
| CC | congestion control |
| ECN | explicit congestion notification |
| CWND | congestion window |
| MPTCP | multipath tcp |
| DRL | deep reinforcement learning |
| RL | reinforcement learning |
| TCP | transmission control protocol |
| UDP | user datagram protocol |

These expansions only apply to BM25; vector search uses your original query.

---

## Generation Backends

### OpenAI Backend (default)

- Model: `gpt-4o` (configurable in `config.yaml`)
- Requires: `openai_api_key.txt`
- Produces concise, well-structured answers with proper citations
- ~1–2s generation latency per question

### Local Backend

- Model: Fine-tuned Llama 3.1 8B (path in `config.yaml`)
- Requires: GPU with ≥ 24 GB VRAM
- Uses Llama 3.1 native chat tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`)
- Greedy decoding (`do_sample: false`) for deterministic output
- Higher latency than API, but no API costs

---

## Exiting the REPL

Type any of:
```
exit
quit
q
```

Or press `Ctrl+C`.

---

## Tips

- **Be specific** — "How does BBR estimate bottleneck bandwidth?" retrieves better than "explain BBR"
- **Use acronyms freely** — they are expanded automatically for BM25 retrieval
- **Check citations** — if the model says a fact came from [Paper X, Section Y], that chunk was retrieved
- **Increase top-k** for broad survey questions (`--top-k 8`), decrease for narrow factual ones (`--top-k 2`)
- **The model will say when it doesn't know** — if the retrieved passages don't cover your question, it will explicitly state this rather than hallucinate

---

## Configuration Reference

Settings that affect `query.py` in `config.yaml`:

```yaml
retrieval:
  reranker_top_n: 5          # default top-k (overridden by --top-k)
  vector_top_k: 30           # candidates from vector search
  bm25_top_k: 30             # candidates from BM25
  rrf_k: 60                  # RRF smoothing constant
  rrf_top_n: 20              # candidates after fusion

generation:
  backend: openai             # default backend (overridden by --backend)
  system_prompt: |            # editable system prompt
    You are an expert in High-Performance Networking...
  openai:
    model: gpt-4o
    max_tokens: 512
    temperature: 0.3
  local:
    max_new_tokens: 512
    temperature: 0.3
    do_sample: false
```
