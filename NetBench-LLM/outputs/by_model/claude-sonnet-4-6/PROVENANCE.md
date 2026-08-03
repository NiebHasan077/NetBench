# Provenance — `claude-sonnet-4-6` (frontier API baseline)

`Anthropic Claude Sonnet 4.6` is an **untrained frontier API baseline**, benchmarked under the
identical protocol as the adapted models: HPN v5.0 (242 questions), **closed-book**
(no retrieval), scored by the **gpt-5.1 judge**. Directly comparable to every
closed-book adapted-model variant.

| Field | Value |
| --- | --- |
| Setting | closed-book (no retrieval) |
| Exact model string invoked | `claude-sonnet-4-6` |
| Provider | Anthropic |
| Provider-side dated snapshot | not pinned (stable alias; see drift caveat) |
| Answer-generation run | `2026-06-15T11:13:39.075410` |
| Judge model | `gpt-5.1` |
| Judge run | `2026-06-15T12:24:58.664900` |
| Benchmark | HPN v5.0 (242 Q) |
| Decoding params | not recorded upstream |
| Source | `master` working tree (only copy); captured 2026-06-18 |

**Drift caveat:** API model identifiers are aliases that providers update
silently. These results pin a *point-in-time* API response (run timestamp above),
not a fixed downloadable artifact; re-running the same model name later may yield
different outputs. Logs were excluded; scores derive from the judged workbook.
