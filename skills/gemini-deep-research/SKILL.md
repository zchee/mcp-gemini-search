---
name: gemini-deep-research
description: Start an asynchronous Gemini Deep Research run with the deep_research tool of the mcp-gemini-search MCP server — an autonomous agent that searches, reads sources, and writes a long citation-rich Markdown report over several minutes. Use this skill whenever the user wants a comprehensive report, competitive analysis, technology comparison, literature survey, market/landscape overview, or any multi-source investigation — even if they just say "research X thoroughly" or "write me a report on Y". Also read it before ever calling deep_research, because a careless call starts a duplicate billed multi-minute run.
---

# deep_research (mcp-gemini-search)

`deep_research` starts a background Deep Research agent run and returns
immediately with `{interaction_id, status}` — it never waits for the report.
The run continues server-side for several minutes and is **billed per run**.
The report is fetched separately with `deep_research_result` (see the
`gemini-deep-research-result` skill).

## The one iron rule

**Call `deep_research` exactly once per research question, and save the
returned `interaction_id` the moment you get it.** Re-issuing the same
question — to "retry", to "check progress", because the first call felt slow —
starts a second full billed run that will also take minutes. Progress is
checked only with `deep_research_result`. The only legitimate reasons to call
`deep_research` again are a genuinely new question, or a deliberate
plan-approval / follow-up call that sets `previous_interaction_id`.

## When to use it (and when not)

- Use for questions whose answer is a *document*: comparisons with tradeoffs,
  state-of-the-art surveys, due-diligence style investigations.
- Do NOT use it for quick factual lookups — `google_search` answers those in
  seconds for a fraction of the cost (see the `gemini-google-search` skill).
- Confirm with the user before starting a run when the request is ambiguous:
  a run is minutes of wall-clock time and real money, so "did you want a full
  research report on this?" is a fair question.

## Calling contract

Input (`additionalProperties: false`):

| field | type | meaning |
|---|---|---|
| `query` (required) | string | The research brief. |
| `plan_only` | boolean (default false) | Return a research plan for review instead of running the research. |
| `previous_interaction_id` | string | Id of an earlier run: approve/refine its plan, or ask a follow-up building on its findings. |
| `agent` | string enum | `deep-research-preview-04-2026` (faster) or `deep-research-max-preview-04-2026` (deeper, slower). Omit to use the server default (`GEMINI_DEEP_RESEARCH_AGENT`). |

Output: `{"interaction_id": "...", "status": "in_progress"}`. The
`interaction_id` is durable — persist it AND tell the user immediately,
before doing anything else. A crashed or interrupted polling loop kills your
process state, and the id is the only handle that survives; losing it
orphans a billed run.

## Write a rich brief

The agent works from your `query` alone, so treat it as a brief, not a search
query. Include:

- **Scope**: what is in and out ("managed Kubernetes offerings on GCP and
  AWS, not on-prem").
- **Constraints**: timeframe, region, language, source preferences ("papers
  from 2024 onward", "prefer primary sources").
- **Desired output shape**: "a comparison table plus recommendation", "an
  annotated bibliography", "an executive summary followed by deep-dives".

Richer briefs produce measurably better reports; a one-line query wastes the
run on the agent guessing your intent. When the user's request is terse, do
not pass it through verbatim — expand it into a brief yourself, stating the
scope and output shape you inferred (and confirm them with the user first
when the run is expensive enough to matter).

## Plan-review workflow (optional but recommended for expensive questions)

1. Call `deep_research` with `plan_only=true` — the run returns a research
   plan instead of executing.
2. Fetch the plan with `deep_research_result` and show it to the user.
3. Approve or refine with a second `deep_research` call that sets
   `previous_interaction_id` to the plan run's id and puts the approval
   ("proceed with this plan") or the corrections in `query`.

Follow-up questions after a completed run use the same mechanism: set
`previous_interaction_id` so the new run builds on the previous findings
instead of starting from zero.

## Choosing the agent

Default (omit `agent`) unless the user signals otherwise. Pick
`deep-research-max-preview-04-2026` when the user asks for exhaustiveness
("be thorough", "leave no stone unturned") and accepts a longer wait; pick
`deep-research-preview-04-2026` explicitly when the server default is Max but
the user wants a faster turnaround.

## Privacy and retention

Background runs require server-side storage (`store=true`); on the paid tier
interactions are retained ~55 days. This differs from the stateless
`google_search` tool — mention it if the research brief contains sensitive
material.

## Errors

- `Invalid arguments for deep_research: …` — schema violation; fix the call.
- `research query cannot be empty` — send a real brief.
- `deep research failed: …` — the start itself failed; it is safe to retry
  once (no run was started when there is no interaction_id), then surface the
  error.
