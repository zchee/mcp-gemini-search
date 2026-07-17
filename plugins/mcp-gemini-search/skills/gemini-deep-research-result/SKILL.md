---
name: gemini-deep-research-result
description: Poll and retrieve Gemini Deep Research runs with the deep_research_result tool of the mcp-gemini-search MCP server. Use this skill whenever a deep_research run has been started and you hold an interaction_id — to check progress, wait for completion, fetch the finished Markdown report with its citations, or re-fetch a report from an earlier session. Also use it when the user asks "is the research done yet?", "get the report", or pastes an interaction_id.
---

# deep_research_result (mcp-gemini-search)

`deep_research_result` fetches the current state of a Deep Research run by
`interaction_id`. It is the **only** correct way to check on or retrieve a
run — never start a new `deep_research` call to "check progress"; that starts
a second billed multi-minute run.

## Calling contract

Input (`additionalProperties: false`):

| field | type | meaning |
|---|---|---|
| `interaction_id` (required) | string | The id returned by `deep_research`. |
| `wait_seconds` | integer 0–60 (default 0) | Seconds the server may hold the request, re-checking the run (~every 5s) before answering. |

Output: `{"interaction_id", "status"}` always; `"text"` and `"sources"` appear
only when `status` is `"completed"`.

## Polling discipline

Runs take several minutes. **Always pass `wait_seconds=60` when you are
waiting for completion** — the server long-polls and each call then costs one
round-trip per minute instead of dozens of instant `in_progress` replies.
`wait_seconds=0` is only for a true instant status peek (e.g. the user asked
"is it done yet?" and you should answer immediately).

The loop:

1. Call with `wait_seconds=60`.
2. `status == "in_progress"` → the report is not ready; call again the same
   way. If you have other useful work queued for the user, do it between
   polls rather than blocking on the loop.
3. `status == "completed"` → the report is in `text`; stop polling.
4. Tool error mentioning `failed` or `cancelled` → terminal; stop (see
   Errors).

Polling is cheap and idempotent — repeated `deep_research_result` calls never
mutate the run or incur a new research charge.

## Reading a completed report

`text` is the full report as normalized Markdown: inline `[n]` citation
markers map to the numbered `## Sources` section at the end. The structured
`sources` array mirrors those entries as
`{"index": n, "title": "...", "uri": "https://..."}` (title/uri may be absent
on some entries; `sources` may be absent when nothing was cited).

Reports are long. When relaying to the user, keep the citation markers and
the Sources section intact — the verifiability is the point of Deep
Research. Summarize on top of the report if asked, but always offer or
provide the full text. Include the `interaction_id` alongside the delivered
report: interactions persist ~55 days, and the id is what lets the user (or
a future session) re-fetch the full text without paying for a new run.

For a `plan_only` run, `text` contains the proposed research plan instead of
a report — show it to the user and approve/refine it via a `deep_research`
call with `previous_interaction_id` (see the `gemini-deep-research` skill).

## Terminal failures

A `failed` or `cancelled` run surfaces as a tool **error** (not a normal
result): the message reads `deep research failed: <detail>` or
`deep research cancelled: <detail>`. These are terminal:

- Do not keep polling that `interaction_id`.
- Do not silently restart the research — a new `deep_research` run is billed
  and slow. Report the failure detail to the user and let them decide.

Other errors:

- `Invalid arguments for deep_research_result: …` — schema violation (e.g.
  `wait_seconds` out of 0–60); fix the call.
- `interaction id cannot be empty` — you lost the id; recover it from your
  notes or the conversation, do not invent one.
- `deep research failed: <backend error>` on an id you know is valid — a
  transient fetch failure is possible; retry the poll once before surfacing.

## Ids outlive the session

Interactions are retained ~55 days on the paid tier, so an `interaction_id`
from an earlier conversation can still be fetched. If the user brings an id
from a past session, poll it directly — no need to re-run the research.
