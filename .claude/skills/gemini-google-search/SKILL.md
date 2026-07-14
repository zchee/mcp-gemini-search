---
name: gemini-google-search
description: Drive the google_search tool of the mcp-gemini-search MCP server — a Google-Search-grounded Gemini call that returns a Markdown answer with inline [n] citations and a numbered Sources list. Use this skill whenever the user asks to search the web, look up current events, verify a fact against live sources, check the latest version/release/pricing/news of anything, or when any claim needs verifiable citations — even if they never say "search" or "google_search". Also use it to decide when a request should instead go to the deep_research tool.
---

# google_search (mcp-gemini-search)

`google_search` runs one Google-Search-grounded Gemini interaction and returns
a Markdown answer with inline `[n]` citation markers and a numbered
`## Sources` list. Depending on how the server is registered, the tool may
appear as `mcp__mcp-gemini-search__google_search` or under another server
alias — match on the trailing `google_search`.

The tool's own schema documents its parameters (`query`, `url_context`,
`code_execution`) and when to flip each flag — trust it. This skill covers
what the schema cannot: choosing this tool, composing queries, and relaying
results without destroying their verifiability.

## Answer vs. report

Use `google_search` when the user wants an *answer*: current events, fresh
facts, version/release checks, any claim that needs verifiable sources.
Prefer it over unsourced recall for anything time-sensitive — training data
is stale, the grounded answer is not.

When the user wants a *document* — a multi-source comparison, survey, or
due-diligence investigation — use the `deep_research` tool instead (see the
`gemini-deep-research` skill).

## Every query must be self-contained

The server is stateless (`store=false`): there is no conversation memory, and
the model never sees your previous queries.

- Never write "the library mentioned above" — name it every time.
- Pack the disambiguators into the query itself: exact version, year, product
  name, the verbatim error string.
- A miss is normal: refine the query and call again. A second, sharper query
  beats guessing — but keep queries purposeful; grounding on Gemini 3 models
  is billed per executed search.

## Fan out multi-facet questions

One facet per query, issued as parallel calls — not one overloaded query.
"Compare X's and Y's latest releases" is two searches (X's release, Y's
release), not one. Calls are independent, so parallel fan-out costs no
correctness and saves wall-clock time.

## Preserve citations when relaying — the step most often skipped

The text's inline `[n]` markers map to the numbered `## Sources` entries,
mirrored in structured content as `{index, title, uri}` objects. When you
relay the answer, keep the claim→source mapping intact: preserve the markers
with their links, or attach each `uri` next to the claim it supports. Never
strip the citations and present the text as your own knowledge —
verifiability is the entire point of a grounded search, and it is the first
thing lost when answers get paraphrased.

## Errors

Failures come back as tool errors with readable text, not protocol errors —
read the message and self-correct:

- `Invalid arguments for google_search: …` — you violated the input schema;
  fix the arguments, never retry verbatim.
- `google search failed: …` — backend/API failure; retry once, then surface
  the error to the user instead of looping.
