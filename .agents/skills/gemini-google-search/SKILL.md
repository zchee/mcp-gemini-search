---
name: gemini-google-search
description: Drive the google_search tool of the mcp-gemini-search MCP server — a Google-Search-grounded Gemini call that returns a Markdown answer with inline [n] citations and a numbered Sources list. Use this skill whenever the user asks to search the web, look up current events, verify a fact against live sources, check the latest version/release/pricing/news of anything, or when any claim needs verifiable citations — even if they never say "search" or "google_search". Also use it to decide when a request should instead go to the deep_research tool.
---

# google_search (mcp-gemini-search)

`google_search` runs one Google-Search-grounded Gemini interaction and returns
the grounded answer. Depending on how the server is registered, the tool may
appear as `mcp__mcp-gemini-search__google_search` or under another server
alias — match on the trailing `google_search`.

## When to use it (and when not)

- Use for current events, fresh facts, version/release checks, and any claim
  that needs verifiable web sources. Results carry real URLs you can cite.
- Do NOT use it for long multi-source investigations that need a structured
  report — that is the `deep_research` tool (see the `gemini-deep-research`
  skill). A good heuristic: if the user wants an *answer*, search; if they
  want a *report*, research.
- Prefer it over unsourced recall for anything time-sensitive: your training
  data is stale, the grounded answer is not.

## Calling contract

Input (`additionalProperties: false` — anything else is rejected):

| field | type | meaning |
|---|---|---|
| `query` (required) | string | Natural-language question. Specific names, versions, and dates sharpen results. |
| `url_context` | boolean | `true` lets the model open and read URLs written in the query. Omit to use the server default (`GEMINI_ENABLE_URL_CONTEXT`). |
| `code_execution` | boolean | `true` lets the model write and run Python for real computation. Omit to use the server default (`GEMINI_ENABLE_CODE_EXECUTION`). |

Set `url_context=true` exactly when the query itself contains URLs the model
should fetch ("summarize https://…", "compare these two pages"). Set
`code_execution=true` when the answer requires computation — unit conversion
over fetched data, aggregating numbers, date math — not plain retrieval.
Passing an explicit `false` deliberately disables a server-side default for
one call.

## Each call is one independent search

The server is stateless (`store=false`): there is no conversation memory
between calls, and the model never sees your previous queries. Consequences:

- Every query must be self-contained. Never write "the library mentioned
  above" — name it.
- If the answer misses, refine the query (add the version, the year, the
  exact error string) and call again. Iterating is cheap and expected.
- For multi-facet questions, fan out several independent calls in parallel
  (one facet per query) instead of one overloaded query.

Google Search grounding on Gemini 3 models is billed per executed search
query, so keep queries purposeful — but a second, sharper query beats
guessing.

## Reading the result

The text content is Markdown: claims carry inline `[n]` markers that map to a
numbered `## Sources` section of links at the end. The structured content
mirrors it:

```json
{"query": "...", "text": "markdown with [n] markers", "sources": [{"index": 1, "title": "...", "uri": "https://..."}]}
```

`sources` may be `null` or absent when nothing was cited. When relaying the
answer, keep the claim→source mapping intact: either preserve the `[n]`
markers with their links or attach the `uri` next to each claim. Do not strip
citations and present the text as your own knowledge.

## Errors

Failures come back as tool errors with readable text, not protocol errors —
read the message and self-correct:

- `Invalid arguments for google_search: …` — you violated the input schema
  (unknown key, wrong type). Fix the arguments; do not retry verbatim.
- `search query cannot be empty` — send a real query.
- `google search failed: …` — backend/API failure; retry once, then surface
  the error to the user instead of looping.
