# mcp-gemini-search

An MCP server that provides Google Search functionality using Gemini's built-in Grounding with Google Search feature.

This repository ports the behavior of [`yukukotani/mcp-gemini-google-search`](https://github.com/yukukotani/mcp-gemini-google-search). This is the Python implementation, built on the official [`google-genai`](https://googleapis.github.io/python-genai/) SDK for Gemini API and Vertex AI access and the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) low-level server over stdio.

## Features

- Exposes `google_search`, `deep_research`, and `deep_research_result` as standard tools
- Uses Gemini's built-in Google Search grounding tool through the [Interactions API](https://ai.google.dev/gemini-api/docs/interactions) (`client.interactions.create`), currently in Beta
- Optionally enables the built-in [URL context](https://ai.google.dev/gemini-api/docs/interactions/url-context) and [code execution](https://ai.google.dev/gemini-api/docs/interactions/code-execution) tools alongside search grounding
- Stateless by design: every search is a single-shot interaction sent with `store=false`, so requests are never retained as server-side interaction history
- Returns the grounded response as clean Markdown: inline `[n]` citation markers plus a linked, ordered source list under a `## Sources` heading, normalized with [`mdformat`](https://github.com/hukkin/mdformat) (GitHub-flavored Markdown)
- Supports both Google AI Studio and Vertex AI

## Requirements

- Python 3.13 or later

## Installation

Install the server as a [`uv`](https://docs.astral.sh/uv/) tool from this repository:

```bash
uv tool install git+https://github.com/zchee/mcp-gemini-search
```

Or run it without installing using `uvx`:

```bash
uvx --from git+https://github.com/zchee/mcp-gemini-search mcp-gemini-search
```

Alternatively, install with `pip`:

```bash
pip install git+https://github.com/zchee/mcp-gemini-search
```

## Configuration

### Google AI Studio

Set either `GEMINI_API_KEY` or `GOOGLE_API_KEY`. When both are set, `GOOGLE_API_KEY` takes precedence.

```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-3.1-pro-preview" # optional
```

### Vertex AI

```bash
export GOOGLE_GENAI_USE_VERTEXAI="true"
export GOOGLE_CLOUD_PROJECT="your-gcp-project"   # required
export GOOGLE_CLOUD_LOCATION="global"            # optional, defaults to "global"
export GEMINI_MODEL="gemini-3.1-pro-preview"     # optional
```

`GOOGLE_GENAI_USE_VERTEXAI` is treated as enabled when set to one of `1`, `true`, `yes`, or `on` (case-insensitive). `GOOGLE_CLOUD_PROJECT` is required in this mode.

If no model is configured, the server defaults to `gemini-3.1-pro-preview`.

### Optional built-in tools

The Gemini `url_context` and `code_execution` built-in tools can be enabled alongside Google Search grounding. The environment variables below set the server-wide defaults, which individual `google_search` requests can override for one call. Both defaults are disabled when the variables are unset:

```bash
export GEMINI_ENABLE_URL_CONTEXT="1"    # let the model fetch URLs mentioned in the query
export GEMINI_ENABLE_CODE_EXECUTION="1" # let the model run Python for computational answers
```

Both flags accept the same truthy spellings as `GOOGLE_GENAI_USE_VERTEXAI` (`1`, `true`, `yes`, `on`; case-insensitive). Enabled tools can increase token usage, and on Gemini 3 models Google Search grounding is billed per executed search query.

## Usage

Run the server over stdio:

```bash
mcp-gemini-search
```

Optional file logging. Logs are written only to the given file (stdout is reserved for the MCP protocol):

```bash
mcp-gemini-search -logpath /tmp/mcp-gemini-search.log
```

## Deep Research

The server uses `deep-research-preview-04-2026` by default. Select the more exhaustive Max variant with:

```bash
export GEMINI_DEEP_RESEARCH_AGENT="deep-research-max-preview-04-2026"
```

Deep Research uses a two-tool start/poll workflow. `deep_research` starts a background run and immediately returns its `interaction_id` and initial `status`; it never waits for the multi-minute run to finish. Poll the same run with `deep_research_result`, passing the `interaction_id`. Its optional `wait_seconds` parameter long-polls for 0-60 seconds and defaults to 0. Continue polling until the status is `completed`, `failed`, or `cancelled`; a completed result contains the formatted Markdown report and cited sources.

Example transcript:

```text
deep_research({"query": "Compare QUIC and TCP for latency-sensitive services"})
→ {"interaction_id": "abc123", "status": "in_progress"}

deep_research_result({"interaction_id": "abc123", "wait_seconds": 60})
→ {"interaction_id": "abc123", "status": "in_progress"}

deep_research_result({"interaction_id": "abc123", "wait_seconds": 60})
→ {"interaction_id": "abc123", "status": "completed", "text": "...", "sources": [...]}
```

The configured agents are preview-tier Gemini agents, so their names and behavior may change. The agent environment variable allows switching versions without a server release.

> [!WARNING]
> Every `deep_research` call starts a multi-minute, billed, multi-step agent run. Never call `deep_research` twice for the same question; retain the returned `interaction_id` and poll `deep_research_result` instead.

Background execution requires server-side interaction storage: `store=false` is incompatible with `background=true`. On the paid tier, Deep Research interactions are retained for approximately 55 days. This is a privacy and retention difference from the stateless `google_search` tool, whose single-shot interactions are sent with `store=false`.

## Tool

### `google_search`

Performs a web search using Google Search (via the Gemini API) and returns the grounded results.

Parameters:

- `query` (string, required): The search query to find information on the web.
- `url_context` (boolean, optional): Override the `GEMINI_ENABLE_URL_CONTEXT` server default for this call only.
- `code_execution` (boolean, optional): Override the `GEMINI_ENABLE_CODE_EXECUTION` server default for this call only.

Output:

- `text` (string): The grounded response formatted as Markdown. Inline citations render as plain `[n]` markers whose numbers match the linked, ordered source list appended under a `## Sources` heading; each source URI appears exactly once, keeping token cost low for LLM consumers. The whole document is normalized with `mdformat` (GFM).
- `sources` (array): The cited sources, each with its 1-based citation `index` plus `title` and `uri` when available.

### `deep_research`

Starts a billed background Deep Research run and returns immediately.

Parameters:

- `query` (string, required): The research question.
- `plan_only` (boolean, optional): Request a collaborative research plan instead of executing it. Defaults to `false`.
- `previous_interaction_id` (string, optional): Continue, refine, or approve a prior Deep Research interaction.
- `agent` (string, optional): Override the server-configured agent for this call only with `deep-research-preview-04-2026` (faster and the default) or `deep-research-max-preview-04-2026` (more comprehensive). When omitted, the server-configured agent is used.

Output:

- `interaction_id` (string): The durable identifier to pass to `deep_research_result`.
- `status` (string): The initial interaction status, typically `in_progress`.

### `deep_research_result`

Fetches or briefly long-polls an existing Deep Research interaction. Reuse this tool instead of starting another run for the same question.

Parameters:

- `interaction_id` (string, required): The identifier returned by `deep_research`.
- `wait_seconds` (integer, optional): Long-poll duration from 0 through 60 seconds. Defaults to `0`.

Output:

- `interaction_id` (string): The fetched interaction identifier.
- `status` (string): `in_progress`, `requires_action`, `completed`, `failed`, or `cancelled`.
- `text` (string, completed results only): The formatted Markdown report.
- `sources` (array, completed results only): The cited sources.

## Development

```bash
uv sync

# Run the test suite
uv run pytest

# Format and lint
uv run ruff format --check
uv run ruff check

# Type-check
uv run ty check

# Benchmarks (deselected by default)
uv run pytest -m benchmark --benchmark-only

# Live API tests (require a real key; deselected by default)
RUN_LIVE_API=1 GEMINI_API_KEY="your-api-key" uv run pytest -m live

# Live Deep Research start/poll/cancel smoke test
RUN_LIVE_API=1 GEMINI_API_KEY="your-api-key" uv run pytest -m live

# Full multi-minute Deep Research run (explicit billed run)
RUN_SLOW=1 RUN_LIVE_API=1 GEMINI_API_KEY="your-api-key" uv run pytest -m "live and slow"
```
