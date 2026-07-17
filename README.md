# mcp-gemini-search

An MCP stdio server for Google-grounded Gemini answers and asynchronous Gemini Deep Research reports.

This repository ports the behavior of [`yukukotani/mcp-gemini-google-search`](https://github.com/yukukotani/mcp-gemini-google-search) and extends it with Deep Research. The Python implementation uses the official [`google-genai`](https://googleapis.github.io/python-genai/) SDK for Gemini API and Vertex AI access and the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) low-level server.

## Features

- Always advertises three standard tools: `google_search`, `deep_research`, and `deep_research_result`
- Runs Google Search grounding through the [Interactions API](https://ai.google.dev/gemini-api/docs/interactions)
- Optionally enables [URL context](https://ai.google.dev/gemini-api/docs/interactions/url-context) and [code execution](https://ai.google.dev/gemini-api/docs/interactions/code-execution) per server or per search request
- Sends every `google_search` as an independent interaction with `store=false`
- Starts Deep Research as a provider-managed background interaction and returns a durable `interaction_id` immediately
- Returns readable text content plus schema-backed `structuredContent`; failures are readable MCP tool errors rather than opaque protocol errors
- Formats successful answers and reports as GitHub-flavored Markdown with inline `[n]` markers and a linked `## Sources` list
- Supports Google AI Studio and Vertex AI

## Choose a tool

| Need                                                               | Tool                   | Behavior                                                                                            |
| ------------------------------------------------------------------ | ---------------------- | --------------------------------------------------------------------------------------------------- |
| A current fact, focused lookup, URL inspection, or computed answer | `google_search`        | Returns one grounded answer synchronously. Refine and call again if needed.                         |
| A comprehensive comparison, survey, or long report                 | `deep_research`        | Starts one billed background run and returns its `interaction_id`; it does not wait for the report. |
| Progress or the final report for an existing run                   | `deep_research_result` | Fetches or long-polls by `interaction_id`; it never starts another research run.                    |

## Requirements

- Python 3.13 or later
- mcp[cli] 2.0.0b1 (pre-release MCP Python SDK v2)

## Installation

Install the server as a [`uv`](https://docs.astral.sh/uv/) tool from this repository:

```bash
uv tool install --prerelease=allow git+https://github.com/zchee/mcp-gemini-search
```

Or run it without installing using `uvx`:

```bash
uvx --prerelease=allow --from git+https://github.com/zchee/mcp-gemini-search mcp-gemini-search
```

Alternatively, install with `pip`:

```bash
pip install --pre git+https://github.com/zchee/mcp-gemini-search
```

### Bundled Codex plugin configuration

The bundled Codex plugin exposes the authentication environment variables from the client process and uses:

```json
{
  "mcpServers": {
    "mcp-gemini-search": {
      "command": "uvx",
      "args": [
        "--prerelease=allow",
        "--from",
        "git+https://github.com/zchee/mcp-gemini-search",
        "mcp-gemini-search"
      ],
      "startup_timeout_sec": 60
    }
  }
}
```

The server itself also parses `$CODEX_HOME/.env` (default `~/.codex/.env`) and `$CLAUDE_HOME/.env` (default `~/.claude/.env`) at startup, so keys stored there are picked up automatically — see [Client dotenv files](#client-dotenv-files).

### Bundled Claude Code plugin

This repository is also a Claude Code plugin (`.claude-plugin/plugin.json` at the root). It registers the MCP server and three skills — `gemini-google-search`, `gemini-deep-research`, and `gemini-deep-research-result` — that teach the client when to search versus research, how to relay answers without dropping citations, and how to poll a Deep Research run without starting a duplicate billed run.

Install it from this repository's marketplace inside Claude Code:

```
/plugin marketplace add zchee/mcp-gemini-search
/plugin install mcp-gemini-search@mcp-gemini-search
```

Or load it for a single session without installing:

```bash
claude --plugin-dir .
```

The plugin starts the server with `uvx` from this repository, so `uv` must be on `PATH`, and the environment variables from [Configuration](#configuration) must be exported in the shell that launches Claude Code — or stored in `~/.claude/.env`, which the server parses at startup ([Client dotenv files](#client-dotenv-files)).

On a cold `uv` cache the first launch clones and builds the package, which can exceed Claude Code's default MCP startup timeout (Claude Code ignores the Codex-only `startup_timeout_sec` field). If the server fails to start once, launch again — the build is cached — or raise the timeout with `MCP_TIMEOUT=60000 claude`.

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

### Client dotenv files

At startup the server parses `$CODEX_HOME/.env` — `~/.codex/.env` when `CODEX_HOME` is unset or empty — and then `$CLAUDE_HOME/.env` — `~/.claude/.env` — with [python-dotenv](https://github.com/theskumar/python-dotenv) and loads the entries into its own environment. Variables that are already exported always take precedence, the Codex file wins over the Claude file when both define a variable, and missing files are a silent no-op. Codex CLI reads its dotenv file for itself but does not pass it to the MCP servers it spawns, so storing `GEMINI_API_KEY` in one of these files is enough for Codex CLI, Claude Code, and this server — no `env` table is needed in `config.toml`. If you relocate a directory via `CODEX_HOME` or `CLAUDE_HOME`, make sure that variable also reaches the server — export it, or set it in the client's `env` configuration.

### Environment reference

| Variable                       | Default                         | Behavior                                                      |
| ------------------------------ | ------------------------------- | ------------------------------------------------------------- |
| `GOOGLE_API_KEY`               | none                            | Google AI Studio key; takes precedence over `GEMINI_API_KEY`. |
| `GEMINI_API_KEY`               | none                            | Google AI Studio key used when `GOOGLE_API_KEY` is empty.     |
| `GEMINI_MODEL`                 | `gemini-3.1-pro-preview`        | Model used by `google_search`.                                |
| `GOOGLE_GENAI_USE_VERTEXAI`    | disabled                        | Selects Vertex AI when set to a recognized truthy value.      |
| `GOOGLE_CLOUD_PROJECT`         | none                            | Required when Vertex AI is enabled.                           |
| `GOOGLE_CLOUD_LOCATION`        | `global`                        | Vertex AI location.                                           |
| `GEMINI_ENABLE_URL_CONTEXT`    | disabled                        | Server default for URL context on `google_search`.            |
| `GEMINI_ENABLE_CODE_EXECUTION` | disabled                        | Server default for code execution on `google_search`.         |
| `GEMINI_DEEP_RESEARCH_AGENT`   | `deep-research-preview-04-2026` | Server default for `deep_research`.                           |
| `GEMINI_SERVICE_TIER`          | none                            | Service tier (`flex`, `standard`, `priority`) for both tools. |
| `CODEX_HOME`                   | `~/.codex`                      | Dotenv directory parsed at startup; exported variables win.   |
| `CLAUDE_HOME`                  | `~/.claude`                     | Second dotenv directory parsed at startup after `CODEX_HOME`. |

Every variable above is also recognized with an `MCP_GEMINI_` prefix — for example `MCP_GEMINI_GEMINI_API_KEY` or `MCP_GEMINI_GEMINI_MODEL`. A non-blank prefixed variable takes precedence over its unprefixed name, so this server can be configured independently of other tools that read the shared names. For the API keys, both prefixed keys take precedence over both unprefixed ones: `MCP_GEMINI_GOOGLE_API_KEY` > `MCP_GEMINI_GEMINI_API_KEY` > `GOOGLE_API_KEY` > `GEMINI_API_KEY`.

### Optional built-in tools

The Gemini `url_context` and `code_execution` built-in tools can be enabled alongside Google Search grounding. The environment variables below set the server-wide defaults, which individual `google_search` requests can override for one call. Both defaults are disabled when the variables are unset:

```bash
export GEMINI_ENABLE_URL_CONTEXT="1"    # let the model fetch URLs mentioned in the query
export GEMINI_ENABLE_CODE_EXECUTION="1" # let the model run Python for computational answers
```

Both flags accept the same truthy spellings as `GOOGLE_GENAI_USE_VERTEXAI` (`1`, `true`, `yes`, `on`; case-insensitive). Enabled tools can increase token usage, and on Gemini 3 models Google Search grounding is billed per executed search query.

Optionally pin the Interactions API service tier for both `google_search` and `deep_research`:

```bash
export GEMINI_SERVICE_TIER="flex"  # or standard, priority
```

## Running

Run the server over stdio:

```bash
mcp-gemini-search
```

Optional file logging. Logs are written only to the given file (stdout is reserved for the MCP protocol):

```bash
mcp-gemini-search --logpath /tmp/mcp-gemini-search.log
```

Both `-logpath` and `--logpath` are accepted. Without this option the server does not emit routine logs; fatal startup errors still go to stderr.

## Deep Research

The server uses `deep-research-preview-04-2026` by default. Select the more exhaustive Max variant with:

```bash
export GEMINI_DEEP_RESEARCH_AGENT="deep-research-max-preview-04-2026"
```

Deep Research uses a two-tool start/poll workflow. `deep_research` starts a background run and immediately returns its `interaction_id` and initial `status`; it never waits for the multi-minute run to finish. Save that id before doing anything else: it is the only handle accepted by `deep_research_result`.

```text
deep_research(query)
        |
        v
interaction_id + initial status
        |
        v
deep_research_result(interaction_id, wait_seconds=60)
        |
        +-- non-terminal status -> poll the same interaction_id again
        +-- completed           -> Markdown text + optional sources
        `-- failed/cancelled    -> MCP tool error; stop polling
```

`deep_research_result` defaults to `wait_seconds=0`, which performs one immediate status fetch. Use `wait_seconds=60` while waiting for completion: the server rechecks the interaction about every five seconds for up to one minute, reducing client round-trips. Any non-terminal response contains `interaction_id` and `status`; a completed response also contains the Markdown report and any cited sources. A failed or cancelled run is returned as `isError=true` with a readable failure message, not as a normal structured status result.

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

`plan_only=true` starts a collaborative-planning interaction rather than immediately producing the full report. Poll that interaction to retrieve the plan, then approve or refine it with a new `deep_research` call whose `previous_interaction_id` is the plan's id. The same field creates a follow-up that builds on a completed run. Each approval, refinement, or follow-up call creates a new interaction and returns a new id.

Deep Research is stateful at the provider because the background run must be retrieved later by id. Treat research prompts and reports according to the current retention policy of your Gemini API or Vertex AI account. This differs from `google_search`, which explicitly sends `store=false`.

## Tool reference

### `google_search`

Performs a web search using Google Search (via the Gemini API) and returns the grounded results.

Parameters:

- `query` (string, required): The search query to find information on the web.
- `url_context` (boolean, optional): Override the `GEMINI_ENABLE_URL_CONTEXT` server default for this call only.
- `code_execution` (boolean, optional): Override the `GEMINI_ENABLE_CODE_EXECUTION` server default for this call only.

Output:

- `query` (string): The original query.
- `text` (string): The grounded response formatted as Markdown. Inline citations render as plain `[n]` markers whose numbers match the linked, ordered source list appended under a `## Sources` heading; each source URI appears exactly once. The whole document is normalized with `mdformat` (GFM).
- `sources` (array, optional): Cited sources. Each source always has a 1-based `index`; `title` and `uri` are included only when available. The key is omitted when the answer has no usable citations.

The MCP text content is the same Markdown stored in `structuredContent.text`.

### `deep_research`

Starts a billed background Deep Research run and returns immediately.

Parameters:

- `query` (string, required): The research question.
- `plan_only` (boolean, optional): Request a collaborative research plan instead of executing it. Defaults to `false`.
- `previous_interaction_id` (string, optional): Continue, refine, or approve a prior Deep Research interaction.
- `agent` (string, optional): Override the server-configured agent for this call only with `deep-research-preview-04-2026` (faster and the default) or `deep-research-max-preview-04-2026` (more comprehensive). When omitted, the server-configured agent is used.

Output:

- `interaction_id` (string): The durable identifier to pass to `deep_research_result`.
- `status` (string): The provider's initial interaction status, normally `in_progress`.

The MCP text content also names the `interaction_id`, the status, and the required next tool so clients do not have to decode `structuredContent` before preserving the id.

### `deep_research_result`

Fetches or briefly long-polls an existing Deep Research interaction. Reuse this tool instead of starting another run for the same question.

Parameters:

- `interaction_id` (string, required): The identifier returned by `deep_research`.
- `wait_seconds` (integer, optional): Long-poll duration from 0 through 60 seconds. Defaults to `0`.

Output:

- `interaction_id` (string): The fetched interaction identifier.
- `status` (string): The current non-terminal status, or `completed`. Non-terminal provider statuses are returned as-is.
- `text` (string, completed results only): The formatted Markdown report.
- `sources` (array, optional, completed results only): Cited sources; omitted when the report has none.

For a non-terminal status, the MCP text content is a progress message. For `completed`, it is the full report. `failed` and `cancelled` are terminal MCP tool errors and therefore do not produce normal structured output.

## Input validation and errors

All three input schemas reject missing required fields, wrong types, and unknown properties. The server validates arguments before dispatch and returns failures as ordinary MCP tool results with `isError=true` and readable text:

- `Invalid arguments for <tool>: ...` for schema violations
- `search query cannot be empty` or `research query cannot be empty` for blank queries
- `interaction id cannot be empty` for a blank Deep Research id
- `google search failed: ...` or `deep research failed: ...` for backend failures
- `deep research cancelled` or `deep research <status>: <detail>` for terminal research failures

Callers should inspect `isError` before reading `structuredContent`.

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
