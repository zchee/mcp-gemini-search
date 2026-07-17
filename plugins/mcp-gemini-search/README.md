# mcp-gemini-search plugin

One plugin directory that serves both Claude Code (`.claude-plugin/`) and Codex (`.codex-plugin/`). Both loaders share the same `skills/` and `.mcp.json`.

## What it provides

- **MCP server** `mcp-gemini-search`, launched with `uvx` from [zchee/mcp-gemini-search](https://github.com/zchee/mcp-gemini-search), exposing `google_search`, `deep_research`, and `deep_research_result`.
- **Three skills** that keep the tools cheap and the citations intact:

| Skill | Teaches |
| --- | --- |
| `gemini-google-search` | When to use grounded search vs. Deep Research, self-contained query composition, relaying answers without dropping `[n]` citations. |
| `gemini-deep-research` | Starting exactly one billed background run per question and saving its `interaction_id`. |
| `gemini-deep-research-result` | Polling with `wait_seconds=60` long-polls and retrieving the finished report without starting duplicate runs. |

## Requirements

- `uv` on `PATH` (the server runs via `uvx`).
- Authentication in the environment that launches the client — either `GEMINI_API_KEY` / `GOOGLE_API_KEY` (Google AI Studio) or `GOOGLE_GENAI_USE_VERTEXAI=true` with `GOOGLE_CLOUD_PROJECT` (Vertex AI). See the [repository README](../../README.md#configuration) for the full environment reference, including `GEMINI_MODEL`, `GEMINI_SERVICE_TIER`, and the URL-context / code-execution toggles.

## Install in Claude Code

```
/plugin marketplace add zchee/mcp-gemini-search
/plugin install mcp-gemini-search@mcp-gemini-search
```

Or load it for one session without installing:

```bash
claude --plugin-dir ./plugins/mcp-gemini-search
```

The first launch on a cold `uv` cache builds the package from git and can exceed Claude Code's default MCP startup timeout; retry once or raise it with `MCP_TIMEOUT=60000 claude`.

## Install in Codex

The `.codex-plugin/plugin.json` manifest points at the same `skills/` and `.mcp.json`; see the repository README's "Bundled Codex plugin configuration" section.
