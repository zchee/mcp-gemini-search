# mcp-gemini-google-search

An MCP server that provides Google Search functionality using Gemini's built-in Grounding with Google Search feature.

This repository ports the behavior of [`yukukotani/mcp-gemini-google-search`](https://github.com/yukukotani/mcp-gemini-google-search). This is the Python implementation, built on the official [`google-genai`](https://googleapis.github.io/python-genai/) SDK for Gemini API and Vertex AI access and the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) low-level server over stdio.

## Features

- Exposes a single MCP tool: `google_search`
- Uses Gemini's built-in Google Search grounding tool
- Returns grounded text with inline citations and an appended source list
- Supports both Google AI Studio and Vertex AI

## Requirements

- Python 3.13 or later

## Installation

Install the server as a [`uv`](https://docs.astral.sh/uv/) tool from this repository:

```bash
uv tool install git+https://github.com/zchee/mcp-gemini-google-search
```

Or run it without installing using `uvx`:

```bash
uvx --from git+https://github.com/zchee/mcp-gemini-google-search mcp-gemini-google-search
```

Alternatively, install with `pip`:

```bash
pip install git+https://github.com/zchee/mcp-gemini-google-search
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

## Usage

Run the server over stdio:

```bash
mcp-gemini-google-search
```

Optional file logging. Logs are written only to the given file (stdout is reserved for the MCP protocol):

```bash
mcp-gemini-google-search -logpath /tmp/mcp-gemini-google-search.log
```

## Tool

### `google_search`

Performs a web search using Google Search (via the Gemini API) and returns the grounded results.

Parameters:

- `query` (string, required): The search query to find information on the web.

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
```
