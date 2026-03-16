# mcp-gemini-google-search

An MCP server that provides Google Search functionality using Gemini's built-in Grounding with Google Search feature.

This repository ports the behavior of [`yukukotani/mcp-gemini-google-search`](https://github.com/yukukotani/mcp-gemini-google-search) to Go and uses [`google.golang.org/genai`](https://pkg.go.dev/google.golang.org/genai) for Gemini API and Vertex AI access.

## Features

- Exposes a single MCP tool: `google_search`
- Uses Gemini's built-in Google Search grounding tool
- Returns grounded text with inline citations and an appended source list
- Supports both Google AI Studio and Vertex AI
- Preserves the upstream environment-variable surface while also accepting the native Go SDK environment variables

## Installation

```bash
go install github.com/zchee/mcp-gemini-google-search@latest
```

## Configuration

### Google AI Studio

Set either `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

```bash
export GEMINI_API_KEY="your-api-key"
export GEMINI_MODEL="gemini-3.1-pro-preview" # optional
```

### Vertex AI

The upstream-compatible variables are:

```bash
export GEMINI_PROVIDER="vertex"
export VERTEX_PROJECT_ID="your-gcp-project"
export VERTEX_LOCATION="global" # optional
export GEMINI_MODEL="gemini-3.1-pro-preview" # optional
```

The Go SDK-native variables are also supported:

```bash
export GOOGLE_GENAI_USE_VERTEXAI="true"
export GOOGLE_CLOUD_PROJECT="your-gcp-project"
export GOOGLE_CLOUD_LOCATION="global"
export GEMINI_MODEL="gemini-3.1-pro-preview" # optional
```

If no model is configured, the server defaults to `gemini-3.1-pro-preview`.

## Usage

Run the server over stdio:

```bash
mcp-gemini-google-search
```

Optional logging:

```bash
mcp-gemini-google-search -logpath /tmp/mcp-gemini-google-search.log
```

## Tool

### `google_search`

Search Google for information through Gemini grounding.

Parameters:

- `query` (string, required): Search query

## Development

```bash
go test ./...
go test -race ./...
go build ./...
```
