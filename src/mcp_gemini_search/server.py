# Copyright 2026 The mcp-gemini-search Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Low-level MCP server exposing the ``google_search`` tool over stdio."""

from __future__ import annotations

from typing import Any

from mcp import types
from mcp.server.lowlevel import Server

from mcp_gemini_search import __version__
from mcp_gemini_search.search import GoogleSearchService

SERVER_NAME = "mcp-gemini-search"
WEBSITE_URL = "https://github.com/zchee/mcp-gemini-search"
TOOL_NAME = "google_search"
TOOL_DESCRIPTION = (
    "Performs a web search using Google Search (via the Gemini API) and returns "
    "the results. This tool is useful for finding information on the internet "
    "based on a query."
)

# inputSchema and outputSchema are pinned byte-for-byte by the golden tools/list
# response (tests/golden/tools_list.json); any change here must update the
# golden in lockstep. The mcp SDK validates tool input against inputSchema and
# structured output against outputSchema. Note the ``sources`` type is
# ["null", "array"] (a nullable slice inherited from the retired Go server's
# schema generator), NOT "array".
_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to find information on the web.",
        }
    },
    "required": ["query"],
    "additionalProperties": False,
}

_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The original search query.",
        },
        "text": {
            "type": "string",
            "description": (
                "The grounded response text formatted as Markdown, with inline citation "
                "markers and an appended Sources section when available."
            ),
        },
        "sources": {
            "type": ["null", "array"],
            "items": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": ("The 1-based citation index shown in the response text."),
                    },
                    "title": {
                        "type": "string",
                        "description": "The title of the cited source.",
                    },
                    "uri": {
                        "type": "string",
                        "description": "The source URL or canonical URI.",
                    },
                },
                "required": ["index"],
                "additionalProperties": False,
            },
            "description": "The sources referenced by the grounded response.",
        },
    },
    "required": ["query", "text"],
    "additionalProperties": False,
}


def create_server(service: GoogleSearchService) -> Server:
    """Create the low-level MCP server wired to ``service``.

    The server identity (name, version, website URL) reproduces the Go server's
    ``serverInfo`` exactly. The single ``google_search`` tool returns its grounded
    text as unstructured content and the structured output dict as
    ``structuredContent``; search failures propagate as exceptions, which the SDK
    converts into an ``isError`` result carrying the exception message.
    """
    server = Server(
        name=SERVER_NAME,
        version=__version__,
        website_url=WEBSITE_URL,
    )

    # The low-level SDK awaits tool handlers, so this must stay async.
    @server.list_tools()
    async def list_tools() -> list[types.Tool]:  # noqa: RUF029
        return [
            types.Tool(
                name=TOOL_NAME,
                description=TOOL_DESCRIPTION,
                inputSchema=_INPUT_SCHEMA,
                outputSchema=_OUTPUT_SCHEMA,
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> tuple[list[types.ContentBlock], dict[str, Any]]:
        if name != TOOL_NAME:
            raise ValueError(f"Unknown tool: {name}")
        output = await service.search(arguments["query"])
        content: list[types.ContentBlock] = [types.TextContent(type="text", text=output.text)]
        return content, output.to_structured()

    return server
