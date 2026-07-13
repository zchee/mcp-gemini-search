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

"""Low-level MCP server exposing search and Deep Research tools."""

from __future__ import annotations

import enum
from typing import Any

from mcp import types
from mcp.server.lowlevel import Server

from mcp_gemini_search import __version__
from mcp_gemini_search.research import (
    DEEP_RESEARCH_AGENT,
    DEEP_RESEARCH_MAX_AGENT,
    DeepResearchService,
)
from mcp_gemini_search.search import GoogleSearchService

SERVER_NAME = "mcp-gemini-search"
WEBSITE_URL = "https://github.com/zchee/mcp-gemini-search"


class ToolName(enum.StrEnum):
    """The MCP tool names this server advertises."""

    GOOGLE_SEARCH = "google_search"
    DEEP_RESEARCH = "deep_research"
    DEEP_RESEARCH_RESULT = "deep_research_result"


SEARCH_TOOL_NAME = ToolName.GOOGLE_SEARCH
SEARCH_TOOL_DESCRIPTION = (
    "Performs a web search using Google Search (via the Gemini API) and returns "
    "the results. This tool is useful for finding information on the internet "
    "based on a query."
)
RESEARCH_TOOL_NAME = ToolName.DEEP_RESEARCH
RESEARCH_TOOL_DESCRIPTION = (
    "Starts a Gemini Deep Research agent run in the background and returns an "
    "interaction_id immediately; research typically takes several minutes and is "
    "a billed multi-step agent run. Never call this tool twice for the same "
    "question — poll deep_research_result with the returned interaction_id "
    "instead. Optional plan_only requests a research plan for review before "
    "execution; previous_interaction_id continues or refines a prior run."
)
RESEARCH_RESULT_TOOL_NAME = ToolName.DEEP_RESEARCH_RESULT
RESEARCH_RESULT_TOOL_DESCRIPTION = (
    "Fetches the status (and, once completed, the formatted report) of a "
    "deep_research run by its interaction_id. Call repeatedly until status is "
    "completed, failed, or cancelled. Optionally long-poll with wait_seconds "
    "(0–60) before returning a non-terminal status."
)

# inputSchema and outputSchema for google_search are pinned byte-for-byte by the
# golden tools/list response (tests/golden/tools_list.json); any change here
# must update the golden in lockstep. The mcp SDK validates tool input against
# inputSchema and structured output against outputSchema. Note the ``sources``
# type is ["null", "array"] (a nullable slice inherited from the retired Go
# server's schema generator), NOT "array".
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

_RESEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The research question or topic to investigate.",
        },
        "plan_only": {
            "type": "boolean",
            "default": False,
            "description": ("Request a research plan for review instead of executing the research immediately."),
        },
        "previous_interaction_id": {
            "type": "string",
            "description": "Continue, refine, or approve a prior deep_research interaction by its id.",
        },
        "agent": {
            "type": "string",
            "enum": [DEEP_RESEARCH_AGENT, DEEP_RESEARCH_MAX_AGENT],
            "description": (
                "Override the server-configured agent for this call only: "
                "'deep-research-preview-04-2026' is faster, "
                "'deep-research-max-preview-04-2026' is more comprehensive. "
                "Falls back to the server-configured agent when omitted."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_RESEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "interaction_id": {
            "type": "string",
            "description": "The interaction id of the started deep research run.",
        },
        "status": {
            "type": "string",
            "description": "The initial status of the background research run (typically in_progress).",
        },
    },
    "required": ["interaction_id", "status"],
    "additionalProperties": False,
}

_RESEARCH_RESULT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "interaction_id": {
            "type": "string",
            "description": "The interaction_id returned by deep_research.",
        },
        "wait_seconds": {
            "type": "integer",
            "minimum": 0,
            "maximum": 60,
            "default": 0,
            "description": (
                "Optionally long-poll up to this many seconds for the run to reach a terminal status before returning."
            ),
        },
    },
    "required": ["interaction_id"],
    "additionalProperties": False,
}

_RESEARCH_RESULT_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "interaction_id": {
            "type": "string",
            "description": "The interaction id of the deep research run.",
        },
        "status": {
            "type": "string",
            "description": "The current status of the research run.",
        },
        "text": {
            "type": "string",
            "description": ("The formatted research report as Markdown, present only when status is completed."),
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
    "required": ["interaction_id", "status"],
    "additionalProperties": False,
}


def create_server(
    service: GoogleSearchService,
    research: DeepResearchService,
) -> Server:
    """Create the low-level MCP server wired to search and research services.

    The server identity (name, version, website URL) reproduces the Go server's
    ``serverInfo`` exactly. The ``google_search`` tool returns its grounded text
    as unstructured content and the structured output dict as
    ``structuredContent``. The ``deep_research`` and ``deep_research_result``
    tools are always advertised.
    Failures propagate as exceptions, which the SDK converts into an ``isError``
    result carrying the exception message.
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
                name=SEARCH_TOOL_NAME,
                description=SEARCH_TOOL_DESCRIPTION,
                inputSchema=_INPUT_SCHEMA,
                outputSchema=_OUTPUT_SCHEMA,
            ),
            types.Tool(
                name=RESEARCH_TOOL_NAME,
                description=RESEARCH_TOOL_DESCRIPTION,
                inputSchema=_RESEARCH_INPUT_SCHEMA,
                outputSchema=_RESEARCH_OUTPUT_SCHEMA,
            ),
            types.Tool(
                name=RESEARCH_RESULT_TOOL_NAME,
                description=RESEARCH_RESULT_TOOL_DESCRIPTION,
                inputSchema=_RESEARCH_RESULT_INPUT_SCHEMA,
                outputSchema=_RESEARCH_RESULT_OUTPUT_SCHEMA,
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> tuple[list[types.ContentBlock], dict[str, Any]]:
        match name:
            case ToolName.GOOGLE_SEARCH:
                output = await service.search(arguments["query"])
                content: list[types.ContentBlock] = [types.TextContent(type="text", text=output.text)]
                return content, output.to_structured()

            case ToolName.DEEP_RESEARCH:
                start = await research.start(
                    arguments["query"],
                    plan_only=arguments.get("plan_only", False),
                    previous_interaction_id=arguments.get("previous_interaction_id", ""),
                    agent=arguments.get("agent", ""),
                )
                start_text = (
                    f"deep research started: interaction_id={start.interaction_id} "
                    f"status={start.status}; call deep_research_result with this "
                    f"interaction_id to check progress"
                )
                return [types.TextContent(type="text", text=start_text)], start.to_structured()

            case ToolName.DEEP_RESEARCH_RESULT:
                result = await research.result(
                    arguments["interaction_id"],
                    wait_seconds=arguments.get("wait_seconds", 0),
                )
                if result.status == "completed":
                    result_text = result.text
                else:
                    result_text = (
                        f"deep research {result.interaction_id} is {result.status}; "
                        f"call deep_research_result again with this interaction_id to check progress"
                    )
                return [types.TextContent(type="text", text=result_text)], result.to_structured()

            case _:
                raise ValueError(f"Unknown tool: {name}")

    return server
