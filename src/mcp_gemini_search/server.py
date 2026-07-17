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
import functools
from typing import Any

import jsonschema
from mcp.server import Server, ServerRequestContext
from mcp_types import (
    CallToolRequestParams,
    CallToolResult,
    ContentBlock,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

from mcp_gemini_search import __version__
from mcp_gemini_search._logging import logger
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


SEARCH_TOOL_DESCRIPTION = (
    "Search the web with Google Search grounding through Gemini. Returns a "
    "Markdown answer whose claims carry inline [n] citation markers that map to "
    "a numbered '## Sources' list of links. Use it for current events, fresh "
    "facts, or any claim that needs verifiable web sources; each call is one "
    "independent search, so refine the query and call again if the answer "
    "misses. Set url_context=true when the query contains URLs the model should "
    "open and read; set code_execution=true when the answer needs real "
    "computation (math, data processing) rather than retrieval alone."
)
RESEARCH_TOOL_DESCRIPTION = (
    "Start an asynchronous Gemini Deep Research run: an autonomous agent that "
    "searches the web, reads sources, and writes a long, citation-rich Markdown "
    "report. Returns interaction_id and status immediately and never waits for "
    "the report — the run continues in the background for several minutes and "
    "is billed per run. Workflow: call this exactly once per research question, "
    "keep the interaction_id, and poll deep_research_result until a terminal "
    "status. Never re-issue the same question to retry or check progress — that "
    "starts a second billed run; call this tool again only for a deliberate "
    "follow-up or plan approval via previous_interaction_id. For quick factual "
    "lookups use google_search instead. Set plan_only=true to receive a "
    "research plan for review first."
)
RESEARCH_RESULT_TOOL_DESCRIPTION = (
    "Fetch the current state of a deep_research run by interaction_id. While "
    "status is 'in_progress' the report is not ready — wait, then call this "
    "tool again, passing wait_seconds=60 so the server long-polls and saves "
    "round-trips. When status is 'completed' the result carries the full "
    "Markdown report (text) with [n] citation markers plus its sources list; "
    "'failed' and 'cancelled' are terminal. Always poll with this tool — never "
    "start a new deep_research run to check on an existing one."
)

# inputSchema and outputSchema for google_search are pinned byte-for-byte by the
# golden tools/list response (tests/golden/tools_list.json); any change here
# must update the golden in lockstep. mcp v2 dropped the SDK-side validation of
# tool arguments, so _call_tool enforces these input schemas itself with
# jsonschema before dispatch; structured output is validated against
# outputSchema on the client side. Note the ``sources`` type is ["null", "array"]
# (a nullable slice inherited from the retired Go server's schema generator),
# NOT "array".
_SEARCH_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "The web search query. Natural-language questions work well; "
                "specific names, versions, and dates sharpen the results."
            ),
        },
        "url_context": {
            "type": "boolean",
            "description": (
                "Set true to let the model open and read URLs written in the "
                "query; false to disable that. Omit to use the server default "
                "(GEMINI_ENABLE_URL_CONTEXT)."
            ),
        },
        "code_execution": {
            "type": "boolean",
            "description": (
                "Set true to let the model write and run Python when the answer "
                "needs real computation; false to disable that. Omit to use the "
                "server default (GEMINI_ENABLE_CODE_EXECUTION)."
            ),
        },
    },
    "required": ["query"],
    "additionalProperties": False,
}

_SOURCE_ITEM_SCHEMA: dict[str, Any] = {
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
}

_SEARCH_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The original search query.",
        },
        "text": {
            "type": "string",
            "description": (
                "The grounded answer as Markdown: inline [n] citation markers "
                "refer to the numbered entries of the trailing '## Sources' "
                "section when sources were cited."
            ),
        },
        "sources": {
            "type": ["null", "array"],
            "items": _SOURCE_ITEM_SCHEMA,
            "description": ("The cited sources; each index matches the [n] markers in text."),
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
            "description": (
                "The research question or task. Include scope, constraints, and "
                "the desired depth or output shape — richer briefs produce "
                "better reports."
            ),
        },
        "plan_only": {
            "type": "boolean",
            "default": False,
            "description": (
                "Set true to get a research plan for review instead of running "
                "the research. Approve or refine the plan with a follow-up "
                "deep_research call that sets previous_interaction_id."
            ),
        },
        "previous_interaction_id": {
            "type": "string",
            "description": (
                "The interaction_id of an earlier deep_research run: use it to "
                "approve or refine a proposed plan, or to ask a follow-up that "
                "builds on that run's findings."
            ),
        },
        "agent": {
            "type": "string",
            "enum": [DEEP_RESEARCH_AGENT, DEEP_RESEARCH_MAX_AGENT],
            "description": (
                "Deep Research variant for this call: "
                "'deep-research-preview-04-2026' returns faster; "
                "'deep-research-max-preview-04-2026' digs deeper and takes "
                "longer. Omit to use the server-configured default."
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
            "description": ("The durable id of the started run. Save it — deep_research_result needs it."),
        },
        "status": {
            "type": "string",
            "description": "The initial run status, normally 'in_progress'.",
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
                "Seconds (0-60) the server may hold this request waiting for "
                "the run to finish before answering. 0 returns the current "
                "status immediately; 60 minimizes polling round-trips."
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
            "description": "The id of the polled run (echoes the request).",
        },
        "status": {
            "type": "string",
            "description": (
                "'in_progress' means poll again; 'completed' means the report "
                "is in text; 'failed' and 'cancelled' are terminal."
            ),
        },
        "text": {
            "type": "string",
            "description": (
                "The finished research report as Markdown with inline [n] "
                "citation markers; present only when status is 'completed'."
            ),
        },
        "sources": {
            "type": ["null", "array"],
            "items": _SOURCE_ITEM_SCHEMA,
            "description": ("The report's cited sources; each index matches the [n] markers in text."),
        },
    },
    "required": ["interaction_id", "status"],
    "additionalProperties": False,
}


_TOOL_INPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    ToolName.GOOGLE_SEARCH: _SEARCH_INPUT_SCHEMA,
    ToolName.DEEP_RESEARCH: _RESEARCH_INPUT_SCHEMA,
    ToolName.DEEP_RESEARCH_RESULT: _RESEARCH_RESULT_INPUT_SCHEMA,
}


async def _list_tools(  # noqa: RUF029
    ctx: ServerRequestContext,
    params: PaginatedRequestParams | None,
) -> ListToolsResult:
    """Return the three tool declarations with their pinned schemas."""
    return ListToolsResult(
        tools=[
            Tool(
                name=ToolName.GOOGLE_SEARCH,
                description=SEARCH_TOOL_DESCRIPTION,
                input_schema=_SEARCH_INPUT_SCHEMA,
                output_schema=_SEARCH_OUTPUT_SCHEMA,
            ),
            Tool(
                name=ToolName.DEEP_RESEARCH,
                description=RESEARCH_TOOL_DESCRIPTION,
                input_schema=_RESEARCH_INPUT_SCHEMA,
                output_schema=_RESEARCH_OUTPUT_SCHEMA,
            ),
            Tool(
                name=ToolName.DEEP_RESEARCH_RESULT,
                description=RESEARCH_RESULT_TOOL_DESCRIPTION,
                input_schema=_RESEARCH_RESULT_INPUT_SCHEMA,
                output_schema=_RESEARCH_RESULT_OUTPUT_SCHEMA,
            ),
        ]
    )


async def _call_tool(
    ctx: ServerRequestContext,
    params: CallToolRequestParams,
    *,
    service: GoogleSearchService,
    research: DeepResearchService,
) -> CallToolResult:
    """Validate the arguments with jsonschema, dispatch to a tool, and wrap errors.

    Every failure path returns ``is_error=True`` with the error text so the
    calling LLM can read and self-correct; mcp v2 would otherwise surface
    raised exceptions as opaque JSON-RPC protocol errors.
    """
    name = params.name
    arguments = params.arguments or {}

    schema = _TOOL_INPUT_SCHEMAS.get(name)
    if schema is None:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            is_error=True,
        )

    try:
        jsonschema.validate(instance=arguments, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Invalid arguments for {name}: {e.message}")],
            is_error=True,
        )

    try:
        content: list[ContentBlock]
        structured: dict[str, Any]
        match name:
            case ToolName.GOOGLE_SEARCH:
                output = await service.search(
                    arguments["query"],
                    url_context=arguments.get("url_context"),
                    code_execution=arguments.get("code_execution"),
                )
                content = [TextContent(type="text", text=output.text)]
                structured = output.to_structured()

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
                content = [TextContent(type="text", text=start_text)]
                structured = start.to_structured()

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
                content = [TextContent(type="text", text=result_text)]
                structured = result.to_structured()

            case _:
                # Unreachable while _TOOL_INPUT_SCHEMAS and this match list the
                # same tools; a tool added to one but not the other fails loudly
                # here instead of as a NameError on `content` below.
                raise AssertionError(f"unhandled tool: {name}")

        return CallToolResult(
            content=content,
            structured_content=structured,
            is_error=False,
        )
    except Exception as e:
        logger.warning("tool %s failed: %s", name, e)
        return CallToolResult(
            content=[TextContent(type="text", text=str(e))],
            is_error=True,
        )


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
    Tool failures are converted here into ``isError`` results carrying the
    exception message so callers retain the server's established error behavior.
    """
    return Server(
        name=SERVER_NAME,
        version=__version__,
        website_url=WEBSITE_URL,
        on_list_tools=_list_tools,
        on_call_tool=functools.partial(_call_tool, service=service, research=research),
    )
