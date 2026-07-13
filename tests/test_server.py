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

"""In-process integration tests of the MCP server against a stub service."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import orjson
import pytest
from google import genai
from mcp import types
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

from mcp_gemini_search import __version__
from mcp_gemini_search.config import DEFAULT_MODEL
from mcp_gemini_search.research import (
    DeepResearchResult,
    DeepResearchService,
    DeepResearchStart,
)
from mcp_gemini_search.search import (
    GoogleSearchOutput,
    GoogleSearchService,
    GoogleSearchSource,
)
from mcp_gemini_search.server import create_server

pytestmark = pytest.mark.anyio

# The Go golden was captured with a client that requested 2025-06-18; the server
# echoes the requested version, so the test client must request the same value.
REQUESTED_PROTOCOL_VERSION = "2025-06-18"

_GOLDEN_DIR = Path(__file__).parent / "golden"


def _load_golden(name: str) -> dict[str, Any]:
    return orjson.loads((_GOLDEN_DIR / name).read_text(encoding="utf-8"))


class _StubService(GoogleSearchService):
    """A service that returns a fixed grounded output for any query."""

    def __init__(self, output: GoogleSearchOutput) -> None:
        super().__init__(model=DEFAULT_MODEL, interactions=None)
        self._output = output

    async def search(self, query: str) -> GoogleSearchOutput:
        return self._output


class _StubResearchService(DeepResearchService):
    """A research service that returns canned start/result values."""

    def __init__(
        self,
        *,
        start: DeepResearchStart | None = None,
        result: DeepResearchResult | None = None,
        error: Exception | None = None,
    ) -> None:
        super().__init__(agent="deep-research-preview-04-2026", interactions=None)
        self._start = start or DeepResearchStart(interaction_id="dr-1", status="in_progress")
        self._result = result or DeepResearchResult(
            interaction_id="dr-1",
            status="completed",
            text="Report body",
            sources=(GoogleSearchSource(index=1, title="Src", uri="https://s.example"),),
        )
        self._error = error

    async def start(
        self,
        query: str,
        *,
        plan_only: bool = False,
        previous_interaction_id: str = "",
    ) -> DeepResearchStart:
        if self._error is not None:
            raise self._error
        return self._start

    async def result(self, interaction_id: str, *, wait_seconds: int = 0) -> DeepResearchResult:
        if self._error is not None:
            raise self._error
        return self._result


_GROUNDED_OUTPUT = GoogleSearchOutput(
    query="who wrote the go language",
    text=(
        "Alpha [1]Beta[1][2]\n\n## Sources\n\n1. [First](https://first.example)\n2. [Second](https://second.example)"
    ),
    sources=(
        GoogleSearchSource(index=1, title="First", uri="https://first.example"),
        GoogleSearchSource(index=2, title="Second", uri="https://second.example"),
    ),
)


@asynccontextmanager
async def _session(
    service: GoogleSearchService,
    research: DeepResearchService | None = None,
) -> AsyncGenerator[tuple[ClientSession, types.InitializeResult]]:
    """Yield a client session connected to the server, initialized at 2025-06-18."""
    server = create_server(service, research)
    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        client_read, client_write = client_streams
        server_read, server_write = server_streams
        async with anyio.create_task_group() as tg:
            tg.start_soon(
                lambda: server.run(
                    server_read,
                    server_write,
                    server.create_initialization_options(),
                    raise_exceptions=True,
                )
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    init = await session.send_request(
                        types.ClientRequest(
                            types.InitializeRequest(
                                params=types.InitializeRequestParams(
                                    protocolVersion=REQUESTED_PROTOCOL_VERSION,
                                    capabilities=types.ClientCapabilities(),
                                    clientInfo=types.Implementation(name="test-client", version="0.0.0"),
                                )
                            )
                        ),
                        types.InitializeResult,
                    )
                    await session.send_notification(types.ClientNotification(types.InitializedNotification()))
                    yield session, init
            finally:
                tg.cancel_scope.cancel()


async def test_initialize_negotiates_golden_protocol_and_server_info() -> None:
    """initialize echoes the requested protocol version and the golden serverInfo."""
    golden = _load_golden("initialize.json")["result"]
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (_session_obj, init):
        assert init.protocolVersion == REQUESTED_PROTOCOL_VERSION
        assert init.protocolVersion == golden["protocolVersion"]

        golden_info = golden["serverInfo"]
        assert init.serverInfo.name == golden_info["name"]
        assert init.serverInfo.version == __version__
        assert init.serverInfo.websiteUrl == golden_info["websiteUrl"]

        # Capabilities are SDK-owned and diverge from the Go golden; only assert
        # that the tools capability is advertised.
        assert init.capabilities.tools is not None


async def test_tools_list_matches_golden_structure() -> None:
    """tools/list matches the Go golden schemas structurally."""
    golden_tool = _load_golden("tools_list.json")["result"]["tools"][0]
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (session, _init):
        result = await session.list_tools()

    assert len(result.tools) == 1
    tool = result.tools[0]
    assert tool.name == golden_tool["name"]
    assert tool.description == golden_tool["description"]
    assert tool.inputSchema == golden_tool["inputSchema"]
    assert tool.outputSchema == golden_tool["outputSchema"]


async def test_call_tool_returns_grounded_text_and_structured_content() -> None:
    """tools/call returns grounded text content plus structuredContent."""
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (session, _init):
        result = await session.call_tool("google_search", {"query": "any"})

    assert not result.isError
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == _GROUNDED_OUTPUT.text
    assert result.structuredContent == _GROUNDED_OUTPUT.to_structured()


async def test_call_tool_empty_query_returns_is_error() -> None:
    """A blank query yields isError with the exact message."""
    client = genai.Client(api_key="dummy")
    service = GoogleSearchService(model=DEFAULT_MODEL, interactions=client.aio.interactions)
    async with _session(service) as (session, _init):
        result = await session.call_tool("google_search", {"query": "  "})

    assert result.isError
    assert len(result.content) >= 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "search query cannot be empty"


async def test_call_tool_unknown_name_returns_is_error() -> None:
    """A tool name other than google_search is rejected without searching."""
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (session, _init):
        result = await session.call_tool("nonexistent_tool", {"query": "x"})

    assert result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "Unknown tool: nonexistent_tool"


async def test_tools_list_without_research_is_single_tool() -> None:
    """Without a research service, list_tools returns only google_search."""
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (session, _init):
        result = await session.list_tools()

    assert [tool.name for tool in result.tools] == ["google_search"]


async def test_tools_list_with_research_returns_three_tools() -> None:
    """With a research service, list_tools returns google_search then the two research tools."""
    golden_tools = _load_golden("tools_list_deep_research.json")["result"]["tools"]
    async with _session(_StubService(_GROUNDED_OUTPUT), research=_StubResearchService()) as (
        session,
        _init,
    ):
        result = await session.list_tools()

    assert len(result.tools) == 3
    assert [tool.name for tool in result.tools] == [
        "google_search",
        "deep_research",
        "deep_research_result",
    ]
    for tool, golden in zip(result.tools, golden_tools, strict=True):
        assert tool.name == golden["name"]
        assert tool.description == golden["description"]
        assert tool.inputSchema == golden["inputSchema"]
        assert tool.outputSchema == golden["outputSchema"]


async def test_call_deep_research_returns_start_text_and_structured() -> None:
    """deep_research returns the start message and structured interaction id."""
    start = DeepResearchStart(interaction_id="dr-99", status="in_progress")
    research = _StubResearchService(start=start)
    async with _session(_StubService(_GROUNDED_OUTPUT), research=research) as (session, _init):
        result = await session.call_tool("deep_research", {"query": "topic"})

    assert not result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert "dr-99" in block.text
    assert "in_progress" in block.text
    assert result.structuredContent == start.to_structured()


async def test_call_deep_research_result_returns_report() -> None:
    """deep_research_result returns the completed report text and structured content."""
    report = DeepResearchResult(
        interaction_id="dr-99",
        status="completed",
        text="Full report\n\n## Sources\n\n1. [Src](https://s.example)",
        sources=(GoogleSearchSource(index=1, title="Src", uri="https://s.example"),),
    )
    research = _StubResearchService(result=report)
    async with _session(_StubService(_GROUNDED_OUTPUT), research=research) as (session, _init):
        result = await session.call_tool(
            "deep_research_result",
            {"interaction_id": "dr-99", "wait_seconds": 0},
        )

    assert not result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == report.text
    assert result.structuredContent == report.to_structured()


async def test_call_deep_research_error_returns_is_error() -> None:
    """Research service ValueError surfaces through call_tool as isError."""
    research = _StubResearchService(error=ValueError("research query cannot be empty"))
    async with _session(_StubService(_GROUNDED_OUTPUT), research=research) as (session, _init):
        result = await session.call_tool("deep_research", {"query": "  "})

    assert result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "research query cannot be empty"


async def test_call_deep_research_runtime_error_returns_is_error() -> None:
    """Research service RuntimeError surfaces through call_tool as isError."""
    research = _StubResearchService(error=RuntimeError("deep research failed: boom"))
    async with _session(_StubService(_GROUNDED_OUTPUT), research=research) as (session, _init):
        result = await session.call_tool("deep_research_result", {"interaction_id": "dr-1"})

    assert result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "deep research failed: boom"


async def test_call_tool_unknown_name_with_research_returns_is_error() -> None:
    """Unknown tool names are rejected even when deep research tools are enabled."""
    async with _session(_StubService(_GROUNDED_OUTPUT), research=_StubResearchService()) as (
        session,
        _init,
    ):
        result = await session.call_tool("nonexistent_tool", {"query": "x"})

    assert result.isError
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "Unknown tool: nonexistent_tool"
