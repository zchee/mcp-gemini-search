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
from typing import Any

import anyio
import pytest
from google import genai
from mcp.client import Client
from mcp.client.session import ClientSession
from mcp.shared.memory import create_client_server_memory_streams
from mcp_types import (
    ClientCapabilities,
    Implementation,
    InitializedNotification,
    InitializeRequest,
    InitializeRequestParams,
    InitializeResult,
    TextContent,
)

from mcp_gemini_search import __version__
from mcp_gemini_search.config import DEFAULT_MODEL
from mcp_gemini_search.research import (
    DEEP_RESEARCH_AGENT,
    DEEP_RESEARCH_MAX_AGENT,
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
from tests._helpers import load_golden

pytestmark = pytest.mark.anyio

# The Go golden was captured with a client that requested 2025-06-18; the server
# echoes the requested version, so the test client must request the same value.
REQUESTED_PROTOCOL_VERSION = "2025-06-18"


class _StubService(GoogleSearchService):
    """A service that returns a fixed grounded output for any query."""

    def __init__(self, output: GoogleSearchOutput) -> None:
        super().__init__(model=DEFAULT_MODEL, interactions=None)
        self._output = output
        self.search_calls: list[dict[str, Any]] = []

    async def search(
        self,
        query: str,
        *,
        url_context: bool | None = None,
        code_execution: bool | None = None,
    ) -> GoogleSearchOutput:
        self.search_calls.append({
            "query": query,
            "url_context": url_context,
            "code_execution": code_execution,
        })
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
        super().__init__(agent=DEEP_RESEARCH_AGENT, interactions=None)
        self._start = start or DeepResearchStart(interaction_id="dr-1", status="in_progress")
        self._result = result or DeepResearchResult(
            interaction_id="dr-1",
            status="completed",
            text="Report body",
            sources=(GoogleSearchSource(index=1, title="Src", uri="https://s.example"),),
        )
        self._error = error
        self.start_calls: list[dict[str, Any]] = []

    async def start(
        self,
        query: str,
        *,
        plan_only: bool = False,
        previous_interaction_id: str = "",
        agent: str = "",
    ) -> DeepResearchStart:
        self.start_calls.append({
            "query": query,
            "plan_only": plan_only,
            "previous_interaction_id": previous_interaction_id,
            "agent": agent,
        })
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
async def _client(
    service: GoogleSearchService,
    research: DeepResearchService | None = None,
) -> AsyncGenerator[Client]:
    """Yield an in-memory v2 client session connected to the server."""
    research_service = research if research is not None else _StubResearchService()
    server = create_server(service, research_service)
    async with Client(server) as client:
        yield client


@asynccontextmanager
async def _raw_session(
    service: GoogleSearchService,
    research: DeepResearchService | None = None,
) -> AsyncGenerator[tuple[ClientSession, InitializeResult]]:
    """Yield a raw session initialized with the protocol version pinned by the golden."""
    research_service = research if research is not None else _StubResearchService()
    server = create_server(service, research_service)
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
                )
            )
            try:
                async with ClientSession(client_read, client_write) as session:
                    init = await session.send_request(
                        InitializeRequest(
                            params=InitializeRequestParams(
                                protocol_version=REQUESTED_PROTOCOL_VERSION,
                                capabilities=ClientCapabilities(),
                                client_info=Implementation(name="test-client", version="0.0.0"),
                            )
                        ),
                        InitializeResult,
                    )
                    await session.send_notification(InitializedNotification())
                    yield session, init
            finally:
                tg.cancel_scope.cancel()


async def test_initialize_negotiates_golden_protocol_and_server_info() -> None:
    """initialize echoes the requested protocol version and the golden serverInfo."""
    golden = load_golden("initialize.json")["result"]
    async with _raw_session(_StubService(_GROUNDED_OUTPUT)) as (_session_obj, init):
        assert init.protocol_version == REQUESTED_PROTOCOL_VERSION
        assert init.protocol_version == golden["protocolVersion"]

        golden_info = golden["serverInfo"]
        assert init.server_info.name == golden_info["name"]
        assert init.server_info.version == __version__
        assert init.server_info.website_url == golden_info["websiteUrl"]

        # Capabilities are SDK-owned and diverge from the Go golden; only assert
        # that the tools capability is advertised.
        assert init.capabilities.tools is not None


async def test_call_tool_returns_grounded_text_and_structured_content() -> None:
    """tools/call returns grounded text content plus structuredContent."""
    async with _client(_StubService(_GROUNDED_OUTPUT)) as client:
        result = await client.call_tool("google_search", {"query": "any"})

    assert not result.is_error
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == _GROUNDED_OUTPUT.text
    assert result.structured_content == _GROUNDED_OUTPUT.to_structured()


async def test_call_google_search_passes_tool_overrides() -> None:
    """google_search passes explicit per-request tool overrides to the service."""
    service = _StubService(_GROUNDED_OUTPUT)
    async with _client(service) as client:
        result = await client.call_tool(
            "google_search",
            {"query": "any", "url_context": True, "code_execution": False},
        )

    assert not result.is_error
    assert service.search_calls == [
        {
            "query": "any",
            "url_context": True,
            "code_execution": False,
        }
    ]


async def test_call_google_search_defaults_tool_overrides_to_none() -> None:
    """google_search passes None tool overrides when both arguments are omitted."""
    service = _StubService(_GROUNDED_OUTPUT)
    async with _client(service) as client:
        result = await client.call_tool("google_search", {"query": "any"})

    assert not result.is_error
    assert service.search_calls == [
        {
            "query": "any",
            "url_context": None,
            "code_execution": None,
        }
    ]


async def test_call_tool_empty_query_returns_is_error() -> None:
    """A blank query yields isError with the exact message."""
    client = genai.Client(api_key="dummy")
    service = GoogleSearchService(model=DEFAULT_MODEL, interactions=client.aio.interactions)
    async with _client(service) as mcp_client:
        result = await mcp_client.call_tool("google_search", {"query": "  "})

    assert result.is_error
    assert len(result.content) >= 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "search query cannot be empty"


async def test_call_tool_unknown_name_returns_is_error() -> None:
    """A name other than the three advertised tools is rejected."""
    async with _client(_StubService(_GROUNDED_OUTPUT)) as client:
        result = await client.call_tool("nonexistent_tool", {"query": "x"})

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "Unknown tool: nonexistent_tool"


@pytest.mark.parametrize(
    ("tool_name", "arguments", "violation"),
    [
        ("google_search", {}, "is a required property"),
        ("google_search", {"query": 42}, "is not of type"),
        ("google_search", {"query": "topic", "unexpected": True}, "Additional properties"),
        ("deep_research", {}, "is a required property"),
        ("deep_research", {"query": 42}, "is not of type"),
        ("deep_research", {"query": "topic", "unexpected": True}, "Additional properties"),
        ("deep_research_result", {}, "is a required property"),
        ("deep_research_result", {"interaction_id": 42}, "is not of type"),
        (
            "deep_research_result",
            {"interaction_id": "dr-1", "unexpected": True},
            "Additional properties",
        ),
    ],
    ids=[
        "google-search-missing",
        "google-search-wrong-type",
        "google-search-extra",
        "deep-research-missing",
        "deep-research-wrong-type",
        "deep-research-extra",
        "deep-research-result-missing",
        "deep-research-result-wrong-type",
        "deep-research-result-extra",
    ],
)
async def test_call_tool_rejects_invalid_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    violation: str,
) -> None:
    """Every tool rejects schema-invalid arguments as a normal error result."""
    async with _client(_StubService(_GROUNDED_OUTPUT)) as client:
        result = await client.call_tool(tool_name, arguments)

    assert result.is_error is True
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert violation in block.text


async def test_tools_list_returns_three_tools() -> None:
    """list_tools returns all three standard tools in golden order."""
    golden_tools = load_golden("tools_list.json")["result"]["tools"]
    async with _client(_StubService(_GROUNDED_OUTPUT)) as client:
        result = await client.list_tools()

    assert len(result.tools) == 3
    assert [tool.name for tool in result.tools] == [
        "google_search",
        "deep_research",
        "deep_research_result",
    ]
    for tool, golden in zip(result.tools, golden_tools, strict=True):
        assert tool.name == golden["name"]
        assert tool.description == golden["description"]
        assert tool.input_schema == golden["inputSchema"]
        assert tool.output_schema == golden["outputSchema"]


async def test_call_deep_research_returns_start_text_and_structured() -> None:
    """deep_research returns the start message and structured interaction id."""
    start = DeepResearchStart(interaction_id="dr-99", status="in_progress")
    research = _StubResearchService(start=start)
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool("deep_research", {"query": "topic"})

    assert not result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert "dr-99" in block.text
    assert "in_progress" in block.text
    assert result.structured_content == start.to_structured()


async def test_call_deep_research_passes_agent_override() -> None:
    """deep_research passes an explicit per-request agent through to the service."""
    research = _StubResearchService()
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool(
            "deep_research",
            {"query": "topic", "agent": DEEP_RESEARCH_MAX_AGENT},
        )

    assert not result.is_error
    assert research.start_calls == [
        {
            "query": "topic",
            "plan_only": False,
            "previous_interaction_id": "",
            "agent": DEEP_RESEARCH_MAX_AGENT,
        }
    ]


async def test_call_deep_research_defaults_agent_override_to_empty() -> None:
    """deep_research passes an empty agent override when the argument is omitted."""
    research = _StubResearchService()
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool("deep_research", {"query": "topic"})

    assert not result.is_error
    assert research.start_calls == [
        {
            "query": "topic",
            "plan_only": False,
            "previous_interaction_id": "",
            "agent": "",
        }
    ]


async def test_call_deep_research_result_returns_report() -> None:
    """deep_research_result returns the completed report text and structured content."""
    report = DeepResearchResult(
        interaction_id="dr-99",
        status="completed",
        text="Full report\n\n## Sources\n\n1. [Src](https://s.example)",
        sources=(GoogleSearchSource(index=1, title="Src", uri="https://s.example"),),
    )
    research = _StubResearchService(result=report)
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool(
            "deep_research_result",
            {"interaction_id": "dr-99", "wait_seconds": 0},
        )

    assert not result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == report.text
    assert result.structured_content == report.to_structured()


async def test_call_deep_research_error_returns_is_error() -> None:
    """Research service ValueError surfaces through call_tool as isError."""
    research = _StubResearchService(error=ValueError("research query cannot be empty"))
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool("deep_research", {"query": "  "})

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "research query cannot be empty"


async def test_call_deep_research_runtime_error_returns_is_error() -> None:
    """Research service RuntimeError surfaces through call_tool as isError."""
    research = _StubResearchService(error=RuntimeError("deep research failed: boom"))
    async with _client(_StubService(_GROUNDED_OUTPUT), research=research) as client:
        result = await client.call_tool("deep_research_result", {"interaction_id": "dr-1"})

    assert result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "deep research failed: boom"
