# Copyright 2026 The mcp-gemini-google-search Authors.
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

from mcp_gemini_google_search.config import DEFAULT_MODEL
from mcp_gemini_google_search.search import (
    GoogleSearchOutput,
    GoogleSearchService,
    GoogleSearchSource,
)
from mcp_gemini_google_search.server import create_server

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
        super().__init__(model=DEFAULT_MODEL, generator=None)
        self._output = output

    async def search(self, query: str) -> GoogleSearchOutput:
        return self._output


_GROUNDED_OUTPUT = GoogleSearchOutput(
    query="who wrote the go language",
    text=(
        "Alpha [1]Beta[1,2]\n\nSources:\n"
        "[1] First (https://first.example)\n"
        "[2] Second (https://second.example)"
    ),
    sources=(
        GoogleSearchSource(index=1, title="First", uri="https://first.example"),
        GoogleSearchSource(index=2, title="Second", uri="https://second.example"),
    ),
)


@asynccontextmanager
async def _session(
    service: GoogleSearchService,
) -> AsyncGenerator[tuple[ClientSession, types.InitializeResult]]:
    """Yield a client session connected to the server, initialized at 2025-06-18."""
    server = create_server(service)
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
                                    clientInfo=types.Implementation(
                                        name="test-client", version="0.0.0"
                                    ),
                                )
                            )
                        ),
                        types.InitializeResult,
                    )
                    await session.send_notification(
                        types.ClientNotification(types.InitializedNotification())
                    )
                    yield session, init
            finally:
                tg.cancel_scope.cancel()


async def test_initialize_negotiates_golden_protocol_and_server_info() -> None:
    golden = _load_golden("initialize.json")["result"]
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (_session_obj, init):
        assert init.protocolVersion == REQUESTED_PROTOCOL_VERSION
        assert init.protocolVersion == golden["protocolVersion"]

        golden_info = golden["serverInfo"]
        assert init.serverInfo.name == golden_info["name"]
        assert init.serverInfo.version == golden_info["version"]
        assert init.serverInfo.websiteUrl == golden_info["websiteUrl"]

        # Capabilities are SDK-owned and diverge from the Go golden; only assert
        # that the tools capability is advertised.
        assert init.capabilities.tools is not None


async def test_tools_list_matches_golden_structure() -> None:
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
    async with _session(_StubService(_GROUNDED_OUTPUT)) as (session, _init):
        result = await session.call_tool("google_search", {"query": "any"})

    assert not result.isError
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == _GROUNDED_OUTPUT.text
    assert result.structuredContent == _GROUNDED_OUTPUT.to_structured()


async def test_call_tool_empty_query_returns_is_error() -> None:
    client = genai.Client(api_key="dummy")
    service = GoogleSearchService(model=DEFAULT_MODEL, generator=client.aio.models)
    async with _session(service) as (session, _init):
        result = await session.call_tool("google_search", {"query": "  "})

    assert result.isError
    assert len(result.content) >= 1
    block = result.content[0]
    assert isinstance(block, types.TextContent)
    assert block.text == "search query cannot be empty"
