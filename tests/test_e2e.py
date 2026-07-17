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

"""End-to-end tests driving the installed console binary over stdio."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any

import anyio
import orjson
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)
from mcp_types import TextContent

from mcp_gemini_search import __version__
from tests._helpers import load_golden

pytestmark = pytest.mark.anyio

# The console script installed by `uv sync`; this is exactly what
# `uv run mcp-gemini-search` execs.
BINARY = Path(sys.executable).parent / "mcp-gemini-search"

_MISSING_API_KEY_ERROR = (
    '"GOOGLE_API_KEY" or "GEMINI_API_KEY" environment variable is required when using Google AI Studio'
)
_STARTUP_LINE = "gemini google search mcp server running on stdio"

_INIT_FRAME = (
    b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":'
    b'{"protocolVersion":"2025-06-18","capabilities":{},'
    b'"clientInfo":{"name":"e2e","version":"0"}}}\n'
)

_SUBPROCESS_TIMEOUT = 20.0


def _clean_env() -> dict[str, str]:
    """Return a minimal env carrying no Gemini/Vertex configuration."""
    keys = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT")
    return {key: os.environ[key] for key in keys if key in os.environ}


def _dummy_key_env() -> dict[str, str]:
    """Return a minimal env with a dummy AI Studio key so config load succeeds."""
    return {**_clean_env(), "GEMINI_API_KEY": "dummy"}


async def _spawn(args: list[str], env: dict[str, str]) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        str(BINARY),
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


async def _kill(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        proc.kill()
        await proc.wait()


async def test_stdio_handshake_reports_golden_server_info_and_tool() -> None:
    """The installed binary serves the golden serverInfo and all standard tools."""
    golden = load_golden("initialize.json")["result"]["serverInfo"]
    golden_tools = load_golden("tools_list.json")["result"]["tools"]
    params = StdioServerParameters(
        command=str(BINARY),
        env={**get_default_environment(), "GEMINI_API_KEY": "dummy"},
    )
    with anyio.fail_after(_SUBPROCESS_TIMEOUT):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = await session.list_tools()

    assert init.server_info.name == golden["name"]
    assert init.server_info.version == __version__
    assert init.server_info.website_url == golden["websiteUrl"]

    assert [tool.name for tool in tools.tools] == [
        "google_search",
        "deep_research",
        "deep_research_result",
    ]
    for tool, golden_tool in zip(tools.tools, golden_tools, strict=True):
        assert tool.name == golden_tool["name"]
        assert tool.description == golden_tool["description"]
        assert tool.input_schema == golden_tool["inputSchema"]
        assert tool.output_schema == golden_tool["outputSchema"]


async def test_logpath_records_startup_line_and_jsonrpc_frames(
    tmp_path: Path,
) -> None:
    """-logpath captures the startup line and direction-tagged JSON-RPC frames."""
    logfile = tmp_path / "server.log"
    params = StdioServerParameters(
        command=str(BINARY),
        args=["-logpath", str(logfile)],
        env={**get_default_environment(), "GEMINI_API_KEY": "dummy"},
    )
    with anyio.fail_after(_SUBPROCESS_TIMEOUT):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()

    content = logfile.read_text(encoding="utf-8")
    assert _STARTUP_LINE in content
    assert "read:" in content
    assert "write:" in content
    assert "time=" in content
    assert "level=" in content
    assert "msg=" in content


async def test_stdin_eof_exits_zero_without_pollution() -> None:
    """stdin EOF exits 0 with protocol-only stdout and a silent stderr."""
    proc = await _spawn([], _dummy_key_env())
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        proc.stdin.write(_INIT_FRAME)
        await proc.stdin.drain()
        first = await asyncio.wait_for(proc.stdout.readline(), _SUBPROCESS_TIMEOUT)
        assert b'"protocolVersion"' in first
        proc.stdin.close()
        returncode = await asyncio.wait_for(proc.wait(), _SUBPROCESS_TIMEOUT)
        rest = await proc.stdout.read()
        stderr = await proc.stderr.read()
    finally:
        await _kill(proc)

    assert returncode == 0
    assert stderr == b""
    for line in (first + rest).split(b"\n"):
        if line.strip():
            orjson.loads(line)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal termination semantics")
async def test_sigterm_exits_one_with_context_canceled() -> None:
    """SIGTERM exits 1 with the Go-identical context-canceled message."""
    proc = await _spawn([], _dummy_key_env())
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    try:
        proc.stdin.write(_INIT_FRAME)
        await proc.stdin.drain()
        await asyncio.wait_for(proc.stdout.readline(), _SUBPROCESS_TIMEOUT)
        # stdin stays open; termination must come from the signal.
        proc.send_signal(signal.SIGTERM)
        returncode = await asyncio.wait_for(proc.wait(), _SUBPROCESS_TIMEOUT)
        stderr = (await proc.stderr.read()).decode("utf-8")
    finally:
        await _kill(proc)

    assert returncode == 1
    assert "serve gemini google search mcp stdio server: context canceled" in stderr


async def test_missing_api_key_exits_one_with_config_error() -> None:
    """A missing API key exits 1 with the exact config error on stderr."""
    proc = await _spawn([], _clean_env())
    stdout, stderr = await asyncio.wait_for(proc.communicate(), _SUBPROCESS_TIMEOUT)

    assert proc.returncode == 1
    assert stderr.decode("utf-8").strip() == _MISSING_API_KEY_ERROR
    assert stdout == b""


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_API") or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="RUN_LIVE_API and a real GEMINI_API_KEY/GOOGLE_API_KEY are required",
)
async def test_live_google_search_returns_grounded_text() -> None:
    """A real API call returns grounded text with a source list."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    params = StdioServerParameters(
        command=str(BINARY),
        env={**get_default_environment(), "GEMINI_API_KEY": api_key},
    )
    with anyio.fail_after(120):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("google_search", {"query": "latest Go release version"})

    assert not result.is_error
    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text.strip()
    if result.structured_content and result.structured_content.get("sources"):
        assert "\n## Sources\n" in block.text


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_API") or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="RUN_LIVE_API and a real GEMINI_API_KEY/GOOGLE_API_KEY are required",
)
async def test_live_deep_research_start_and_poll_then_cancel() -> None:
    """Start a deep research run, poll once, then cancel the billed background job."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    params = StdioServerParameters(
        command=str(BINARY),
        env={**get_default_environment(), "GEMINI_API_KEY": api_key},
    )
    interaction_id = ""
    with anyio.fail_after(120):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                start = await session.call_tool(
                    "deep_research",
                    {"query": "one-sentence summary of the Go programming language"},
                )
                assert not start.is_error
                assert start.structured_content is not None
                interaction_id = start.structured_content["interaction_id"]
                assert interaction_id

                poll = await session.call_tool(
                    "deep_research_result",
                    {"interaction_id": interaction_id, "wait_seconds": 5},
                )
                assert not poll.is_error
                assert poll.structured_content is not None
                assert poll.structured_content["status"] in {"in_progress", "completed"}

    # Cancel via the SDK so the billed background run does not linger.
    client = genai.Client(api_key=api_key)
    await client.aio.interactions.cancel(interaction_id)


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("RUN_SLOW"),
    reason="set RUN_SLOW=1 to run the full multi-minute deep research live test",
)
@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_API") or not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="RUN_LIVE_API and a real GEMINI_API_KEY/GOOGLE_API_KEY are required",
)
async def test_live_deep_research_full_run_to_completion() -> None:
    """Gated by RUN_SLOW=1 in addition to the live marker so `-m live` alone never runs this multi-minute billed job."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    params = StdioServerParameters(
        command=str(BINARY),
        env={**get_default_environment(), "GEMINI_API_KEY": api_key},
    )
    with anyio.fail_after(15 * 60):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                start = await session.call_tool(
                    "deep_research",
                    {"query": "one-paragraph overview of the MCP protocol"},
                )
                assert not start.is_error
                assert start.structured_content is not None
                interaction_id = start.structured_content["interaction_id"]

                status = "in_progress"
                report_text = ""
                sources: list[Any] | None = None
                while status not in {"completed", "failed", "cancelled"}:
                    poll = await session.call_tool(
                        "deep_research_result",
                        {"interaction_id": interaction_id, "wait_seconds": 60},
                    )
                    assert not poll.is_error
                    assert poll.structured_content is not None
                    status = poll.structured_content["status"]
                    report_text = poll.structured_content.get("text", "")
                    sources = poll.structured_content.get("sources")

                assert status == "completed"
                assert report_text.strip()
                assert sources
                assert len(sources) >= 1
