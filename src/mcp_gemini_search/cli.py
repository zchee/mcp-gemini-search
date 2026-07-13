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

"""Command-line entry point: argument parsing, wiring, and the serve loop."""

from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from mcp_gemini_search import _logging
from mcp_gemini_search.config import load_config_from_env
from mcp_gemini_search.research import DeepResearchService
from mcp_gemini_search.search import GoogleSearchService
from mcp_gemini_search.server import create_server

_STARTUP_MESSAGE = "gemini google search mcp server running on stdio"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments, accepting both -logpath and --logpath."""
    parser = argparse.ArgumentParser(
        prog="mcp-gemini-search",
        description=("MCP server providing Google Search via Gemini's Grounding with Google Search."),
    )
    parser.add_argument(
        "-logpath",
        "--logpath",
        dest="logpath",
        default="",
        help="if set, enable MCP server logging",
    )
    return parser.parse_args(argv)


def _backend_options() -> dict[str, bool]:
    """Select uvloop for the anyio asyncio backend when it is installed.

    uvloop is a platform-conditional dependency (non-Windows); when it is
    absent the default asyncio event loop is used.
    """
    if importlib.util.find_spec("uvloop") is None:
        return {}
    return {"use_uvloop": True}


def main() -> None:
    """Console-script entry point.

    Any fatal error is printed to stderr as its bare message and the process
    exits with status 1, mirroring Go's ``fmt.Fprintln(os.Stderr, err)`` +
    ``os.Exit(1)``. Clean shutdown (stdin EOF or SIGINT/SIGTERM) exits 0.
    """
    args = _parse_args(None)
    try:
        _run(args.logpath)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


def _run(logpath: str) -> None:
    """Configure logging, build the server, and serve until shutdown."""
    handle = _logging.setup_logging(logpath)
    try:
        config = load_config_from_env(lambda key: os.environ.get(key, ""))

        try:
            client = config.new_client()
        except Exception as e:
            raise RuntimeError(f"create genai client: {e}") from e

        service = GoogleSearchService(
            model=config.model,
            interactions=client.aio.interactions,
            url_context=config.url_context,
            code_execution=config.code_execution,
        )
        research = None
        if config.deep_research:
            research = DeepResearchService(
                agent=config.deep_research_agent,
                interactions=client.aio.interactions,
            )
        server = create_server(service, research)

        tools_msg = ", ".join(tool["type"] for tool in service.tools)
        if config.deep_research:
            _logging.logger.info(
                "gemini interactions tools: %s; deep research agent: %s",
                tools_msg,
                config.deep_research_agent,
            )
        else:
            _logging.logger.info("gemini interactions tools: %s", tools_msg)
        _logging.logger.info(_STARTUP_MESSAGE)
        try:
            anyio.run(_serve, server, logpath, backend_options=_backend_options())
        except Exception as e:
            _logging.logger.error("serve gemini google search mcp stdio server: %s", e)
            raise RuntimeError(f"serve gemini google search mcp stdio server: {e}") from e
    finally:
        if handle is not None:
            handle.close()


async def _serve(server: Server, logpath: str) -> None:
    """Run the MCP server over stdio with signal-driven cancellation.

    SIGINT/SIGTERM (non-Windows) cancel the serve scope for a clean shutdown;
    stdin EOF ends the read stream and shuts the server down the same way.
    """
    init_options = server.create_initialization_options()
    async with anyio.create_task_group() as tg:
        if sys.platform != "win32":
            tg.start_soon(_watch_signals)
        async with stdio_server() as (read_stream, write_stream):
            if logpath:
                async with _logging.frame_logging_streams(read_stream, write_stream) as (framed_read, framed_write):
                    await server.run(framed_read, framed_write, init_options)
            else:
                await server.run(read_stream, write_stream, init_options)
        tg.cancel_scope.cancel()


async def _watch_signals() -> None:
    """Terminate on the first SIGINT or SIGTERM, matching the Go server.

    The Go server's context is cancelled on signal, ``srv.Run`` returns
    ``context.Canceled``, and ``main`` exits 1 after printing
    ``serve gemini google search mcp stdio server: context canceled``. Because
    stdio_server reads stdin in an uncancellable worker thread, a cooperative
    anyio unwind would block on that thread; ``os._exit`` reproduces Go's prompt
    ``os.Exit(1)`` instead. Log lines are already flushed per record.
    """
    with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
        async for _ in signals:
            message = "serve gemini google search mcp stdio server: context canceled"
            _logging.logger.error(message)
            print(message, file=sys.stderr)
            sys.stderr.flush()
            os._exit(1)
