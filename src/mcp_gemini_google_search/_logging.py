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

"""File logging setup and JSON-RPC frame logging for the MCP server.

Parity notes:

- No ``-logpath`` attaches a ``NullHandler`` to the root logger, mirroring Go's
  ``slog.DiscardHandler``: nothing is written to stdout or stderr.
- With ``-logpath`` the file is opened with the exact Go flags
  (``os.O_RDWR | os.O_CREAT``, mode ``0o666`` -- no truncation, no append) and a
  DEBUG handler using a Go ``slog`` ``TextHandler``-style formatter is attached.
- Frame logging is a semantic port of the go-sdk ``LoggingTransport`` (plan D3):
  every inbound/outbound JSON-RPC message is logged with a ``read:``/``write:``
  direction prefix. The exact byte format is intentionally not reproduced.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TextIO

import anyio
import orjson
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage

_LOGGER_NAME = "mcp_gemini_google_search"

logger = logging.getLogger(_LOGGER_NAME)

# Map Python logging level numbers to Go slog level names.
_SLOG_LEVELS = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARN",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "ERROR",
}


def _slog_quote(value: str) -> str:
    """Quote a value the way slog does when it needs quoting.

    slog wraps a value in double quotes (with escaping) when it is empty or
    contains whitespace, ``=`` or ``"``; otherwise it is written verbatim.
    """
    if value == "" or any(ch <= " " or ch in '="' for ch in value):
        return orjson.dumps(value).decode()
    return value


class _SlogTextFormatter(logging.Formatter):
    """Render records as ``time=RFC3339 level=LEVEL msg=...`` like slog."""

    def format(self, record: logging.LogRecord) -> str:
        """Format ``record`` in Go ``slog`` ``TextHandler`` style."""
        timestamp = datetime.fromtimestamp(record.created).astimezone().isoformat()
        level = _SLOG_LEVELS.get(record.levelno, record.levelname)
        return f"time={timestamp} level={level} msg={_slog_quote(record.getMessage())}"


def setup_logging(logpath: str) -> TextIO | None:
    """Configure root logging and return the open log file handle or ``None``.

    Without ``logpath``, attach a ``NullHandler`` to the root logger and return
    ``None`` (total silence, never to stdout). With ``logpath``, open the file
    (``os.O_RDWR | os.O_CREAT``, ``0o666``), attach a DEBUG handler using the
    slog-style formatter, and return the file handle for the caller to close.

    Raises:
        RuntimeError: If ``logpath`` cannot be opened (message ``open "<path>"
            file: <error>``, mirroring Go's ``open %q file`` wrapping).
    """
    root = logging.getLogger()
    if not logpath:
        root.addHandler(logging.NullHandler())
        return None

    try:
        fd = os.open(logpath, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as e:
        raise RuntimeError(f'open "{logpath}" file: {e}') from e
    handle: TextIO = os.fdopen(fd, "w", encoding="utf-8")

    handler = logging.StreamHandler(handle)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(_SlogTextFormatter())
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    return handle


def _log_frame(direction: str, item: SessionMessage | Exception) -> None:
    """Log one JSON-RPC frame with its direction prefix at DEBUG level."""
    if isinstance(item, Exception):
        logger.debug("%s: %r", direction, item)
        return
    payload = item.message.model_dump_json(by_alias=True, exclude_none=True)
    logger.debug("%s: %s", direction, payload)


@asynccontextmanager
async def frame_logging_streams(
    read_stream: MemoryObjectReceiveStream[SessionMessage | Exception],
    write_stream: MemoryObjectSendStream[SessionMessage],
) -> AsyncGenerator[
    tuple[
        MemoryObjectReceiveStream[SessionMessage | Exception],
        MemoryObjectSendStream[SessionMessage],
    ]
]:
    """Tee the stdio streams through frame logging.

    Yields a ``(read, write)`` pair to hand to ``Server.run``. Two pump tasks
    forward every message between the yielded pair and the underlying stdio
    streams, logging each inbound frame with a ``read:`` prefix and each outbound
    frame with a ``write:`` prefix. Closing the stdio read stream (stdin EOF)
    propagates cleanly so the server shuts down.
    """
    server_read_send, server_read_recv = anyio.create_memory_object_stream[
        SessionMessage | Exception
    ](0)
    server_write_send, server_write_recv = anyio.create_memory_object_stream[
        SessionMessage
    ](0)

    async def pump_inbound() -> None:
        async with server_read_send:
            async for item in read_stream:
                _log_frame("read", item)
                await server_read_send.send(item)

    async def pump_outbound() -> None:
        # Closing write_stream when the server stops writing signals the stdio
        # writer task to finish; without it stdio_server's task group would wait
        # on the writer forever and never shut down.
        async with write_stream:
            async for item in server_write_recv:
                _log_frame("write", item)
                await write_stream.send(item)

    async with anyio.create_task_group() as tg:
        tg.start_soon(pump_inbound)
        tg.start_soon(pump_outbound)
        try:
            yield server_read_recv, server_write_send
        finally:
            tg.cancel_scope.cancel()
