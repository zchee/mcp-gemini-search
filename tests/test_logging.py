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

"""Tests for file logging setup parity."""

from __future__ import annotations

import logging
from pathlib import Path

from mcp_gemini_search import _logging


def test_setup_logging_overwrites_in_place_without_truncation(tmp_path: Path) -> None:
    """The log file keeps Go O_RDWR semantics: overwrite in place, never truncate."""
    logfile = tmp_path / "server.log"
    logfile.write_text("X" * 4096, encoding="utf-8")

    root = logging.getLogger()
    handlers_before = root.handlers[:]
    level_before = root.level
    handle = _logging.setup_logging(str(logfile))
    assert handle is not None
    try:
        _logging.logger.debug("overwrite-check")
    finally:
        for handler in root.handlers[:]:
            if handler not in handlers_before:
                root.removeHandler(handler)
                handler.close()
        root.setLevel(level_before)
        if not handle.closed:
            handle.close()

    content = logfile.read_text(encoding="utf-8")
    assert "msg=overwrite-check" in content
    assert len(content) == 4096
    assert content.endswith("X")
