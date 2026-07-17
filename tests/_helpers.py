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

"""Shared interaction factories and golden-file helpers for the test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
from google.genai import interactions

GOLDEN_DIR = Path(__file__).parent / "golden"


def load_golden(name: str) -> dict[str, Any]:
    """Load a golden JSON-RPC response from ``tests/golden`` by file name."""
    return orjson.loads((GOLDEN_DIR / name).read_text(encoding="utf-8"))


def golden_tool(name: str) -> dict[str, Any]:
    """Return one tool declaration from the golden ``tools/list`` response."""
    tools: list[dict[str, Any]] = load_golden("tools_list.json")["result"]["tools"]
    return next(tool for tool in tools if tool["name"] == name)


def url_citation(url: str, title: str, end_index: int) -> interactions.URLCitation:
    """Build a ``url_citation`` annotation ending at ``end_index``."""
    return interactions.URLCitation(url=url, title=title, start_index=0, end_index=end_index)


def text_block(text: str, *annotations: interactions.URLCitation) -> interactions.TextContent:
    """Build a text content block, attaching ``annotations`` when given."""
    return interactions.TextContent(text=text, annotations=list(annotations) or None)


def model_output(*blocks: Any) -> interactions.ModelOutputStep:
    """Build a ``model_output`` step from content blocks."""
    return interactions.ModelOutputStep(content=list(blocks))


def interaction(
    *steps: Any,
    status: str = "completed",
    interaction_id: str | None = None,
) -> interactions.Interaction:
    """Build an interaction in ``status`` carrying ``steps``."""
    return interactions.Interaction(id=interaction_id, status=status, steps=list(steps))
