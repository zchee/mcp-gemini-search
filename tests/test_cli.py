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

"""Tests for CLI event-loop backend selection."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from mcp_gemini_search import cli
from mcp_gemini_search.config import (
    DEFAULT_DEEP_RESEARCH_AGENT,
    ENV_CLAUDE_HOME,
    ENV_CODEX_HOME,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_DEEP_RESEARCH_AGENT,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_GOOGLE_GENAI_USE_VERTEXAI,
)
from mcp_gemini_search.research import DeepResearchService
from mcp_gemini_search.search import GoogleSearchService


@pytest.mark.skipif(sys.platform == "win32", reason="uvloop is a non-Windows dependency")
def test_backend_options_enable_uvloop() -> None:
    """uvloop is installed on POSIX platforms and selected for anyio."""
    assert cli._backend_options() == {"use_uvloop": True}


@pytest.mark.parametrize(
    ("agent", "want_agent"),
    [
        ("", DEFAULT_DEEP_RESEARCH_AGENT),
        ("deep-research-max-preview-04-2026", "deep-research-max-preview-04-2026"),
    ],
    ids=["default agent", "max agent"],
)
def test_run_wires_deep_research_service_agent(
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
    want_agent: str,
) -> None:
    """The CLI always constructs Deep Research with the configured agent."""
    for key in (
        ENV_GOOGLE_API_KEY,
        ENV_GOOGLE_CLOUD_PROJECT,
        ENV_GOOGLE_GENAI_USE_VERTEXAI,
        ENV_GEMINI_DEEP_RESEARCH_AGENT,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(ENV_GEMINI_API_KEY, "dummy")
    if agent:
        monkeypatch.setenv(ENV_GEMINI_DEEP_RESEARCH_AGENT, agent)

    sentinel = object()
    created: dict[str, object] = {}
    run_call: dict[str, object] = {}

    def fake_create_server(
        service: GoogleSearchService,
        research: DeepResearchService,
    ) -> object:
        created["service"] = service
        created["research"] = research
        return sentinel

    def fake_run(func: object, *args: object, **kwargs: object) -> None:
        run_call["func"] = func
        run_call["args"] = args
        run_call["kwargs"] = kwargs

    monkeypatch.setattr(cli, "create_server", fake_create_server)
    monkeypatch.setattr(cli.anyio, "run", fake_run)

    cli._run("")

    assert isinstance(created["service"], GoogleSearchService)
    research = created["research"]
    assert isinstance(research, DeepResearchService)
    assert research._agent == want_agent
    assert run_call["func"] is cli._serve
    assert run_call["args"] == (sentinel, "")
    kwargs = run_call["kwargs"]
    assert isinstance(kwargs, dict)
    assert "backend_options" in kwargs


def test_run_loads_codex_dotenv_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GEMINI_API_KEY stored only in the Codex dotenv is enough to start the server."""
    (tmp_path / ".env").write_text(f'{ENV_GEMINI_API_KEY}="codex-key"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    for key in (
        ENV_GOOGLE_API_KEY,
        ENV_GEMINI_API_KEY,
        ENV_GOOGLE_CLOUD_PROJECT,
        ENV_GOOGLE_GENAI_USE_VERTEXAI,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "create_server", lambda service, research: object())
    monkeypatch.setattr(cli.anyio, "run", lambda *args, **kwargs: None)

    with caplog.at_level(logging.INFO, logger="mcp_gemini_search"):
        cli._run("")

    assert "parsed codex dotenv" in caplog.text


def test_run_loads_claude_dotenv_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A GEMINI_API_KEY stored only in the Claude dotenv is enough to start the server."""
    (tmp_path / ".env").write_text(f'{ENV_GEMINI_API_KEY}="claude-key"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_CLAUDE_HOME, str(tmp_path))
    for key in (
        ENV_GOOGLE_API_KEY,
        ENV_GEMINI_API_KEY,
        ENV_GOOGLE_CLOUD_PROJECT,
        ENV_GOOGLE_GENAI_USE_VERTEXAI,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "create_server", lambda service, research: object())
    monkeypatch.setattr(cli.anyio, "run", lambda *args, **kwargs: None)

    with caplog.at_level(logging.INFO, logger="mcp_gemini_search"):
        cli._run("")

    assert "parsed claude dotenv" in caplog.text
