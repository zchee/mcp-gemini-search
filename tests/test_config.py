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

"""Tests for environment-driven configuration loading and client construction."""

import logging
import os
from pathlib import Path

import pytest

from mcp_gemini_search.config import (
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    ENV_CLAUDE_HOME,
    ENV_CODEX_HOME,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_DEEP_RESEARCH_AGENT,
    ENV_GEMINI_ENABLE_CODE_EXECUTION,
    ENV_GEMINI_ENABLE_URL_CONTEXT,
    ENV_GEMINI_MODEL,
    ENV_GEMINI_SERVICE_TIER,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_CLOUD_LOCATION,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_GOOGLE_GENAI_USE_VERTEXAI,
    ServerConfig,
    _first_non_empty,
    _is_enabled,
    load_claude_env,
    load_codex_env,
    load_config_from_env,
)


@pytest.mark.parametrize(
    ("env", "want"),
    [
        (
            {
                ENV_GOOGLE_API_KEY: "google-key",
                ENV_GEMINI_MODEL: "gemini-2.0-flash",
            },
            ServerConfig(model="gemini-2.0-flash", api_key="google-key"),
        ),
        (
            {ENV_GOOGLE_API_KEY: "test-key"},
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key"),
        ),
        (
            {
                ENV_GOOGLE_CLOUD_PROJECT: "project-1",
                ENV_GOOGLE_GENAI_USE_VERTEXAI: "true",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                vertexai=True,
                project="project-1",
                location=DEFAULT_LOCATION,
            ),
        ),
        (
            {
                ENV_GOOGLE_CLOUD_PROJECT: "project-2",
                ENV_GOOGLE_CLOUD_LOCATION: "asia-northeast1",
                ENV_GOOGLE_GENAI_USE_VERTEXAI: "true",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                vertexai=True,
                project="project-2",
                location="asia-northeast1",
            ),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_ENABLE_URL_CONTEXT: "1",
                ENV_GEMINI_ENABLE_CODE_EXECUTION: "true",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                api_key="test-key",
                url_context=True,
                code_execution=True,
            ),
        ),
        (
            {
                ENV_GOOGLE_CLOUD_PROJECT: "project-1",
                ENV_GOOGLE_GENAI_USE_VERTEXAI: "true",
                ENV_GEMINI_ENABLE_CODE_EXECUTION: "on",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                vertexai=True,
                project="project-1",
                location=DEFAULT_LOCATION,
                code_execution=True,
            ),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_ENABLE_URL_CONTEXT: "0",
                ENV_GEMINI_ENABLE_CODE_EXECUTION: "off",
            },
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key"),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_DEEP_RESEARCH_AGENT: "deep-research-max-preview-04-2026",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                api_key="test-key",
                deep_research_agent="deep-research-max-preview-04-2026",
            ),
        ),
        (
            {
                ENV_GOOGLE_CLOUD_PROJECT: "project-1",
                ENV_GOOGLE_GENAI_USE_VERTEXAI: "true",
                ENV_GEMINI_DEEP_RESEARCH_AGENT: "deep-research-max-preview-04-2026",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                vertexai=True,
                project="project-1",
                location=DEFAULT_LOCATION,
                deep_research_agent="deep-research-max-preview-04-2026",
            ),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_SERVICE_TIER: "flex",
            },
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key", service_tier="flex"),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_SERVICE_TIER: "standard",
            },
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key", service_tier="standard"),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_SERVICE_TIER: "priority",
            },
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key", service_tier="priority"),
        ),
        (
            {
                ENV_GOOGLE_API_KEY: "test-key",
                ENV_GEMINI_SERVICE_TIER: " Flex ",
            },
            ServerConfig(model=DEFAULT_MODEL, api_key="test-key", service_tier="flex"),
        ),
        (
            {
                ENV_GOOGLE_CLOUD_PROJECT: "project-1",
                ENV_GOOGLE_GENAI_USE_VERTEXAI: "true",
                ENV_GEMINI_SERVICE_TIER: "priority",
            },
            ServerConfig(
                model=DEFAULT_MODEL,
                vertexai=True,
                project="project-1",
                location=DEFAULT_LOCATION,
                service_tier="priority",
            ),
        ),
    ],
    ids=[
        "google api key",
        "gemini api key fallback and custom model",
        "vertex provider compatibility env",
        "vertex native go sdk env fallback",
        "optional tools enabled",
        "vertex optional code execution",
        "optional tools falsy values stay disabled",
        "ai studio deep research max agent",
        "vertex deep research max agent",
        "service tier flex",
        "service tier standard",
        "service tier priority",
        "service tier whitespace and case normalized",
        "vertex service tier priority",
    ],
)
def test_load_config_from_env(env: dict[str, str], want: ServerConfig) -> None:
    """Each supported environment combination resolves to the expected ServerConfig."""

    def getenv(key: str) -> str:
        return env.get(key, "")

    assert load_config_from_env(getenv) == want


@pytest.mark.parametrize(
    ("env", "want_err"),
    [
        (
            {},
            '"GOOGLE_API_KEY" or "GEMINI_API_KEY" environment variable is required when using Google AI Studio',
        ),
        (
            {ENV_GOOGLE_GENAI_USE_VERTEXAI: "true"},
            '"GOOGLE_CLOUD_PROJECT" environment variable is required when using Google Vertex AI',
        ),
        (
            {ENV_GOOGLE_API_KEY: "test-key", ENV_GEMINI_SERVICE_TIER: "turbo"},
            '"GEMINI_SERVICE_TIER" must be one of "flex", "standard", "priority" (or unset); got \'turbo\'',
        ),
    ],
    ids=["missing api key", "missing vertex project", "invalid service tier"],
)
def test_load_config_from_env_errors(env: dict[str, str], want_err: str) -> None:
    """Missing required variables raise ValueError with the Go-identical message."""

    def getenv(key: str) -> str:
        return env.get(key, "")

    with pytest.raises(ValueError) as exc_info:
        load_config_from_env(getenv)
    assert str(exc_info.value) == want_err


def test_load_config_from_env_tolerates_none_lookups() -> None:
    """os.getenv-style lookups returning None for missing keys resolve defaults."""
    env = {ENV_GOOGLE_API_KEY: "test-key"}

    config = load_config_from_env(env.get)

    assert config == ServerConfig(model=DEFAULT_MODEL, api_key="test-key")


def test_server_config_new_client_rejects_mutually_exclusive_settings() -> None:
    """google-genai rejects an API key combined with a Vertex project."""
    cfg = ServerConfig(model=DEFAULT_MODEL, api_key="test-key", project="project-1")
    with pytest.raises(ValueError):
        cfg.new_client()


def test_first_non_empty() -> None:
    """The first non-blank value wins after trimming."""
    assert _first_non_empty("", "  ", "value", "other") == "value"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_is_enabled_truthy(value: str) -> None:
    """Truthy spellings enable the Vertex switch."""
    assert _is_enabled(value) is True


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_is_enabled_falsy(value: str) -> None:
    """Falsy spellings leave the Vertex switch disabled."""
    assert _is_enabled(value) is False


def test_new_client_resolves_gemini_backend_ignoring_vertex_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit backend bool blocks the SDK env re-derivation."""
    monkeypatch.delenv(ENV_GOOGLE_API_KEY, raising=False)
    monkeypatch.delenv(ENV_GOOGLE_GENAI_USE_VERTEXAI, raising=False)
    monkeypatch.delenv(ENV_GOOGLE_CLOUD_PROJECT, raising=False)
    monkeypatch.setenv(ENV_GEMINI_API_KEY, "gemini-key")
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")

    config = load_config_from_env(os.getenv)
    assert config.vertexai is False

    client = config.new_client()
    assert client.vertexai is False


def test_new_client_vertex_config_resolves_vertex_backend() -> None:
    """A Vertex ServerConfig resolves to the Vertex backend."""
    config = ServerConfig(
        model=DEFAULT_MODEL,
        vertexai=True,
        project="project-1",
        location=DEFAULT_LOCATION,
    )
    client = config.new_client()
    assert client.vertexai is True


_CLIENT_TEST_VAR = "MCP_GEMINI_SEARCH_CLIENT_TEST_VAR"


def _write_client_env(directory: Path) -> Path:
    """Create ``<directory>/.env`` holding one marker variable and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    env_file = directory / ".env"
    env_file.write_text(f'{_CLIENT_TEST_VAR}="from-dotenv"\n', encoding="utf-8")
    return env_file


def test_load_codex_env_loads_from_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """$CODEX_HOME/.env entries are parsed into os.environ."""
    env_file = _write_client_env(tmp_path)
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_codex_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-dotenv"


@pytest.mark.parametrize("codex_home", [None, ""], ids=["unset", "empty"])
def test_load_codex_env_defaults_to_home_codex(
    codex_home: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """An unset or empty CODEX_HOME falls back to ~/.codex/.env."""
    env_file = _write_client_env(tmp_path / ".codex")
    monkeypatch.setenv("HOME", str(tmp_path))
    if codex_home is None:
        monkeypatch.delenv(ENV_CODEX_HOME, raising=False)
    else:
        monkeypatch.setenv(ENV_CODEX_HOME, codex_home)
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_codex_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-dotenv"


def test_load_codex_env_expands_tilde_codex_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """A leading ~ in CODEX_HOME resolves against the home directory."""
    env_file = _write_client_env(tmp_path / "codex-home")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ENV_CODEX_HOME, "~/codex-home")
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_codex_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-dotenv"


def test_load_codex_env_never_overrides_process_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """Variables already exported win over Codex dotenv values."""
    env_file = _write_client_env(tmp_path)
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    monkeypatch.setenv(_CLIENT_TEST_VAR, "from-process")

    assert load_codex_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-process"


def test_load_codex_env_missing_file_is_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """A CODEX_HOME without a .env file is a silent no-op."""
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_codex_env() is None
    assert _CLIENT_TEST_VAR not in os.environ


@pytest.mark.parametrize(
    "kind",
    [
        "directory",
        pytest.param("fifo", marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is POSIX-only")),
    ],
)
def test_load_codex_env_non_regular_file_warns_and_skips(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A directory or FIFO at the dotenv path warns and is skipped without blocking startup."""
    env_file = tmp_path / ".env"
    if kind == "directory":
        env_file.mkdir()
    else:
        os.mkfifo(env_file)
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    with caplog.at_level(logging.WARNING, logger="mcp_gemini_search"):
        assert load_codex_env() is None

    assert "not a regular file" in caplog.text
    assert _CLIENT_TEST_VAR not in os.environ


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root bypasses file permission bits")
def test_load_codex_env_unreadable_file_warns_and_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unreadable dotenv file logs a warning and leaves the environment untouched."""
    env_file = _write_client_env(tmp_path)
    env_file.chmod(0o000)
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    with caplog.at_level(logging.WARNING, logger="mcp_gemini_search"):
        assert load_codex_env() is None

    assert "skip codex dotenv" in caplog.text
    assert _CLIENT_TEST_VAR not in os.environ


def test_load_codex_env_undecodable_file_warns_and_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A non-UTF-8 dotenv file logs a warning instead of crashing startup."""
    (tmp_path / ".env").write_bytes(b"\xff\xfe\x00KEY=1\n")
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))

    with caplog.at_level(logging.WARNING, logger="mcp_gemini_search"):
        assert load_codex_env() is None

    assert "skip codex dotenv" in caplog.text


def test_load_codex_env_feeds_load_config_from_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """A GEMINI_API_KEY stored in the Codex dotenv satisfies config resolution."""
    (tmp_path / ".env").write_text(f'{ENV_GEMINI_API_KEY}="codex-key"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_CODEX_HOME, str(tmp_path))
    for key in (
        ENV_GOOGLE_API_KEY,
        ENV_GEMINI_API_KEY,
        ENV_GOOGLE_GENAI_USE_VERTEXAI,
        ENV_GEMINI_MODEL,
        ENV_GEMINI_SERVICE_TIER,
        ENV_GEMINI_DEEP_RESEARCH_AGENT,
        ENV_GEMINI_ENABLE_URL_CONTEXT,
        ENV_GEMINI_ENABLE_CODE_EXECUTION,
    ):
        monkeypatch.delenv(key, raising=False)

    assert load_codex_env() == tmp_path / ".env"
    assert load_config_from_env(os.getenv) == ServerConfig(model=DEFAULT_MODEL, api_key="codex-key")


def test_load_claude_env_loads_from_claude_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """$CLAUDE_HOME/.env entries are parsed into os.environ."""
    env_file = _write_client_env(tmp_path)
    monkeypatch.setenv(ENV_CLAUDE_HOME, str(tmp_path))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_claude_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-dotenv"


@pytest.mark.parametrize("claude_home", [None, ""], ids=["unset", "empty"])
def test_load_claude_env_defaults_to_home_claude(
    claude_home: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """An unset or empty CLAUDE_HOME falls back to ~/.claude/.env."""
    env_file = _write_client_env(tmp_path / ".claude")
    monkeypatch.setenv("HOME", str(tmp_path))
    if claude_home is None:
        monkeypatch.delenv(ENV_CLAUDE_HOME, raising=False)
    else:
        monkeypatch.setenv(ENV_CLAUDE_HOME, claude_home)
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_claude_env() == env_file
    assert os.environ[_CLIENT_TEST_VAR] == "from-dotenv"


def test_codex_dotenv_wins_over_claude_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_environ: None,
) -> None:
    """With the CLI's codex-then-claude load order, the Codex value wins under override=False."""
    codex_dir = tmp_path / "codex"
    claude_dir = tmp_path / "claude"
    codex_dir.mkdir()
    claude_dir.mkdir()
    (codex_dir / ".env").write_text(f'{_CLIENT_TEST_VAR}="from-codex"\n', encoding="utf-8")
    (claude_dir / ".env").write_text(f'{_CLIENT_TEST_VAR}="from-claude"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_CODEX_HOME, str(codex_dir))
    monkeypatch.setenv(ENV_CLAUDE_HOME, str(claude_dir))
    monkeypatch.delenv(_CLIENT_TEST_VAR, raising=False)

    assert load_codex_env() == codex_dir / ".env"
    assert load_claude_env() == claude_dir / ".env"
    assert os.environ[_CLIENT_TEST_VAR] == "from-codex"
