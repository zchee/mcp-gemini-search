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

import os

import pytest

from mcp_gemini_search.config import (
    DEFAULT_LOCATION,
    DEFAULT_MODEL,
    ENV_GEMINI_API_KEY,
    ENV_GEMINI_DEEP_RESEARCH_AGENT,
    ENV_GEMINI_ENABLE_CODE_EXECUTION,
    ENV_GEMINI_ENABLE_URL_CONTEXT,
    ENV_GEMINI_MODEL,
    ENV_GOOGLE_API_KEY,
    ENV_GOOGLE_CLOUD_LOCATION,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_GOOGLE_GENAI_USE_VERTEXAI,
    ServerConfig,
    _first_non_empty,
    _is_enabled,
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
    ],
    ids=["missing api key", "missing vertex project"],
)
def test_load_config_from_env_errors(env: dict[str, str], want_err: str) -> None:
    """Missing required variables raise ValueError with the Go-identical message."""

    def getenv(key: str) -> str:
        return env.get(key, "")

    with pytest.raises(ValueError) as exc_info:
        load_config_from_env(getenv)
    assert str(exc_info.value) == want_err


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

    config = load_config_from_env(lambda key: os.environ.get(key, ""))
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
