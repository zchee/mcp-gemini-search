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

"""Environment-driven configuration and client construction for the server."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from mcp_gemini_search._logging import logger
from mcp_gemini_search.research import DEEP_RESEARCH_AGENT as DEFAULT_DEEP_RESEARCH_AGENT

ENV_PREFIX = "MCP_GEMINI_"

ENV_GEMINI_MODEL = "GEMINI_MODEL"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
ENV_GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"
ENV_GOOGLE_GENAI_USE_VERTEXAI = "GOOGLE_GENAI_USE_VERTEXAI"
ENV_GEMINI_ENABLE_URL_CONTEXT = "GEMINI_ENABLE_URL_CONTEXT"
ENV_GEMINI_ENABLE_CODE_EXECUTION = "GEMINI_ENABLE_CODE_EXECUTION"
ENV_GEMINI_DEEP_RESEARCH_AGENT = "GEMINI_DEEP_RESEARCH_AGENT"
ENV_GEMINI_SERVICE_TIER = "GEMINI_SERVICE_TIER"
ENV_CODEX_HOME = "CODEX_HOME"
ENV_CLAUDE_HOME = "CLAUDE_HOME"

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"

_SERVICE_TIERS = ("flex", "standard", "priority")


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration resolved from the environment."""

    model: str
    vertexai: bool = False
    api_key: str = ""
    project: str = ""
    location: str = ""
    url_context: bool = False
    code_execution: bool = False
    deep_research_agent: str = DEFAULT_DEEP_RESEARCH_AGENT
    service_tier: str = ""

    def new_client(self) -> genai.Client:
        """Create a new genai.Client based on the server configuration."""
        return genai.Client(
            vertexai=self.vertexai,
            api_key=self.api_key or None,
            project=self.project or None,
            location=self.location or None,
        )


def load_config_from_env(getenv: Callable[[str], str | None]) -> ServerConfig:
    """Resolve the server configuration from environment variable lookups.

    ``getenv`` may return ``None`` for missing keys (``os.getenv`` style);
    missing values are treated as empty strings. Every variable is also
    recognized under the ``MCP_GEMINI_`` prefix (e.g. ``MCP_GEMINI_GEMINI_MODEL``),
    and a non-blank prefixed value wins over the unprefixed name so this server
    can be configured independently of tools sharing the generic variables. For
    the API key pair, both prefixed keys win over both unprefixed keys.
    """

    def raw(key: str) -> str:
        return getenv(key) or ""

    def resolve(key: str) -> tuple[str, str]:
        prefixed_key = ENV_PREFIX + key
        prefixed = raw(prefixed_key)
        if prefixed.strip():
            return prefixed_key, prefixed
        return key, raw(key)

    def lookup(key: str) -> str:
        return resolve(key)[1]

    model = lookup(ENV_GEMINI_MODEL).strip() or DEFAULT_MODEL
    url_context = _is_enabled(lookup(ENV_GEMINI_ENABLE_URL_CONTEXT))
    code_execution = _is_enabled(lookup(ENV_GEMINI_ENABLE_CODE_EXECUTION))
    deep_research_agent = lookup(ENV_GEMINI_DEEP_RESEARCH_AGENT).strip() or DEFAULT_DEEP_RESEARCH_AGENT
    service_tier_var, raw_service_tier = resolve(ENV_GEMINI_SERVICE_TIER)
    service_tier = raw_service_tier.strip().lower()
    if service_tier and service_tier not in _SERVICE_TIERS:
        allowed = ", ".join(f'"{tier}"' for tier in _SERVICE_TIERS)
        raise ValueError(f'"{service_tier_var}" must be one of {allowed} (or unset); got {raw_service_tier!r}')

    if _is_enabled(lookup(ENV_GOOGLE_GENAI_USE_VERTEXAI)):
        project = lookup(ENV_GOOGLE_CLOUD_PROJECT)
        if not project:
            raise ValueError(
                f'"{ENV_GOOGLE_CLOUD_PROJECT}" environment variable is required when using Google Vertex AI'
            )
        location = lookup(ENV_GOOGLE_CLOUD_LOCATION) or DEFAULT_LOCATION
        return ServerConfig(
            model=model,
            vertexai=True,
            project=project,
            location=location,
            url_context=url_context,
            code_execution=code_execution,
            deep_research_agent=deep_research_agent,
            service_tier=service_tier,
        )

    api_key = _first_non_empty(
        raw(ENV_PREFIX + ENV_GOOGLE_API_KEY),
        raw(ENV_PREFIX + ENV_GEMINI_API_KEY),
        raw(ENV_GOOGLE_API_KEY),
        raw(ENV_GEMINI_API_KEY),
    )
    if not api_key:
        raise ValueError(
            f'"{ENV_GOOGLE_API_KEY}" or "{ENV_GEMINI_API_KEY}" environment '
            "variable is required when using Google AI Studio"
        )
    return ServerConfig(
        model=model,
        api_key=api_key,
        url_context=url_context,
        code_execution=code_execution,
        deep_research_agent=deep_research_agent,
        service_tier=service_tier,
    )


def load_codex_env() -> Path | None:
    """Load Codex CLI dotenv entries into the process environment.

    Codex CLI keeps user secrets in ``$CODEX_HOME/.env`` (``~/.codex/.env``
    when ``CODEX_HOME`` is unset or empty) without exporting them to the MCP
    servers it spawns, so the file is parsed here before the configuration is
    resolved. Variables already present in the process environment always win,
    and a missing, non-regular, or unreadable file is skipped so non-Codex
    launches are unaffected.

    Returns the dotenv path when it contained entries, ``None`` otherwise.
    """
    return _load_home_env(ENV_CODEX_HOME, ".codex")


def load_claude_env() -> Path | None:
    """Load Claude Code dotenv entries into the process environment.

    Mirrors ``load_codex_env`` for ``$CLAUDE_HOME/.env`` (``~/.claude/.env``
    when ``CLAUDE_HOME`` is unset or empty), with the same precedence and
    failure handling. When both client files define a variable, the caller's
    load order decides: the CLI loads Codex first, so its value wins under
    ``override=False``.

    Returns the dotenv path when it contained entries, ``None`` otherwise.
    """
    return _load_home_env(ENV_CLAUDE_HOME, ".claude")


def _load_home_env(env_var: str, default_dirname: str) -> Path | None:
    """Parse ``$<env_var>/.env`` (``~/<default_dirname>/.env`` fallback) into os.environ.

    ``MCP_GEMINI_<env_var>`` overrides ``<env_var>`` so the server's dotenv
    directory can be relocated without moving the client's own home.
    """
    label = default_dirname.removeprefix(".")
    try:
        home = os.getenv(ENV_PREFIX + env_var) or os.getenv(env_var) or ""
        base = Path(home).expanduser() if home else Path.home() / default_dirname
    except RuntimeError as e:
        logger.warning("skip %s dotenv: cannot resolve home directory: %s", label, e)
        return None
    env_file = base / ".env"
    if not env_file.is_file():
        # Regular files only: python-dotenv would block forever opening a FIFO.
        if env_file.exists():
            logger.warning("skip %s dotenv %s: not a regular file", label, env_file)
        return None
    try:
        loaded = load_dotenv(env_file, override=False)
    except (OSError, ValueError) as e:
        logger.warning("skip %s dotenv %s: %s", label, env_file, e)  # %s, not %r: reprs can embed file bytes
        return None
    return env_file if loaded else None


def _first_non_empty(*values: str) -> str:
    """Return the first value that is non-empty after stripping whitespace."""
    for value in values:
        stripped = value.strip()
        if stripped:
            return stripped
    return ""


def _is_enabled(value: str) -> bool:
    """Report whether the value is a recognized truthy flag string."""
    return value.strip().lower() in {"1", "true", "yes", "on"}
