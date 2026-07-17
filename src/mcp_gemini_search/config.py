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
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values
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

CONFIG_ENV_VARS = (
    ENV_GEMINI_MODEL,
    ENV_GOOGLE_API_KEY,
    ENV_GEMINI_API_KEY,
    ENV_GOOGLE_CLOUD_PROJECT,
    ENV_GOOGLE_CLOUD_LOCATION,
    ENV_GOOGLE_GENAI_USE_VERTEXAI,
    ENV_GEMINI_ENABLE_URL_CONTEXT,
    ENV_GEMINI_ENABLE_CODE_EXECUTION,
    ENV_GEMINI_DEEP_RESEARCH_AGENT,
    ENV_GEMINI_SERVICE_TIER,
)
"""Unprefixed configuration names; each is also recognized under ``ENV_PREFIX``."""

_CLIENT_IMPORT_VARS = frozenset(CONFIG_ENV_VARS) | {ENV_PREFIX + name for name in CONFIG_ENV_VARS}

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
                f'"{ENV_GOOGLE_CLOUD_PROJECT}" or "{ENV_PREFIX}{ENV_GOOGLE_CLOUD_PROJECT}" environment '
                "variable is required when using Google Vertex AI"
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
            f'"{ENV_GOOGLE_API_KEY}" or "{ENV_GEMINI_API_KEY}" (or an "{ENV_PREFIX}"-prefixed variant) '
            "environment variable is required when using Google AI Studio"
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
    """Import recognized Codex CLI dotenv entries into the process environment.

    Codex CLI keeps user secrets in ``$CODEX_HOME/.env`` (``~/.codex/.env``
    when ``CODEX_HOME`` is unset or blank) without exporting them to the MCP
    servers it spawns, so the file is parsed before the configuration is
    resolved. Only the names in ``CONFIG_ENV_VARS`` and their ``ENV_PREFIX``
    variants are imported; every other entry — including the home variables
    themselves and ``PYTHON_DOTENV_DISABLED`` — never reaches ``os.environ``.
    Variables already present in the process environment always win, and a
    missing, non-regular, or unreadable file is skipped so non-Codex launches
    are unaffected.

    Returns the dotenv path when it contained recognized entries, ``None``
    otherwise.
    """
    env_file = _client_env_file("codex", ENV_CODEX_HOME, ".codex")
    if env_file is None:
        return None
    return _import_client_env("codex", env_file)


def _client_env_file(label: str, env_var: str, default_dirname: str) -> Path | None:
    """Resolve ``$<env_var>/.env`` (``~/<default_dirname>/.env`` fallback).

    ``MCP_GEMINI_<env_var>`` overrides ``<env_var>`` so the server's dotenv
    directory can be relocated without moving the client's own home; blank
    values fall through like every other prefixed variable. Neither name is
    importable from the dotenv file itself, so both must come from the
    process environment.
    """
    home = ""
    for candidate in (os.getenv(ENV_PREFIX + env_var), os.getenv(env_var)):
        if candidate and candidate.strip():
            home = candidate
            break
    try:
        base = Path(home).expanduser() if home else Path.home() / default_dirname
    except RuntimeError as e:
        logger.warning("skip %s dotenv: cannot resolve home directory: %s", label, e)
        return None
    return base / ".env"


def _import_client_env(label: str, env_file: Path) -> Path | None:
    """Import recognized ``env_file`` entries without overriding ``os.environ``.

    The file is opened once and required by ``fstat`` to be a regular file so
    a FIFO swapped in after a path check can never block startup; symlinks are
    followed deliberately. ``${VAR}`` interpolation is disabled so a value
    from one source can never be assembled from another. Honors python-dotenv's
    ``PYTHON_DOTENV_DISABLED`` opt-out.
    """
    if _dotenv_disabled():
        logger.debug("skip %s dotenv %s: disabled by PYTHON_DOTENV_DISABLED", label, env_file)
        return None
    try:
        fd = os.open(env_file, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        logger.warning("skip %s dotenv %s: %s", label, env_file, e)  # %s, not %r: reprs can embed file bytes
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            logger.warning("skip %s dotenv %s: not a regular file", label, env_file)
            return None
        with os.fdopen(fd, encoding="utf-8") as stream:
            fd = -1
            parsed = dotenv_values(stream=stream, interpolate=False)
    except (OSError, ValueError) as e:
        logger.warning("skip %s dotenv %s: %s", label, env_file, e)
        return None
    finally:
        if fd >= 0:
            os.close(fd)
    recognized = {key: value for key, value in parsed.items() if key in _CLIENT_IMPORT_VARS and value is not None}
    for key, value in recognized.items():
        os.environ.setdefault(key, value)
    return env_file if recognized else None


def _dotenv_disabled() -> bool:
    """Mirror python-dotenv's ``PYTHON_DOTENV_DISABLED`` opt-out for stream parsing."""
    return os.getenv("PYTHON_DOTENV_DISABLED", "").casefold() in {"1", "true", "t", "yes", "y"}


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
