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

from collections.abc import Callable
from dataclasses import dataclass

from google import genai

ENV_GEMINI_MODEL = "GEMINI_MODEL"
ENV_GOOGLE_API_KEY = "GOOGLE_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_GOOGLE_CLOUD_PROJECT = "GOOGLE_CLOUD_PROJECT"
ENV_GOOGLE_CLOUD_LOCATION = "GOOGLE_CLOUD_LOCATION"
ENV_GOOGLE_GENAI_USE_VERTEXAI = "GOOGLE_GENAI_USE_VERTEXAI"

DEFAULT_MODEL = "gemini-3.1-pro-preview"
DEFAULT_LOCATION = "global"


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration resolved from the environment."""

    model: str
    vertexai: bool = False
    api_key: str = ""
    project: str = ""
    location: str = ""

    def new_client(self) -> genai.Client:
        """Create a new genai.Client based on the server configuration."""
        return genai.Client(
            vertexai=self.vertexai,
            api_key=self.api_key or None,
            project=self.project or None,
            location=self.location or None,
        )


def load_config_from_env(getenv: Callable[[str], str]) -> ServerConfig:
    """Resolve the server configuration from environment variable lookups."""
    model = getenv(ENV_GEMINI_MODEL).strip() or DEFAULT_MODEL

    if _is_enabled(getenv(ENV_GOOGLE_GENAI_USE_VERTEXAI)):
        project = getenv(ENV_GOOGLE_CLOUD_PROJECT)
        if not project:
            raise ValueError(
                f'"{ENV_GOOGLE_CLOUD_PROJECT}" environment variable is required when using Google Vertex AI'
            )
        location = getenv(ENV_GOOGLE_CLOUD_LOCATION) or DEFAULT_LOCATION
        return ServerConfig(
            model=model,
            vertexai=True,
            project=project,
            location=location,
        )

    api_key = _first_non_empty(
        getenv(ENV_GOOGLE_API_KEY),
        getenv(ENV_GEMINI_API_KEY),
    )
    if not api_key:
        raise ValueError(
            f'"{ENV_GOOGLE_API_KEY}" or "{ENV_GEMINI_API_KEY}" environment '
            "variable is required when using Google AI Studio"
        )
    return ServerConfig(model=model, api_key=api_key)


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
