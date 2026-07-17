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

"""Shared pytest fixtures for the test suite."""

import os

import pytest

from mcp_gemini_search.config import CONFIG_ENV_VARS, ENV_CODEX_HOME, ENV_PREFIX


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio-backed async tests to the asyncio event loop."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_client_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CODEX_HOME at a non-directory so a developer's real dotenv file never leaks into tests.

    The MCP_GEMINI_-prefixed home variable is deleted rather than set: it
    outranks the unprefixed name, so a leftover devnull value would shadow the
    homes that individual tests configure.
    """
    monkeypatch.setenv(ENV_CODEX_HOME, os.devnull)
    monkeypatch.delenv(ENV_PREFIX + ENV_CODEX_HOME, raising=False)


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap os.environ for a scrubbed copy so host settings and dotenv writes never leak across the test boundary.

    Every supported configuration variable — prefixed and unprefixed — plus
    python-dotenv's PYTHON_DOTENV_DISABLED switch is removed from the copy, so
    a developer's exported MCP_GEMINI_* settings cannot change test results.
    """
    env = os.environ.copy()
    for name in CONFIG_ENV_VARS:
        env.pop(name, None)
        env.pop(ENV_PREFIX + name, None)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    monkeypatch.setattr(os, "environ", env)
