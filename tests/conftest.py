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


@pytest.fixture
def anyio_backend() -> str:
    """Pin anyio-backed async tests to the asyncio event loop."""
    return "asyncio"


@pytest.fixture(autouse=True)
def _no_codex_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point CODEX_HOME at a non-directory so a developer's real Codex dotenv never leaks into tests."""
    monkeypatch.setenv("CODEX_HOME", os.devnull)


@pytest.fixture
def isolated_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap os.environ for a copy so dotenv writes never leak out of a test."""
    monkeypatch.setattr(os, "environ", os.environ.copy())
