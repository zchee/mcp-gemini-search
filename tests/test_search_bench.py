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

"""Benchmarks for the Google Search grounding service.

Ported from ``search_bench_test.go``. Marked ``benchmark`` so they are
deselected by default and run serially with ``-m benchmark``.
"""

from collections.abc import Mapping, Sequence

import anyio
import pytest
from google.genai import interactions
from pytest_benchmark.fixture import BenchmarkFixture

from mcp_gemini_search.search import (
    GoogleSearchService,
    format_interaction,
)


def _cite(url: str, title: str, end_index: int) -> interactions.URLCitation:
    return interactions.URLCitation(url=url, title=title, start_index=0, end_index=end_index)


def _benchmark_interaction() -> interactions.Interaction:
    first = _cite("https://first.example", "First", 6)
    return interactions.Interaction(
        status="completed",
        steps=[
            interactions.ModelOutputStep(
                content=[
                    interactions.TextContent(text="Alpha ", annotations=[first]),
                    interactions.TextContent(
                        text="Beta ",
                        annotations=[
                            _cite("https://first.example", "First", 5),
                            _cite("https://second.example", "Second", 5),
                        ],
                    ),
                    interactions.TextContent(
                        text="Gamma Delta Epsilon Zeta Eta Theta",
                        annotations=[
                            _cite("https://third.example", "Third", 5),
                            _cite("https://second.example", "Second", 13),
                            _cite("https://third.example", "Third", 13),
                            _cite("https://fourth.example", "Fourth", 13),
                            _cite("https://fourth.example", "Fourth", 21),
                            _cite("https://first.example", "First", 26),
                            _cite("https://fourth.example", "Fourth", 26),
                            _cite("https://second.example", "Second", 30),
                            _cite("https://first.example", "First", 34),
                            _cite("https://third.example", "Third", 34),
                            _cite("https://fourth.example", "Fourth", 34),
                        ],
                    ),
                ]
            )
        ],
    )


class _BenchStub:
    def __init__(self, interaction: interactions.Interaction) -> None:
        self._interaction = interaction

    async def create(
        self,
        *,
        model: str,
        input: str,
        tools: Sequence[Mapping[str, str]],
        store: bool,
    ) -> interactions.Interaction:
        return self._interaction


@pytest.mark.benchmark
def test_benchmark_format_interaction(benchmark: BenchmarkFixture) -> None:
    """Benchmark the interaction formatter."""
    interaction = _benchmark_interaction()
    text, _ = benchmark(format_interaction, interaction)
    assert text.startswith("Alpha [1]")


@pytest.mark.benchmark
def test_benchmark_google_search_service_search(benchmark: BenchmarkFixture) -> None:
    """Benchmark the full search path against a stub interactions API."""
    svc = GoogleSearchService("gemini-3.5-flash", _BenchStub(_benchmark_interaction()))

    def run() -> str:
        return anyio.run(svc.search, "latest golang release notes").text

    text = benchmark(run)
    assert text.startswith("Alpha [1]")
