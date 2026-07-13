# Copyright 2026 The mcp-gemini-google-search Authors.
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

import anyio
import pytest
from google.genai import types

from mcp_gemini_google_search.search import (
    GoogleSearchService,
    format_grounded_response,
)


def _benchmark_response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Alpha "),
                        types.Part(text="Beta "),
                        types.Part(text="Gamma Delta Epsilon Zeta Eta Theta"),
                    ],
                ),
                grounding_metadata=types.GroundingMetadata(
                    grounding_chunks=[
                        types.GroundingChunk(
                            web=types.GroundingChunkWeb(
                                title="First", uri="https://first.example"
                            )
                        ),
                        types.GroundingChunk(
                            maps=types.GroundingChunkMaps(
                                title="Second", uri="https://second.example"
                            )
                        ),
                        types.GroundingChunk(
                            retrieved_context=types.GroundingChunkRetrievedContext(
                                title="Third", uri="https://third.example"
                            )
                        ),
                        types.GroundingChunk(
                            image=types.GroundingChunkImage(
                                title="Fourth", source_uri="https://fourth.example"
                            )
                        ),
                    ],
                    grounding_supports=[
                        _support(0, 6, [0]),
                        _support(1, 5, [0, 1]),
                        _support(2, 5, [2]),
                        _support(2, 13, [1, 2, 3]),
                        _support(2, 21, [3]),
                        _support(2, 26, [0, 3]),
                        _support(2, 30, [1]),
                        _support(2, 39, [0, 2, 3]),
                    ],
                ),
            )
        ]
    )


def _support(
    part_index: int, end_index: int, indices: list[int]
) -> types.GroundingSupport:
    return types.GroundingSupport(
        segment=types.Segment(part_index=part_index, end_index=end_index),
        grounding_chunk_indices=indices,
    )


class _BenchStub:
    def __init__(self, resp: types.GenerateContentResponse) -> None:
        self._resp = resp

    async def generate_content(
        self,
        *,
        model: str,
        contents: types.ContentListUnion,
        config: types.GenerateContentConfig | None = None,
    ) -> types.GenerateContentResponse:
        return self._resp


@pytest.mark.benchmark
def test_benchmark_format_grounded_response(benchmark) -> None:
    resp = _benchmark_response()
    text, _ = benchmark(format_grounded_response, resp)
    assert text.startswith("Alpha [1]")


@pytest.mark.benchmark
def test_benchmark_google_search_service_search(benchmark) -> None:
    svc = GoogleSearchService("gemini-2.5-flash", _BenchStub(_benchmark_response()))

    def run() -> str:
        return anyio.run(svc.search, "latest golang release notes").text

    text = benchmark(run)
    assert text.startswith("Alpha [1]")
