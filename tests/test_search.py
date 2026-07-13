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

"""Tests for the Google Search grounding service."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jsonschema
import orjson
import pytest
from google.genai import types

from mcp_gemini_google_search.search import (
    ContentGenerator,
    GoogleSearchOutput,
    GoogleSearchService,
    GoogleSearchSource,
    _citation_text,
    _grounding_source,
    format_grounded_response,
)


class StubGenerator:
    """Records the request and returns a canned response or raises an error."""

    def __init__(
        self,
        *,
        resp: types.GenerateContentResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.resp = resp
        self.error = error
        self.got_model: str | None = None
        self.got_contents: types.ContentListUnion | None = None
        self.got_config: types.GenerateContentConfig | None = None

    async def generate_content(
        self,
        *,
        model: str,
        contents: types.ContentListUnion,
        config: types.GenerateContentConfig | None = None,
    ) -> types.GenerateContentResponse:
        self.got_model = model
        self.got_contents = contents
        self.got_config = config
        if self.error is not None:
            raise self.error
        if self.resp is None:
            raise RuntimeError("stub generator is misconfigured")
        return self.resp


def _part(text: str, *, thought: bool = False) -> types.Part:
    return types.Part(text=text, thought=thought)


def _web(title: str, uri: str) -> types.GroundingChunk:
    return types.GroundingChunk(web=types.GroundingChunkWeb(title=title, uri=uri))


def _support(
    part_index: int, end_index: int, indices: Sequence[int]
) -> types.GroundingSupport:
    return types.GroundingSupport(
        segment=types.Segment(part_index=part_index, end_index=end_index),
        grounding_chunk_indices=list(indices),
    )


def _response(
    parts: list[types.Part],
    chunks: list[types.GroundingChunk] | None = None,
    supports: list[types.GroundingSupport] | None = None,
    *,
    metadata: bool = True,
) -> types.GenerateContentResponse:
    grounding = (
        types.GroundingMetadata(
            grounding_chunks=chunks or [],
            grounding_supports=supports or [],
        )
        if metadata
        else None
    )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=parts),
                grounding_metadata=grounding,
            )
        ]
    )


@pytest.mark.anyio
async def test_search_happy_path() -> None:
    stub = StubGenerator(
        resp=_response(
            [_part("Answer")],
            [_web("Example", "https://example.com")],
            [_support(0, 6, [0])],
        )
    )
    svc = GoogleSearchService("gemini-2.5-flash", stub)

    got = await svc.search("golang")

    assert got.query == "golang"
    assert got.text == "Answer[1]\n\nSources:\n[1] Example (https://example.com)"
    assert len(got.sources) == 1
    assert stub.got_model == "gemini-2.5-flash"

    contents = stub.got_contents
    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    assert isinstance(content, types.Content)
    assert content.parts is not None
    assert len(content.parts) == 1
    assert content.parts[0].text == "golang"

    config = stub.got_config
    assert config is not None
    assert config.tools is not None
    assert len(config.tools) == 1
    tool = config.tools[0]
    assert isinstance(tool, types.Tool)
    assert tool.google_search is not None


@pytest.mark.anyio
async def test_search_not_configured() -> None:
    svc = GoogleSearchService("gemini-2.5-flash", None)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search service is not configured"


@pytest.mark.anyio
async def test_search_empty_query() -> None:
    svc = GoogleSearchService("gemini-2.5-flash", StubGenerator())
    with pytest.raises(ValueError) as excinfo:
        await svc.search("   ")
    assert str(excinfo.value) == "search query cannot be empty"


@pytest.mark.anyio
async def test_search_backend_error() -> None:
    backend_error = RuntimeError("backend failed")
    svc = GoogleSearchService("gemini-2.5-flash", StubGenerator(error=backend_error))
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: backend failed"
    assert excinfo.value.__cause__ is backend_error


@pytest.mark.anyio
async def test_search_wraps_format_error() -> None:
    stub = StubGenerator(resp=types.GenerateContentResponse())
    svc = GoogleSearchService("gemini-2.5-flash", stub)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: no response from Gemini model"
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "no response from Gemini model"


def test_format_grounded_response() -> None:
    resp = _response(
        [_part("Alpha "), _part("Beta")],
        [
            _web("First", "https://first.example"),
            types.GroundingChunk(
                maps=types.GroundingChunkMaps(
                    title="Second", uri="https://second.example"
                )
            ),
        ],
        [_support(0, 6, [0]), _support(1, 4, [0, 1])],
    )

    text, sources = format_grounded_response(resp)

    assert text == (
        "Alpha [1]Beta[1,2]\n\nSources:\n"
        "[1] First (https://first.example)\n"
        "[2] Second (https://second.example)"
    )
    assert len(sources) == 2
    assert sources[1].title == "Second"
    assert sources[1].uri == "https://second.example"


def test_format_grounded_response_no_text() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        format_grounded_response(types.GenerateContentResponse())
    assert str(excinfo.value) == "no response from Gemini model"


def test_format_grounded_response_none() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        format_grounded_response(None)
    assert str(excinfo.value) == "no response from Gemini model"


def test_format_grounded_response_orders_and_deduplicates_citations() -> None:
    resp = _response(
        [_part("Alpha "), _part("Beta")],
        [
            _web("One", "https://one.example"),
            _web("Two", "https://two.example"),
        ],
        [
            _support(1, 4, [1, 0, 1]),
            _support(0, 6, [0]),
            _support(1, 4, [0]),
        ],
    )

    text, sources = format_grounded_response(resp)

    assert text == (
        "Alpha [1]Beta[1,2]\n\nSources:\n"
        "[1] One (https://one.example)\n"
        "[2] Two (https://two.example)"
    )
    assert len(sources) == 2


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (None, ("", "")),
        (types.GroundingChunk(), ("", "")),
        (
            types.GroundingChunk(web=types.GroundingChunkWeb(title="W", uri="w://u")),
            ("W", "w://u"),
        ),
        (
            types.GroundingChunk(maps=types.GroundingChunkMaps(title="M", uri="m://u")),
            ("M", "m://u"),
        ),
        (
            types.GroundingChunk(
                retrieved_context=types.GroundingChunkRetrievedContext(
                    title="R", uri="r://u"
                )
            ),
            ("R", "r://u"),
        ),
        (
            types.GroundingChunk(
                image=types.GroundingChunkImage(
                    title="Image Result",
                    source_uri="https://source.example",
                    image_uri="https://image.example",
                )
            ),
            ("Image Result", "https://source.example"),
        ),
        (
            types.GroundingChunk(
                image=types.GroundingChunkImage(
                    title="OnlyImage", image_uri="https://image.example"
                )
            ),
            ("OnlyImage", "https://image.example"),
        ),
    ],
)
def test_grounding_source(
    chunk: types.GroundingChunk | None, expected: tuple[str, str]
) -> None:
    assert _grounding_source(chunk) == expected


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [
        ([], "[]"),
        ([1, 2, 3], "[1,2,3]"),
        ([1, 12345], "[1,12345]"),
    ],
)
def test_citation_text(numbers: list[int], expected: str) -> None:
    assert _citation_text(numbers) == expected


def test_citation_text_concatenation() -> None:
    assert "Answer" + _citation_text([]) == "Answer[]"
    assert "Answer" + _citation_text([1, 2, 3]) == "Answer[1,2,3]"
    assert "Answer" + _citation_text([1, 12345]) == "Answer[1,12345]"


def test_citation_marker_multibyte_japanese() -> None:
    # end_index=9 bytes = 3 characters of 3 bytes each ("日本語").
    text, _ = format_grounded_response(
        _response(
            [_part("日本語のテキスト")],
            [_web("J", "https://j.example")],
            [_support(0, 9, [0])],
        )
    )
    assert text == "日本語[1]のテキスト\n\nSources:\n[1] J (https://j.example)"


def test_citation_marker_japanese_ascii_multipart() -> None:
    # part 0 "日本" = 6 bytes (base 0), part 1 "test" = 4 bytes (base 6).
    text, _ = format_grounded_response(
        _response(
            [_part("日本"), _part("test")],
            [_web("X", "https://x.example"), _web("Y", "https://y.example")],
            [_support(0, 6, [0]), _support(1, 4, [1])],
        )
    )
    assert text == (
        "日本[1]test[2]\n\nSources:\n"
        "[1] X (https://x.example)\n"
        "[2] Y (https://y.example)"
    )


def test_citation_marker_emoji_boundary() -> None:
    # end_index=4 lands on the 4-byte boundary of the emoji, before "ok".
    text, _ = format_grounded_response(
        _response(
            [_part("\U0001f600ok")],
            [_web("E", "https://e.example")],
            [_support(0, 4, [0])],
        )
    )
    assert text == "\U0001f600[1]ok\n\nSources:\n[1] E (https://e.example)"


def test_end_index_equals_byte_length_accepted() -> None:
    # "café" = 5 bytes (é is 2 bytes); end_index=5 == byte length is accepted.
    text, _ = format_grounded_response(
        _response(
            [_part("café")],
            [_web("C", "https://c.example")],
            [_support(0, 5, [0])],
        )
    )
    assert text == "café[1]\n\nSources:\n[1] C (https://c.example)"


def test_end_index_beyond_byte_length_skipped() -> None:
    # "café" = 5 bytes; end_index=6 exceeds it, so the support is skipped and no
    # marker is inserted, while the source itself is still listed.
    text, _ = format_grounded_response(
        _response(
            [_part("café")],
            [_web("C", "https://c.example")],
            [_support(0, 6, [0])],
        )
    )
    assert text == "café\n\nSources:\n[1] C (https://c.example)"


def test_thought_part_excluded_and_offsets_aligned() -> None:
    text, _ = format_grounded_response(
        _response(
            [_part("Alpha "), _part("THOUGHT", thought=True), _part("Beta")],
            [_web("X", "https://x.example"), _web("Y", "https://y.example")],
            [_support(0, 6, [0]), _support(2, 4, [1])],
        )
    )
    assert "THOUGHT" not in text
    assert text == (
        "Alpha [1]Beta[2]\n\nSources:\n"
        "[1] X (https://x.example)\n"
        "[2] Y (https://y.example)"
    )


def test_empty_source_skipped_and_indices_preserved() -> None:
    text, sources = format_grounded_response(
        _response(
            [_part("hello")],
            [
                _web("A", "https://a.example"),
                types.GroundingChunk(),
                _web("C", "https://c.example"),
            ],
        )
    )
    assert [source.index for source in sources] == [1, 3]
    assert text == (
        "hello\n\nSources:\n[1] A (https://a.example)\n[3] C (https://c.example)"
    )


def test_no_grounding_metadata_returns_plain_text() -> None:
    text, sources = format_grounded_response(
        _response([_part("hello")], metadata=False)
    )
    assert text == "hello"
    assert sources == ()


def test_supports_without_usable_sources_no_insertion() -> None:
    text, sources = format_grounded_response(
        _response(
            [_part("hello")],
            [types.GroundingChunk()],
            [_support(0, 5, [0])],
        )
    )
    assert text == "hello"
    assert sources == ()


def test_to_structured_omits_empty_fields() -> None:
    out = GoogleSearchOutput(
        query="q",
        text="t",
        sources=(
            GoogleSearchSource(index=1, title="Title", uri=""),
            GoogleSearchSource(index=2, title="", uri="https://u.example"),
        ),
    )
    assert out.to_structured() == {
        "query": "q",
        "text": "t",
        "sources": [
            {"index": 1, "title": "Title"},
            {"index": 2, "uri": "https://u.example"},
        ],
    }


def test_to_structured_omits_empty_sources() -> None:
    out = GoogleSearchOutput(query="q", text="t")
    assert out.to_structured() == {"query": "q", "text": "t"}


def _golden_output_schema() -> dict[str, Any]:
    path = Path(__file__).parent / "golden" / "tools_list.json"
    data = orjson.loads(path.read_text(encoding="utf-8"))
    tool: dict[str, Any] = data["result"]["tools"][0]
    return tool["outputSchema"]


def test_to_structured_validates_against_golden_schema() -> None:
    schema = _golden_output_schema()

    populated = GoogleSearchOutput(
        query="q",
        text="t",
        sources=(
            GoogleSearchSource(index=1, title="Title", uri="https://u.example"),
            GoogleSearchSource(index=2, title="OnlyTitle"),
        ),
    )
    jsonschema.validate(populated.to_structured(), schema)

    empty = GoogleSearchOutput(query="q", text="t")
    jsonschema.validate(empty.to_structured(), schema)


@pytest.mark.anyio
async def test_stub_generator_is_usable_content_generator() -> None:
    canned = _response([_part("Hi")])
    generator: ContentGenerator = StubGenerator(resp=canned)
    resp = await generator.generate_content(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[types.Part(text="q")])],
        config=None,
    )
    assert resp is canned
