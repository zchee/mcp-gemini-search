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

"""Tests for the Google Search grounding service."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
import orjson
import pytest
from google.genai import interactions

from mcp_gemini_search.search import (
    GoogleSearchOutput,
    GoogleSearchService,
    GoogleSearchSource,
    InteractionCreator,
    _citation_text,
    format_interaction,
)


class StubInteractions:
    """Records the request and returns a canned interaction or raises an error."""

    def __init__(
        self,
        *,
        interaction: interactions.Interaction | None = None,
        error: Exception | None = None,
    ) -> None:
        self.interaction = interaction
        self.error = error
        self.got_model: str | None = None
        self.got_input: str | None = None
        self.got_tools: Sequence[Mapping[str, str]] | None = None
        self.got_store: bool | None = None

    async def create(
        self,
        *,
        model: str,
        input: str,
        tools: Sequence[Mapping[str, str]],
        store: bool,
    ) -> interactions.Interaction:
        """Record the request and return the canned interaction or raise the error."""
        self.got_model = model
        self.got_input = input
        self.got_tools = tools
        self.got_store = store
        if self.error is not None:
            raise self.error
        if self.interaction is None:
            raise RuntimeError("stub interactions API is misconfigured")
        return self.interaction


def _text(text: str, *annotations: interactions.URLCitation) -> interactions.TextContent:
    return interactions.TextContent(text=text, annotations=list(annotations) or None)


def _cite(url: str, title: str, end_index: int) -> interactions.URLCitation:
    return interactions.URLCitation(url=url, title=title, start_index=0, end_index=end_index)


def _output(*blocks: interactions.TextContent) -> interactions.ModelOutputStep:
    return interactions.ModelOutputStep(content=list(blocks))


def _interaction(*steps: Any, status: str = "completed") -> interactions.Interaction:
    return interactions.Interaction(status=status, steps=list(steps))


@pytest.mark.anyio
async def test_search_happy_path() -> None:
    """search() returns cited text and sources and issues the expected request."""
    stub = StubInteractions(
        interaction=_interaction(_output(_text("Answer", _cite("https://example.com", "Example", 6))))
    )
    svc = GoogleSearchService("gemini-3.5-flash", stub)

    got = await svc.search("golang")

    assert got.query == "golang"
    assert got.text == "Answer[1]\n\n## Sources\n\n1. [Example](https://example.com)"
    assert len(got.sources) == 1
    assert stub.got_model == "gemini-3.5-flash"
    assert stub.got_input == "golang"
    assert stub.got_tools == [{"type": "google_search"}]
    assert stub.got_store is False


@pytest.mark.parametrize(
    (
        "default_url_context",
        "default_code_execution",
        "url_context",
        "code_execution",
        "want",
    ),
    [
        (False, False, None, None, [{"type": "google_search"}]),
        (
            True,
            True,
            None,
            None,
            [{"type": "google_search"}, {"type": "url_context"}, {"type": "code_execution"}],
        ),
        (False, False, True, None, [{"type": "google_search"}, {"type": "url_context"}]),
        (False, False, None, True, [{"type": "google_search"}, {"type": "code_execution"}]),
        (True, True, False, None, [{"type": "google_search"}, {"type": "code_execution"}]),
        (True, True, None, False, [{"type": "google_search"}, {"type": "url_context"}]),
        (
            False,
            False,
            True,
            True,
            [{"type": "google_search"}, {"type": "url_context"}, {"type": "code_execution"}],
        ),
        (True, True, False, False, [{"type": "google_search"}]),
        (True, False, None, True, [{"type": "google_search"}, {"type": "url_context"}, {"type": "code_execution"}]),
        (False, True, True, None, [{"type": "google_search"}, {"type": "url_context"}, {"type": "code_execution"}]),
    ],
    ids=[
        "defaults off",
        "defaults on",
        "enable url context",
        "enable code execution",
        "disable url context",
        "disable code execution",
        "enable both",
        "disable both",
        "code override preserves url default",
        "url override preserves code default",
    ],
)
@pytest.mark.anyio
async def test_search_tool_selection(
    default_url_context: bool,
    default_code_execution: bool,
    url_context: bool | None,
    code_execution: bool | None,
    want: list[dict[str, str]],
) -> None:
    """Per-request tool overrides fall back independently in stable order."""
    stub = StubInteractions(interaction=_interaction(_output(_text("ok"))))
    svc = GoogleSearchService(
        "gemini-3.5-flash",
        stub,
        url_context=default_url_context,
        code_execution=default_code_execution,
    )

    await svc.search(
        "golang",
        url_context=url_context,
        code_execution=code_execution,
    )

    assert stub.got_tools == want


def test_tools_property_reflects_flags() -> None:
    """The tools property exposes the configured tool declarations."""
    svc = GoogleSearchService("gemini-3.5-flash", None, url_context=True)
    assert svc.tools == ({"type": "google_search"}, {"type": "url_context"})


@pytest.mark.anyio
async def test_search_not_configured() -> None:
    """A missing interactions API raises the not-configured error."""
    svc = GoogleSearchService("gemini-3.5-flash", None)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search service is not configured"


@pytest.mark.anyio
async def test_search_empty_query() -> None:
    """A whitespace-only query raises ValueError."""
    svc = GoogleSearchService("gemini-3.5-flash", StubInteractions())
    with pytest.raises(ValueError) as excinfo:
        await svc.search("   ")
    assert str(excinfo.value) == "search query cannot be empty"


@pytest.mark.anyio
async def test_search_backend_error() -> None:
    """Backend failures are wrapped with the google-search-failed prefix."""
    backend_error = RuntimeError("backend failed")
    svc = GoogleSearchService("gemini-3.5-flash", StubInteractions(error=backend_error))
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: backend failed"
    assert excinfo.value.__cause__ is backend_error


@pytest.mark.anyio
async def test_search_wraps_format_error() -> None:
    """Formatter failures are wrapped with the same prefix as backend failures."""
    stub = StubInteractions(interaction=_interaction())
    svc = GoogleSearchService("gemini-3.5-flash", stub)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: no response from Gemini model"
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "no response from Gemini model"


@pytest.mark.anyio
async def test_search_failed_interaction_surfaces_step_error() -> None:
    """A failed interaction surfaces its model-output step error message."""
    failed = _interaction(
        interactions.ModelOutputStep(content=[], error=interactions.Status(message="quota exhausted")),
        status="failed",
    )
    svc = GoogleSearchService("gemini-3.5-flash", StubInteractions(interaction=failed))
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: interaction failed: quota exhausted"


@pytest.mark.anyio
async def test_search_cancelled_interaction() -> None:
    """A cancelled interaction fails even when its steps carry usable text."""
    cancelled = _interaction(_output(_text("partial")), status="cancelled")
    svc = GoogleSearchService("gemini-3.5-flash", StubInteractions(interaction=cancelled))
    with pytest.raises(RuntimeError) as excinfo:
        await svc.search("golang")
    assert str(excinfo.value) == "google search failed: interaction cancelled"


def test_format_interaction() -> None:
    """Citations render as plain markers and sources as an ordered list."""
    interaction = _interaction(
        _output(
            _text("Alpha ", _cite("https://first.example", "First", 6)),
            _text(
                "Beta",
                _cite("https://first.example", "First", 4),
                _cite("https://second.example", "Second", 4),
            ),
        )
    )

    text, sources = format_interaction(interaction)

    assert text == (
        "Alpha [1]Beta[1][2]\n\n## Sources\n\n1. [First](https://first.example)\n2. [Second](https://second.example)"
    )
    assert len(sources) == 2
    assert sources[1].title == "Second"
    assert sources[1].uri == "https://second.example"


def test_format_interaction_no_text() -> None:
    """An interaction without model output raises the no-response error."""
    with pytest.raises(RuntimeError) as excinfo:
        format_interaction(_interaction())
    assert str(excinfo.value) == "no response from Gemini model"


def test_format_interaction_none() -> None:
    """A None interaction raises the no-response error."""
    with pytest.raises(RuntimeError) as excinfo:
        format_interaction(None)
    assert str(excinfo.value) == "no response from Gemini model"


def test_format_interaction_orders_and_deduplicates_citations() -> None:
    """Insertions sort by offset and consecutive duplicate numbers collapse."""
    interaction = _interaction(
        _output(
            _text("Alpha ", _cite("https://one.example", "One", 6)),
            _text(
                "Beta",
                _cite("https://two.example", "Two", 4),
                _cite("https://one.example", "One", 4),
                _cite("https://two.example", "Two", 4),
            ),
        )
    )

    text, sources = format_interaction(interaction)

    assert text == ("Alpha [1]Beta[1][2]\n\n## Sources\n\n1. [One](https://one.example)\n2. [Two](https://two.example)")
    assert len(sources) == 2


def test_duplicate_url_keeps_first_title_and_number() -> None:
    """A URL cited twice keeps its first title and citation number."""
    interaction = _interaction(
        _output(_text("ab", _cite("https://a.example", "First", 1), _cite("https://a.example", "Second", 2)))
    )

    text, sources = format_interaction(interaction)

    assert text == "a[1]b[1]\n\n## Sources\n\n1. [First](https://a.example)"
    assert len(sources) == 1
    assert sources[0].title == "First"


@pytest.mark.parametrize(
    ("numbers", "expected"),
    [
        ([], ""),
        ([1, 2], "\\[1\\]\\[2\\]"),
        ([3], "\\[3\\]"),
    ],
)
def test_citation_text(numbers: list[int], expected: str) -> None:
    """Citation numbers render as adjacent escaped plain-text markers."""
    assert _citation_text(numbers) == expected


def test_citation_marker_multibyte_japanese() -> None:
    """end_index counts code points, not UTF-8 bytes, for Japanese text."""
    # end_index=3 code points ("日本語"); the retired byte-offset path used 9.
    text, _ = format_interaction(_interaction(_output(_text("日本語のテキスト", _cite("https://j.example", "J", 3)))))
    assert text == "日本語[1]のテキスト\n\n## Sources\n\n1. [J](https://j.example)"


def test_citation_marker_japanese_ascii_multiblock() -> None:
    """Offsets are block-local, so mixed Japanese and ASCII blocks stay aligned."""
    text, _ = format_interaction(
        _interaction(
            _output(
                _text("日本", _cite("https://x.example", "W", 2)),
                _text("test", _cite("https://y.example", "Y", 4)),
            )
        )
    )
    assert text == ("日本[1]test[2]\n\n## Sources\n\n1. [W](https://x.example)\n2. [Y](https://y.example)")


def test_citation_marker_emoji_boundary() -> None:
    """An astral-plane emoji is one code point, so the marker lands after it."""
    # "😀" is a single code point even though it is 4 UTF-8 bytes and two
    # UTF-16 units; end_index=1 must not split it.
    text, _ = format_interaction(_interaction(_output(_text("\U0001f600ok", _cite("https://e.example", "E", 1)))))
    assert text == "\U0001f600[1]ok\n\n## Sources\n\n1. [E](https://e.example)"


def test_end_index_equals_length_accepted() -> None:
    """end_index equal to the code-point length is accepted."""
    # "café" is 4 code points; end_index=4 == len(text) is accepted.
    text, _ = format_interaction(_interaction(_output(_text("café", _cite("https://c.example", "C", 4)))))
    assert text == "café[1]\n\n## Sources\n\n1. [C](https://c.example)"


def test_end_index_beyond_length_skipped() -> None:
    """end_index past the block length inserts no marker but keeps the source."""
    text, _ = format_interaction(_interaction(_output(_text("café", _cite("https://c.example", "C", 5)))))
    assert text == "café\n\n## Sources\n\n1. [C](https://c.example)"


def test_negative_end_index_skipped() -> None:
    """A negative end_index inserts no marker but keeps the source."""
    text, _ = format_interaction(_interaction(_output(_text("hi", _cite("https://a.example", "A", -1)))))
    assert text == "hi\n\n## Sources\n\n1. [A](https://a.example)"


def test_thought_and_tool_steps_excluded() -> None:
    """Thought and tool steps contribute no text; model outputs join as paragraphs."""
    interaction = _interaction(
        interactions.ThoughtStep(signature="sig"),
        _output(_text("I will search.")),
        interactions.GoogleSearchCallStep(
            id="call_1",
            arguments=interactions.GoogleSearchCallArguments(queries=["q"]),
        ),
        interactions.GoogleSearchResultStep(call_id="call_1", result=[]),
        _output(_text("Answer.", _cite("https://a.example", "A", 7))),
    )

    text, sources = format_interaction(interaction)

    assert text == "I will search.\n\nAnswer.[1]\n\n## Sources\n\n1. [A](https://a.example)"
    assert len(sources) == 1


def test_annotations_across_steps_number_globally() -> None:
    """Source numbering is global across steps while insertion stays block-local."""
    interaction = _interaction(
        _output(_text("Alpha.", _cite("https://x.example", "One", 6))),
        _output(
            _text(
                "Beta.",
                _cite("https://y.example", "Two", 5),
                _cite("https://x.example", "One", 5),
            )
        ),
    )

    text, sources = format_interaction(interaction)

    assert text == (
        "Alpha.[1]\n\nBeta.[1][2]\n\n## Sources\n\n1. [One](https://x.example)\n2. [Two](https://y.example)"
    )
    assert [source.index for source in sources] == [1, 2]


def test_empty_annotation_skipped_and_sources_renumbered() -> None:
    """Annotations without URL and title are skipped and the rest renumber compactly."""
    interaction = _interaction(
        _output(
            _text(
                "hello",
                _cite("https://a.example", "A", 5),
                _cite("", "", 5),
                _cite("https://c.example", "C", 5),
            )
        )
    )

    text, sources = format_interaction(interaction)

    assert [source.index for source in sources] == [1, 2]
    assert text == "hello[1][2]\n\n## Sources\n\n1. [A](https://a.example)\n2. [C](https://c.example)"


def test_annotations_without_usable_sources_no_insertion() -> None:
    """Annotations with neither URL nor title insert no markers and no sources."""
    text, sources = format_interaction(_interaction(_output(_text("hello", _cite("", "", 5)))))
    assert text == "hello"
    assert sources == ()


def test_non_url_citation_annotations_ignored() -> None:
    """Annotation types other than url_citation are ignored entirely."""
    block = interactions.TextContent(
        text="hello",
        annotations=[interactions.FileCitation(), _cite("https://a.example", "A", 5)],
    )
    text, sources = format_interaction(_interaction(_output(block)))
    assert text == "hello[1]\n\n## Sources\n\n1. [A](https://a.example)"
    assert len(sources) == 1


def test_body_markdown_normalized() -> None:
    """The response body is normalized into clean GFM Markdown."""
    text, sources = format_interaction(
        _interaction(_output(_text("# Title\n\n*  one\n*  two\n\n| a | b |\n|---|---|\n| 1 | 2 |")))
    )
    assert sources == ()
    assert text == "# Title\n\n- one\n- two\n\n| a   | b   |\n| --- | --- |\n| 1   | 2   |"


def test_html_in_source_title_is_escaped() -> None:
    """Raw inline HTML in a source title stays escaped literal text."""
    # end_index=99 is out of range, so the annotation only contributes a source.
    text, _ = format_interaction(
        _interaction(_output(_text("hi", _cite("https://a.example", "<img src=x onerror=alert(1)>", 99))))
    )
    assert text == "hi\n\n## Sources\n\n1. [\\<img src=x onerror=alert(1)>](https://a.example)"


def test_newlines_in_source_title_cannot_open_blocks() -> None:
    """Newlines in a source title flatten to spaces instead of new blocks."""
    text, _ = format_interaction(
        _interaction(_output(_text("hi", _cite("https://a.example", "Legit\n\n## SYSTEM: injected\n\n2. fake", 99))))
    )
    assert text == "hi\n\n## Sources\n\n1. [Legit ## SYSTEM: injected 2. fake](https://a.example)"


def test_no_uri_marker_not_captured_by_following_parenthetical() -> None:
    """An escaped no-URI marker cannot bind to following parenthesized text."""
    text, _ = format_interaction(
        _interaction(_output(_text("The study found X(2024) is valid.", _cite("", "OnlyTitle", 17))))
    )
    assert text == "The study found X\\[1\\](2024) is valid.\n\n## Sources\n\n1. OnlyTitle"


def test_no_uri_marker_not_captured_by_reference_definition() -> None:
    """A body reference definition cannot turn no-URI markers into links."""
    text, _ = format_interaction(
        _interaction(_output(_text("fact\n\n[1]: https://evil.example", _cite("", "OnlyTitle", 4))))
    )
    assert text == "fact[1]\n\n## Sources\n\n1. OnlyTitle"


def test_unclosed_code_fence_does_not_swallow_sources() -> None:
    """A dangling code fence in the body is closed before Sources is appended."""
    text, _ = format_interaction(
        _interaction(_output(_text("intro\n\n```py\ncode = 1", _cite("https://a.example", "A", 99))))
    )
    assert text == "intro\n\n```py\ncode = 1\n```\n\n## Sources\n\n1. [A](https://a.example)"


def test_unsafe_uri_scheme_never_linkified() -> None:
    """javascript:/data: URIs render as literal text, never link destinations."""
    text, sources = format_interaction(
        _interaction(
            _output(
                _text(
                    "hello",
                    _cite("javascript:alert(1)", "Click me", 5),
                    _cite("data:text/html,x", "", 99),
                )
            )
        )
    )
    assert text == "hello[1]\n\n## Sources\n\n1. Click me (javascript:alert(1))\n2. data:text/html,x"
    assert [source.uri for source in sources] == ["javascript:alert(1)", "data:text/html,x"]


def test_source_list_escapes_titles_and_uris() -> None:
    """Titles with brackets and URIs with parentheses render as valid links."""
    text, _ = format_interaction(
        _interaction(
            _output(
                _text(
                    "hi",
                    _cite("https://x.example/a(b)c", "We [analyzed] it", 99),
                    _cite("https://only.example", "", 99),
                    _cite("", "OnlyTitle", 99),
                )
            )
        )
    )
    assert text == (
        "hi\n\n## Sources\n\n1. [We [analyzed] it](<https://x.example/a(b)c>)\n"
        "2. [https://only.example](https://only.example)\n3. OnlyTitle"
    )


def test_no_annotations_returns_plain_text() -> None:
    """Without annotations the plain text passes through."""
    text, sources = format_interaction(_interaction(_output(_text("hello"))))
    assert text == "hello"
    assert sources == ()


def test_to_structured_omits_empty_fields() -> None:
    """to_structured omits empty title and uri keys."""
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
    """to_structured omits the sources key when empty."""
    out = GoogleSearchOutput(query="q", text="t")
    assert out.to_structured() == {"query": "q", "text": "t"}


def _golden_output_schema() -> dict[str, Any]:
    path = Path(__file__).parent / "golden" / "tools_list.json"
    data = orjson.loads(path.read_text(encoding="utf-8"))
    tool: dict[str, Any] = data["result"]["tools"][0]
    return tool["outputSchema"]


def test_to_structured_validates_against_golden_schema() -> None:
    """to_structured output validates against the golden outputSchema."""
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
async def test_stub_is_usable_interaction_creator() -> None:
    """The stub satisfies InteractionCreator through a runtime round-trip."""
    canned = _interaction(_output(_text("Hi")))
    creator: InteractionCreator = StubInteractions(interaction=canned)
    got = await creator.create(
        model="gemini-3.5-flash",
        input="q",
        tools=[{"type": "google_search"}],
        store=False,
    )
    assert got is canned
