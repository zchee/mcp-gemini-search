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

"""Tests for the Markdown rendering helpers."""

import pytest

from mcp_gemini_search._markdown import (
    autolink_or_link,
    escape_link_text,
    format_destination,
    format_document,
    is_safe_uri,
    link,
)


def test_format_document_normalizes_gfm() -> None:
    """Bullets, tables, and spacing normalize; the trailing newline is stripped."""
    src = "# Heading\n\n*  item one\n*  item two\n\n\n\n| a | b |\n|---|---|\n| 1 | 2 |"
    assert format_document(src) == (
        "# Heading\n\n- item one\n- item two\n\n| a   | b   |\n| --- | --- |\n| 1   | 2   |"
    )


def test_format_document_keeps_consecutive_numbering() -> None:
    """Ordered lists keep consecutive 1. 2. 3. numbering instead of all-ones."""
    assert format_document("1. one\n2. two\n3. three") == "1. one\n2. two\n3. three"


def test_format_document_escapes_checkbox_lookalike_list_items() -> None:
    """List items starting with an [x]-style link escape to avoid GFM checkboxes."""
    assert format_document("1. [X](https://x.example)") == "1. \\[X\\](https://x.example)"


def test_format_document_preserves_citation_links_and_markers() -> None:
    """Citation-style links and bare markers survive normalization unescaped."""
    src = "fact[[1]](https://a.example)[[2]](https://b.example) and plain [3]."
    assert format_document(src) == src


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain", "plain"),
        ("a[b]c", "a\\[b\\]c"),
        ("back\\slash", "back\\\\slash"),
        ("a<em>b</em>", "a\\<em>b\\</em>"),
        ("a\nb", "a b"),
        ("a\r\nb", "a  b"),
    ],
)
def test_escape_link_text(text: str, expected: str) -> None:
    """Brackets, backslashes, ``<``, and newlines neutralize to literal text."""
    assert escape_link_text(text) == expected


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://x.example/path", True),
        ("HTTP://x.example", True),
        ("mailto:a@b.example", True),
        ("x.example/no-scheme", True),
        ("//x.example/relative", True),
        ("javascript:alert(1)", False),
        ("DATA:text/html,x", False),
        ("vbscript:x", False),
        (" javascript:alert(1)", False),
        ("\tdata:text/html,x", False),
    ],
)
def test_is_safe_uri(uri: str, expected: bool) -> None:
    """Only schemeless URIs and allowlisted schemes count as safe."""
    assert is_safe_uri(uri) is expected


def test_format_document_falls_back_on_pathological_input() -> None:
    """Deeply nested emphasis falls back to the raw text instead of raising."""
    src = "*" * 2000 + "a" + "*" * 2000
    assert format_document(src) == src


def test_format_document_closes_dangling_code_fence() -> None:
    """An unterminated code fence gains a closing fence on its own line."""
    assert format_document("intro\n\n```py\ncode = 1") == "intro\n\n```py\ncode = 1\n```"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://x.example/path", "https://x.example/path"),
        ("https://x.example/a b", "https://x.example/a%20b"),
        ("https://x.example/<v>", "https://x.example/%3Cv%3E"),
        ("https://x.example/a\nb", "https://x.example/a%0Ab"),
        ("https://x.example/a\\b", "https://x.example/a%5Cb"),
        ("https://x.example/a(b)c", "<https://x.example/a(b)c>"),
        ("https://x.example/a(b", "<https://x.example/a(b>"),
        ("https://x.example/日本語", "https://x.example/日本語"),
    ],
)
def test_format_destination(uri: str, expected: str) -> None:
    """Unsafe characters percent-encode and parenthesized URIs wrap in angles."""
    assert format_destination(uri) == expected


def test_link_escapes_text_and_destination() -> None:
    """link() escapes both sides so the result always parses as one link."""
    assert link("a[b]c", "https://x.example") == "[a\\[b\\]c](https://x.example)"
    assert link("T", "https://x.example/a b") == "[T](https://x.example/a%20b)"
    assert link("T", "https://x.example/a(b") == "[T](<https://x.example/a(b>)"


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("https://x.example/path", "<https://x.example/path>"),
        ("x.example/no-scheme", "[x.example/no-scheme](x.example/no-scheme)"),
        ("https://x.example/a b", "[https://x.example/a b](https://x.example/a%20b)"),
    ],
)
def test_autolink_or_link(uri: str, expected: str) -> None:
    """URIs render as autolinks when CommonMark allows, else as inline links."""
    assert autolink_or_link(uri) == expected
