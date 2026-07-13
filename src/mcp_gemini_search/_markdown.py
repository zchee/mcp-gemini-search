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

"""Markdown rendering helpers for grounded search output."""

from __future__ import annotations

import re

import mdformat

from mcp_gemini_search._logging import logger

_MDFORMAT_OPTIONS: dict[str, bool] = {"number": True}
_MDFORMAT_EXTENSIONS = frozenset({"gfm"})

# ``<`` must be escaped so titles cannot smuggle raw inline HTML; newlines
# must be flattened so titles cannot open a new Markdown block.
_LINK_TEXT_ESCAPES = str.maketrans({"\\": "\\\\", "[": "\\[", "]": "\\]", "<": "\\<", "\n": " ", "\r": " "})

# CommonMark absolute-URI autolink: a 2-32 character scheme, a colon, then any
# characters other than whitespace, ``<``, and ``>``.
_AUTOLINK_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*")

_SCHEME_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_SAFE_SCHEMES = frozenset({"http", "https", "mailto"})


def format_document(text: str) -> str:
    """Normalize ``text`` into clean GitHub-flavored Markdown.

    Runs mdformat with the GFM extension (tables, strikethrough, autolinks)
    and consecutive ordered-list numbering, then strips the trailing newline
    mdformat guarantees so the result behaves like a plain text field.

    Normalization is cosmetic, so any mdformat or plugin failure — e.g.
    markdown-it's recursive-descent parser overflowing on thousands of nested
    emphasis markers — logs a warning and returns the raw text unformatted
    instead of failing the tool call.

    The input is fed to mdformat with a guaranteed trailing newline: without
    one, mdformat glues the closing fence of an unterminated code block onto
    its last content line, which un-closes the fence on the next parse.
    """
    try:
        return mdformat.text(f"{text}\n", options=_MDFORMAT_OPTIONS, extensions=_MDFORMAT_EXTENSIONS).rstrip("\n")
    except Exception as e:
        logger.warning("markdown normalization failed, returning unformatted text: %r", e)
        return text.rstrip("\n")


def link(text: str, uri: str) -> str:
    """Render an inline link, escaping ``text`` and ``uri`` so both always parse."""
    return f"[{escape_link_text(text)}]({format_destination(uri)})"


def autolink_or_link(uri: str) -> str:
    """Render ``uri`` as an autolink when CommonMark allows it, else as an inline link."""
    if _AUTOLINK_RE.fullmatch(uri):
        return f"<{uri}>"
    return link(uri, uri)


def escape_link_text(text: str) -> str:
    """Escape ``text`` so it stays literal: no links, inline HTML, or new blocks.

    Backslashes, square brackets, and ``<`` are backslash-escaped; newlines and
    carriage returns are flattened to spaces.
    """
    return text.translate(_LINK_TEXT_ESCAPES)


def is_safe_uri(uri: str) -> bool:
    """Report whether ``uri`` is schemeless or uses an allowlisted scheme.

    Grounding sources legitimately carry only web URIs; anything like
    ``javascript:`` or ``data:`` must never become a link destination in the
    rendered Markdown. Surrounding whitespace is stripped before scheme
    detection so padding cannot smuggle a scheme past the allowlist.
    """
    match = _SCHEME_RE.match(uri.strip())
    return match is None or match.group()[:-1].lower() in _SAFE_SCHEMES


def format_destination(uri: str) -> str:
    """Render ``uri`` as a link destination that always parses as one.

    Whitespace, angle brackets, and non-printable characters are
    percent-encoded; destinations containing parentheses are wrapped in angle
    brackets so unbalanced parentheses cannot terminate the link early.
    """
    encoded = "".join(_encode_destination_char(ch) for ch in uri)
    if "(" in encoded or ")" in encoded:
        return f"<{encoded}>"
    return encoded


def _encode_destination_char(ch: str) -> str:
    if ch in "<>\\" or ch.isspace() or not ch.isprintable():
        return "".join(f"%{byte:02X}" for byte in ch.encode("utf-8"))
    return ch
