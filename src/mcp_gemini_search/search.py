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

"""Google Search grounding service backed by the Gemini API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google.genai import types

from mcp_gemini_search import _markdown


@dataclass(frozen=True, slots=True)
class GoogleSearchSource:
    """A single source referenced by a grounded response."""

    index: int
    title: str = ""
    uri: str = ""


@dataclass(frozen=True, slots=True)
class GoogleSearchOutput:
    """The result of a grounded Google Search query."""

    query: str
    text: str
    sources: tuple[GoogleSearchSource, ...] = ()

    def to_structured(self) -> dict[str, Any]:
        """Return the MCP structured-content dict with Go ``omitempty`` semantics.

        The ``query`` and ``text`` keys are always present. Each source always
        carries ``index``; ``title`` and ``uri`` are included only when
        non-empty. The ``sources`` key is omitted entirely when there are no
        sources.
        """
        structured: dict[str, Any] = {"query": self.query, "text": self.text}
        if self.sources:
            source_list: list[dict[str, Any]] = []
            for source in self.sources:
                entry: dict[str, Any] = {"index": source.index}
                if source.title:
                    entry["title"] = source.title
                if source.uri:
                    entry["uri"] = source.uri
                source_list.append(entry)
            structured["sources"] = source_list
        return structured


class ContentGenerator(Protocol):
    """Structural type for the async Gemini content generator.

    Structurally satisfied by ``google.genai`` ``client.aio.models``.
    """

    async def generate_content(
        self,
        *,
        model: str,
        contents: types.ContentListUnion,
        config: types.GenerateContentConfig | None = None,
    ) -> types.GenerateContentResponse:
        """Generate content for the given model, contents, and config."""
        ...


class GoogleSearchService:
    """Runs Google-Search-grounded Gemini queries and formats the results."""

    def __init__(self, model: str, generator: ContentGenerator | None) -> None:
        """Store the model name and the injected content generator."""
        self._model = model
        self._generator = generator

    @property
    def model(self) -> str:
        """Return the Gemini model identifier used for generation."""
        return self._model

    async def search(self, query: str) -> GoogleSearchOutput:
        """Run a grounded Google Search for ``query`` and return the output.

        Raises:
            RuntimeError: If the service is not configured, if the backend call
                fails, or if the response cannot be formatted.
            ValueError: If ``query`` is empty or whitespace only.
        """
        if self._generator is None:
            raise RuntimeError("google search service is not configured")
        if not query.strip():
            raise ValueError("search query cannot be empty")

        try:
            resp = await self._generator.generate_content(
                model=self._model,
                contents=[types.Content(role="user", parts=[types.Part(text=query)])],
                config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())]),
            )
        except Exception as e:
            raise RuntimeError(f"google search failed: {e}") from e

        try:
            text, sources = format_grounded_response(resp)
        except Exception as e:
            raise RuntimeError(f"google search failed: {e}") from e

        return GoogleSearchOutput(query=query, text=text, sources=sources)


def format_grounded_response(
    resp: types.GenerateContentResponse | None,
) -> tuple[str, tuple[GoogleSearchSource, ...]]:
    """Format a Gemini response into clean Markdown text and its cited sources.

    Citation markers are inserted at UTF-8 byte offsets: Gemini's
    ``Segment.end_index`` is a byte offset into the UTF-8 encoding of the part
    text, so all offset arithmetic runs on ``str.encode("utf-8")`` and the cited
    text is assembled from byte slices and decoded exactly once.

    Sources are numbered compactly in grounding-chunk order (chunks without a
    title and URI are skipped), inline citations render as ``[[n]](uri)``
    Markdown links pointing at the cited source, and a trailing ``## Sources``
    section lists every source as an ordered list whose labels match the
    inline citation numbers. The whole document is normalized with mdformat.

    Raises:
        RuntimeError: If ``resp`` is ``None`` or contains no usable text.
    """
    if resp is None:
        raise RuntimeError("no response from Gemini model")

    text = _candidate_text(resp)
    if not text.strip():
        raise RuntimeError("no response from Gemini model")

    candidates = resp.candidates
    if not candidates or candidates[0] is None:
        return _markdown.format_document(text), ()

    candidate = candidates[0]
    metadata = candidate.grounding_metadata
    if metadata is None:
        return _markdown.format_document(text), ()

    sources: list[GoogleSearchSource] = []
    number_by_chunk: dict[int, int] = {}
    for idx, chunk in enumerate(metadata.grounding_chunks or []):
        title, uri = _grounding_source(chunk)
        if not title and not uri:
            continue
        number_by_chunk[idx] = len(sources) + 1
        sources.append(GoogleSearchSource(index=len(sources) + 1, title=title, uri=uri))

    formatted = text
    content = candidate.content
    parts = content.parts if content is not None else None
    supports = metadata.grounding_supports or []
    if parts and supports and sources:
        uri_by_number = {source.index: source.uri for source in sources}
        formatted = _insert_citations(text, parts, supports, number_by_chunk, uri_by_number)

    # Normalize the body on its own first: mdformat closes any dangling code
    # fence, so the appended Sources section cannot be swallowed by one.
    document = _markdown.format_document(formatted)
    if sources:
        document = _markdown.format_document(f"{document}\n\n## Sources\n\n{_render_source_list(sources)}")
    return document, tuple(sources)


def _render_source_list(sources: Sequence[GoogleSearchSource]) -> str:
    """Render sources as an ordered Markdown list whose labels match citations.

    URIs with a non-allowlisted scheme are rendered as literal text, never as
    link destinations.
    """
    lines: list[str] = []
    for source in sources:
        safe = _markdown.is_safe_uri(source.uri)
        if source.title and source.uri and safe:
            entry = _markdown.link(source.title, source.uri)
        elif source.uri and safe:
            entry = _markdown.autolink_or_link(source.uri)
        elif source.title and source.uri:
            entry = f"{_markdown.escape_link_text(source.title)} ({_markdown.escape_link_text(source.uri)})"
        elif source.uri:
            entry = _markdown.escape_link_text(source.uri)
        else:
            entry = _markdown.escape_link_text(source.title)
        lines.append(f"{source.index}. {entry}")
    return "\n".join(lines)


def _candidate_text(resp: types.GenerateContentResponse) -> str:
    """Join the visible text parts of the first candidate.

    Uses the same part filter as citation insertion (non-thought, non-empty
    text) so the joined text and byte offsets stay aligned.
    """
    candidates = resp.candidates
    if not candidates:
        return ""
    candidate = candidates[0]
    if candidate is None or candidate.content is None or candidate.content.parts is None:
        return ""
    pieces: list[str] = []
    for part in candidate.content.parts:
        if part.thought or not part.text:
            continue
        pieces.append(part.text)
    return "".join(pieces)


def _insert_citations(
    text: str,
    parts: list[types.Part],
    supports: list[types.GroundingSupport],
    number_by_chunk: Mapping[int, int],
    uri_by_number: Mapping[int, str],
) -> str:
    """Insert Markdown citation links into ``text`` at UTF-8 byte offsets.

    ``number_by_chunk`` maps grounding-chunk indices to compact source numbers;
    supports citing a chunk without a usable source insert no marker.
    """
    text_bytes = text.encode("utf-8")
    part_count = len(parts)
    part_offsets = [0] * part_count
    part_has_text = [False] * part_count
    part_lengths = [0] * part_count
    total_length = 0
    for idx, part in enumerate(parts):
        if part.thought or not part.text:
            continue
        part_has_text[idx] = True
        part_offsets[idx] = total_length
        part_lengths[idx] = len(part.text.encode("utf-8"))
        total_length += part_lengths[idx]

    insertions: list[tuple[int, int]] = []
    for support in supports:
        segment = support.segment
        indices = support.grounding_chunk_indices
        if segment is None or not indices:
            continue

        part_index = segment.part_index or 0
        if part_index < 0 or part_index >= part_count or not part_has_text[part_index]:
            continue

        base_offset = part_offsets[part_index]
        end_index = segment.end_index or 0
        if end_index < 0 or end_index > part_lengths[part_index]:
            continue

        global_offset = base_offset + end_index
        for chunk_index in indices:
            number = number_by_chunk.get(chunk_index)
            if number is None:
                continue
            insertions.append((global_offset, number))

    if not insertions:
        return text

    insertions.sort()
    total_bytes = len(text_bytes)
    result = bytearray()
    last_offset = 0
    i = 0
    n = len(insertions)
    while i < n:
        offset = insertions[i][0]
        if offset < last_offset or offset > total_bytes:
            i += 1
            continue

        result += text_bytes[last_offset:offset]
        numbers: list[int] = []
        j = i
        while j < n and insertions[j][0] == offset:
            number = insertions[j][1]
            if not (numbers and numbers[-1] == number):
                numbers.append(number)
            j += 1

        result += _citation_text(numbers, uri_by_number).encode("utf-8")
        last_offset = offset
        i = j

    result += text_bytes[last_offset:]
    return result.decode("utf-8")


def _grounding_source(chunk: types.GroundingChunk | None) -> tuple[str, str]:
    """Extract the ``(title, uri)`` pair from a grounding chunk.

    Mirrors Go's discriminated union: web, then maps, then retrieved context,
    then image (whose URI prefers ``source_uri`` over ``image_uri``).
    """
    if chunk is None:
        return "", ""
    if chunk.web is not None:
        return chunk.web.title or "", chunk.web.uri or ""
    if chunk.maps is not None:
        return chunk.maps.title or "", chunk.maps.uri or ""
    if chunk.retrieved_context is not None:
        return chunk.retrieved_context.title or "", chunk.retrieved_context.uri or ""
    if chunk.image is not None:
        return (
            chunk.image.title or "",
            chunk.image.source_uri or chunk.image.image_uri or "",
        )
    return "", ""


def _citation_text(numbers: list[int], uri_by_number: Mapping[int, str]) -> str:
    """Render citation numbers as adjacent ``[[n]](uri)`` Markdown links.

    Numbers without a usable URI (missing or with a non-allowlisted scheme)
    render as escaped ``\\[n\\]`` markers so surrounding body text can never
    capture them into a link; an empty list renders as an empty string.
    """
    pieces: list[str] = []
    for number in numbers:
        uri = uri_by_number.get(number, "")
        if uri and _markdown.is_safe_uri(uri):
            pieces.append(f"[[{number}]]({_markdown.format_destination(uri)})")
        else:
            pieces.append(f"\\[{number}\\]")
    return "".join(pieces)
