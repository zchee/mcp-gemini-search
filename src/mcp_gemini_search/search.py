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

"""Google Search grounding service backed by the Gemini Interactions API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from google.genai import interactions

from mcp_gemini_search import _markdown


@dataclass(frozen=True, slots=True)
class GoogleSearchSource:
    """A single source referenced by a grounded response."""

    index: int
    title: str = ""
    uri: str = ""

    def to_structured(self) -> dict[str, Any]:
        """Return the source entry dict: ``index`` always; ``title`` and ``uri`` omit-empty."""
        entry: dict[str, Any] = {"index": self.index}
        if self.title:
            entry["title"] = self.title
        if self.uri:
            entry["uri"] = self.uri
        return entry


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
            structured["sources"] = [source.to_structured() for source in self.sources]
        return structured


class InteractionCreator(Protocol):
    """Structural type for the async Gemini interactions API.

    Structurally satisfied by ``google.genai`` ``client.aio.interactions``.
    """

    async def create(
        self,
        *,
        model: str,
        input: str,
        tools: Sequence[Mapping[str, str]],
        store: bool,
        service_tier: str = ...,
    ) -> Any:
        """Create an interaction for the given model, input, and tools.

        Declared as ``Any`` because the SDK method returns
        ``Interaction | AsyncStream[InteractionSSEEvent]``; this service never
        sets ``stream``, so the result is always an ``Interaction``.
        """
        ...


def _build_tools(url_context: bool, code_execution: bool) -> list[dict[str, str]]:
    """Return the interaction tool declarations for the enabled optional tools."""
    tools: list[dict[str, str]] = [{"type": "google_search"}]
    if url_context:
        tools.append({"type": "url_context"})
    if code_execution:
        tools.append({"type": "code_execution"})
    return tools


class GoogleSearchService:
    """Runs Google-Search-grounded Gemini interactions and formats the results."""

    def __init__(
        self,
        model: str,
        interactions: InteractionCreator | None,
        *,
        url_context: bool = False,
        code_execution: bool = False,
        service_tier: str = "",
    ) -> None:
        """Store the model name, the injected interactions API, the tool set, and the service tier."""
        self._model = model
        self._interactions = interactions
        self._url_context = url_context
        self._code_execution = code_execution
        self._service_tier = service_tier

    @property
    def model(self) -> str:
        """Return the Gemini model identifier used for generation."""
        return self._model

    @property
    def tools(self) -> tuple[dict[str, str], ...]:
        """Return the server-default tool declarations, before per-request overrides."""
        return tuple(_build_tools(self._url_context, self._code_execution))

    async def search(
        self,
        query: str,
        *,
        url_context: bool | None = None,
        code_execution: bool | None = None,
    ) -> GoogleSearchOutput:
        """Run a grounded Google Search for ``query`` and return the output.

        Every search is a stateless single-shot request (``store=False``): no
        follow-up turn ever references the interaction, so nothing needs the
        server-side history that storing would retain.

        Args:
            query: The search query to send to the model.
            url_context: Whether to let the model fetch URLs mentioned in the
                query. ``None`` uses the server-configured default; a boolean
                overrides that default for this request only.
            code_execution: Whether to let the model run Python for
                computational answers. ``None`` uses the server-configured
                default; a boolean overrides that default for this request
                only.

        Raises:
            RuntimeError: If the service is not configured, if the backend call
                fails, or if the interaction cannot be formatted.
            ValueError: If ``query`` is empty or whitespace only.
        """
        if self._interactions is None:
            raise RuntimeError("google search service is not configured")
        if not query.strip():
            raise ValueError("search query cannot be empty")

        tools = _build_tools(
            self._url_context if url_context is None else url_context,
            self._code_execution if code_execution is None else code_execution,
        )

        extra: dict[str, Any] = {}
        if self._service_tier:
            extra["service_tier"] = self._service_tier
        try:
            interaction = await self._interactions.create(
                model=self._model,
                input=query,
                tools=tools,
                store=False,
                **extra,
            )
        except Exception as e:
            raise RuntimeError(f"google search failed: {e}") from e

        try:
            text, sources = format_interaction(interaction)
        except Exception as e:
            raise RuntimeError(f"google search failed: {e}") from e

        return GoogleSearchOutput(query=query, text=text, sources=sources)


def format_interaction(
    interaction: interactions.Interaction | None,
) -> tuple[str, tuple[GoogleSearchSource, ...]]:
    """Format an interaction into clean Markdown text and its cited sources.

    Citation markers are inserted at code-point offsets: a ``url_citation``
    annotation's ``end_index`` indexes the Python string of its own text block
    (the documented Python usage slices ``str`` directly), so all insertion
    arithmetic runs on ``str`` and stays local to each block.

    Sources are numbered compactly in annotation encounter order and
    deduplicated by URL (annotations without a URL and title are skipped),
    inline citations render as plain ``[n]`` markers, and a trailing
    ``## Sources`` section links every source as an ordered list whose labels
    match the inline citation numbers. Keeping the markers plain and each URI
    in the list only minimizes token cost for LLM consumers. Text blocks
    within one ``model_output`` step are continuation runs and join directly;
    text from distinct steps (separated by tool or thought steps) joins as
    paragraphs. The whole document is normalized with mdformat.

    Raises:
        RuntimeError: If ``interaction`` is ``None``, did not complete, or
            contains no usable text.
    """
    if interaction is None:
        raise RuntimeError("no response from Gemini model")
    _raise_on_failure(interaction)
    return _render_cited_document(_model_output_blocks(interaction.steps or []))


def _render_cited_document(
    grouped: Sequence[Sequence[interactions.TextContent]],
) -> tuple[str, tuple[GoogleSearchSource, ...]]:
    """Cite each block group with shared source numbering and render the document.

    Groups join as paragraphs. The body is normalized on its own first —
    mdformat closes any dangling code fence, so the appended Sources section
    cannot be swallowed by one — then normalized again with the section.

    Raises:
        RuntimeError: If the groups contain no usable text.
    """
    sources: list[GoogleSearchSource] = []
    number_by_key: dict[str, int] = {}
    step_texts = ["".join(_cite_block(block, sources, number_by_key) for block in blocks) for blocks in grouped]

    body = "\n\n".join(step_texts)
    if not body.strip():
        raise RuntimeError("no response from Gemini model")

    document = _markdown.format_document(body)
    if sources:
        document = _markdown.format_document(f"{document}\n\n## Sources\n\n{_render_source_list(sources)}")
    return document, tuple(sources)


def _raise_on_failure(interaction: interactions.Interaction) -> None:
    """Raise ``RuntimeError`` unless the interaction completed successfully."""
    status = interaction.status
    if status == "completed":
        return
    detail = _step_error_message(interaction)
    if detail:
        raise RuntimeError(f"interaction {status}: {detail}")
    raise RuntimeError(f"interaction {status}")


def _step_error_message(interaction: interactions.Interaction) -> str:
    """Return the first model-output step error message, or the empty string."""
    for step in interaction.steps or []:
        if isinstance(step, interactions.ModelOutputStep) and step.error is not None and step.error.message:
            return step.error.message
    return ""


def _model_output_blocks(
    steps: Sequence[object],
) -> list[list[interactions.TextContent]]:
    """Group the non-empty text content blocks of each ``model_output`` step.

    Thought and tool steps carry no user-facing prose, and non-text content
    (images, audio) has no Markdown rendering here, so both are skipped.
    """
    grouped: list[list[interactions.TextContent]] = []
    for step in steps:
        if not isinstance(step, interactions.ModelOutputStep):
            continue
        blocks = [block for block in step.content or [] if isinstance(block, interactions.TextContent) and block.text]
        if blocks:
            grouped.append(blocks)
    return grouped


def _cite_block(
    block: interactions.TextContent,
    sources: list[GoogleSearchSource],
    number_by_key: dict[str, int],
) -> str:
    """Insert escaped ``\\[n\\]`` markers into one text block, collecting sources.

    New sources append to ``sources`` in annotation encounter order;
    ``number_by_key`` deduplicates them across blocks, keyed by URL (or by
    title for URL-less annotations, using a separator no URL can contain). An
    annotation whose ``end_index`` falls outside the block still registers its
    source but inserts no marker, mirroring the retired byte-offset clamping.
    """
    text = block.text or ""
    insertions: list[tuple[int, int]] = []
    for annotation in block.annotations or []:
        if not isinstance(annotation, interactions.URLCitation):
            continue
        url = annotation.url or ""
        title = annotation.title or ""
        if not url and not title:
            continue
        key = url or f"title\x00{title}"
        number = number_by_key.get(key)
        if number is None:
            number = len(sources) + 1
            number_by_key[key] = number
            sources.append(GoogleSearchSource(index=number, title=title, uri=url))
        end_index = annotation.end_index or 0
        if end_index < 0 or end_index > len(text):
            continue
        insertions.append((end_index, number))

    if not insertions:
        return text

    insertions.sort()
    pieces: list[str] = []
    last = 0
    i = 0
    n = len(insertions)
    while i < n:
        offset = insertions[i][0]
        pieces.append(text[last:offset])
        numbers: list[int] = []
        j = i
        while j < n and insertions[j][0] == offset:
            number = insertions[j][1]
            if not (numbers and numbers[-1] == number):
                numbers.append(number)
            j += 1
        pieces.append(_citation_text(numbers))
        last = offset
        i = j
    pieces.append(text[last:])
    return "".join(pieces)


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
            entry = _markdown.link(source.uri, source.uri)
        elif source.title and source.uri:
            entry = f"{_markdown.escape_link_text(source.title)} ({_markdown.escape_link_text(source.uri)})"
        elif source.uri:
            entry = _markdown.escape_link_text(source.uri)
        else:
            entry = _markdown.escape_link_text(source.title)
        lines.append(f"{source.index}. {entry}")
    return "\n".join(lines)


def _citation_text(numbers: list[int]) -> str:
    """Render citation numbers as adjacent escaped ``\\[n\\]`` markers.

    Markers are plain text, never links: the URI lives once in the
    ``## Sources`` list, which keeps token cost low for LLM consumers. The
    escaping stops surrounding body text (parentheticals, reference
    definitions) from capturing a marker into a link; mdformat drops the
    backslashes wherever that is provably safe.
    """
    return "".join(f"\\[{number}\\]" for number in numbers)
