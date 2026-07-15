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

"""Deep Research service backed by the Gemini Interactions API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anyio
from google.genai import interactions

from mcp_gemini_search._logging import logger
from mcp_gemini_search.search import (
    GoogleSearchSource,
    _model_output_blocks,
    _render_cited_document,
    _step_error_message,
)

_TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
_MAX_WAIT_SECONDS = 60
DEEP_RESEARCH_AGENT = "deep-research-preview-04-2026"
DEEP_RESEARCH_MAX_AGENT = "deep-research-max-preview-04-2026"


@dataclass(frozen=True, slots=True)
class DeepResearchStart:
    """The result of starting a background Deep Research agent run."""

    interaction_id: str
    status: str  # e.g. "in_progress"

    def to_structured(self) -> dict[str, Any]:
        """Return the MCP structured-content dict with both fields always present."""
        return {"interaction_id": self.interaction_id, "status": self.status}


@dataclass(frozen=True, slots=True)
class DeepResearchResult:
    """The status (and, once completed, the report) of a Deep Research run."""

    interaction_id: str
    status: str  # completed | in_progress | failed | cancelled | ...
    text: str = ""
    sources: tuple[GoogleSearchSource, ...] = ()

    def to_structured(self) -> dict[str, Any]:
        """Return the MCP structured-content dict with omit-empty semantics.

        ``interaction_id`` and ``status`` are always present. ``text`` is
        included only when non-empty. Each source always carries ``index``;
        ``title`` and ``uri`` are included only when non-empty. The ``sources``
        key is omitted entirely when there are no sources.
        """
        structured: dict[str, Any] = {
            "interaction_id": self.interaction_id,
            "status": self.status,
        }
        if self.text:
            structured["text"] = self.text
        if self.sources:
            structured["sources"] = [source.to_structured() for source in self.sources]
        return structured


class ResearchInteractions(Protocol):
    """Structural type for the async Gemini interactions API subset this service needs.

    Declared with Any returns for the same SDK-union reason documented in
    search.py's InteractionCreator Protocol. Structurally satisfied by
    client.aio.interactions.
    """

    async def create(self, **kwargs: Any) -> Any:
        """Create a background Deep Research interaction.

        Declared as ``Any`` because the SDK method returns
        ``Interaction | AsyncStream[InteractionSSEEvent]``; this service never
        sets ``stream``, so the result is always an ``Interaction``.
        """
        ...

    async def get(self, interaction_id: str, /) -> Any:
        """Fetch an interaction by id.

        Declared as ``Any`` for the same SDK-union reason as ``create``. Called
        positionally so the Protocol parameter name need not match the SDK's
        ``id`` keyword.
        """
        ...


class DeepResearchService:
    """Starts and polls Gemini Deep Research agent runs."""

    def __init__(
        self,
        agent: str,
        interactions: ResearchInteractions | None,
        *,
        poll_interval: float = 5.0,
        service_tier: str = "",
    ) -> None:
        """Store the agent name, the injected interactions API, the poll interval, and the service tier."""
        self._agent = agent
        self._interactions = interactions
        self._poll_interval = poll_interval
        self._service_tier = service_tier

    async def start(
        self,
        query: str,
        *,
        plan_only: bool = False,
        previous_interaction_id: str = "",
        agent: str = "",
    ) -> DeepResearchStart:
        """Start a background Deep Research run and return its interaction id.

        Args:
            query: The research question or topic to investigate.
            plan_only: Whether to request a collaborative research plan only.
            previous_interaction_id: A prior interaction to continue or refine.
            agent: A per-request agent override, or empty to use the configured agent.

        Raises:
            RuntimeError: If the service is not configured, if the backend call
                fails, or if the response lacks an interaction id.
            ValueError: If ``query`` is empty or whitespace only.
        """
        if self._interactions is None:
            raise RuntimeError("deep research service is not configured")
        if not query.strip():
            raise ValueError("research query cannot be empty")

        body: dict[str, Any] = {
            "agent": agent or self._agent,
            "input": query,
            "background": True,
        }
        if plan_only:
            body["agent_config"] = {
                "type": "deep-research",
                "collaborative_planning": True,
            }
        if previous_interaction_id:
            body["previous_interaction_id"] = previous_interaction_id
        if self._service_tier:
            body["service_tier"] = self._service_tier

        try:
            interaction = await self._interactions.create(**body)
        except Exception as e:
            raise RuntimeError(f"deep research failed: {e}") from e

        if not interaction.id:
            raise RuntimeError("deep research failed: missing interaction id")

        logger.info(
            "deep research started: id=%s agent=%s",
            interaction.id,
            self._agent,
        )
        return DeepResearchStart(interaction_id=interaction.id, status=interaction.status)

    async def result(
        self,
        interaction_id: str,
        *,
        wait_seconds: int = 0,
    ) -> DeepResearchResult:
        """Fetch (and optionally long-poll) a Deep Research run by id.

        Raises:
            RuntimeError: If the service is not configured, if the backend call
                fails, if the run fails or is cancelled, or if the completed
                report cannot be formatted.
            ValueError: If ``interaction_id`` is empty or whitespace only.
        """
        if self._interactions is None:
            raise RuntimeError("deep research service is not configured")
        if not interaction_id.strip():
            raise ValueError("interaction id cannot be empty")

        wait_seconds = max(0, min(_MAX_WAIT_SECONDS, wait_seconds))
        deadline = anyio.current_time() + wait_seconds
        interaction: Any = None
        status = ""

        while True:
            try:
                interaction = await self._interactions.get(interaction_id)
            except Exception as e:
                raise RuntimeError(f"deep research failed: {e}") from e

            status = interaction.status
            logger.info("deep research poll: id=%s status=%s", interaction_id, status)

            if status in _TERMINAL_STATUSES:
                break

            # wait_seconds == 0 must never sleep, even under clock jitter.
            remaining = 0.0 if wait_seconds == 0 else deadline - anyio.current_time()
            if remaining <= 0:
                return DeepResearchResult(interaction_id=interaction_id, status=status)

            await anyio.sleep(min(self._poll_interval, remaining))

        if status == "completed":
            try:
                text, sources = format_research_report(interaction)
            except Exception as e:
                raise RuntimeError(f"deep research failed: {e}") from e
            logger.info(
                "deep research completed: id=%s text_len=%d sources=%d",
                interaction_id,
                len(text),
                len(sources),
            )
            return DeepResearchResult(
                interaction_id=interaction_id,
                status="completed",
                text=text,
                sources=sources,
            )

        # failed or cancelled
        top_error = getattr(interaction, "error", None)
        if top_error is not None and not isinstance(top_error, str):
            top_error = str(top_error)
        detail = _step_error_message(interaction) or (top_error or "")
        if detail:
            raise RuntimeError(f"deep research {status}: {detail}")
        raise RuntimeError(f"deep research {status}")


def format_research_report(
    interaction: interactions.Interaction,
) -> tuple[str, tuple[GoogleSearchSource, ...]]:
    """Format the last consecutive model-output run into Markdown and sources.

    Selects the last maximal consecutive run of ``ModelOutputStep`` steps in
    ``interaction.steps``, tolerating trailing non-model-output steps, and
    renders it through the shared citation pipeline: ``_cite_block`` numbering,
    mdformat normalization, and a trailing ``## Sources`` section.

    Raises:
        RuntimeError: If the selected run contains no usable text.
    """
    steps = interaction.steps or []
    end = len(steps)
    while end > 0 and not isinstance(steps[end - 1], interactions.ModelOutputStep):
        end -= 1
    start = end
    while start > 0 and isinstance(steps[start - 1], interactions.ModelOutputStep):
        start -= 1
    return _render_cited_document(_model_output_blocks(steps[start:end]))
