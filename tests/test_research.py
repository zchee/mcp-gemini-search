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

"""Tests for the Deep Research service."""

from __future__ import annotations

import logging
from typing import Any

import anyio
import jsonschema
import pytest
from google.genai import interactions

from mcp_gemini_search import research as research_mod
from mcp_gemini_search.research import (
    DeepResearchResult,
    DeepResearchService,
    DeepResearchStart,
    format_research_report,
)
from mcp_gemini_search.search import GoogleSearchSource
from tests._helpers import golden_tool
from tests._helpers import interaction as _interaction
from tests._helpers import model_output as _output
from tests._helpers import text_block as _text
from tests._helpers import url_citation as _cite


class StubInteractions:
    """Records create kwargs and returns scripted get/create responses."""

    def __init__(
        self,
        *,
        create_response: interactions.Interaction | None = None,
        create_error: Exception | None = None,
        get_responses: list[interactions.Interaction] | None = None,
        get_error: Exception | None = None,
    ) -> None:
        self.create_response = create_response
        self.create_error = create_error
        self.get_responses = list(get_responses or [])
        self.get_error = get_error
        self.create_kwargs: dict[str, Any] | None = None
        self.create_calls = 0
        self.get_calls = 0
        self.get_ids: list[str] = []

    async def create(self, **kwargs: Any) -> interactions.Interaction:
        """Record kwargs and return the canned interaction or raise the error."""
        self.create_calls += 1
        self.create_kwargs = dict(kwargs)
        if self.create_error is not None:
            raise self.create_error
        if self.create_response is None:
            raise RuntimeError("stub interactions API is misconfigured")
        return self.create_response

    async def get(self, interaction_id: str) -> interactions.Interaction:
        """Return the next scripted response or raise the configured error."""
        self.get_calls += 1
        self.get_ids.append(interaction_id)
        if self.get_error is not None:
            raise self.get_error
        if not self.get_responses:
            raise RuntimeError("stub interactions API has no more get responses")
        return self.get_responses.pop(0)


class _NoneResponses:
    """Returns None from both API calls to exercise the defensive guards."""

    async def create(self, **kwargs: Any) -> Any:
        """Return None in place of an interaction."""
        return None

    async def get(self, interaction_id: str, /) -> Any:
        """Return None in place of an interaction."""
        return None


@pytest.mark.anyio
async def test_start_issues_background_create() -> None:
    """start() issues create with agent, input, and background=True only."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-1"),
    )
    svc = DeepResearchService("deep-research-preview-04-2026", stub)

    got = await svc.start("q")

    assert got == DeepResearchStart(interaction_id="dr-1", status="in_progress")
    assert stub.create_calls == 1
    assert stub.create_kwargs == {
        "agent": "deep-research-preview-04-2026",
        "input": "q",
        "background": True,
    }


@pytest.mark.anyio
async def test_start_passes_service_tier() -> None:
    """When configured, start() includes service_tier in the create kwargs."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-tier"),
    )
    svc = DeepResearchService("deep-research-preview-04-2026", stub, service_tier="priority")

    await svc.start("q")

    assert stub.create_kwargs is not None
    assert stub.create_kwargs["service_tier"] == "priority"


@pytest.mark.anyio
async def test_start_omits_service_tier_when_unset() -> None:
    """When service_tier is unset, create kwargs do not include service_tier."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-no-tier"),
    )
    svc = DeepResearchService("deep-research-preview-04-2026", stub)

    await svc.start("q")

    assert stub.create_kwargs is not None
    assert "service_tier" not in stub.create_kwargs


@pytest.mark.anyio
async def test_start_plan_only_and_previous_interaction() -> None:
    """plan_only and previous_interaction_id add the expected create kwargs."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-2"),
    )
    svc = DeepResearchService("deep-research-preview-04-2026", stub)

    await svc.start("q", plan_only=True, previous_interaction_id="i-1")

    assert stub.create_kwargs == {
        "agent": "deep-research-preview-04-2026",
        "input": "q",
        "background": True,
        "agent_config": {"type": "deep-research", "collaborative_planning": True},
        "previous_interaction_id": "i-1",
    }


@pytest.mark.parametrize(
    ("agent", "expected_agent"),
    [
        (research_mod.DEEP_RESEARCH_MAX_AGENT, research_mod.DEEP_RESEARCH_MAX_AGENT),
        (None, research_mod.DEEP_RESEARCH_AGENT),
        ("", research_mod.DEEP_RESEARCH_AGENT),
    ],
    ids=["override", "omitted", "empty"],
)
@pytest.mark.anyio
async def test_start_agent_override(
    agent: str | None,
    expected_agent: str,
) -> None:
    """A non-empty request agent overrides the configured agent; empty values fall back."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-agent"),
    )
    svc = DeepResearchService(research_mod.DEEP_RESEARCH_AGENT, stub)

    if agent is None:
        await svc.start("q")
    else:
        await svc.start("q", agent=agent)

    assert stub.create_kwargs == {
        "agent": expected_agent,
        "input": "q",
        "background": True,
    }


@pytest.mark.parametrize("query", ["", "   ", "\t\n"], ids=["empty", "spaces", "whitespace"])
@pytest.mark.anyio
async def test_start_empty_query(query: str) -> None:
    """Empty or whitespace-only research queries raise ValueError."""
    svc = DeepResearchService("agent", StubInteractions())
    with pytest.raises(ValueError) as excinfo:
        await svc.start(query)
    assert str(excinfo.value) == "research query cannot be empty"


@pytest.mark.anyio
async def test_start_not_configured() -> None:
    """start() without an interactions API raises the not-configured error."""
    svc = DeepResearchService("agent", None)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.start("q")
    assert str(excinfo.value) == "deep research service is not configured"


@pytest.mark.anyio
async def test_result_not_configured() -> None:
    """result() without an interactions API raises the not-configured error."""
    svc = DeepResearchService("agent", None)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-1")
    assert str(excinfo.value) == "deep research service is not configured"


@pytest.mark.anyio
async def test_start_backend_error() -> None:
    """Backend create failures are wrapped with the deep-research-failed prefix."""
    backend_error = RuntimeError("backend failed")
    svc = DeepResearchService("agent", StubInteractions(create_error=backend_error))
    with pytest.raises(RuntimeError) as excinfo:
        await svc.start("q")
    assert str(excinfo.value) == "deep research failed: backend failed"
    assert excinfo.value.__cause__ is backend_error


@pytest.mark.anyio
async def test_start_missing_interaction_id() -> None:
    """A create response without an id raises the missing-id error."""
    stub = StubInteractions(
        create_response=interactions.Interaction(status="in_progress", steps=[]),
    )
    svc = DeepResearchService("agent", stub)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.start("q")
    assert str(excinfo.value) == "deep research failed: missing interaction id"


@pytest.mark.parametrize("interaction_id", ["", "   ", "\t"], ids=["empty", "spaces", "tab"])
@pytest.mark.anyio
async def test_result_empty_interaction_id(interaction_id: str) -> None:
    """Empty or whitespace-only interaction ids raise ValueError."""
    svc = DeepResearchService("agent", StubInteractions())
    with pytest.raises(ValueError) as excinfo:
        await svc.result(interaction_id)
    assert str(excinfo.value) == "interaction id cannot be empty"


@pytest.mark.anyio
async def test_result_wait_seconds_zero_single_poll_no_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_seconds=0 performs exactly one get and never sleeps."""
    sleep_calls: list[float] = []

    async def tracking_sleep(seconds: float) -> None:  # noqa: RUF029
        sleep_calls.append(seconds)
        raise AssertionError(f"anyio.sleep should not be called, got {seconds}")

    monkeypatch.setattr(research_mod.anyio, "sleep", tracking_sleep)
    stub = StubInteractions(
        get_responses=[_interaction(status="in_progress", interaction_id="i-1")],
    )
    svc = DeepResearchService("agent", stub)

    with anyio.fail_after(1):
        got = await svc.result("i-1", wait_seconds=0)

    assert stub.get_calls == 1
    assert sleep_calls == []
    assert got == DeepResearchResult(interaction_id="i-1", status="in_progress", text="", sources=())


@pytest.mark.anyio
async def test_result_wait_seconds_negative_clamps_like_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_seconds below 0 clamps to 0: one get, no sleep, status-only return."""
    sleep_calls: list[float] = []

    async def tracking_sleep(seconds: float) -> None:  # noqa: RUF029
        sleep_calls.append(seconds)

    monkeypatch.setattr(research_mod.anyio, "sleep", tracking_sleep)
    stub = StubInteractions(
        get_responses=[_interaction(status="in_progress", interaction_id="i-1")],
    )
    svc = DeepResearchService("agent", stub)

    with anyio.fail_after(1):
        got = await svc.result("i-1", wait_seconds=-5)

    assert stub.get_calls == 1
    assert sleep_calls == []
    assert got.status == "in_progress"
    assert got.text == ""
    assert got.sources == ()


@pytest.mark.anyio
async def test_result_wait_seconds_above_max_clamps_like_sixty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wait_seconds above 60 clamps to 60 for the same poll/sleep budget as 60."""

    class FakeClock:
        def __init__(self) -> None:
            self.now = 1000.0
            self.sleeps: list[float] = []

        def current_time(self) -> float:
            return self.now

        async def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

    async def run_with_wait(wait_seconds: int) -> tuple[int, list[float]]:
        clock = FakeClock()
        monkeypatch.setattr(research_mod.anyio, "current_time", clock.current_time)
        monkeypatch.setattr(research_mod.anyio, "sleep", clock.sleep)
        # Always-in-progress queue of responses; stop after the service returns.
        responses = [_interaction(status="in_progress", interaction_id="i-1") for _ in range(100)]
        stub = StubInteractions(get_responses=responses)
        svc = DeepResearchService("agent", stub, poll_interval=5.0)
        with anyio.fail_after(1):
            got = await svc.result("i-1", wait_seconds=wait_seconds)
        assert got.status == "in_progress"
        return stub.get_calls, list(clock.sleeps)

    calls_60, sleeps_60 = await run_with_wait(60)
    calls_300, sleeps_300 = await run_with_wait(300)

    assert calls_60 == calls_300
    assert sleeps_60 == sleeps_300
    assert calls_60 > 1
    assert sleeps_60
    assert sum(sleeps_60) == pytest.approx(60.0)


@pytest.mark.anyio
async def test_result_long_poll_until_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    """wait_seconds>0 polls until completed, sleeping between non-terminal gets."""
    sleep_calls: list[float] = []
    clock = {"now": 0.0}

    def current_time() -> float:
        return clock["now"]

    async def fake_sleep(seconds: float) -> None:  # noqa: RUF029
        sleep_calls.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(research_mod.anyio, "current_time", current_time)
    monkeypatch.setattr(research_mod.anyio, "sleep", fake_sleep)

    completed = _interaction(
        _output(_text("Report A")),
        _output(_text("Report B", _cite("https://example.com", "Example", 8))),
        status="completed",
        interaction_id="i-1",
    )
    stub = StubInteractions(
        get_responses=[
            _interaction(status="in_progress", interaction_id="i-1"),
            _interaction(status="in_progress", interaction_id="i-1"),
            completed,
        ],
    )
    svc = DeepResearchService("agent", stub, poll_interval=5.0)

    with anyio.fail_after(1):
        got = await svc.result("i-1", wait_seconds=15)

    assert stub.get_calls == 3
    assert len(sleep_calls) == 2
    assert sleep_calls == [5.0, 5.0]
    assert got.status == "completed"
    assert got.interaction_id == "i-1"
    assert got.text == "Report A\n\nReport B[1]\n\n## Sources\n\n1. [Example](https://example.com)"
    assert len(got.sources) == 1
    assert got.sources[0].title == "Example"
    assert got.sources[0].uri == "https://example.com"


def test_format_research_report_trailing_run_skips_plan() -> None:
    """Only the last consecutive model-output run is formatted into the report."""
    interaction = _interaction(
        _output(_text("plan echo")),
        interactions.GoogleSearchCallStep(
            id="call_1",
            arguments=interactions.GoogleSearchCallArguments(queries=["q"]),
        ),
        interactions.GoogleSearchResultStep(call_id="call_1", result=[]),
        _output(_text("Report A")),
        _output(_text("Report B", _cite("https://example.com", "Example", 8))),
    )

    text, sources = format_research_report(interaction)

    assert "plan echo" not in text
    assert text == "Report A\n\nReport B[1]\n\n## Sources\n\n1. [Example](https://example.com)"
    assert len(sources) == 1


def test_format_research_report_tolerates_trailing_thought() -> None:
    """A trailing ThoughtStep after the report still extracts the model output."""
    interaction = _interaction(
        _output(_text("Report")),
        interactions.ThoughtStep(signature="sig"),
    )

    text, sources = format_research_report(interaction)

    assert text == "Report"
    assert sources == ()


def test_format_research_report_skips_image_content() -> None:
    """Image content blocks inside model_output are skipped without error."""
    interaction = _interaction(
        _output(
            interactions.ImageContent(uri="https://example.com/x.png"),
            _text("Only text"),
        ),
    )

    text, sources = format_research_report(interaction)

    assert text == "Only text"
    assert sources == ()


@pytest.mark.anyio
async def test_result_failed_surfaces_step_error() -> None:
    """A failed interaction surfaces its model-output step error message."""
    failed = _interaction(
        interactions.ModelOutputStep(content=[], error=interactions.Status(message="quota exhausted")),
        status="failed",
        interaction_id="i-fail",
    )
    stub = StubInteractions(get_responses=[failed])
    svc = DeepResearchService("agent", stub)

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-fail", wait_seconds=0)
    assert str(excinfo.value) == "deep research failed: quota exhausted"


@pytest.mark.anyio
async def test_result_cancelled_without_detail() -> None:
    """A cancelled interaction with no step error raises the bare status message."""
    cancelled = _interaction(status="cancelled", interaction_id="i-cancel")
    stub = StubInteractions(get_responses=[cancelled])
    svc = DeepResearchService("agent", stub)

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-cancel", wait_seconds=0)
    assert str(excinfo.value) == "deep research cancelled"


@pytest.mark.anyio
async def test_result_failed_surfaces_top_level_error() -> None:
    """A failed run with no step error falls back to the top-level error extra.

    ``Interaction`` declares no ``error`` field, but the model allows extras,
    so the payload of a failed background run can still carry one.
    """
    failed = interactions.Interaction.model_validate({
        "id": "i-top",
        "status": "failed",
        "steps": [],
        "error": {"message": "backend exploded"},
    })
    stub = StubInteractions(get_responses=[failed])
    svc = DeepResearchService("agent", stub)

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-top", wait_seconds=0)
    assert str(excinfo.value) == "deep research failed: backend exploded"


@pytest.mark.anyio
async def test_result_failed_top_level_error_without_message() -> None:
    """A top-level error extra with no message field falls back to its str form."""
    failed = interactions.Interaction.model_validate({
        "id": "i-code",
        "status": "failed",
        "steps": [],
        "error": {"code": 500},
    })
    stub = StubInteractions(get_responses=[failed])
    svc = DeepResearchService("agent", stub)

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-code", wait_seconds=0)
    assert str(excinfo.value) == "deep research failed: {'code': 500}"


@pytest.mark.anyio
async def test_result_none_response_raises() -> None:
    """A None get response raises the no-response error instead of AttributeError."""
    svc = DeepResearchService("agent", _NoneResponses())

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-none", wait_seconds=0)
    assert str(excinfo.value) == "deep research failed: no response from Gemini API"


@pytest.mark.anyio
async def test_start_none_response_raises() -> None:
    """A None create response raises the missing-id error instead of AttributeError."""
    svc = DeepResearchService("agent", _NoneResponses())

    with pytest.raises(RuntimeError) as excinfo:
        await svc.start("q")
    assert str(excinfo.value) == "deep research failed: missing interaction id"


@pytest.mark.anyio
async def test_result_completed_without_output_raises() -> None:
    """A completed run with no model output raises the wrapped no-response error."""
    stub = StubInteractions(get_responses=[_interaction(status="completed", interaction_id="i-empty")])
    svc = DeepResearchService("agent", stub)

    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-empty", wait_seconds=0)
    assert str(excinfo.value) == "deep research failed: no response from Gemini model"


@pytest.mark.parametrize(
    "status",
    ["requires_action", "incomplete", "budget_exceeded"],
    ids=["requires_action", "incomplete", "budget_exceeded"],
)
@pytest.mark.anyio
async def test_result_non_terminal_status_returns_status_only(status: str) -> None:
    """Non-terminal non-progress statuses return a status-only result at wait=0."""
    stub = StubInteractions(
        get_responses=[_interaction(status=status, interaction_id="i-1")],
    )
    svc = DeepResearchService("agent", stub)

    got = await svc.result("i-1", wait_seconds=0)

    assert got == DeepResearchResult(interaction_id="i-1", status=status, text="", sources=())
    assert "text" not in got.to_structured()
    assert "sources" not in got.to_structured()


def test_to_structured_start() -> None:
    """DeepResearchStart.to_structured always includes interaction_id and status."""
    start = DeepResearchStart(interaction_id="dr-1", status="in_progress")
    assert start.to_structured() == {"interaction_id": "dr-1", "status": "in_progress"}


def test_to_structured_result_omits_empty_fields() -> None:
    """DeepResearchResult.to_structured omits empty text, sources, title, and uri."""
    empty = DeepResearchResult(interaction_id="i-1", status="in_progress")
    assert empty.to_structured() == {"interaction_id": "i-1", "status": "in_progress"}

    populated = DeepResearchResult(
        interaction_id="i-1",
        status="completed",
        text="Report",
        sources=(
            GoogleSearchSource(index=1, title="Title", uri=""),
            GoogleSearchSource(index=2, title="", uri="https://u.example"),
        ),
    )
    assert populated.to_structured() == {
        "interaction_id": "i-1",
        "status": "completed",
        "text": "Report",
        "sources": [
            {"index": 1, "title": "Title"},
            {"index": 2, "uri": "https://u.example"},
        ],
    }


def test_to_structured_validates_against_golden_schemas() -> None:
    """to_structured output validates against the deep-research golden schemas."""
    start_schema = golden_tool("deep_research")["outputSchema"]
    result_schema = golden_tool("deep_research_result")["outputSchema"]

    start = DeepResearchStart(interaction_id="dr-1", status="in_progress")
    jsonschema.validate(start.to_structured(), start_schema)

    populated = DeepResearchResult(
        interaction_id="i-1",
        status="completed",
        text="Report",
        sources=(GoogleSearchSource(index=1, title="Title", uri="https://u.example"),),
    )
    jsonschema.validate(populated.to_structured(), result_schema)

    status_only = DeepResearchResult(interaction_id="i-1", status="in_progress")
    jsonschema.validate(status_only.to_structured(), result_schema)


@pytest.mark.anyio
async def test_start_logs_interaction_id(caplog: pytest.LogCaptureFixture) -> None:
    """start() logs the new interaction id at INFO."""
    stub = StubInteractions(
        create_response=_interaction(status="in_progress", interaction_id="dr-log"),
    )
    svc = DeepResearchService("deep-research-preview-04-2026", stub)

    with caplog.at_level(logging.INFO, logger="mcp_gemini_search"):
        await svc.start("q")

    assert any("deep research started" in rec.message and "dr-log" in rec.message for rec in caplog.records)


@pytest.mark.anyio
async def test_result_logs_poll_and_completion(caplog: pytest.LogCaptureFixture) -> None:
    """result() logs poll status transitions and completion details."""
    completed = _interaction(_output(_text("Done")), status="completed", interaction_id="i-log")
    stub = StubInteractions(get_responses=[completed])
    svc = DeepResearchService("agent", stub)

    with caplog.at_level(logging.INFO, logger="mcp_gemini_search"):
        await svc.result("i-log", wait_seconds=0)

    messages = [rec.message for rec in caplog.records]
    assert any("deep research poll" in msg and "i-log" in msg for msg in messages)
    assert any("deep research completed" in msg and "i-log" in msg for msg in messages)


@pytest.mark.anyio
async def test_result_get_backend_error() -> None:
    """Backend get failures are wrapped with the deep-research-failed prefix."""
    backend_error = RuntimeError("get failed")
    stub = StubInteractions(get_error=backend_error)
    svc = DeepResearchService("agent", stub)
    with pytest.raises(RuntimeError) as excinfo:
        await svc.result("i-1")
    assert str(excinfo.value) == "deep research failed: get failed"
    assert excinfo.value.__cause__ is backend_error
