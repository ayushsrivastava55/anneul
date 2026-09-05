"""Tests for anneal.tracing.

All tests run offline. The single "tracing on" test uses neatlogs with export disabled and
an in-memory OpenTelemetry exporter, then shuts neatlogs down so other tests stay offline.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from anneal import tracing


@pytest.fixture(autouse=True)
def _reset_tracing_state(monkeypatch: pytest.MonkeyPatch):
    """Every test starts with tracing off, no key, and an empty run context."""
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    monkeypatch.setenv("NEATLOGS_DISABLE_EXPORT", "1")
    tracing.clear_run_context()
    tracing._reset_for_tests()
    yield
    tracing.clear_run_context()
    tracing._reset_for_tests()


# --- init / lifecycle ---------------------------------------------------------------


def test_init_tracing_without_key_is_noop_and_warns(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, logger="anneal.tracing"):
        assert tracing.init_tracing() is False
    assert tracing.is_tracing_enabled() is False
    assert any("NEATLOGS_API_KEY" in r.getMessage() for r in caplog.records)


def test_wrap_client_returns_client_unchanged_when_off():
    client = object()
    assert tracing.wrap_client(client) is client


def test_flush_and_shutdown_are_safe_when_off():
    assert tracing.flush() is False
    assert tracing.shutdown() is False


def test_current_trace_id_is_none_when_off():
    assert tracing.current_trace_id() is None


# --- decorators degrade to plain calls ---------------------------------------------------


def test_sync_decorators_return_wrapped_values_when_off():
    @tracing.node_span("planner")
    def plan(x: int) -> int:
        return x + 1

    @tracing.tool_span("lookup")
    def lookup(x: int, *, y: int = 0) -> dict[str, int]:
        return {"x": x, "y": y}

    @tracing.llm_span("executor.llm")
    def call(prompt: str) -> str:
        return prompt.upper()

    assert plan(1) == 2
    assert lookup(1, y=2) == {"x": 1, "y": 2}
    assert call("hi") == "HI"
    assert plan.__name__ == "plan"
    assert lookup.__name__ == "lookup"


def test_async_decorators_return_wrapped_values_when_off():
    @tracing.node_span("planner")
    async def plan(x: int) -> int:
        return x * 2

    @tracing.tool_span("lookup")
    async def lookup(x: int) -> int:
        return x * 3

    @tracing.llm_span("executor.llm")
    async def call(x: int) -> int:
        return x * 4

    async def main() -> tuple[int, int, int]:
        return await plan(1), await lookup(1), await call(1)

    assert asyncio.run(main()) == (2, 3, 4)
    assert asyncio.iscoroutinefunction(plan)


def test_decorator_exceptions_propagate_when_off():
    @tracing.tool_span("boom")
    def boom() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        boom()


# --- run context ----------------------------------------------------------------------


def test_run_context_set_get_clear():
    assert tracing.get_run_context() == tracing.RunContext()
    tracing.set_run_context(candidate_id="c1", iteration=2, domain="airline", split="search")
    ctx = tracing.get_run_context()
    assert (ctx.candidate_id, ctx.iteration, ctx.domain, ctx.split) == (
        "c1",
        2,
        "airline",
        "search",
    )
    assert ctx.as_tags() == ["candidate_id:c1", "iteration:2", "domain:airline", "split:search"]
    assert ctx.as_attributes() == {
        "anneal.candidate_id": "c1",
        "anneal.iteration": 2,
        "anneal.domain": "airline",
        "anneal.split": "search",
    }
    tracing.clear_run_context()
    assert tracing.get_run_context() == tracing.RunContext()


def test_run_context_partial_update_keeps_other_fields():
    tracing.set_run_context(candidate_id="c1", domain="airline")
    tracing.set_run_context(iteration=3)
    ctx = tracing.get_run_context()
    assert (ctx.candidate_id, ctx.iteration, ctx.domain, ctx.split) == ("c1", 3, "airline", None)
    assert ctx.as_tags() == ["candidate_id:c1", "iteration:3", "domain:airline"]


def test_run_context_manager_restores_previous_value():
    tracing.set_run_context(candidate_id="outer")
    with tracing.run_context(candidate_id="inner", split="holdout"):
        assert tracing.get_run_context().candidate_id == "inner"
        assert tracing.get_run_context().split == "holdout"
    assert tracing.get_run_context().candidate_id == "outer"
    assert tracing.get_run_context().split is None


def test_run_context_propagates_into_asyncio_tasks_and_decorated_calls():
    seen: dict[str, tracing.RunContext] = {}

    @tracing.node_span("executor")
    async def executor(label: str) -> str:
        await asyncio.sleep(0)
        seen[label] = tracing.get_run_context()
        return label

    async def main() -> None:
        with tracing.run_context(candidate_id="cA", iteration=1, domain="airline", split="search"):
            task_a = asyncio.create_task(executor("a"))
        with tracing.run_context(candidate_id="cB", iteration=1, domain="airline", split="search"):
            task_b = asyncio.create_task(executor("b"))
        await asyncio.gather(task_a, task_b)

    asyncio.run(main())
    assert seen["a"].candidate_id == "cA"
    assert seen["b"].candidate_id == "cB"


# --- tracing on: tags land on spans -------------------------------------------------------


@pytest.fixture
def in_memory_neatlogs(monkeypatch: pytest.MonkeyPatch):
    """Initialise neatlogs with a private in-memory exporter; shut it down afterwards."""
    neatlogs = pytest.importorskip("neatlogs")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setenv("NEATLOGS_API_KEY", "test-key")
    assert tracing.init_tracing(
        tags=["unit"],
        _init_kwargs={
            "disable_export": True,
            "tracer_provider": provider,
            "register_shutdown_handlers": False,
        },
    )
    yield exporter
    neatlogs.shutdown()


def test_init_tracing_is_called_once(in_memory_neatlogs):
    assert tracing.is_tracing_enabled()
    assert tracing.init_tracing() is True  # second call is a no-op, not an SDK re-init


def test_spans_carry_kind_name_and_run_context(in_memory_neatlogs):
    exporter = in_memory_neatlogs

    @tracing.node_span("planner")
    def plan(x: int) -> int:
        return x + 1

    @tracing.tool_span("search_flights")
    def search(q: str) -> str:
        return q

    @tracing.llm_span("planner.llm")
    def call(prompt: str) -> str:
        assert tracing.current_trace_id() is not None
        return prompt

    with tracing.run_context(candidate_id="c1", iteration=2, domain="airline", split="search"):
        assert plan(1) == 2
        assert search("SFO") == "SFO"
        assert call("hi") == "hi"

    spans = {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}
    assert spans["planner"]["openinference.span.kind"] == "AGENT"
    assert spans["search_flights"]["openinference.span.kind"] == "TOOL"
    assert spans["planner.llm"]["openinference.span.kind"] == "LLM"
    for attrs in spans.values():
        assert attrs["anneal.candidate_id"] == "c1"
        assert attrs["anneal.iteration"] == 2
        assert attrs["anneal.domain"] == "airline"
        assert attrs["anneal.split"] == "search"
        assert set(attrs["tag.tags"]) >= {
            "candidate_id:c1",
            "iteration:2",
            "domain:airline",
            "split:search",
        }


def test_async_spans_carry_run_context(in_memory_neatlogs):
    exporter = in_memory_neatlogs

    @tracing.tool_span("async_tool")
    async def tool(x: int) -> int:
        return x

    async def main() -> int:
        with tracing.run_context(candidate_id="c9", iteration=0, domain="d", split="search"):
            return await tool(7)

    assert asyncio.run(main()) == 7
    (span,) = [s for s in exporter.get_finished_spans() if s.name == "async_tool"]
    assert span.attributes["anneal.candidate_id"] == "c9"
    assert "candidate_id:c9" in span.attributes["tag.tags"]


def test_current_trace_id_inside_span_is_hex(in_memory_neatlogs):
    captured: list[str | None] = []

    @tracing.node_span("root")
    def root() -> None:
        captured.append(tracing.current_trace_id())

    root()
    assert captured[0] is not None
    assert len(captured[0]) == 32
    int(captured[0], 16)
    assert tracing.current_trace_id() is None
