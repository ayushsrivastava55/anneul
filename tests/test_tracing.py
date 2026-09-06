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


# --- bounded serializer: the guard against neatlogs' unbounded _serialize_obj -------------
#
# neatlogs 1.4.21 serializes every decorated function's arguments by recursing into
# __dict__ with no depth limit and no cycle detection. An argument that reaches a module
# object sent it walking the whole import graph: 99% CPU for 16+ minutes before the first
# LLM call. These tests pin the replacement's bounds and that init actually installs it.


def test_bounded_serialize_refuses_modules_classes_and_callables():
    import json as json_module

    out = tracing._bounded_serialize(json_module)
    assert isinstance(out, str)  # a short str(), not a walk of the module's globals

    assert isinstance(tracing._bounded_serialize(dict), str)
    assert isinstance(tracing._bounded_serialize(lambda: None), str)


def test_bounded_serialize_survives_cycles():
    a: dict = {"name": "a"}
    a["self"] = a
    out = tracing._bounded_serialize(a)
    assert out["name"] == "a"
    assert out["self"] == "<cycle>"


def test_bounded_serialize_truncates_depth_items_and_strings():
    deep: dict = {"leaf": "x"}
    for _ in range(10):
        deep = {"child": deep}
    flat = tracing._bounded_serialize(deep)
    for _ in range(tracing._MAX_DEPTH - 1):
        flat = flat["child"]
    assert isinstance(flat["child"], str)  # depth floor reached: stringified, not recursed

    wide = tracing._bounded_serialize(list(range(1000)))
    assert len(wide) == tracing._MAX_ITEMS + 1
    assert wide[-1] == f"...[{1000 - tracing._MAX_ITEMS} more]"

    long = tracing._bounded_serialize("y" * (tracing._MAX_STR + 5))
    assert len(long) == tracing._MAX_STR + len("...[truncated]")


def test_bounded_serialize_keeps_model_dump_style_protocols():
    class Spec:
        def to_dict(self) -> dict:
            return {"topology": "single", "nodes": [1, 2]}

    assert tracing._bounded_serialize(Spec()) == {"topology": "single", "nodes": [1, 2]}


def test_bounded_serialize_reads_plain_object_dicts():
    class Row:
        def __init__(self) -> None:
            self.task_id = "t1"
            self._private = "hidden"

    assert tracing._bounded_serialize(Row()) == {"task_id": "t1"}


def test_init_tracing_installs_the_bounded_serializer(in_memory_neatlogs):
    from neatlogs.decorators import _base as nl_base

    assert nl_base._serialize_obj is tracing._bounded_serialize


def test_span_with_a_module_reaching_argument_completes_and_stays_bounded(in_memory_neatlogs):
    """Regression: this exact shape (arg whose __dict__ reaches a module) froze real runs."""
    import time

    class Domain:
        def __init__(self) -> None:
            import json as json_module

            self.name = "airline"
            self.module = json_module  # the poison: __dict__ walk reaches a module object

    @tracing.node_span("runner.run_split")
    def run_split(domain: Domain) -> str:
        return domain.name

    start = time.monotonic()
    assert run_split(Domain()) == "airline"
    assert time.monotonic() - start < 5  # unpatched, this path burned minutes of CPU

    exporter = in_memory_neatlogs
    (span,) = [s for s in exporter.get_finished_spans() if s.name == "runner.run_split"]
    input_value = span.attributes.get("input.value", "")
    assert len(input_value) < 20_000  # bounded evidence, not a heap dump
