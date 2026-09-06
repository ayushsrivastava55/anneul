"""Neatlogs tracing for Anneal: init, span decorators, run-context tags.

Every node, tool call and LLM call in the runtime is a Neatlogs span (CLAUDE.md rule 3).
This module wraps the ``neatlogs`` SDK so that:

- ``init_tracing()`` calls ``neatlogs.init`` exactly once, and is a no-op that returns
  ``False`` when ``NEATLOGS_API_KEY`` is empty.
- ``node_span`` / ``tool_span`` / ``llm_span`` decorate sync or async callables. When
  tracing is off they are plain pass-throughs, so unit tests run offline.
- ``set_run_context`` / ``run_context`` stash ``candidate_id``, ``iteration``, ``domain``,
  ``split`` in a contextvar. Every span created while that context is active carries them
  as ``anneal.*`` attributes and as ``tag.tags`` entries (``"candidate_id:<v>"``) so
  ``diagnose.py`` can search traces by tag over MCP.

Divergences from docs/SPONSORS.md, verified against the installed SDK (neatlogs 1.4.21):

- ``neatlogs.span`` rejects ``kind="LLM"`` (valid: WORKFLOW, AGENT, CHAIN, TOOL, RETRIEVER,
  EMBEDDING, GUARDRAIL, EVALUATOR, MEMORY, MCP_TOOL). ``llm_span`` therefore uses the
  ``neatlogs.trace(name, kind="LLM")`` context manager. The real per-request LLM span
  (model, tokens, messages) is emitted by ``neatlogs.wrap(client)``; ``llm_span`` is the
  call-site parent that carries the run-context tags.
- The SDK exposes no trace-id getter. ``current_trace_id()`` reads the active neatlogs span
  through OpenTelemetry and returns ``None`` outside a recording span.
- Neatlogs always runs in *isolated* mode: its active span is not the OpenTelemetry
  current span. Run-context attributes are attached via ``active_neatlogs_context``.
- ``neatlogs.decorators._base._serialize_obj`` (1.4.21) recurses into ``__dict__`` of
  arbitrary objects with no depth limit, no cycle detection and no memoisation. Any
  decorated function whose arguments reach a module object (a domain, a client, a spec
  holding callables) sends it walking the interpreter's import graph -- observed as 99%
  CPU for 16+ minutes before the first LLM call, with the faulthandler stack looping
  through ``_serialize_obj`` -> ``importlib._module_repr_from_spec``. ``init_tracing``
  therefore replaces it with :func:`_bounded_serialize` (same output shape, but depth-,
  size- and cycle-bounded) before ``neatlogs.init``. Every neatlogs call site funnels
  through ``_base._safe_json_dumps``, which resolves ``_serialize_obj`` at call time,
  so patching that one module-level name covers them all.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import functools
import logging
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger("anneal.tracing")

_UNSET = object()
_enabled = False

# --- run context -----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RunContext:
    """Identifies which candidate / iteration / domain / split a span belongs to."""

    candidate_id: str | None = None
    iteration: int | None = None
    domain: str | None = None
    split: str | None = None

    def as_attributes(self) -> dict[str, str | int]:
        """Span attributes, ``anneal.<field>``, for every field that is set."""
        return {f"anneal.{k}": v for k, v in dataclasses.asdict(self).items() if v is not None}

    def as_tags(self) -> list[str]:
        """Neatlogs tags, ``<field>:<value>``, for every field that is set."""
        return [f"{k}:{v}" for k, v in dataclasses.asdict(self).items() if v is not None]


_run_context: contextvars.ContextVar[RunContext] = contextvars.ContextVar("anneal_run_context")


def get_run_context() -> RunContext:
    """Return the run context for the current task/thread."""
    return _run_context.get(RunContext())


def set_run_context(
    *,
    candidate_id: str | None | object = _UNSET,
    iteration: int | None | object = _UNSET,
    domain: str | None | object = _UNSET,
    split: str | None | object = _UNSET,
) -> contextvars.Token[RunContext]:
    """Update the run context in place; omitted fields keep their current value.

    Returns the contextvar token so callers can ``_run_context.reset(token)``; prefer the
    ``run_context`` context manager which does that for you.
    """
    current = get_run_context()
    updates = {
        k: v
        for k, v in {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "domain": domain,
            "split": split,
        }.items()
        if v is not _UNSET
    }
    return _run_context.set(dataclasses.replace(current, **updates))


def clear_run_context() -> None:
    """Reset the run context to empty."""
    _run_context.set(RunContext())


@contextlib.contextmanager
def run_context(**fields: Any) -> Iterator[RunContext]:
    """Scoped ``set_run_context``: previous values are restored on exit."""
    token = set_run_context(**fields)
    try:
        yield get_run_context()
    finally:
        _run_context.reset(token)


# --- neatlogs serializer guard -----------------------------------------------------------

# Bounds for span-attribute serialization. Deep nesting past _MAX_DEPTH, containers past
# _MAX_ITEMS and strings past _MAX_STR are truncated: span attributes are evidence, not
# a pickle of the process.
_MAX_DEPTH = 4
_MAX_ITEMS = 25
_MAX_STR = 2_000


def _short(obj: Any) -> str:
    """A truncated ``str()`` of ``obj`` that can never raise or explode."""
    try:
        text = str(obj)
    except Exception:  # noqa: BLE001 - reprs of arbitrary objects can do anything
        text = f"<unprintable {type(obj).__name__}>"
    return text if len(text) <= _MAX_STR else text[:_MAX_STR] + "...[truncated]"


def _bounded_serialize(obj: Any, _depth: int = 0, _seen: set[int] | None = None) -> Any:
    """Drop-in replacement for neatlogs' ``_serialize_obj`` with hard bounds.

    Same shape on the happy paths (primitives, containers, ``model_dump``-style protocols,
    ``__dict__``), but with a depth limit, per-container item caps, string truncation,
    cycle/sharing detection, and a refusal to introspect modules, classes and callables.
    """
    import types

    if obj is None or isinstance(obj, (int, float, bool)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= _MAX_STR else obj[:_MAX_STR] + "...[truncated]"
    if isinstance(obj, (types.ModuleType, type)) or callable(obj):
        return _short(obj)
    if _depth >= _MAX_DEPTH:
        return _short(obj)
    seen = _seen if _seen is not None else set()
    if id(obj) in seen:
        return "<cycle>"
    seen.add(id(obj))

    if isinstance(obj, (list, tuple, set, frozenset)):
        items = list(obj)[:_MAX_ITEMS]
        out: Any = [_bounded_serialize(v, _depth + 1, seen) for v in items]
        if len(obj) > _MAX_ITEMS:
            out.append(f"...[{len(obj) - _MAX_ITEMS} more]")
        return out
    if isinstance(obj, dict):
        pairs = list(obj.items())[:_MAX_ITEMS]
        mapped = {_short(k): _bounded_serialize(v, _depth + 1, seen) for k, v in pairs}
        if len(obj) > _MAX_ITEMS:
            mapped["..."] = f"[{len(obj) - _MAX_ITEMS} more]"
        return mapped

    # Pydantic / dataclass-style protocols, as in the original serializer.
    import json as _json

    for method in ("model_dump", "dict", "to_dict", "to_json", "as_dict"):
        candidate = getattr(obj, method, None)
        if callable(candidate):
            try:
                result = candidate()
                if method == "to_json" and isinstance(result, str):
                    result = _json.loads(result)
                return _bounded_serialize(result, _depth + 1, seen)
            except Exception:  # noqa: BLE001 - mirror the original: try the next protocol
                continue

    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict):
        try:
            picked = {
                k: _bounded_serialize(v, _depth + 1, seen)
                for k, v in list(attrs.items())[:_MAX_ITEMS]
                if isinstance(k, str) and not k.startswith("_") and not callable(v)
            }
            if picked:
                return picked
        except Exception:  # noqa: BLE001
            pass
    return _short(obj)


def _patch_neatlogs_serializer() -> bool:
    """Install :func:`_bounded_serialize` over neatlogs' unbounded ``_serialize_obj``.

    Returns False (with a warning) if the SDK layout changed and the seam is gone; in
    that case tracing still works but heavy arguments may make spans slow again.
    """
    try:
        from neatlogs.decorators import _base as nl_base
    except Exception:  # noqa: BLE001 - private module: absence must not kill tracing
        logger.warning(
            "could not patch neatlogs._serialize_obj (SDK layout changed?); "
            "span serialization is unbounded again -- watch for CPU spins"
        )
        return False
    nl_base._serialize_obj = _bounded_serialize
    return True


# --- lifecycle -------------------------------------------------------------------------


def is_tracing_enabled() -> bool:
    """True once ``init_tracing`` has successfully initialised neatlogs."""
    return _enabled


def init_tracing(
    tags: list[str] | None = None,
    *,
    _init_kwargs: dict[str, Any] | None = None,
) -> bool:
    """Initialise neatlogs exactly once. Returns False (and warns) when the key is unset.

    ``_init_kwargs`` is a test hook forwarded to ``neatlogs.init`` (e.g. a private
    tracer provider with export disabled).
    """
    global _enabled
    if _enabled:
        return True
    # via config, not os.environ: config owns .env loading and drops blank exported vars,
    # which otherwise shadow a filled-in .env and disable tracing with only this warning to
    # go on -- the console then sits on "waiting for your first trace" and nothing says why.
    from anneal import config

    api_key = (config.env("NEATLOGS_API_KEY") or "").strip()
    if not api_key:
        logger.warning("NEATLOGS_API_KEY is empty; tracing disabled, spans degrade to plain calls")
        return False
    import neatlogs

    _patch_neatlogs_serializer()
    workflow = (config.env("NEATLOGS_WORKFLOW") or "").strip() or "anneal"
    neatlogs.init(
        api_key=api_key,
        workflow_name=workflow,
        tags=list(tags or []),
        **(_init_kwargs or {}),
    )
    _enabled = True
    logger.info("neatlogs tracing initialised", extra={"workflow": workflow, "tags": tags})
    return True


def wrap_client(client: Any) -> Any:
    """Return ``neatlogs.wrap(client)`` when tracing is on, else the client unchanged."""
    if not _enabled:
        return client
    import neatlogs

    return neatlogs.wrap(client)


def flush(timeout_millis: int = 30_000) -> bool:
    """Flush pending spans. False when tracing is off."""
    if not _enabled:
        return False
    import neatlogs

    return bool(neatlogs.flush(timeout_millis=timeout_millis))


def shutdown(timeout_millis: int = 30_000) -> bool:
    """Flush and shut neatlogs down. False when tracing is off."""
    global _enabled
    if not _enabled:
        return False
    import neatlogs

    _enabled = False
    return bool(neatlogs.shutdown(timeout_millis=timeout_millis))


def _reset_for_tests() -> None:
    """Forget the enabled flag without touching the SDK (tests shut the SDK down)."""
    global _enabled
    _enabled = False


# --- current span helpers ------------------------------------------------------------


def _active_span() -> Any | None:
    """The recording neatlogs span for this context, or None."""
    if not _enabled:
        return None
    from opentelemetry import trace as otel_trace

    try:
        from neatlogs._wrap_utils import active_neatlogs_context
    except ImportError:  # private path moved in a newer SDK: fall back to the OTel span
        span = otel_trace.get_current_span()
    else:
        ctx = active_neatlogs_context()
        span = otel_trace.get_current_span(ctx) if ctx is not None else None
    return span if span is not None and span.is_recording() else None


def current_trace_id() -> str | None:
    """32-char hex trace id of the active neatlogs span, or None when not in a span.

    The neatlogs SDK has no public trace-id accessor; this reads it via OpenTelemetry.
    """
    span = _active_span()
    if span is None:
        return None
    trace_id = span.get_span_context().trace_id
    return format(trace_id, "032x") if trace_id else None


def _attach_run_context(base_tags: list[str]) -> None:
    """Stamp run-context attributes and tags on the active span."""
    span = _active_span()
    if span is None:
        return
    ctx = get_run_context()
    for key, value in ctx.as_attributes().items():
        span.set_attribute(key, value)
    span.set_attribute("tag.tags", [*base_tags, *ctx.as_tags()])


# --- decorators --------------------------------------------------------------------------


def _with_run_context[F: Callable[..., Any]](func: F, base_tags: list[str]) -> F:
    """Inner wrapper: attach run context to the active span, then call ``func``."""
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_inner(*args: Any, **kwargs: Any) -> Any:
            _attach_run_context(base_tags)
            return await func(*args, **kwargs)

        return async_inner  # type: ignore[return-value]

    @functools.wraps(func)
    def inner(*args: Any, **kwargs: Any) -> Any:
        _attach_run_context(base_tags)
        return func(*args, **kwargs)

    return inner  # type: ignore[return-value]


def _traced_via_span[F: Callable[..., Any]](func: F, name: str, kind: str, tags: list[str]) -> F:
    """Build the traced variant with ``neatlogs.span``; built lazily on first traced call."""
    import neatlogs

    return neatlogs.span(kind=kind, name=name, tags=tags)(_with_run_context(func, tags))


def _traced_via_trace[F: Callable[..., Any]](func: F, name: str, kind: str, tags: list[str]) -> F:
    """Build the traced variant with ``neatlogs.trace`` for kinds ``span`` rejects."""
    import neatlogs

    inner = _with_run_context(func, tags)
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_traced(*args: Any, **kwargs: Any) -> Any:
            with neatlogs.trace(name, kind=kind):
                return await inner(*args, **kwargs)

        return async_traced  # type: ignore[return-value]

    @functools.wraps(func)
    def traced(*args: Any, **kwargs: Any) -> Any:
        with neatlogs.trace(name, kind=kind):
            return inner(*args, **kwargs)

    return traced  # type: ignore[return-value]


def _make_decorator(name: str, kind: str, tags: list[str] | None, via_trace: bool) -> Any:
    """Decorator that dispatches to the traced variant only when tracing is on."""
    base_tags = list(tags or [])
    builder = _traced_via_trace if via_trace else _traced_via_span

    def decorate[F: Callable[..., Any]](func: F) -> F:
        traced: list[Callable[..., Any]] = []

        def get_traced() -> Callable[..., Any]:
            if not traced:
                traced.append(builder(func, name, kind, base_tags))
            return traced[0]

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not _enabled:
                    return await func(*args, **kwargs)
                return await get_traced()(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _enabled:
                return func(*args, **kwargs)
            return get_traced()(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorate


def node_span(name: str, tags: list[str] | None = None) -> Callable[[Callable[..., Any]], Any]:
    """Trace a harness node (planner, executor, critic, router...) as an AGENT span."""
    return _make_decorator(name, "AGENT", tags, via_trace=False)


def tool_span(name: str, tags: list[str] | None = None) -> Callable[[Callable[..., Any]], Any]:
    """Trace a tool invocation as a TOOL span."""
    return _make_decorator(name, "TOOL", tags, via_trace=False)


def mcp_span(name: str, tags: list[str] | None = None) -> Callable[[Callable[..., Any]], Any]:
    """Trace a call to a third-party MCP server tool as an MCP_TOOL span."""
    return _make_decorator(name, "MCP_TOOL", tags, via_trace=False)


def llm_span(name: str, tags: list[str] | None = None) -> Callable[[Callable[..., Any]], Any]:
    """Trace an LLM call site as an LLM span (parent of the wrapped-client request span)."""
    return _make_decorator(name, "LLM", tags, via_trace=True)


# --- smoke ----------------------------------------------------------------------------------


def _smoke() -> int:
    """Emit one traced span, or report 'ready pending key' when the key is empty."""
    with contextlib.suppress(ImportError):
        from anneal import config  # noqa: F401  imported for its .env-loading side effect

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not init_tracing(tags=["smoke"]):
        print("ready pending key: set NEATLOGS_API_KEY in .env then run:")
        print("  uv run python -m anneal.tracing")
        return 0

    @node_span("smoke.node")
    def node() -> str:
        return f"trace_id={current_trace_id()}"

    with run_context(candidate_id="smoke", iteration=0, domain="smoke", split="search"):
        print(node())
    flushed = flush()
    import neatlogs

    print(f"flushed={flushed} diagnostics={neatlogs.get_delivery_diagnostics()}")
    shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(_smoke())
