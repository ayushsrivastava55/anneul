"""Diagnose: classify failed tasks into the failure taxonomy and upsert the ledger.

For every failed row (``score < THRESHOLD`` or ``hard_fail``):

1. deterministic pre-checks, in descending severity: ``hard_fail`` -> ``unsafe_action``,
   ``schema_error`` -> ``output_format``, ``hit_step_budget`` -> ``loop_or_timeout``, and a
   truncation error or an over-budget token total -> ``context_overflow``;
2. otherwise an LLM classifier (tier ``cheap`` via :mod:`anneal.llm`; an injected ``client``
   such as ``tests.fakes.FakeClient`` keeps tests offline)
   constrained to the ``detect: llm`` class ids in ``specs/failure_taxonomy.yaml``, given the
   row output, the task input, the trace context and the deterministic signals below.

The classifier can only ever return an id listed in the taxonomy. When its reply is
unparseable, names an id outside that list, or omits the class, we fall back to the
highest-severity deterministic guess -- a heuristic signal when one fired (``missing_capability``
for an action no tool provides or an identical failing tool call retried, ``context_overflow``
for a near-budget token total), else the highest-severity classifier class -- and record
``"fallback": true`` on the ledger entry so the ledger never hides a guess.

Trace context comes from a :class:`TraceSource`: :class:`NeatlogsMCP` (JSON-RPC over the
Neatlogs streamable-HTTP MCP endpoint, rate-limited to 60 req/min) when ``NEATLOGS_API_KEY``
is set, else :class:`LocalTraces`, which serves the ``trace`` recorded in the run rows.

Task input is only ever read from the ``search`` split. Rows for tasks outside that split are
skipped entirely, so this module can never see a gated task (CLAUDE.md rule 1).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

from anneal import config, llm
from anneal.tracing import llm_span, tool_span

logger = logging.getLogger("anneal.diagnose")

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "specs" / "failure_taxonomy.yaml"
MCP_URL = "https://ingest.neatlogs.com/mcp"
SEARCH_SPLIT = "search"
CLASSIFIER_TIER = "cheap"
MAX_CONTEXT_CHARS = 6000

# Row ``tokens_in``/``tokens_out`` are cumulative over every step of the task, not the peak
# prompt size, so this default is sized for a whole multi-step run rather than one window.
CONTEXT_TOKEN_LIMIT_ENV = "ANNEAL_CONTEXT_TOKEN_LIMIT"
DEFAULT_CONTEXT_TOKEN_LIMIT = 100_000
# Two identical failing calls to the same tool is already a retry loop, not a one-off blip.
REPEAT_RETRY_THRESHOLD = 2

TRUNCATION_RE = re.compile(
    r"context_length_exceeded|maximum context length|context window|"
    r"prompt is too long|too many tokens|reduce the length of the messages",
    re.IGNORECASE,
)
UNKNOWN_TOOL_RE = re.compile(r"unknown tool|no such tool|not a (?:known|valid) tool", re.IGNORECASE)
TOOL_ERROR_RE = re.compile(r"^\s*(?:error|exception|traceback|failed)\b", re.IGNORECASE)

Issue = dict[str, Any]
Row = dict[str, Any]

# (row flag, class id) in priority order. The first flag that is set wins.
DETERMINISTIC_CHECKS: tuple[tuple[str, str], ...] = (
    ("hard_fail", "unsafe_action"),
    ("schema_error", "output_format"),
    ("hit_step_budget", "loop_or_timeout"),
)
# Class ids this module decides without any model call. Kept next to the checks above so
# tests can assert the union with ``llm_classes`` covers the whole taxonomy.
DETERMINISTIC_CLASSES: tuple[str, ...] = (*(c for _, c in DETERMINISTIC_CHECKS), "context_overflow")


# --- taxonomy -----------------------------------------------------------------------------


def load_taxonomy(path: Path | str | None = None) -> dict[str, dict[str, Any]]:
    """Return ``{class_id: {severity, detect, description, operators}}`` from the taxonomy."""
    with open(Path(path) if path else TAXONOMY_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    classes = data.get("classes") if isinstance(data, dict) else None
    if not classes:
        raise ValueError(f"{path or TAXONOMY_PATH}: expected top-level 'classes'")
    return {str(c["id"]): {k: v for k, v in c.items() if k != "id"} for c in classes}


def llm_classes(taxonomy: dict[str, dict[str, Any]]) -> list[str]:
    """Class ids the LLM classifier may choose from (``detect: llm``)."""
    return [cid for cid, spec in taxonomy.items() if spec.get("detect") == "llm"]


def deterministic_classes() -> list[str]:
    """Class ids this module decides from run flags and trace shape alone."""
    return list(DETERMINISTIC_CLASSES)


def severity(taxonomy: dict[str, dict[str, Any]], cls: str) -> float:
    """Severity weight of ``cls``, or 0 when the taxonomy does not list it."""
    return float(taxonomy.get(cls, {}).get("severity", 0))


# --- trace sources ------------------------------------------------------------------------


class TraceSource(Protocol):
    """Where diagnose gets span-level context for a failed task."""

    def search_traces(self, tags: list[str]) -> list[str]:
        """Trace ids carrying every tag in ``tags``."""
        ...

    def get_trace_context(self, trace_id: str) -> dict[str, Any] | None:
        """Span tree / tool calls for one trace, or None when unknown."""
        ...


class LocalTraces:
    """Trace context from the run rows themselves (``trace`` and ``output`` fields)."""

    def __init__(self, rows: Iterable[Row]) -> None:
        self._by_trace: dict[str, Row] = {}
        for row in rows:
            key = row.get("trace_id") or row.get("task_id")
            if key:
                self._by_trace[str(key)] = row

    def search_traces(self, tags: list[str]) -> list[str]:
        wanted = {t.split(":", 1) for t in tags if ":" in t}
        return [
            tid
            for tid, row in self._by_trace.items()
            if all(str(row.get(k)) == v for k, v in wanted)
        ]

    def get_trace_context(self, trace_id: str) -> dict[str, Any] | None:
        row = self._by_trace.get(trace_id)
        if row is None:
            return None
        return {"trace": row.get("trace", []), "output": row.get("output")}


class NeatlogsMCP:
    """Minimal MCP client for ``https://ingest.neatlogs.com/mcp`` (streamable HTTP).

    One JSON-RPC ``tools/call`` per request, paced to at most 60 requests per minute.
    ``http`` is injectable so tests never touch the network.
    """

    def __init__(
        self,
        api_key: str,
        *,
        url: str = MCP_URL,
        http: Any | None = None,
        min_interval_s: float = 1.0,
        timeout_s: float = 30.0,
    ) -> None:
        self._url = url
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        self._http = http or httpx.Client(timeout=timeout_s)
        self._min_interval = min_interval_s
        self._last_call = 0.0
        self._next_id = 0
        self._initialised = False

    def _pace(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """One paced JSON-RPC request; echoes ``Mcp-Session-Id`` once the server issues it."""
        self._pace()
        self._next_id += 1
        body = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        response = self._http.post(self._url, json=body, headers=self._headers)
        response.raise_for_status()
        session = response.headers.get("mcp-session-id")
        if session:
            self._headers["Mcp-Session-Id"] = str(session)
        payload = _decode_mcp_response(response)
        if "error" in payload:
            raise RuntimeError(f"neatlogs mcp {method}: {payload['error']}")
        return payload

    def _initialise(self) -> None:
        """Streamable-HTTP handshake; a server that rejects it is treated as stateless."""
        params = {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "anneal.diagnose", "version": "0.1"},
        }
        try:
            self._rpc("initialize", params)
        except RuntimeError as exc:
            logger.info("mcp initialize rejected; continuing stateless", extra={"err": str(exc)})
        self._initialised = True

    @tool_span("neatlogs.mcp.call")
    def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Invoke one MCP tool and return its decoded content (JSON if possible)."""
        if not self._initialised:
            self._initialise()
        payload = self._rpc("tools/call", {"name": tool, "arguments": arguments})
        return _mcp_content(payload.get("result", {}))

    def search_traces(self, tags: list[str]) -> list[str]:
        result = self.call("search_traces", {"tags": tags})
        items = result.get("traces", result) if isinstance(result, dict) else result
        ids = [t.get("trace_id") or t.get("id") if isinstance(t, dict) else t for t in items or []]
        return [str(i) for i in ids if i]

    def get_trace_context(self, trace_id: str) -> dict[str, Any] | None:
        try:
            result = self.call("get_trace_context", {"trace_id": trace_id})
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            logger.warning(
                "trace context unavailable", extra={"trace_id": trace_id, "err": str(exc)}
            )
            return None
        return result if isinstance(result, dict) else {"context": result}


def _decode_mcp_response(response: Any) -> dict[str, Any]:
    """Parse a JSON or SSE (``data:`` lines) MCP response body into the JSON-RPC envelope."""
    content_type = str(response.headers.get("content-type", ""))
    if "text/event-stream" not in content_type:
        return response.json()
    last: dict[str, Any] = {}
    for line in response.text.splitlines():
        if line.startswith("data:"):
            last = json.loads(line[5:].strip() or "{}")
    return last


def _mcp_content(result: dict[str, Any]) -> Any:
    """Flatten MCP ``content`` blocks; decode the text as JSON when it is JSON."""
    if "structuredContent" in result:
        return result["structuredContent"]
    texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
    joined = "\n".join(texts)
    try:
        return json.loads(joined)
    except ValueError:
        return joined


def default_trace_source(rows: Iterable[Row]) -> TraceSource:
    """``NeatlogsMCP`` when ``NEATLOGS_API_KEY`` is set, else ``LocalTraces`` over ``rows``."""
    key = (os.environ.get("NEATLOGS_API_KEY") or "").strip()
    return NeatlogsMCP(key) if key else LocalTraces(rows)


# --- ledger -------------------------------------------------------------------------------


def load_ledger(path: Path | str) -> list[Issue]:
    """Read ``ledger.json``; a missing or empty file is an empty ledger."""
    p = Path(path)
    if not p.exists() or not p.read_text(encoding="utf-8").strip():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{p}: ledger must be a JSON list")
    return data


def save_ledger(path: Path | str, ledger: list[Issue]) -> None:
    """Write the ledger atomically (tmp file + rename)."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    tmp.replace(p)


def _next_id(ledger: list[Issue]) -> str:
    highest = 0
    for issue in ledger:
        match = re.fullmatch(r"L-(\d+)", str(issue.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"L-{highest + 1:04d}"


def upsert(
    ledger: list[Issue], cls: str, node: str, evidence: Iterable[str], *, fallback: bool = False
) -> Issue:
    """Increment the ``(class, node)`` issue (creating it if needed) and dedupe evidence.

    ``fallback=True`` marks the issue as holding at least one row whose class came from the
    deterministic fallback rather than a usable classifier reply. It is sticky once set.
    """
    for issue in ledger:
        if issue["class"] == cls and issue["node"] == node:
            issue["count"] += 1
            issue["evidence"].extend(e for e in evidence if e not in issue["evidence"])
            issue["fallback"] = bool(issue.get("fallback")) or fallback
            return issue
    issue: Issue = {
        "id": _next_id(ledger),
        "class": cls,
        "node": node,
        "count": 1,
        "evidence": list(dict.fromkeys(evidence)),
        "status": "open",
        "operators_tried": [],
        "fallback": fallback,
    }
    ledger.append(issue)
    return issue


def rank(issues: Iterable[Issue], taxonomy: dict[str, dict[str, Any]] | None = None) -> list[Issue]:
    """Open issues, highest ``count * severity`` first (ties: lower id first)."""
    tax = taxonomy or load_taxonomy()

    def weight(issue: Issue) -> float:
        return issue["count"] * float(tax.get(issue["class"], {}).get("severity", 1))

    open_issues = [i for i in issues if i.get("status", "open") == "open"]
    return sorted(open_issues, key=lambda i: (-weight(i), i["id"]))


# --- classification -----------------------------------------------------------------------


def is_failure(row: Row, threshold: float) -> bool:
    """A row fails when it is a hard fail or scores below the domain threshold."""
    return bool(row.get("hard_fail")) or float(row.get("score", 0.0)) < threshold


def context_token_limit() -> int:
    """Cumulative-token ceiling above which a failed run counts as ``context_overflow``.

    Read from ``$ANNEAL_CONTEXT_TOKEN_LIMIT`` at call time; a missing or unparseable value
    falls back to :data:`DEFAULT_CONTEXT_TOKEN_LIMIT`.
    """
    raw = config.env(CONTEXT_TOKEN_LIMIT_ENV)
    try:
        limit = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_TOKEN_LIMIT
    return limit if limit > 0 else DEFAULT_CONTEXT_TOKEN_LIMIT


def _as_text(value: Any) -> str:
    """Best-effort string view of a trace result / output for pattern matching."""
    return value if isinstance(value, str) else json.dumps(value, default=str)


def _trace_steps(row: Row) -> list[dict[str, Any]]:
    return [step for step in (row.get("trace") or []) if isinstance(step, dict)]


def _row_texts(row: Row) -> list[str]:
    """Every free-text surface of a row an error can show up in."""
    texts = [_as_text(row.get("output")), _as_text(row.get("error") or "")]
    texts.extend(_as_text(step.get("result")) for step in _trace_steps(row))
    return texts


def context_overflow(row: Row, *, token_limit: int | None = None) -> bool:
    """True when the run hit a truncation error or burned more tokens than the ceiling."""
    if any(TRUNCATION_RE.search(text) for text in _row_texts(row)):
        return True
    limit = context_token_limit() if token_limit is None else token_limit
    total = int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)
    return total > limit


def _spec_tools(spec: Any) -> set[str]:
    names: set[str] = set()
    for node in getattr(spec, "nodes", None) or []:
        names.update(str(t) for t in (getattr(node, "tools", None) or []))
    return names


def missing_capability_signal(row: Row, spec: Any) -> bool:
    """True when the agent reached for an action no tool provides.

    Either it called a tool the spec does not grant (the runtime answers
    ``Error: unknown tool ...``), or it retried one identical call that kept erroring.
    """
    known = _spec_tools(spec)
    attempts: dict[tuple[str, str], int] = {}
    for step in _trace_steps(row):
        name = str(step.get("tool", ""))
        result = _as_text(step.get("result"))
        if UNKNOWN_TOOL_RE.search(result) or (known and name and name not in known):
            return True
        if TOOL_ERROR_RE.search(result):
            key = (name, json.dumps(step.get("args"), sort_keys=True, default=str))
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] >= REPEAT_RETRY_THRESHOLD:
                return True
    return False


def deterministic_class(row: Row, *, token_limit: int | None = None) -> str | None:
    """Class id from run flags and trace shape alone, or None when the LLM must decide."""
    for flag, cls in DETERMINISTIC_CHECKS:
        if row.get(flag):
            return cls
    if context_overflow(row, token_limit=token_limit):
        return "context_overflow"
    return None


def heuristic_guesses(row: Row, spec: Any, taxonomy: dict[str, dict[str, Any]]) -> list[str]:
    """Classifier classes the deterministic signals point at, highest severity first."""
    guesses = [
        cls
        for cls, fired in (("missing_capability", missing_capability_signal(row, spec)),)
        if fired and cls in taxonomy
    ]
    return sorted(guesses, key=lambda c: -severity(taxonomy, c))


def fallback_class(
    guesses: list[str], allowed: list[str], taxonomy: dict[str, dict[str, Any]]
) -> str:
    """Safe class for an unusable classifier reply: never an id outside the taxonomy."""
    if guesses:
        return guesses[0]
    return max(allowed, key=lambda c: severity(taxonomy, c))


def _node_names(spec: Any) -> list[str]:
    nodes = getattr(spec, "nodes", None) or []
    return [n.name if hasattr(n, "name") else str(n) for n in nodes]


def default_node(row: Row, spec: Any) -> str:
    """Node blamed for a deterministic failure: the last tool-calling node, else the executor."""
    names = _node_names(spec)
    trace = row.get("trace") or []
    for step in reversed(trace):
        node = step.get("node") if isinstance(step, dict) else None
        if node in names:
            return str(node)
    for n in getattr(spec, "nodes", None) or []:
        if getattr(n, "tools", None):
            return str(n.name)
    return names[0] if names else "unknown"


def _truncate(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= MAX_CONTEXT_CHARS else text[:MAX_CONTEXT_CHARS] + "...<truncated>"


def build_prompt(
    row: Row,
    task_input: Any,
    context: dict[str, Any] | None,
    classes: dict[str, dict[str, Any]],
    nodes: list[str],
    guesses: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Chat messages asking the classifier for ``{"class": ..., "node": ...}``."""
    class_lines = "\n".join(
        f"- {cid}: {spec.get('description', '')}" for cid, spec in classes.items()
    )
    system = (
        "You classify why an LLM agent failed a task. Reply with a single JSON object "
        '{"class": <one class id>, "node": <one node name>} and nothing else. Use only the '
        "class ids listed below; never invent one.\n"
        f"Allowed class ids:\n{class_lines}\nAllowed nodes: {', '.join(nodes)}"
    )
    signals = (
        f"Deterministic signals suggest: {', '.join(guesses)}. "
        "Weigh them against the evidence; they are hints, not the answer.\n\n"
        if guesses
        else ""
    )
    user = (
        f"{signals}"
        f"Task input:\n{_truncate(task_input)}\n\n"
        f"Agent output:\n{_truncate(row.get('output'))}\n\n"
        f"Trace context:\n{_truncate(context or {})}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@llm_span("diagnose.classify")
def classify_with_llm(
    messages: list[dict[str, Any]],
    allowed: list[str],
    nodes: list[str],
    client: Any | None = None,
) -> tuple[str | None, str | None]:
    """Ask the ``cheap`` tier via the gateway; ``client`` overrides it (tests inject a fake).

    Returns ``(class, node)``. The class is None -- never an invented id -- when the reply is
    unparseable or names something outside ``allowed``; the caller then picks the fallback.
    """
    reply, _usage = llm.chat(CLASSIFIER_TIER, messages, client=client, temperature=0)
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            parsed = {}
    cls: str | None = str(parsed.get("class", "")).strip()
    if cls not in allowed:
        logger.warning("classifier returned unusable class", extra={"reply": reply[:200]})
        cls = None
    node = parsed.get("node")
    return cls, str(node) if node in nodes else None


# --- entry point --------------------------------------------------------------------------


def _search_inputs(domain: Any) -> dict[str, Any] | None:
    """``{task_id: input}`` for the search split, or None when the domain has no loader."""
    loader = getattr(getattr(domain, "eval", None), "load_tasks", None)
    if loader is None:
        return None
    out: dict[str, Any] = {}
    for task in loader(SEARCH_SPLIT):
        tid = task.get("id") if isinstance(task, dict) else getattr(task, "id", None)
        inp = task.get("input") if isinstance(task, dict) else getattr(task, "input", None)
        if tid is not None:
            out[str(tid)] = inp
    return out


def _classify_row(
    row: Row,
    task_input: Any,
    spec: Any,
    traces: TraceSource,
    taxonomy: dict[str, dict[str, Any]],
    client: Any | None,
) -> tuple[str, str, bool]:
    """``(class, node, fell_back)`` for one failed row. Deterministic checks skip the model."""
    cls = deterministic_class(row)
    if cls is not None:
        return cls, default_node(row, spec), False
    nodes = _node_names(spec)
    allowed = llm_classes(taxonomy)
    guesses = heuristic_guesses(row, spec, taxonomy)
    context = traces.get_trace_context(str(row.get("trace_id") or row.get("task_id")))
    classes = {cid: taxonomy[cid] for cid in allowed}
    messages = build_prompt(row, task_input, context, classes, nodes, guesses)
    cls, node = classify_with_llm(messages, allowed, nodes, client)
    fell_back = cls is None
    if cls is None:
        cls = fallback_class(guesses, allowed, taxonomy)
        logger.warning("classifier fell back", extra={"class": cls, "guesses": guesses})
    return cls, node or default_node(row, spec), fell_back


def diagnose(
    rows: Iterable[Row],
    domain: Any,
    spec: Any,
    *,
    ledger_path: Path | str = "ledger.json",
    traces: TraceSource | None = None,
    client: Any | None = None,
    taxonomy_path: Path | str | None = None,
) -> list[Issue]:
    """Classify every failed row, upsert the ledger on disk and return the touched issues."""
    rows = list(rows)
    taxonomy = load_taxonomy(taxonomy_path)
    threshold = float(getattr(getattr(domain, "eval", None), "THRESHOLD", 1.0))
    inputs = _search_inputs(domain)
    source = traces or default_trace_source(rows)
    ledger = load_ledger(ledger_path)
    touched: dict[str, Issue] = {}
    for row in rows:
        task_id = str(row.get("task_id"))
        if inputs is not None and task_id not in inputs:
            logger.info("skipping row outside search split", extra={"task_id": task_id})
            continue
        if not is_failure(row, threshold):
            continue
        task_input = inputs.get(task_id) if inputs is not None else None
        cls, node, fell_back = _classify_row(row, task_input, spec, source, taxonomy, client)
        evidence = [str(row.get("trace_id") or task_id)]
        issue = upsert(ledger, cls, node, evidence, fallback=fell_back)
        touched[issue["id"]] = issue
        logger.info(
            "diagnosed",
            extra={"task_id": task_id, "class": cls, "node": node, "fallback": fell_back},
        )
    save_ledger(ledger_path, ledger)
    return list(touched.values())
