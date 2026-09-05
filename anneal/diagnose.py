"""Diagnose: classify failed tasks into the failure taxonomy and upsert the ledger.

For every failed row (``score < THRESHOLD`` or ``hard_fail``):

1. deterministic pre-checks: ``hard_fail`` -> ``unsafe_action``, ``schema_error`` ->
   ``output_format``, ``hit_step_budget`` -> ``loop_or_timeout``;
2. otherwise an LLM classifier (tier ``cheap`` via :mod:`anneal.llm`; an injected ``client``
   such as ``tests.fakes.FakeClient`` keeps tests offline)
   constrained to the ``detect: llm`` class ids in ``specs/failure_taxonomy.yaml``, given the
   row output, the task input and the trace context.

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

from anneal import llm
from anneal.tracing import llm_span, tool_span

logger = logging.getLogger("anneal.diagnose")

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "specs" / "failure_taxonomy.yaml"
MCP_URL = "https://ingest.neatlogs.com/mcp"
SEARCH_SPLIT = "search"
CLASSIFIER_TIER = "cheap"
MAX_CONTEXT_CHARS = 6000

Issue = dict[str, Any]
Row = dict[str, Any]

# (row flag, class id) in priority order. The first flag that is set wins.
DETERMINISTIC_CHECKS: tuple[tuple[str, str], ...] = (
    ("hard_fail", "unsafe_action"),
    ("schema_error", "output_format"),
    ("hit_step_budget", "loop_or_timeout"),
)


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


def upsert(ledger: list[Issue], cls: str, node: str, evidence: Iterable[str]) -> Issue:
    """Increment the ``(class, node)`` issue (creating it if needed) and dedupe evidence."""
    for issue in ledger:
        if issue["class"] == cls and issue["node"] == node:
            issue["count"] += 1
            issue["evidence"].extend(e for e in evidence if e not in issue["evidence"])
            return issue
    issue: Issue = {
        "id": _next_id(ledger),
        "class": cls,
        "node": node,
        "count": 1,
        "evidence": list(dict.fromkeys(evidence)),
        "status": "open",
        "operators_tried": [],
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


def deterministic_class(row: Row) -> str | None:
    """Class id from run flags alone, or None when the LLM must decide."""
    for flag, cls in DETERMINISTIC_CHECKS:
        if row.get(flag):
            return cls
    return None


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
) -> list[dict[str, Any]]:
    """Chat messages asking the classifier for ``{"class": ..., "node": ...}``."""
    class_lines = "\n".join(
        f"- {cid}: {spec.get('description', '')}" for cid, spec in classes.items()
    )
    system = (
        "You classify why an LLM agent failed a task. Reply with a single JSON object "
        '{"class": <one class id>, "node": <one node name>} and nothing else.\n'
        f"Allowed class ids:\n{class_lines}\nAllowed nodes: {', '.join(nodes)}"
    )
    user = (
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
) -> tuple[str, str | None]:
    """Ask the ``cheap`` tier via the gateway; ``client`` overrides it (tests inject a fake).

    The reply is coerced to an allowed class; garbage falls back to the first allowed class.
    """
    reply, _usage = llm.chat(CLASSIFIER_TIER, messages, client=client, temperature=0)
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except ValueError:
            parsed = {}
    cls = str(parsed.get("class", "")).strip()
    if cls not in allowed:
        logger.warning("classifier returned unknown class", extra={"reply": reply[:200]})
        cls = allowed[0]
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
) -> tuple[str, str]:
    cls = deterministic_class(row)
    if cls is not None:
        return cls, default_node(row, spec)
    nodes = _node_names(spec)
    allowed = llm_classes(taxonomy)
    context = traces.get_trace_context(str(row.get("trace_id") or row.get("task_id")))
    classes = {cid: taxonomy[cid] for cid in allowed}
    cls, node = classify_with_llm(
        build_prompt(row, task_input, context, classes, nodes), allowed, nodes, client
    )
    return cls, node or default_node(row, spec)


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
        cls, node = _classify_row(row, task_input, spec, source, taxonomy, client)
        evidence = [str(row.get("trace_id") or task_id)]
        issue = upsert(ledger, cls, node, evidence)
        touched[issue["id"]] = issue
        logger.info("diagnosed", extra={"task_id": task_id, "class": cls, "node": node})
    save_ledger(ledger_path, ledger)
    return list(touched.values())
