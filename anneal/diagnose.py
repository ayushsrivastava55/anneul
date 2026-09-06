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
from anneal.mcp import decode_response as _decode_mcp_response
from anneal.mcp import mcp_content as _mcp_content
from anneal.tracing import llm_span, tool_span

logger = logging.getLogger("anneal.diagnose")

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "specs" / "failure_taxonomy.yaml"
MCP_URL = "https://ingest.neatlogs.com/mcp"
SEARCH_SPLIT = "search"
# Classification is the reasoning step the whole improvement loop turns on: a misread failure
# selects the wrong operator, the gate correctly rejects the result, and the loop learns
# nothing. Running it on the weakest tier was a design error -- in a live run the cheap tier
# returned an unusable class repeatedly and fell back to a deterministic guess. Diagnosis is
# one call per failed task, so the strongest tier is the right trade even under a budget.
# Override with ANNEAL_CLASSIFIER_TIER when a domain proves a cheaper tier is sufficient.
CLASSIFIER_TIER = os.environ.get("ANNEAL_CLASSIFIER_TIER") or "frontier"
MAX_CONTEXT_CHARS = 6000

# --- diagnosis confidence ---------------------------------------------------------------
# Published failure-attribution accuracy for LLM classifiers of this kind is 14-48%, so a
# diagnosis is a hypothesis, not a fact. Confidence travels with it and gates whether we act.

CERTAIN_CONFIDENCE = 1.0
"""Deterministic classes. A forbidden tool call or a schema violation is observed from the
row by rule, not inferred by a model, so there is nothing to be unsure about."""

FALLBACK_CONFIDENCE = 0.2
"""The classifier reply was unusable and ``fallback_class`` guessed from heuristics. The
ledger already recorded this as ``fallback`` and then nothing used it; a guessed class is
precisely a low-confidence diagnosis, so it now scores as one."""

DEFAULT_CONFIDENCE = 0.3
"""Ranking weight for an issue whose confidence was never reported. Ordering only.

It does NOT block, and that distinction is the whole design: a reply that parsed cleanly but
omitted the field tells us nothing about certainty, which is not the same as telling us
certainty is low. Blocking on absence would mean any model that ignores the field silently
stops the optimiser dead -- a much worse failure than trying a fix on an unscored diagnosis.
Reported-low blocks; unknown proceeds and says so in the log."""

CONFIDENCE_FLOOR = 0.5
"""Mean confidence an issue needs before Mutate will spend an operator on it, once
confidence has actually been reported. Below this the issue stays open and ranked but is
skipped in favour of the next one, because applying a typed fix to a misdiagnosed failure
costs a full gate cycle and teaches us nothing."""

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
    ledger: list[Issue],
    cls: str,
    node: str,
    evidence: Iterable[str],
    *,
    fallback: bool = False,
    confidence: float | None = None,
) -> Issue:
    """Increment the ``(class, node)`` issue (creating it if needed) and dedupe evidence.

    ``fallback=True`` marks the issue as holding at least one row whose class came from the
    deterministic fallback rather than a usable classifier reply. It is sticky once set.

    ``confidence`` accumulates as a running mean over the rows folded into the issue, kept
    with its own sample count (``confidence_n``) so repeated diagnoses of the same failure
    converge instead of the last row overwriting everything before it. One shaky diagnosis
    among nine solid ones should barely move the issue; nine shaky ones should sink it. A row
    that reported no confidence contributes nothing to the mean and leaves ``confidence_n``
    alone, so ``confidence_n == 0`` means "never scored" rather than "scored zero".
    """
    for issue in ledger:
        if issue["class"] == cls and issue["node"] == node:
            issue["count"] += 1
            issue["evidence"].extend(e for e in evidence if e not in issue["evidence"])
            issue["fallback"] = bool(issue.get("fallback")) or fallback
            if confidence is not None:
                seen = int(issue.get("confidence_n", 0))
                prior = float(issue.get("confidence", 0.0))
                issue["confidence"] = (prior * seen + confidence) / (seen + 1)
                issue["confidence_n"] = seen + 1
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
        "confidence": DEFAULT_CONFIDENCE if confidence is None else confidence,
        "confidence_n": 0 if confidence is None else 1,
    }
    ledger.append(issue)
    return issue


def confidence_of(issue: Issue) -> float:
    """Mean reported diagnosis confidence, or ``DEFAULT_CONFIDENCE`` when never reported."""
    if not int(issue.get("confidence_n", 0)):
        return DEFAULT_CONFIDENCE
    return float(issue.get("confidence", DEFAULT_CONFIDENCE))


def confidence_known(issue: Issue) -> bool:
    """Whether any row folded into ``issue`` actually reported a confidence."""
    return bool(int(issue.get("confidence_n", 0)))


def actionable(issue: Issue) -> bool:
    """Whether to spend an operator on this issue.

    Refusing to fix a diagnosis we do not believe is the point of scoring confidence: a typed
    fix aimed at a misdiagnosed failure costs a whole gate cycle and teaches us nothing, and
    if it happens to pass the gate we have promoted a change for a reason that was not real.

    Only *reported* low confidence blocks. An issue nothing ever scored is unknown, not
    doubted, and proceeds -- otherwise a classifier that ignores the field would stop the
    optimiser dead while looking like a principled refusal.
    """
    return not confidence_known(issue) or confidence_of(issue) >= CONFIDENCE_FLOOR


def rank(issues: Iterable[Issue], taxonomy: dict[str, dict[str, Any]] | None = None) -> list[Issue]:
    """Open issues, highest ``count * severity * confidence`` first (ties: lower id first).

    Confidence is a factor, not a filter, so a frequent severe failure we are unsure about
    still outranks a rare mild one we are certain of -- it just has to be believed more
    before it outranks an equally common failure we understand.
    """
    tax = taxonomy or load_taxonomy()

    def weight(issue: Issue) -> float:
        severity_ = float(tax.get(issue["class"], {}).get("severity", 1))
        return issue["count"] * severity_ * confidence_of(issue)

    open_issues = [i for i in issues if i.get("status", "open") == "open"]
    return sorted(open_issues, key=lambda i: (-weight(i), i["id"]))


def ledger_stats(ledger: Iterable[Issue]) -> dict[str, Any]:
    """Compact snapshot of the ledger for per-iteration summaries.

    This is the 'memory growing over iterations' number: every iteration's summary.json
    carries one, so the growth of what the optimiser knows (issues seen, observations
    accumulated, repairs attempted) is a plottable series, not a claim.
    """
    issues = list(ledger)
    by_class: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for issue in issues:
        by_class[issue["class"]] = by_class.get(issue["class"], 0) + 1
        status = issue.get("status", "open")
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "issues": len(issues),
        "observations": sum(int(i.get("count", 1)) for i in issues),
        "operators_tried": sum(len(i.get("operators_tried") or ()) for i in issues),
        "by_class": dict(sorted(by_class.items())),
        "by_status": dict(sorted(by_status.items())),
    }


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
    """Chat messages asking for ``{"class": ..., "node": ..., "confidence": ...}``."""
    class_lines = "\n".join(
        f"- {cid}: {spec.get('description', '')}" for cid, spec in classes.items()
    )
    system = (
        "You classify why an LLM agent failed a task. Reply with a single JSON object "
        '{"class": <one class id>, "node": <one node name>, "confidence": <0.0-1.0>} and '
        "nothing else. Use only the class ids listed below; never invent one.\n"
        "confidence is how sure you are of the class, given the evidence you were shown. "
        "Be honest and use the low end: if the trace does not show why the task failed, say "
        "so with a low number. A low score is useful -- it stops us applying a fix for a "
        "problem you are guessing at. Do not default to a high number.\n"
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
def _parse_confidence(raw: Any) -> float | None:
    """Clamp a classifier-reported confidence into [0, 1]. None when it reported none.

    None means "unknown", not "low": a junk or absent value must neither become 1.0 by
    accident (letting the least trustworthy replies carry the most weight) nor be treated as
    an explicit refusal. See ``DEFAULT_CONFIDENCE``.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    return max(0.0, min(1.0, value))


def classify_with_llm(
    messages: list[dict[str, Any]],
    allowed: list[str],
    nodes: list[str],
    client: Any | None = None,
) -> tuple[str | None, str | None, float | None]:
    """Ask the ``cheap`` tier via the gateway; ``client`` overrides it (tests inject a fake).

    Returns ``(class, node, confidence)``, confidence None when the reply reported none. The
    class is None -- never an invented id -- when the reply is unparseable or names something
    outside ``allowed``; the caller then picks the fallback. Published attribution accuracy
    for this kind of classifier runs 14-48%, so confidence travels with the diagnosis and
    decides whether we act on it.
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
    return cls, str(node) if node in nodes else None, _parse_confidence(parsed.get("confidence"))


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
) -> tuple[str, str, bool, float | None]:
    """``(class, node, fell_back, confidence)`` for one failed row, confidence None if unreported.

    Deterministic checks skip the model and are certain by construction -- except a step-budget
    death. Hitting the budget is a *symptom*: an agent that reaches for a tool the spec does not
    grant (no way to run tests, no way to fetch a receipt) loops until the budget kills it, and
    repairing the loop (more steps, another topology) leaves the real gap in place. So when the
    missing-capability signal fires the row is classed as ``missing_capability``, and when it
    does not, the model is asked for the cause with ``loop_or_timeout`` still on the menu.
    """
    cls = deterministic_class(row)
    budget_death = cls == "loop_or_timeout"
    if cls is not None and not budget_death:
        return cls, default_node(row, spec), False, CERTAIN_CONFIDENCE
    if budget_death and missing_capability_signal(row, spec):
        return "missing_capability", default_node(row, spec), False, CERTAIN_CONFIDENCE
    nodes = _node_names(spec)
    allowed = llm_classes(taxonomy)
    guesses = heuristic_guesses(row, spec, taxonomy)
    if budget_death:
        allowed = [*allowed, "loop_or_timeout"]
        guesses = [*guesses, "loop_or_timeout"]
    context = traces.get_trace_context(str(row.get("trace_id") or row.get("task_id")))
    classes = {cid: taxonomy[cid] for cid in allowed}
    messages = build_prompt(row, task_input, context, classes, nodes, guesses)
    cls, node, confidence = classify_with_llm(messages, allowed, nodes, client)
    fell_back = cls is None
    if cls is None:
        cls = "loop_or_timeout" if budget_death else fallback_class(guesses, allowed, taxonomy)
        confidence = FALLBACK_CONFIDENCE
        logger.warning("classifier fell back", extra={"class": cls, "guesses": guesses})
    return cls, node or default_node(row, spec), fell_back, confidence


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
        cls, node, fell_back, confidence = _classify_row(
            row, task_input, spec, source, taxonomy, client
        )
        evidence = [str(row.get("trace_id") or task_id)]
        issue = upsert(ledger, cls, node, evidence, fallback=fell_back, confidence=confidence)
        touched[issue["id"]] = issue
        logger.info(
            "diagnosed",
            extra={"task_id": task_id, "class": cls, "node": node, "fallback": fell_back},
        )
    save_ledger(ledger_path, ledger)
    return list(touched.values())
