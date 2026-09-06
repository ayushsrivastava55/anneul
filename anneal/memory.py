"""Episodic memory: contextual rules the agent learns from its own failed traces.

The loop writes here, the runtime reads here, and the store survives the process:

- ``reflect(rows, domain, client=...)`` looks at the tasks a run FAILED, together with the
  tool results those runs actually saw, and asks the model (cheap tier, through
  :mod:`anneal.llm`) for one to three imperative rules that would have changed the outcome
  -- e.g. *"basic economy reservations cannot be modified; check the cabin with
  get_reservation_details before offering a change"*. Each rule cites the tool result that
  justifies it. New rules are deduplicated against the store by token overlap.
- ``recall(task, k)`` returns the top ``k`` active entries for a task by keyword/tool
  overlap. It is deterministic (ties break on entry id) and records what it handed out so
  the loop can credit it later.
- ``credit(entries, passed)`` / ``credit_run(rows, threshold)`` move ``hits``/``wins``/
  ``losses``, and ``prune()`` retires entries whose losses outgrow their wins after a
  minimum sample. Memory that cannot unlearn is just a growing prompt.

The store is JSON at ``runs/<domain>/memory.json`` (the path is injectable; tests pass a
tmp path). Nothing domain-specific lives here: a rule is text the model wrote about the
tool results of one domain, carried in a ``domain``-tagged entry.

The active store for a run is a contextvar (``activate``/``active``) so the runtime can
reach it without a signature change and so ``runner``'s thread pool -- which copies the
caller's context -- shares one instance. Nothing is auto-created from disk inside the
runtime: with no active memory, the runtime behaves exactly as it did before.

Only the search split is ever reflected on; the reserved evaluation split is the gate's
business and never reaches this module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from anneal import llm, tracing

logger = logging.getLogger("anneal.memory")

#: Model tier used for reflection (never a model id -- those live in specs/models.yaml).
REFLECT_TIER = "cheap"
#: Heading the runtime injects the recalled rules under.
MEMORY_HEADING = "Learned from previous runs"
#: Most rules one reflection may add.
MAX_NEW_RULES = 3
#: Failed tasks shown to the reflector in one prompt.
MAX_FAILURES = 5
#: Characters of one tool result kept in the reflection prompt.
RESULT_CHARS = 400
#: Token-overlap (Jaccard) at or above which two rules are considered the same rule.
DEDUPE_THRESHOLD = 0.6
#: Injections an entry needs before ``prune`` is allowed to judge it.
MIN_SAMPLE = 3

_STOPWORDS = frozenset(
    """a an and are as at be before but by can cannot do does for from has have if in into is it
    its must never no not of on only or should that the their then there these they this to use
    used using was were what when which while who will with without you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

REFLECT_SYSTEM = (
    "You study an agent's failed runs and write down what it should have known.\n"
    f"Reply with a JSON array of at most {MAX_NEW_RULES} objects, each "
    '{"text": ..., "tool": ..., "evidence": ...}.\n'
    "- text: ONE imperative, specific, reusable rule, grounded in what a tool result "
    "actually showed. Good: 'basic economy reservations cannot be modified; call "
    "get_reservation_details and check the cabin before offering a change'. "
    "Bad: 'be more careful'.\n"
    "- tool: the tool whose result justifies the rule, or null.\n"
    "- evidence: the exact tool result fragment that justifies it.\n"
    "Write no rule you cannot ground in a tool result below. Fewer rules is better than "
    "vague rules. Reply with the JSON array and nothing else."
)


# --- entries -----------------------------------------------------------------------------


@dataclass
class Entry:
    """One learned rule and its track record."""

    id: str
    text: str
    domain: str
    source_task_ids: list[str] = field(default_factory=list)
    source_trace_ids: list[str] = field(default_factory=list)
    tool: str | None = None
    evidence: str | None = None
    created_iteration: int = 0
    hits: int = 0
    wins: int = 0
    losses: int = 0
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entry:
        known = {f: data.get(f) for f in cls.__dataclass_fields__ if f in data}
        known.setdefault("id", uuid.uuid4().hex[:12])
        return cls(**known)  # type: ignore[arg-type]


def tokens(text: str) -> set[str]:
    """Lowercased word tokens with stopwords and one/two-letter noise removed."""
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 2 and t not in _STOPWORDS}


def overlap(left: set[str], right: set[str]) -> float:
    """Jaccard overlap of two token sets (0.0 when either is empty)."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def task_id_of(task: Any) -> str:
    """``task.id`` / ``task["id"]`` as a string, else the task's text."""
    for get in (lambda: task.id, lambda: task["id"]):
        with contextlib.suppress(AttributeError, TypeError, KeyError, IndexError):
            return str(get())
    return task_text(task)


def task_text(task: Any) -> str:
    """Best-effort text of a task: its ``input`` when it has one, else the whole task."""
    payload = getattr(task, "input", None)
    if payload is None and isinstance(task, dict):
        payload = task.get("input", task)
    if payload is None:
        payload = task
    return payload if isinstance(payload, str) else json.dumps(payload, default=str)


# --- the store ---------------------------------------------------------------------------


class Memory:
    """A persistent list of learned rules plus the bookkeeping for one run."""

    def __init__(self, path: str | Path, entries: Iterable[Entry] | None = None) -> None:
        self.path = Path(path)
        self.entries: list[Entry] = list(entries or [])
        self._lock = threading.RLock()
        #: task_id -> entry ids handed to that task this run (set by ``recall``).
        self.injections: dict[str, list[str]] = {}
        #: task_id -> {"trace": [...], "output": ...} recorded by the runtime this run.
        self.observations: dict[str, dict[str, Any]] = {}

    # --- persistence ---

    @classmethod
    def load(cls, path: str | Path) -> Memory:
        """Read the store at ``path``; a missing or malformed file yields an empty memory."""
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(p)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(json.dumps({"event": "memory_unreadable", "path": str(p),
                                       "error": str(exc)}))
            return cls(p)
        rows = data.get("entries", data) if isinstance(data, dict) else data
        return cls(p, [Entry.from_dict(r) for r in rows if isinstance(r, dict)])

    def save(self) -> Path:
        """Write the store atomically (tmp + rename) and return its path."""
        return tracing.node_span("memory.save")(self._save)()

    def _save(self) -> Path:
        with self._lock:
            body = json.dumps(
                {"entries": [e.to_dict() for e in self.entries]}, indent=2, sort_keys=True
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(body + "\n", encoding="utf-8")
        tmp.replace(self.path)
        return self.path

    # --- per-run bookkeeping ---

    def begin_run(self) -> None:
        """Forget which entries went to which task; called before each scored run."""
        with self._lock:
            self.injections.clear()
            self.observations.clear()

    def observe(self, task_id: str, trace: list[dict[str, Any]], output: Any) -> None:
        """Record the tool results one task saw, so ``reflect`` can ground rules in them."""
        with self._lock:
            self.observations[str(task_id)] = {"trace": list(trace or []), "output": output}

    def observed(self, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.observations.get(str(task_id), {}).get("trace") or [])

    @property
    def active(self) -> list[Entry]:
        with self._lock:
            return [e for e in self.entries if e.status == "active"]

    def by_id(self, entry_id: str) -> Entry | None:
        return next((e for e in self.entries if e.id == entry_id), None)

    def injected_ids(self) -> list[str]:
        """Distinct entry ids injected since the last ``begin_run``, in first-use order."""
        with self._lock:
            seen: dict[str, None] = {}
            for ids in self.injections.values():
                seen.update(dict.fromkeys(ids))
            return list(seen)

    # --- recall ---

    def recall(self, task: Any, k: int = 3) -> list[Entry]:
        """Top ``k`` active entries for ``task`` by keyword/tool overlap, best first.

        Deterministic: entries score on token overlap with the task text (a tool named in
        both is worth an extra point) and ties break on entry id.
        """
        return tracing.node_span("memory.recall")(self._recall)(task, k)

    def _recall(self, task: Any, k: int) -> list[Entry]:
        if k <= 0:
            return []
        text = task_text(task)
        want = tokens(text)
        lowered = text.lower()
        scored = [
            (self._score(entry, want, lowered), entry.id, entry) for entry in self.active
        ]
        ranked = [t for t in sorted(scored, key=lambda t: (-t[0], t[1])) if t[0] > 0][:k]
        hits = [entry for _, _, entry in ranked]
        with self._lock:
            self.injections.setdefault(task_id_of(task), []).extend(e.id for e in hits)
        return hits

    @staticmethod
    def _score(entry: Entry, want: set[str], lowered: str) -> float:
        score = overlap(want, tokens(entry.text))
        if entry.tool and entry.tool.lower() in lowered:
            score += 1.0
        return score

    def prompt_block(self, entries: list[Entry]) -> str:
        """The recalled rules rendered for a system prompt, or '' when there are none."""
        if not entries:
            return ""
        lines = "\n".join(f"- {e.text}" for e in entries)
        return f"# {MEMORY_HEADING}\n\n{lines}"

    # --- credit and pruning ---

    def credit(self, entries: Iterable[Entry | str], passed: bool) -> None:
        """Record one outcome against every entry that was injected for that task."""
        with self._lock:
            for item in entries:
                entry = self.by_id(item) if isinstance(item, str) else item
                if entry is None:
                    continue
                entry.hits += 1
                if passed:
                    entry.wins += 1
                else:
                    entry.losses += 1

    def credit_run(self, rows: Iterable[dict[str, Any]], threshold: float) -> int:
        """Credit every row of a finished run against the entries it was given."""
        return int(tracing.node_span("memory.credit")(self._credit_run)(list(rows), threshold))

    def _credit_run(self, rows: list[dict[str, Any]], threshold: float) -> int:
        n = 0
        for row in rows:
            ids = self.injections.get(str(row.get("task_id")))
            if not ids:
                continue
            self.credit(ids, float(row.get("score") or 0.0) >= threshold)
            n += 1
        return n

    def prune(self, *, min_sample: int = MIN_SAMPLE) -> list[Entry]:
        """Retire entries that have been tried enough and lose more often than they win."""
        return tracing.node_span("memory.prune")(self._prune)(min_sample)

    def _prune(self, min_sample: int) -> list[Entry]:
        retired: list[Entry] = []
        with self._lock:
            for entry in self.entries:
                if entry.status == "active" and entry.hits >= min_sample and (
                    entry.losses > entry.wins
                ):
                    entry.status = "retired"
                    retired.append(entry)
        for entry in retired:
            logger.info(json.dumps({"event": "memory_retired", "id": entry.id,
                                    "wins": entry.wins, "losses": entry.losses}))
        return retired

    # --- reflection ---

    def add(self, entry: Entry) -> Entry | None:
        """Append ``entry`` unless an active entry already says the same thing."""
        candidate = tokens(entry.text)
        with self._lock:
            for existing in self.active:
                if overlap(candidate, tokens(existing.text)) >= DEDUPE_THRESHOLD:
                    _merge(existing, entry)
                    return None
            self.entries.append(entry)
        return entry

    def reflect(
        self,
        rows: Iterable[dict[str, Any]],
        domain: Any,
        *,
        client: Any | None = None,
        iteration: int = 0,
        threshold: float | None = None,
        traces: Any = None,
    ) -> list[Entry]:
        """Learn from the failed rows of one run; returns the entries actually added.

        ``traces`` is an optional :class:`anneal.diagnose.TraceSource` (the Neatlogs MCP
        client when it is configured) used to enrich rows whose trace the runtime did not
        record. Reflection never raises: a gateway that is not configured logs and yields
        no new rules.
        """
        return tracing.node_span("memory.reflect")(self._reflect)(
            list(rows), domain, client, iteration, threshold, traces
        )

    def _reflect(
        self, rows: list[dict[str, Any]], domain: Any, client: Any,
        iteration: int, threshold: float | None, traces: Any,
    ) -> list[Entry]:
        limit = _threshold(domain) if threshold is None else threshold
        failures = [r for r in rows if float(r.get("score") or 0.0) < limit][:MAX_FAILURES]
        if not failures:
            return []
        cases = [(row, self._trace_for(row, traces)) for row in failures]
        messages = [
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": _reflect_prompt(domain, cases)},
        ]
        try:
            reply, _usage = llm.chat(REFLECT_TIER, messages, client=client, temperature=0)
        except Exception as exc:  # noqa: BLE001 - reflection is best effort, never fatal
            logger.warning(json.dumps({"event": "reflect_failed", "error": repr(exc)}))
            return []
        grounded = _grounded(parse_rules(reply), any(trace for _, trace in cases))
        added = [
            e for e in (
                self.add(_entry(rule, failures, getattr(domain, "name", ""), iteration))
                for rule in grounded
            ) if e is not None
        ]
        logger.info(json.dumps({"event": "memory_reflect", "domain": getattr(domain, "name", ""),
                                "iteration": iteration, "failures": len(failures),
                                "added": len(added), "total": len(self.active)}))
        return added

    def _trace_for(self, row: dict[str, Any], traces: Any) -> list[dict[str, Any]]:
        """The tool calls one failed task made: the row's own, else what the runtime saw."""
        trace = [s for s in (row.get("trace") or []) if isinstance(s, dict)]
        if trace:
            return trace
        trace = self.observed(str(row.get("task_id")))
        if trace or traces is None:
            return trace
        key = str(row.get("trace_id") or row.get("task_id"))
        context = traces.get_trace_context(key) or {}
        return [s for s in (context.get("trace") or []) if isinstance(s, dict)]


def _grounded(rules: list[dict[str, Any]], had_tool_results: bool) -> list[dict[str, Any]]:
    """Drop rules that cite no tool result, but only when there were tool results to cite.

    A rule the model cannot point at a tool result for is a guess, and guesses are what
    memory is supposed to stop accumulating. Runs that made no tool call at all (a crash, a
    schema failure on the first turn) have nothing to cite, so their rules pass through.
    """
    if not had_tool_results:
        return rules
    kept = [r for r in rules if str(r.get("evidence") or "").strip()]
    for dropped in [r for r in rules if r not in kept]:
        logger.info(json.dumps({"event": "memory_ungrounded", "text": str(dropped.get("text"))}))
    return kept


def _merge(existing: Entry, new: Entry) -> None:
    """Fold a duplicate rule's provenance into the entry that already covers it."""
    for field_name in ("source_task_ids", "source_trace_ids"):
        merged = dict.fromkeys([*getattr(existing, field_name), *getattr(new, field_name)])
        setattr(existing, field_name, list(merged))
    existing.tool = existing.tool or new.tool
    existing.evidence = existing.evidence or new.evidence


def _threshold(domain: Any) -> float:
    return float(getattr(getattr(domain, "eval", None), "THRESHOLD", 1.0))


def _entry(rule: dict[str, Any], rows: list[dict[str, Any]], domain: str, iteration: int) -> Entry:
    named = {str(t) for t in (rule.get("task_ids") or rule.get("tasks") or [])}
    used = [r for r in rows if str(r.get("task_id")) in named] or rows
    return Entry(
        id=uuid.uuid4().hex[:12],
        text=str(rule["text"]).strip(),
        domain=domain,
        source_task_ids=[str(r.get("task_id")) for r in used],
        source_trace_ids=[str(r["trace_id"]) for r in used if r.get("trace_id")],
        tool=str(rule["tool"]) if rule.get("tool") else None,
        evidence=str(rule["evidence"])[:RESULT_CHARS] if rule.get("evidence") else None,
        created_iteration=iteration,
    )


def _reflect_prompt(domain: Any, cases: list[tuple[dict[str, Any], list[dict[str, Any]]]]) -> str:
    """The failed tasks, what their tools returned and what the agent answered."""
    parts = [f"# Goal\n\n{getattr(domain, 'goal', '')[:1500]}", "# Failed runs"]
    for row, trace in cases:
        steps = "\n".join(
            f"  - {s.get('tool')}({json.dumps(s.get('args'), default=str)}) -> "
            f"{str(s.get('result'))[:RESULT_CHARS]}"
            for s in trace
        ) or "  - (no tool calls)"
        parts.append(
            f"## task {row.get('task_id')} (score {row.get('score')})\n"
            f"tool results:\n{steps}\n"
            f"agent answer: {str(row.get('output'))[:RESULT_CHARS]}"
        )
    parts.append("What should the agent have known before starting these tasks?")
    return "\n\n".join(parts)


def parse_rules(reply: str) -> list[dict[str, Any]]:
    """Parse the reflector's JSON array; tolerates fences and surrounding prose."""
    text = (reply or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    rules = [r for r in data if isinstance(r, dict) and str(r.get("text") or "").strip()]
    return rules[:MAX_NEW_RULES]


# --- the active store --------------------------------------------------------------------

_ACTIVE: ContextVar[Memory | None] = ContextVar("anneal_memory", default=None)


def active() -> Memory | None:
    """The memory the current run is using, or None when memory is not in play."""
    return _ACTIVE.get()


@contextlib.contextmanager
def activate(memory: Memory | None) -> Iterator[Memory | None]:
    """Make ``memory`` the active store for the duration of the block."""
    token = _ACTIVE.set(memory)
    try:
        yield memory
    finally:
        _ACTIVE.reset(token)


def memory_path(runs_dir: str | Path, domain_name: str) -> Path:
    """``<runs_dir>/<domain>/memory.json`` -- one store per domain, kept across runs."""
    return Path(runs_dir) / domain_name / "memory.json"
