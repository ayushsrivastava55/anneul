"""Episodic memory: what the agent learns about a domain from its own runs.

The loop writes here, the runtime reads here, and the store survives the process. Two kinds
of entry are learned, both grounded in what the tools actually returned:

- a ``rule`` -- a contextual constraint extracted from a FAILED task, e.g. *"basic economy
  reservations cannot be modified; call get_reservation_details and check the cabin before
  offering a change"*. The model must quote the tool result that justifies it.
- a ``procedure`` -- an ordered tool recipe extracted from a SUCCESSFUL task that took at
  least ``MIN_PROCEDURE_STEPS`` tool calls: which tools to call, in what order, with what
  parameters, and the pitfalls. A procedure is kept only when every tool it names was
  actually called by that task, which is a stronger check than a quoted string.

``recall(task, k)`` retrieves the top ``k`` active entries for a task. The index is SQLite
FTS5 (``sqlite3`` ships with Python; no new dependency) with the porter stemmer, ranked by
bm25 and tie-broken on entry id so recall is deterministic. Builds of Python without FTS5
fall back to token-overlap scoring; both paths are tested.

``credit(entries, passed)`` / ``credit_run(rows, threshold)`` move ``hits``/``wins``/
``losses`` for the entries a task was actually given, and ``prune()`` retires entries whose
losses outgrow their wins after a minimum sample. Memory that cannot unlearn is just a
growing prompt.

The store of record is SQLite at ``runs/<domain>/memory.db``; ``save`` also writes a
``memory.json`` export next to it so the dashboard and a human can read it without SQL. The
path is injectable and either name may be handed to ``Memory``/``Memory.load``.

Nothing domain-specific lives here: an entry is text the model wrote about one domain's tool
results, carried in a ``domain``-tagged row. The active store for a run is a contextvar
(``activate``/``active``) so the runtime can reach it without a signature change and so
``runner``'s thread pool -- which copies the caller's context -- shares one instance. With no
active store the runtime behaves exactly as it did before memory existed.

Only the search split is ever reflected on; the reserved evaluation split is the gate's
business and never reaches this module.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable, Iterator
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from anneal import llm, tracing

logger = logging.getLogger("anneal.memory")

#: Model tier used for reflection (never a model id -- those live in specs/models.yaml).
REFLECT_TIER = "cheap"
#: Heading the runtime injects the recalled entries under.
MEMORY_HEADING = "Learned from previous runs"
#: Entry kinds.
RULE, PROCEDURE = "rule", "procedure"
#: Most entries one reflection pass may add.
MAX_NEW_RULES = 3
#: Failed / successful tasks shown to the reflector in one prompt.
MAX_FAILURES = 5
MAX_SUCCESSES = 3
#: Tool calls a successful task needs before its procedure is worth writing down.
MIN_PROCEDURE_STEPS = 5
#: Pseudo-tool the tool_router records its choice under; not a real step.
ROUTE_TOOL = "route"
#: Characters of one tool result kept in the reflection prompt.
RESULT_CHARS = 400
#: Token-overlap (Jaccard) at or above which two entries are considered the same lesson.
DEDUPE_THRESHOLD = 0.6
#: Injections an entry needs before ``prune`` is allowed to judge it.
MIN_SAMPLE = 3
#: FTS5 tokenizer: porter stemming so "reservations" finds "reservation".
FTS_TOKENIZER = "porter unicode61"

_STOPWORDS = frozenset(
    """a an and are as at be before but by can cannot do does for from has have if in into is it
    its must never no not of on only or should that the their then there these they this to use
    used using was were what when which while who will with without you your""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_WORD_RE = re.compile(r"[a-z0-9]+")

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

PROCEDURE_SYSTEM = (
    "You study an agent's SUCCESSFUL multi-step runs and write down the procedure, so the "
    "next run gets there in fewer steps.\n"
    f"Reply with a JSON array of at most {MAX_NEW_RULES} objects, each "
    '{"text": ..., "steps": [...], "tool": ..., "evidence": ...}.\n'
    "- steps: the tool names to call, in the order they must be called. Use only tools that "
    "appear in the runs below.\n"
    "- text: the recipe as short numbered steps naming the parameters that matter, ending "
    "with the pitfalls that would have broken it. Good: '1. get_reservation_details(id) to "
    "read the cabin. 2. search_flights(origin, destination, date) ... Pitfall: never call "
    "book before the payment method is confirmed.'\n"
    "- tool: the tool the procedure starts with.\n"
    "- evidence: the tool result that shows the procedure worked.\n"
    "Write down only a procedure that generalises to other tasks of this kind. If the run "
    "was a one-off, reply with []. Reply with the JSON array and nothing else."
)


# --- entries -----------------------------------------------------------------------------


@dataclass
class Entry:
    """One learned rule or procedure and its track record."""

    id: str
    text: str
    domain: str
    kind: str = RULE
    steps: list[str] = field(default_factory=list)
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

    def body(self) -> str:
        """The text the full-text index searches: the entry plus its tool vocabulary."""
        return " ".join([self.text, self.tool or "", *self.steps])


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


def tool_steps(trace: Iterable[dict[str, Any]]) -> list[str]:
    """The real tool names a trace called, in order (the router's pseudo-step excluded)."""
    return [
        str(s.get("tool"))
        for s in trace or ()
        if isinstance(s, dict) and s.get("tool") and s.get("tool") != ROUTE_TOOL
    ]


# --- full-text index ---------------------------------------------------------------------


@lru_cache(maxsize=1)
def fts5_available() -> bool:
    """True when this Python's sqlite3 was built with FTS5 (checked once)."""
    try:
        with sqlite3.connect(":memory:") as conn:
            conn.execute(f"CREATE VIRTUAL TABLE probe USING fts5(body, tokenize='{FTS_TOKENIZER}')")
    except sqlite3.Error as exc:
        logger.warning(json.dumps({"event": "fts5_unavailable", "error": str(exc)}))
        return False
    return True


def fts_query(text: str) -> str:
    """``"term" OR "term" ...`` from a task's words; '' when nothing is worth searching.

    Terms come from ``[a-z0-9]+`` only, so no task text can inject FTS5 query syntax.
    """
    terms = {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}
    return " OR ".join(f'"{t}"' for t in sorted(terms))


class _Index:
    """An in-memory FTS5 index over the active entries, rebuilt when the store changes."""

    def __init__(self) -> None:
        self._conn: sqlite3.Connection | None = None
        self._signature: tuple[tuple[str, str], ...] = ()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        return self._conn

    def refresh(self, entries: list[Entry]) -> None:
        """Rebuild the index when the set of active entries has changed."""
        signature = tuple((e.id, e.status) for e in entries)
        if signature == self._signature and self._conn is not None:
            return
        conn = self._connect()
        with conn:
            conn.execute("DROP TABLE IF EXISTS mem")
            conn.execute(
                "CREATE VIRTUAL TABLE mem USING "
                f"fts5(entry_id UNINDEXED, body, tokenize='{FTS_TOKENIZER}')"
            )
            conn.executemany(
                "INSERT INTO mem (entry_id, body) VALUES (?, ?)",
                [(e.id, e.body()) for e in entries],
            )
        self._signature = signature

    def search(self, query: str) -> dict[str, float]:
        """entry id -> relevance for the entries matching ``query`` (higher is better)."""
        if not query or self._conn is None:
            return {}
        rows = self._conn.execute(
            "SELECT entry_id, bm25(mem) FROM mem WHERE mem MATCH ?", (query,)
        ).fetchall()
        # FTS5 bm25 is negative-is-better; flip it so every score in this module rises.
        return {str(entry_id): -float(rank) for entry_id, rank in rows}


# --- the store ---------------------------------------------------------------------------


class Memory:
    """A persistent set of learned entries plus the bookkeeping for one run."""

    def __init__(self, path: str | Path, entries: Iterable[Entry] | None = None) -> None:
        self.path = Path(path)
        #: store of record, and the human-readable export beside it
        self.db_path = self.path.with_suffix(".db")
        self.json_path = self.path.with_suffix(".json")
        self.entries: list[Entry] = list(entries or [])
        self._lock = threading.RLock()
        self._index = _Index()
        #: task_id -> entry ids handed to that task this run (set by ``recall``).
        self.injections: dict[str, list[str]] = {}
        #: task_id -> {"trace": [...], "output": ...} recorded by the runtime this run.
        self.observations: dict[str, dict[str, Any]] = {}

    # --- persistence ---

    @classmethod
    def load(cls, path: str | Path) -> Memory:
        """Read the store at ``path``: the SQLite db, else the JSON export, else empty."""
        store = cls(path)
        rows = _read_db(store.db_path)
        if rows is None:
            rows = _read_json(store.json_path)
        store.entries = [Entry.from_dict(r) for r in rows if isinstance(r, dict)]
        return store

    def save(self) -> Path:
        """Write the SQLite store and the JSON export; returns the db path."""
        return tracing.node_span("memory.save")(self._save)()

    def _save(self) -> Path:
        with self._lock:
            rows = [e.to_dict() for e in self.entries]
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        _write_db(self.db_path, rows)
        _write_json(self.json_path, rows)
        return self.db_path

    # --- per-run bookkeeping ---

    def begin_run(self) -> None:
        """Forget which entries went to which task; called before each scored run."""
        with self._lock:
            self.injections.clear()
            self.observations.clear()

    def observe(self, task_id: str, trace: list[dict[str, Any]], output: Any) -> None:
        """Record the tool results one task saw, so ``reflect`` can ground entries in them."""
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

    def by_kind(self) -> dict[str, int]:
        """Active entry count per kind, e.g. ``{"rule": 4, "procedure": 1}``."""
        counts = {RULE: 0, PROCEDURE: 0}
        for entry in self.active:
            counts[entry.kind] = counts.get(entry.kind, 0) + 1
        return counts

    def injected_ids(self) -> list[str]:
        """Distinct entry ids injected since the last ``begin_run``, in first-use order."""
        with self._lock:
            seen: dict[str, None] = {}
            for ids in self.injections.values():
                seen.update(dict.fromkeys(ids))
            return list(seen)

    # --- recall ---

    def recall(self, task: Any, k: int = 3) -> list[Entry]:
        """Top ``k`` active entries for ``task``, best first.

        Ranked by SQLite FTS5 bm25 over the entry text, tool name and step list (porter
        stemming, so "reservations" finds "reservation"), plus a point for an entry whose
        tool the task names. Ties break on entry id, so recall is deterministic.
        """
        return tracing.node_span("memory.recall")(self._recall)(task, k)

    def _recall(self, task: Any, k: int) -> list[Entry]:
        if k <= 0:
            return []
        text = task_text(task)
        lowered = text.lower()
        with self._lock:
            entries = self.active
            relevance = self._relevance(entries, text)
        scored = [
            (score + (1.0 if e.tool and e.tool.lower() in lowered else 0.0), e.id, e)
            for e in entries
            for score in [relevance.get(e.id, 0.0)]
        ]
        ranked = [t for t in sorted(scored, key=lambda t: (-t[0], t[1])) if t[0] > 0][:k]
        hits = [entry for _, _, entry in ranked]
        with self._lock:
            self.injections.setdefault(task_id_of(task), []).extend(e.id for e in hits)
        return hits

    def _relevance(self, entries: list[Entry], text: str) -> dict[str, float]:
        """entry id -> text relevance: FTS5 bm25 when available, else token overlap."""
        if fts5_available():
            self._index.refresh(entries)
            return self._index.search(fts_query(text))
        want = tokens(text)
        scores = {e.id: overlap(want, tokens(e.body())) for e in entries}
        return {entry_id: s for entry_id, s in scores.items() if s > 0}

    def prompt_block(self, entries: list[Entry]) -> str:
        """The recalled entries rendered for a system prompt, or '' when there are none."""
        if not entries:
            return ""
        parts = [f"# {MEMORY_HEADING}"]
        for kind, label in ((RULE, "Rules"), (PROCEDURE, "Procedures")):
            chosen = [e for e in entries if e.kind == kind]
            if chosen:
                parts.append(f"## {label}\n\n" + "\n".join(f"- {e.text}" for e in chosen))
        return "\n\n".join(parts)

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
        """Append ``entry`` unless an active entry of the same kind already covers it.

        Procedures are compared on their step list first: two recipes over the same tool
        vocabulary read alike, so text overlap alone would merge distinct ones.
        """
        candidate = tokens(entry.text)
        with self._lock:
            for existing in self.active:
                if existing.kind != entry.kind:
                    continue
                same_steps = bool(entry.steps) and existing.steps == entry.steps
                if same_steps or overlap(candidate, tokens(existing.text)) >= DEDUPE_THRESHOLD:
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
        """Learn from one run: rules from what failed, procedures from what worked.

        ``traces`` is an optional :class:`anneal.diagnose.TraceSource` (the Neatlogs MCP
        client when it is configured) used to enrich rows whose trace the runtime did not
        record. Reflection never raises: a gateway that is not configured logs and yields
        nothing new.
        """
        return tracing.node_span("memory.reflect")(self._reflect)(
            list(rows), domain, client, iteration, threshold, traces
        )

    def _reflect(
        self, rows: list[dict[str, Any]], domain: Any, client: Any,
        iteration: int, threshold: float | None, traces: Any,
    ) -> list[Entry]:
        limit = _threshold(domain) if threshold is None else threshold
        cases = [(row, self._trace_for(row, traces)) for row in rows]
        failures = [c for c in cases if float(c[0].get("score") or 0.0) < limit][:MAX_FAILURES]
        wins = [
            c for c in cases
            if float(c[0].get("score") or 0.0) >= limit
            and len(tool_steps(c[1])) >= MIN_PROCEDURE_STEPS
        ][:MAX_SUCCESSES]
        added = self._learn(RULE, failures, domain, client, iteration)
        added += self._learn(PROCEDURE, wins, domain, client, iteration)
        logger.info(json.dumps({"event": "memory_reflect", "domain": getattr(domain, "name", ""),
                                "iteration": iteration, "failures": len(failures),
                                "successes": len(wins), "added": len(added),
                                "total": len(self.active), "by_kind": self.by_kind()}))
        return added

    def _learn(
        self, kind: str, cases: list[tuple[dict[str, Any], list[dict[str, Any]]]],
        domain: Any, client: Any, iteration: int,
    ) -> list[Entry]:
        """One reflection pass: ask for entries of ``kind``, keep the grounded ones."""
        if not cases:
            return []
        system = REFLECT_SYSTEM if kind == RULE else PROCEDURE_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _reflect_prompt(domain, cases, kind)},
        ]
        try:
            reply, _usage = llm.chat(REFLECT_TIER, messages, client=client, temperature=0)
        except Exception as exc:  # noqa: BLE001 - reflection is best effort, never fatal
            logger.warning(json.dumps({"event": "reflect_failed", "kind": kind,
                                       "error": repr(exc)}))
            return []
        rows = [row for row, _ in cases]
        grounded = _grounded(parse_rules(reply), kind, cases)
        made = (
            _entry(r, kind, rows, getattr(domain, "name", ""), iteration) for r in grounded
        )
        return [e for e in (self.add(m) for m in made) if e is not None]

    def _trace_for(self, row: dict[str, Any], traces: Any) -> list[dict[str, Any]]:
        """The tool calls one task made: the row's own, else what the runtime saw."""
        trace = [s for s in (row.get("trace") or []) if isinstance(s, dict)]
        if trace:
            return trace
        trace = self.observed(str(row.get("task_id")))
        if trace or traces is None:
            return trace
        key = str(row.get("trace_id") or row.get("task_id"))
        context = traces.get_trace_context(key) or {}
        return [s for s in (context.get("trace") or []) if isinstance(s, dict)]


# --- persistence helpers -----------------------------------------------------------------

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    created_iteration INTEGER NOT NULL,
    data TEXT NOT NULL
)
"""


def _write_db(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace the store's contents in one transaction."""
    with sqlite3.connect(path) as conn:
        conn.execute(_DB_SCHEMA)
        conn.execute("DELETE FROM entries")
        conn.executemany(
            "INSERT INTO entries (id, kind, status, created_iteration, data) VALUES (?,?,?,?,?)",
            [
                (r["id"], r.get("kind", RULE), r.get("status", "active"),
                 int(r.get("created_iteration") or 0), json.dumps(r, sort_keys=True))
                for r in rows
            ],
        )
    conn.close()


def _read_db(path: Path) -> list[dict[str, Any]] | None:
    """Rows from the SQLite store, or None when it is absent or unreadable."""
    if not path.is_file():
        return None
    try:
        with sqlite3.connect(path) as conn:
            raw = conn.execute("SELECT data FROM entries ORDER BY rowid").fetchall()
        conn.close()
        return [json.loads(row[0]) for row in raw]
    except (sqlite3.Error, json.JSONDecodeError) as exc:
        logger.warning(json.dumps({"event": "memory_db_unreadable", "path": str(path),
                                   "error": str(exc)}))
        return None


def _write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    """The human/dashboard-readable export, written atomically."""
    body = json.dumps({"entries": rows}, indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Rows from the JSON export; missing or malformed files read as empty."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(json.dumps({"event": "memory_unreadable", "path": str(path),
                                   "error": str(exc)}))
        return []
    rows = data.get("entries", data) if isinstance(data, dict) else data
    return rows if isinstance(rows, list) else []


# --- reflection helpers ------------------------------------------------------------------


def _grounded(
    rules: list[dict[str, Any]], kind: str,
    cases: list[tuple[dict[str, Any], list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    """Keep only the entries the runs below actually support.

    A rule must quote a tool result -- unless the runs made no tool call at all, in which
    case there is nothing to quote and its lesson still counts. A procedure must name only
    tools those runs really called, in a non-empty order; that is checkable, so it is
    checked rather than trusted.
    """
    called = {t for _, trace in cases for t in tool_steps(trace)}
    if kind == PROCEDURE:
        kept = [r for r in rules if r.get("steps") and set(map(str, r["steps"])) <= called]
    elif called:
        kept = [r for r in rules if str(r.get("evidence") or "").strip()]
    else:
        kept = rules
    for dropped in [r for r in rules if r not in kept]:
        logger.info(json.dumps({"event": "memory_ungrounded", "kind": kind,
                                "text": str(dropped.get("text"))}))
    return kept


def _merge(existing: Entry, new: Entry) -> None:
    """Fold a duplicate entry's provenance into the one that already covers it."""
    for field_name in ("source_task_ids", "source_trace_ids"):
        merged = dict.fromkeys([*getattr(existing, field_name), *getattr(new, field_name)])
        setattr(existing, field_name, list(merged))
    existing.tool = existing.tool or new.tool
    existing.evidence = existing.evidence or new.evidence


def _threshold(domain: Any) -> float:
    return float(getattr(getattr(domain, "eval", None), "THRESHOLD", 1.0))


def _entry(
    rule: dict[str, Any], kind: str, rows: list[dict[str, Any]], domain: str, iteration: int
) -> Entry:
    named = {str(t) for t in (rule.get("task_ids") or rule.get("tasks") or [])}
    used = [r for r in rows if str(r.get("task_id")) in named] or rows
    return Entry(
        id=uuid.uuid4().hex[:12],
        text=str(rule["text"]).strip(),
        domain=domain,
        kind=kind,
        steps=[str(s) for s in (rule.get("steps") or [])],
        source_task_ids=[str(r.get("task_id")) for r in used],
        source_trace_ids=[str(r["trace_id"]) for r in used if r.get("trace_id")],
        tool=str(rule["tool"]) if rule.get("tool") else None,
        evidence=str(rule["evidence"])[:RESULT_CHARS] if rule.get("evidence") else None,
        created_iteration=iteration,
    )


def _reflect_prompt(
    domain: Any, cases: list[tuple[dict[str, Any], list[dict[str, Any]]]], kind: str
) -> str:
    """The runs, what their tools returned and what the agent answered."""
    label = "Failed runs" if kind == RULE else "Successful runs"
    parts = [f"# Goal\n\n{getattr(domain, 'goal', '')[:1500]}", f"# {label}"]
    for row, trace in cases:
        steps = "\n".join(
            f"  {i}. {s.get('tool')}({json.dumps(s.get('args'), default=str)}) -> "
            f"{str(s.get('result'))[:RESULT_CHARS]}"
            for i, s in enumerate(trace, start=1)
        ) or "  - (no tool calls)"
        parts.append(
            f"## task {row.get('task_id')} (score {row.get('score')})\n"
            f"tool calls in order:\n{steps}\n"
            f"agent answer: {str(row.get('output'))[:RESULT_CHARS]}"
        )
    question = (
        "What should the agent have known before starting these tasks?"
        if kind == RULE
        else "What is the reusable procedure these runs followed?"
    )
    parts.append(question)
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
    """``<runs_dir>/<domain>/memory.db`` -- one store per domain, kept across runs."""
    return Path(runs_dir) / domain_name / "memory.db"
