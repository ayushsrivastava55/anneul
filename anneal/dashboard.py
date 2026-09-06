"""The Anneal console: a single server-rendered page showing the loop as it runs.

Everything rendered here is read from disk: ``runs/<domain>/<iter>/summary.json`` for the
iteration curves, ``ledger.json`` for the issue table and ``runs/<domain>/anneal/pareto.json``
for the Pareto scatter. No number is ever hardcoded; a missing file renders an empty state.

Charts are plain inline SVG built server-side, so the page is fully readable with JavaScript
disabled. One short inline script re-fetches the fragments every 10s; there is no CDN, no
framework and no build step. Presentation follows ``.stitch/DESIGN.md``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from anneal import vocab

log = logging.getLogger("anneal.dashboard")

# summary.json / pareto.json field aliases, most specific first. cli.py writes mean_score /
# pass3_rate / p95_latency_ms at the top level and the per-candidate search metrics (which is
# where cost_per_task lives) under summary["search"][<candidate id>]; anneal.py's pareto points
# use their own names. Read them all; note top-level "cost_usd" is a split total, not $/task.
SCORE_KEYS = ("score", "mean_score", "accuracy")
PASS3_KEYS = ("pass3", "pass3_rate", "pass_cubed", "pass^3")
COST_KEYS = ("cost_per_task", "usd_per_task", "dollars_per_task", "cost_usd_per_task")
P95_KEYS = ("p95_latency_ms", "latency_p95_ms", "p95_ms", "p95")

# The four things the loop is trying to move, named the way the person paying for it would
# name them. "pass^3" and "p95" are the statistics; what they answer is whether the agent is
# dependable and whether a slow day is still bearable, so that is what the chart is titled.
METRICS: tuple[tuple[str, str, str], ...] = (
    ("score", "how often it is right", "{:.3f}"),
    ("pass3", "right 3 times running", "{:.3f}"),
    ("cost_per_task", "cost per task", "${:.4f}"),
    ("p95_latency_ms", "slowest runs (ms)", "{:.0f}"),
)

# Ledger table: the class, how bad it is, how often it fired, and the operator Mutate would
# reach for next. Severity and the operator ladder come from specs/failure_taxonomy.yaml.
# "operator" is one column, not two: it shows the repair the loop will try next and, beneath
# it, the ones already spent on this issue. Seven columns did not fit the detail panel, and
# splitting next-from-tried put the loop's memory in the column that got clipped.
# The first two are keys the console never prints; the rest are the column headings a reader
# sees, so they are phrased as questions about the agent rather than as field names.
LEDGER_COLUMNS = (
    "domain", "id", "what went wrong", "where", "severity", "times", "status", "fix to try",
)

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "specs" / "failure_taxonomy.yaml"

STAGES = ("Architect", "Run", "Diagnose", "Mutate", "Gate", "Anneal")

DASH = "\u2014"  # every absent value renders as this, never as a zero


@dataclass(frozen=True)
class Point:
    """One iteration (or one Pareto candidate) with whatever metrics were on disk."""

    label: str
    score: float | None
    pass3: float | None
    cost_per_task: float | None
    p95_latency_ms: float | None
    extra: dict[str, Any]

    def get(self, metric: str) -> float | None:
        return {
            "score": self.score,
            "pass3": self.pass3,
            "cost_per_task": self.cost_per_task,
            "p95_latency_ms": self.p95_latency_ms,
        }[metric]


# --- reading ------------------------------------------------------------------------------


def _num(data: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        return float(value)
    return None


def _first(*values: float | None) -> float | None:
    return next((v for v in values if v is not None), None)


def _read_json(path: Path) -> Any | None:
    """Parse ``path``; malformed or unreadable JSON is skipped with a log line, never raised.

    A file that simply is not there yet is not a fault: every domain lacks a pareto.json
    until the downshift stage runs for it, and warning about that on every page render makes
    a normal dashboard look broken.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        event = {"event": "dashboard_bad_json", "path": str(path), "error": str(exc)}
        log.warning(json.dumps(event))
        return None


def _point(data: dict[str, Any], label: str) -> Point:
    return Point(
        label=label,
        score=_num(data, SCORE_KEYS),
        pass3=_num(data, PASS3_KEYS),
        cost_per_task=_num(data, COST_KEYS),
        p95_latency_ms=_num(data, P95_KEYS),
        extra=data,
    )


def _search_block(data: dict[str, Any]) -> dict[str, Any]:
    """``summary["search"][winner]`` — the per-candidate metrics ``anneal report`` reads."""
    search = data.get("search")
    if not isinstance(search, dict):
        return {}
    for key in ("winner_id", "candidate_id", "incumbent_id"):
        block = search.get(str(data.get(key)))
        if isinstance(block, dict):
            return block
    return {}


def _summary_point(data: dict[str, Any], label: str) -> Point:
    """One iteration. $/task and p95 come from the winner's search block when it is there."""
    block = _search_block(data)
    point = _point(data, label)
    return Point(
        label=point.label,
        score=point.score,
        pass3=point.pass3,
        cost_per_task=_first(_num(block, COST_KEYS), point.cost_per_task),
        p95_latency_ms=_first(_num(block, P95_KEYS), point.p95_latency_ms),
        extra=data,
    )


def _iteration_sort_key(path: Path) -> tuple[int, str]:
    name = path.parent.name
    return (int(name), name) if name.isdigit() else (10**9, name)


def load_summaries(runs_dir: Path | str) -> dict[str, list[Point]]:
    """``{domain: [Point per iteration]}`` from ``runs/<domain>/<iter>/summary.json``."""
    root = Path(runs_dir)
    domains: dict[str, list[Point]] = {}
    if not root.is_dir():
        return domains
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        paths = sorted(domain_dir.glob("*/summary.json"), key=_iteration_sort_key)
        points = []
        for path in paths:
            data = _read_json(path)
            if isinstance(data, dict):
                label = str(data.get("iteration", path.parent.name))
                points.append(_summary_point(data, label))
        if points:
            domains[domain_dir.name] = points
    return domains


def load_ledger(ledger_path: Path | str) -> list[dict[str, Any]]:
    """Issue list from ``ledger.json``; a missing, empty or malformed ledger reads as empty."""
    path = Path(ledger_path)
    if not path.is_file():
        return []
    data = _read_json(path)
    if isinstance(data, dict):
        data = data.get("issues")
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def load_ledgers(runs_dir: Path | str, ledger_path: Path | str) -> list[dict[str, Any]]:
    """Every issue the optimiser knows about, stamped with its domain.

    The run loop writes one ledger per domain at ``runs/<domain>/ledger.json`` (issue ids
    are keyed on (class, node) and node names repeat across domains), so a dashboard that
    reads only the ``--ledger`` file shows an empty table against a default run. An
    explicit ``--ledger`` file still wins when it exists; otherwise the per-domain ledgers
    under ``runs_dir`` are aggregated.
    """
    explicit = Path(ledger_path)
    if explicit.is_file():
        return [{"domain": explicit.parent.name, **row} for row in load_ledger(explicit)]
    root = Path(runs_dir)
    issues: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("*/ledger.json")):
            issues.extend({"domain": path.parent.name, **row} for row in load_ledger(path))
    return issues


def _pareto_point(row: dict[str, Any], front: Any) -> Point:
    """One ``anneal.py`` pareto point; ``on_front`` comes from the payload's ``front`` id list."""
    label = str(
        row.get("config_id") or row.get("label") or row.get("candidate_id") or row.get("id") or "?"
    )
    on_front = label in front if isinstance(front, list) else bool(row.get("on_front"))
    return _point({**row, "on_front": on_front}, label)


def load_pareto(runs_dir: Path | str) -> dict[str, list[Point]]:
    """``{domain: [Point per candidate]}`` from ``runs/<domain>/anneal/pareto.json``."""
    root = Path(runs_dir)
    fronts: dict[str, list[Point]] = {}
    if not root.is_dir():
        return fronts
    for domain_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        data = _read_json(domain_dir / "anneal" / "pareto.json")
        rows = data.get("points") if isinstance(data, dict) else data
        if not isinstance(rows, list):
            continue
        front = data.get("front") if isinstance(data, dict) else None
        points = [_pareto_point(row, front) for row in rows if isinstance(row, dict)]
        if points:
            fronts[domain_dir.name] = points
    return fronts


def credit_balance() -> float | None:
    """Dodo credit balance from ``anneal.billing`` when that module exists, else ``None``."""
    try:
        from anneal import billing  # type: ignore[attr-defined]
    except Exception:  # billing lands in a sibling session; the dashboard must not depend on it
        return None
    for name in ("balance", "get_balance", "current_balance", "cached_balance"):
        fn = getattr(billing, name, None)
        try:
            value = fn() if callable(fn) else fn
        except Exception as exc:
            log.warning(json.dumps({"event": "dashboard_balance_failed", "error": str(exc)}))
            return None
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, dict):
            found = _num(value, ("balance", "credits", "remaining", "balance_usd"))
            if found is not None:
                return found
    return None



# --- the current iteration ----------------------------------------------------------------


def _read_yaml(path: Path) -> dict[str, Any] | None:
    """Parse a candidate spec; malformed yaml is logged and skipped, exactly like _read_json."""
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        event = {"event": "dashboard_bad_yaml", "path": str(path), "error": str(exc)}
        log.warning(json.dumps(event))
        return None
    return data if isinstance(data, dict) else None


_TAXONOMY: dict[str, dict[str, Any]] = {}


def taxonomy() -> dict[str, dict[str, Any]]:
    """``{class_id: {severity, operators}}`` from specs/failure_taxonomy.yaml, read once.

    A class the taxonomy does not know still renders — its severity and next operator are
    em-dashes rather than a guess.
    """
    if _TAXONOMY:
        return _TAXONOMY
    data = _read_yaml(TAXONOMY_PATH) or {}
    for row in data.get("classes") or []:
        if isinstance(row, dict) and row.get("id"):
            _TAXONOMY[str(row["id"])] = row
    return _TAXONOMY


def list_domains(runs_dir: Path | str) -> list[str]:
    """Domains that have at least one iteration summary under ``runs/``."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return [d.name for d in sorted(root.iterdir()) if d.is_dir() and any(d.glob("*/summary.json"))]


def _iteration_dirs(runs_dir: Path | str, domain: str) -> list[Path]:
    root = Path(runs_dir) / domain
    if not root.is_dir():
        return []
    return [p.parent for p in sorted(root.glob("*/summary.json"), key=_iteration_sort_key)]


def pending_iterations(runs_dir: Path | str, domain: str) -> tuple[str, ...]:
    """Iteration directories that exist but hold no readable summary.json.

    These are iterations the loop has started (or failed to finish) and for which we hold no
    measurement. The chart draws them hollow on a dotted line so a reader can never mistake
    them for something we scored.
    """
    root = Path(runs_dir) / domain
    if not root.is_dir():
        return ()
    measured = {path.name for path in _iteration_dirs(runs_dir, domain)}
    started = [d for d in sorted(root.iterdir()) if d.is_dir() and d.name.isdigit()]
    return tuple(d.name for d in started if d.name not in measured)


LIVE_KEYS = (
    "task_id", "score", "hard_fail", "hit_step_budget", "schema_error", "latency_ms", "trace_id",
)


def load_live_rows(iter_dir: Path, candidate_id: str | None) -> tuple[str | None, list[dict]]:
    """Per-task rows for one candidate's search run: ``(candidate_id, rows)``.

    Only the ``search`` split is ever opened here; the reserved split belongs to the gate and
    this module must never read it. ``output`` and ``trace`` are dropped on the way in — the
    console shows a score, a latency and a status, and those payloads are megabytes.
    """
    files = sorted(iter_dir.glob(f"{candidate_id}.search.s*.jsonl")) if candidate_id else []
    if not files:
        files = sorted(iter_dir.glob("*.search.s*.jsonl"))
    if not files:
        return None, []
    path = files[0]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        event = {"event": "dashboard_bad_jsonl", "path": str(path), "error": str(exc)}
        log.warning(json.dumps(event))
        return None, []
    rows: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            log.warning(json.dumps({"event": "dashboard_bad_jsonl_row", "path": str(path)}))
            continue
        if isinstance(row, dict):
            rows.append({key: row.get(key) for key in LIVE_KEYS})
    return path.name.split(".search.")[0], rows


@dataclass(frozen=True)
class Iteration:
    """Everything the console shows for one domain's latest iteration."""

    domain: str
    path: Path | None
    label: str
    summary: dict[str, Any]
    gate: dict[str, Any]
    specs: dict[str, dict[str, Any]]
    live_candidate: str | None
    live: list[dict[str, Any]]
    has_pareto: bool

    @property
    def search(self) -> dict[str, dict[str, Any]]:
        block = self.summary.get("search")
        return block if isinstance(block, dict) else {}


def _find_spec(runs_dir: Path | str, domain: str, candidate_id: str) -> dict[str, Any] | None:
    """The candidate's spec yaml, newest iteration first.

    ``summary["specs"]`` holds paths relative to whatever directory the run was launched
    from, so they are not resolvable here; globbing the runs tree is.
    """
    root = Path(runs_dir) / domain
    for parent in [*reversed(_iteration_dirs(runs_dir, domain)), root / "anneal", root]:
        spec = _read_yaml(parent / f"{candidate_id}.yaml")
        if spec is not None:
            return spec
    return None


def load_iteration(runs_dir: Path | str, domain: str) -> Iteration:
    """The latest iteration for ``domain``: summary, gate, specs and the live run rows."""
    dirs = _iteration_dirs(runs_dir, domain)
    path = dirs[-1] if dirs else None
    summary = (_read_json(path / "summary.json") if path else None) or {}
    summary = summary if isinstance(summary, dict) else {}
    gate = (_read_json(path / "gate.json") if path else None) or {}
    gate = gate if isinstance(gate, dict) else {}
    ids = [str(cid) for cid in (summary.get("search") or {})] if isinstance(
        summary.get("search"), dict
    ) else []
    specs = {cid: spec for cid in ids if (spec := _find_spec(runs_dir, domain, cid)) is not None}
    challenger = summary.get("candidate_id") or summary.get("winner_id")
    live_id, live = load_live_rows(path, str(challenger) if challenger else None) if path else (
        None, [],
    )
    return Iteration(
        domain=domain,
        path=path,
        label=str(summary.get("iteration", path.name if path else DASH)),
        summary=summary,
        gate=gate,
        specs=specs,
        live_candidate=live_id,
        live=live,
        has_pareto=(Path(runs_dir) / domain / "anneal" / "pareto.json").is_file(),
    )


def pipeline_stages(
    it: Iteration, issues: list[dict[str, Any]]
) -> list[tuple[str, str, str]]:
    """``[(stage, state, detail)]`` derived only from the artefacts that exist on disk.

    A stage is ``done`` when its artefact is there, the first stage without one is ``active``
    and everything after it is ``todo``. Nothing here is hardcoded to an iteration number.
    """
    specs_on_disk = len(list(it.path.glob("cand-*.yaml"))) if it.path else 0
    # only this domain's own issues count: the ledger panel may fall back to every domain's
    # rows, but a stage that says "done" must point at an artefact in this domain's directory.
    mine = [row for row in issues if row.get("domain") in (None, it.domain)]
    issue = it.summary.get("issue")
    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    diagnosed = plural(len(mine), "fault") if mine else (
        vocab.failure(issue.get("class"))
        if isinstance(issue, dict) and issue.get("class") else DASH
    )
    search = it.search
    operator = it.summary.get("operator")
    decision = (it.gate or {}).get("decision")
    # Each stage's one-line status is what the reader sees beside it in the flow, so it says
    # what happened rather than naming the file or the function that made it happen.
    facts: list[tuple[str, bool, str]] = [
        ("Architect", specs_on_disk > 0,
         plural(specs_on_disk, "design") if specs_on_disk else DASH),
        ("Run", bool(search), plural(len(search), "design") + " scored" if search else DASH),
        ("Diagnose", diagnosed != DASH, diagnosed),
        ("Mutate", bool(operator), vocab.operator(operator) if operator else DASH),
        ("Gate", bool(it.gate), vocab.decision(decision) if decision else DASH),
        ("Anneal", it.has_pareto, "cheaper mixes found" if it.has_pareto else DASH),
    ]
    stages: list[tuple[str, str, str]] = []
    active_taken = False
    for name, present, detail in facts:
        if present:
            state = "done"
        elif not active_taken:
            state, active_taken = "active", True
        else:
            state = "todo"
        stages.append((name, state, detail))
    return stages


# --- value formatting ---------------------------------------------------------------------


def num(value: Any, fmt: str = "{:.3f}") -> str:
    """A number as the console shows it, or an em-dash. Never a stand-in zero."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return DASH
    return fmt.format(value)


def txt(value: Any) -> str:
    """A string as the console shows it, escaped, or an em-dash when it is absent."""
    if value is None or value == "":
        return DASH
    return escape(str(value))


def _panel(pid: str, title: str, body: str, meta: str = "") -> str:
    return (
        f'<section id="{pid}" class="panel"><header class="ph"><h2>{escape(title)}</h2>'
        f'<span class="meta mono">{meta}</span></header>{body}</section>'
    )


def _note(text: str) -> str:
    """An empty state: a composed line saying what happens next, never the bare word."""
    return f'<p class="note">{text}</p>'


# --- svg ----------------------------------------------------------------------------------


def _scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) / (hi - lo) * (out_hi - out_lo)


def line_chart(
    points: list[Point], metric: str, title: str, fmt: str, pending: tuple[str, ...] = ()
) -> str:
    """``metric`` over iterations: measured points solid and filled, pending ones hollow.

    ``pending`` are iterations we hold no measurement for. They are drawn on a dotted Ash
    line with hollow dots and no value, so the chart cannot imply a number we never took.
    """
    pairs = [(p.label, p.get(metric)) for p in points]
    known = [(label, v) for label, v in pairs if v is not None]
    if not known:
        return (
            f'<div class="chart {escape(metric)} empty"><h4>{escape(title)}</h4>'
            f'<p class="note">no {escape(title)} recorded in these summaries</p></div>'
        )
    w, h, pad = 260.0, 130.0, 26.0
    values = [v for _, v in known]
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    lo, hi = lo - span * 0.1, hi + span * 0.1
    slots = len(known) + len(pending)
    xs = [_scale(i, 0, max(slots - 1, 1), pad, w - pad / 2) for i in range(slots)]
    ys = [_scale(v, lo, hi, h - pad, pad / 2) for v in values]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=False))
    rest = ""
    if pending:
        mid = (h - pad + pad / 2) / 2
        tail = [(x, ys[-1] if ys else mid) for x in xs[len(known):]]
        anchor = (xs[len(known) - 1], ys[-1]) if known else (xs[0], mid)
        d = " ".join(f"L{x:.1f},{y:.1f}" for x, y in tail)
        rest = (
            f'<path d="M{anchor[0]:.1f},{anchor[1]:.1f} {d}" class="pending"/>'
            + "".join(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" class="hollow"><title>'
                f"{escape(label)}: not measured yet</title></circle>"
                for label, (x, y) in zip(pending, tail, strict=True)
            )
        )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"><title>{escape(label)}: '
        f"{escape(fmt.format(v))}</title></circle>"
        for (label, v), x, y in zip(known, xs, ys, strict=False)
    )
    return (
        f'<div class="chart {escape(metric)}"><h4>{escape(title)}</h4>'
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{escape(title)}">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad / 2}" y2="{h - pad}" class="axis"/>'
        f'<line x1="{pad}" y1="{pad / 2}" x2="{pad}" y2="{h - pad}" class="axis"/>'
        f'<polyline points="{path}" class="series"/>{rest}{dots}'
        f'<text x="{pad}" y="{pad / 2 - 2}" class="tick">{escape(fmt.format(hi))}</text>'
        f'<text x="{pad}" y="{h - pad + 12}" class="tick">{escape(fmt.format(lo))}</text>'
        f"</svg>"
        f'<p class="latest">now <b class="mono">{escape(fmt.format(known[-1][1]))}</b>'
        f", after round "
        f'<b class="mono">{escape(known[-1][0])}</b></p></div>'
    )


def pareto_chart(points: list[Point]) -> str:
    """Score against $/task: dominated points dim, the front connected, radius by p95.

    Nothing with an ``r="`` attribute may precede the marks in this SVG — the radius test
    reads them positionally.
    """
    usable = [p for p in points if p.score is not None and p.cost_per_task is not None]
    if not usable:
        return (
            '<div class="chart empty"><h4>Pareto</h4>'
            '<p class="note">No pareto.json yet — run <code>anneal anneal</code> to cool the '
            "winner down a model tier at a time.</p></div>"
        )
    w, h, pad = 720.0, 260.0, 44.0
    costs = [p.cost_per_task or 0.0 for p in usable]
    scores = [p.score or 0.0 for p in usable]
    p95s = [p.p95_latency_ms for p in usable if p.p95_latency_ms is not None]
    lo_p, hi_p = (min(p95s), max(p95s)) if p95s else (0.0, 0.0)
    marks, front_xy = [], []
    for point in usable:
        x = _scale(point.cost_per_task or 0.0, min(costs), max(costs), pad, w - pad)
        y = _scale(point.score or 0.0, min(scores), max(scores), h - pad, pad)
        r = 5.0 if point.p95_latency_ms is None else _scale(point.p95_latency_ms, lo_p, hi_p, 4, 13)
        p95_text = "n/a" if point.p95_latency_ms is None else f"{point.p95_latency_ms:.0f}ms"
        on_front = bool(point.extra.get("on_front"))
        if on_front:
            front_xy.append((point.cost_per_task or 0.0, x, y))
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'class="pt{" front" if on_front else ""}"><title>'
            f"{escape(point.label)}{' (front)' if on_front else ''}: score {point.score:.3f}, "
            f"${point.cost_per_task:.4f}/task, p95 {escape(p95_text)}</title></circle>"
        )
    line = ""
    if len(front_xy) > 1:
        pts = " ".join(f"{x:.1f},{y:.1f}" for _, x, y in sorted(front_xy))
        line = f'<polyline points="{pts}" class="front-line"/>'
    return (
        '<div class="chart wide"><h4>score vs $/task &mdash; point size is p95, '
        "filled points are the front</h4>"
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="Pareto scatter">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="axis"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" class="axis"/>'
        f'<text x="{pad}" y="{pad - 10}" class="tick">score {max(scores):.3f}</text>'
        f'<text x="{pad}" y="{h - pad + 16}" class="tick">${min(costs):.4f}</text>'
        f'<text x="{w - pad - 60}" y="{h - pad + 16}" class="tick">${max(costs):.4f}</text>'
        f"{''.join(marks)}{line}</svg></div>"
    )


# --- the contract ---------------------------------------------------------------------------
#
# goal.md, tools.yaml and eval.py are the only three things Anneal is given. They are read
# here as text (never imported: anneal/ does not depend on domains/) and shown as definition
# rows above the timeline, because everything below them is an answer to that contract.


def _domain_dir(runs_dir: Path | str, domain: str, summary: dict[str, Any]) -> Path | None:
    """The domain's input directory: the path the run recorded, else this repo's copy."""
    recorded = summary.get("domain_path")
    candidates = [Path(str(recorded))] if recorded else []
    candidates.append(Path(__file__).resolve().parent.parent / "domains" / domain)
    candidates.append(Path(runs_dir).parent / "domains" / domain)
    return next((p for p in candidates if p.is_dir()), None)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _goal_line(path: Path) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line
    return None


def _tools_line(path: Path) -> str | None:
    data = _read_yaml(path)
    tools = (data or {}).get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    names = [str(t.get("name")) for t in tools if isinstance(t, dict) and t.get("name")]
    shown = ", ".join(names[:3])
    more = f" +{len(names) - 3}" if len(names) > 3 else ""
    return f"{len(tools)} tools · {shown}{more}" if names else f"{len(tools)} tools"


def _scorer_line(path: Path) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("THRESHOLD"):
            _, _, value = line.partition("=")
            return f"eval.py · THRESHOLD {value.strip()}"
    return "eval.py"


def vocab_strip_goal(line: str | None) -> str | None:
    """Drop a leading "Goal:" from goal.md's heading; the row already says GOAL."""
    if not line:
        return line
    lowered = line.lower()
    for prefix in vocab.GOAL_PREFIXES:
        if lowered.startswith(prefix):
            return line[len(prefix):].strip() or line
    return line


def load_contract(runs_dir: Path | str, domain: str, summary: dict[str, Any]) -> dict[str, str]:
    """``{GOAL, TOOLS, SCORER}``; anything not on disk stays absent and renders em-dashed."""
    root = _domain_dir(runs_dir, domain, summary)
    if root is None:
        return {}
    found = {
        # the heading in goal.md reads "# Goal: <thing>"; the row is already labelled GOAL, so
        # printing the prefix again would show "GOAL  Goal: airline customer support agent"
        "goal": vocab_strip_goal(_goal_line(root / "goal.md")),
        "tools": _tools_line(root / "tools.yaml"),
        "scorer": _scorer_line(root / "eval.py"),
    }
    return {k: v for k, v in found.items() if v}


def agent_title(runs_dir: Path | str, domain: str, summary: dict[str, Any]) -> str:
    """What this agent is *for*, in plain English, for a reader who has never seen the repo.

    A directory name like ``airline`` or ``invoices`` means nothing to someone arriving at the
    console; it is a CLI argument, not a description. Every goal.md opens with a title line
    ("# Goal: AP invoice triage"), including the ones ``anneal init`` writes from the user's own
    answer, so that line is the name the console shows. The slug stays visible beside it because
    it is what you type to run the thing; it is just no longer the only label.
    """
    root = _domain_dir(runs_dir, domain, summary)
    line = _goal_line(root / "goal.md") if root else None
    if not line:
        return domain.replace("_", " ").replace("-", " ")
    lowered = line.lower()
    for prefix in vocab.GOAL_PREFIXES:
        if lowered.startswith(prefix):
            line = line[len(prefix):].strip()
            break
    return line[:60] or domain


def agent_titles(runs_dir: Path | str, domains: list[str]) -> dict[str, str]:
    """``{slug: plain-English name}`` for every domain the switcher offers."""
    return {d: agent_title(runs_dir, d, {}) for d in domains}


CONTRACT_ROWS = (
    ("goal", "GOAL", "what the agent is asked to do"),
    ("tools", "TOOLS", "what it is allowed to call"),
    ("scorer", "SCORER", "what counts as a pass"),
)


def render_contract(
    contract: dict[str, str], domain: str | None, title: str | None = None
) -> str:
    """The three inputs, always visible: mono label left, value right."""
    rows = "".join(
        f'<div class="drow"><span class="dk mono">{escape(label)}</span>'
        f'<span class="dv mono">{txt(contract.get(key))}</span>'
        f'<span class="dh">{escape(hint)}</span></div>'
        for key, label, hint in CONTRACT_ROWS
    )
    name = escape(title or domain or DASH)
    slug = f'<span class="dslug mono">{escape(domain)}</span>' if domain else ""
    return (
        f'<div id="contract" class="contract"><div class="cname">{name}{slug}</div>'
        f"{rows}</div>"
    )


# --- the step timeline ----------------------------------------------------------------------

# The stage keys are the loop's own and stay in URLs and run files. What the reader sees is
# vocab.STEPS, so the console's wording changes in one file rather than in string literals here.
STEP_META: tuple[tuple[str, str, str], ...] = tuple(
    (key, name, blurb) for key, (name, blurb) in vocab.STEPS.items()
)

CHEVRON = (
    '<svg class="chev" viewBox="0 0 12 12" aria-hidden="true">'
    '<path d="M4.5 2.5 L8 6 L4.5 9.5"/></svg>'
)


def default_step(stages: list[tuple[str, str, str]]) -> str:
    """The step the loop is on: the first without its artefact, else the last one done."""
    for (key, _, _), (_, state, _) in zip(STEP_META, stages, strict=True):
        if state == "active":
            return key
    return STEP_META[-1][0]


def _step_row(key: str, name: str, desc: str, state: str, status: str, index: int,
              domain: str | None, open_key: str) -> str:
    # The flow sits well down the page, so a bare "?step=" link would reload to the top and
    # lose the reader's place. The fragment lands them back on the flow; .flow's
    # scroll-margin-top keeps the header from covering it.
    href = f"?step={key}" + (f"&amp;domain={escape(domain)}" if domain else "") + "#flow"
    is_open = "true" if key == open_key else "false"
    return (
        f'<li class="step" data-state="{state}" data-open="{is_open}" style="--i:{index}">'
        f'<a href="{href}"><span class="dot"></span>'
        f'<span class="sname mono">{escape(name)}</span>'
        f'<span class="sdesc">{escape(desc)}</span>'
        f'<span class="sstat mono">{escape(status)}</span>{CHEVRON}</a></li>'
    )


def render_timeline(
    it: Iteration,
    issues: list[dict[str, Any]],
    fronts: dict[str, list[Point]],
    step: str | None = None,
) -> str:
    """The six steps, one expanded. Only the open step's artefact is in the document."""
    stages = pipeline_stages(it, issues)
    open_key = step if any(step == key for key, _, _ in STEP_META) else default_step(stages)
    rows = "".join(
        _step_row(key, name, desc, state, status, i, it.domain if it.path else None, open_key)
        for i, ((key, name, desc), (_, state, status)) in enumerate(
            zip(STEP_META, stages, strict=True)
        )
    )
    title = next(name for key, name, _ in STEP_META if key == open_key)
    detail = step_detail(open_key, it, issues, fronts)
    return (
        f'<div id="timeline" class="flow" tabindex="-1"><ol class="steps">{rows}</ol>'
        f'<section class="detail" data-step="{escape(open_key)}">'
        f'<header class="ph"><h2>{escape(title)}</h2>'
        f'<span class="meta">round {escape(it.label)}</span></header>'
        f"{detail}</section></div>"
    )


def step_detail(
    key: str, it: Iteration, issues: list[dict[str, Any]], fronts: dict[str, list[Point]]
) -> str:
    """The artefact belonging to one step, and nothing else."""
    if key == "architect":
        return render_specs(it)
    if key == "run":
        return render_run(it)
    if key == "diagnose":
        return render_ledger(issues)
    if key == "mutate":
        return render_mutation(it)
    if key == "gate":
        return render_gate(it.gate)
    return render_pareto(fronts)


# --- step artefacts -------------------------------------------------------------------------


def _node_pills(spec: dict[str, Any] | None) -> str:
    """The topology as connected node pills, each carrying its model tier badge."""
    nodes = (spec or {}).get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return f'<span class="pills"><span class="pill mono absent">{DASH}</span></span>'
    pills = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = escape(vocab.role(node.get("role") or node.get("name")))
        tier = node.get("model_tier")
        cls = "tier" if tier else "tier absent"
        badge = escape(vocab.tier(tier)) if tier else DASH
        pills.append(f'<span class="pill">{name}<b class="{cls}">{badge}</b></span>')
    return '<span class="pills">' + '<i class="link"></i>'.join(pills) + "</span>"


def _roles(it: Iteration, cid: str) -> str:
    summary = it.summary
    chips = []
    if cid == summary.get("incumbent_id"):
        chips.append('<span class="chip">current best</span>')
    if cid == summary.get("candidate_id"):
        decision = str(summary.get("decision") or "")
        cls = {"promote": "chip ok", "reject": "chip bad"}.get(decision, "chip")
        chips.append('<span class="chip">this round\'s try</span>')
        if decision:
            chips.append(f'<span class="{cls}">{escape(vocab.decision(decision))}</span>')
    if cid == summary.get("winner_id"):
        chips.append('<span class="chip ok">winner</span>')
    return "".join(chips)


def render_specs(it: Iteration) -> str:
    """Architect's artefact: the candidate specs this iteration is working from."""
    ids = sorted(it.search) or sorted(it.specs)
    if not ids:
        return _note(
            "No candidate specs on disk for this iteration — "
            "<code>uv run anneal run domains/&lt;name&gt;</code> writes them here."
        )
    rows = "".join(
        f'<tr style="--i:{i}"><td>{escape(vocab.design_name(cid))}'
        f'<span class="cid mono">{escape(cid)}</span>{_roles(it, cid)}</td>'
        f'<td>{escape(vocab.topology((it.specs.get(cid) or {}).get("topology")))}'
        f"{_node_pills(it.specs.get(cid))}</td>"
        f'<td class="mono num">{num((it.specs.get(cid) or {}).get("step_budget"), "{:.0f}")}</td>'
        "</tr>"
        for i, cid in enumerate(ids)
    )
    # Three columns, not four: how an agent is wired and the parts it is wired from are one
    # idea, and four columns of identifiers overflowed the panel and were clipped on the right.
    return (
        '<div class="tblwrap"><table class="tbl"><thead><tr><th>design</th>'
        "<th>how it is wired</th><th>step limit</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _pips(score: Any) -> str:
    if score is None or isinstance(score, bool) or not isinstance(score, int | float):
        return f'<span class="pips"><span class="mono absent">{DASH}</span></span>'
    filled = round(max(0.0, min(1.0, float(score))) * 5)
    dots = "".join(f'<i class="{"on" if i < filled else ""}"></i>' for i in range(5))
    return f'<span class="pips">{dots}</span>'


def _status(row: dict[str, Any]) -> str:
    flags = [
        ("hard_fail", "chip bad", "hard fail"),
        ("hit_step_budget", "chip warn", "step budget"),
        ("schema_error", "chip warn", "schema"),
    ]
    chips = [f'<span class="{cls}">{label}</span>' for key, cls, label in flags if row.get(key)]
    return "".join(chips) or f'<span class="mono absent">{DASH}</span>'


def render_run(it: Iteration) -> str:
    """Run's artefact: what each candidate scored, then the task rows behind that score."""
    scores = "".join(
        f'<tr style="--i:{i}"><td class="mono id">{escape(cid)}{_roles(it, cid)}</td>'
        f'<td class="scorecell"><span class="bar">'
        f'{_score_bar(_num(m if isinstance(m, dict) else {}, SCORE_KEYS))}</span>'
        f'<span class="mono">{num(_num(m if isinstance(m, dict) else {}, SCORE_KEYS))}</span></td>'
        f'<td class="mono num">{num(_num(m if isinstance(m, dict) else {}, COST_KEYS), "${:.4f}")}'
        "</td>"
        f'<td class="mono num">'
        f'{num(_num(m if isinstance(m, dict) else {}, P95_KEYS), "{:.0f}ms")}</td>'
        f'<td class="mono num">'
        f'{num((m if isinstance(m, dict) else {}).get("hard_fails"), "{:.0f}")}</td></tr>'
        for i, (cid, m) in enumerate(sorted(it.search.items()))
    )
    if not scores and not it.live:
        return _note(
            "No scored tasks for this iteration yet — the runner appends one line per task "
            "as it finishes."
        )
    table = (
        '<table class="tbl"><thead><tr><th>candidate</th><th>score</th><th>$/task</th>'
        f"<th>p95</th><th>hard fails</th></tr></thead><tbody>{scores}</tbody></table>"
        if scores
        else ""
    )
    if not it.live:
        return table + _note("No per-task rows on disk for this candidate yet.")
    hard = sum(1 for row in it.live if row.get("hard_fail"))
    rows = "".join(
        f'<tr class="lrow{" bad" if row.get("hard_fail") else ""}" style="--i:{i}">'
        f'<td class="mono">{txt(row.get("task_id"))}</td>'
        f"<td>{_pips(row.get('score'))}</td>"
        f'<td class="mono num">{num(row.get("score"))}</td>'
        f'<td class="mono num">{num(row.get("latency_ms"), "{:.0f}ms")}</td>'
        f'<td class="st">{_status(row)}</td></tr>'
        for i, row in enumerate(it.live[:60])
    )
    label = (
        f'<div class="sublabel mono">per task · {escape(it.live_candidate or DASH)} · '
        f"{len(it.live)} tasks · {hard} hard fails</div>"
    )
    return (
        table + label + '<div class="scroll"><table class="tbl"><thead><tr><th>task</th>'
        "<th>score</th><th></th><th>latency</th><th>status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _score_bar(score: float | None) -> str:
    return "" if score is None else f'<i style="width:{max(0.0, min(1.0, score)) * 100:.1f}%"></i>'


def _next_operator(row: dict[str, Any]) -> str:
    """The operator Mutate would reach for next: first in the class ladder not yet tried."""
    entry = taxonomy().get(str(row.get("class")))
    operators = (entry or {}).get("operators")
    tried = row.get("operators_tried")
    tried = tried if isinstance(tried, list) else []
    if not isinstance(operators, list) or not operators:
        return DASH  # a class the taxonomy does not carry; guessing an operator would be a lie
    remaining = [op for op in operators if op not in tried]
    return escape(vocab.operator(remaining[0])) if remaining else "every repair tried"


def _severity(row: dict[str, Any]) -> Any:
    return (taxonomy().get(str(row.get("class"))) or {}).get("severity")


def _rank(row: dict[str, Any]) -> float:
    """severity x count, so the class costing the most accuracy sits at the top."""
    count = row.get("count")
    count = float(count) if isinstance(count, int | float) and not isinstance(count, bool) else 0.0
    severity = _severity(row)
    severity = float(severity) if isinstance(severity, int | float) else 0.0
    return severity * count


def _tried(row: dict[str, Any]) -> str:
    """The operators already spent on this issue, under the one coming next.

    This is the visible evidence that the loop does not retry what it has already tried, so it
    belongs beside the next operator rather than in a column of its own.
    """
    tried = [vocab.operator(o) for o in row.get("operators_tried") or []]
    if not tried:
        return ""
    return f'<span class="tried">already tried: {escape("; ".join(tried))}</span>'


def render_ledger(ledger: list[dict[str, Any]]) -> str:
    """Diagnose's artefact: failure classes ranked by severity x count, with the next operator."""
    if not ledger:
        return _note("Ledger is empty — no failures have been diagnosed from traces yet.")
    ranked = sorted(ledger, key=_rank, reverse=True)
    head = "".join(f"<th>{escape(c)}</th>" for c in LEDGER_COLUMNS[2:])
    rows = "".join(
        f'<tr class="lgrow" style="--i:{i}" title="{escape(str(row.get("id") or ""))} · '
        f'{escape(str(row.get("domain") or ""))}">'
        f'<td class="cls">{escape(vocab.failure(row.get("class")))}'
        f'<span class="dslug mono">{txt(row.get("class"))}</span></td>'
        f"<td>{escape(vocab.role(row.get('node')))}</td>"
        f'<td class="mono num">{num(_severity(row), "{:.0f}")}</td>'
        f'<td class="mono num">{txt(row.get("count"))}</td>'
        f'<td class="mono">{txt(row.get("status"))}</td>'
        f'<td class="op">{_next_operator(row)}{_tried(row)}</td></tr>'
        for i, row in enumerate(ranked)
    )
    # A wide table scrolls inside its own panel rather than pushing the page sideways
    # (DESIGN.md 5): the rightmost column is "operators tried", which grows every iteration.
    return (
        f'<div class="tblwrap"><table class="tbl"><thead><tr>{head}</tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
    )


def _node_names(spec: dict[str, Any] | None) -> list[str]:
    nodes = (spec or {}).get("nodes")
    if not isinstance(nodes, list):
        return []
    return [str(n.get("name")) for n in nodes if isinstance(n, dict) and n.get("name")]


def _prompt_changes(parent: dict[str, Any], child: dict[str, Any]) -> str:
    """Which node prompts the operator rewrote — a prompt operator changes no nodes at all."""
    def refs(spec: dict[str, Any]) -> dict[str, str]:
        nodes = spec.get("nodes")
        if not isinstance(nodes, list):
            return {}
        return {
            str(n.get("name")): str(n.get("system_prompt_ref"))
            for n in nodes
            if isinstance(n, dict) and n.get("name") and n.get("system_prompt_ref")
        }

    before, after = refs(parent), refs(child)
    moved = [
        f"{name}: {before[name]} &rarr; {ref}"
        for name, ref in after.items()
        if name in before and before[name] != ref
    ]
    return escape(", ".join(moved)).replace("&amp;rarr;", "&rarr;") if moved else DASH


def render_mutation(it: Iteration) -> str:
    """Mutate's artefact: the operator, the issue it answers, and how the spec changed."""
    operator = it.summary.get("operator")
    child_id = it.summary.get("candidate_id")
    if not operator or not child_id:
        return _note(
            "No mutation this iteration — Mutate runs once the ledger names an issue with an "
            "operator left to try."
        )
    child = it.specs.get(str(child_id)) or {}
    parent_id = (child.get("lineage") or {}).get("parent") or it.summary.get("incumbent_id")
    parent = it.specs.get(str(parent_id)) or {}
    issue = it.summary.get("issue") if isinstance(it.summary.get("issue"), dict) else {}
    before, after = _node_names(parent), _node_names(child)
    added = [n for n in after if n not in before]
    removed = [n for n in before if n not in after]
    rows = (
        ("OPERATOR", txt(operator)),
        ("ISSUE", f'{txt(issue.get("class"))} · {txt(issue.get("node"))} · {txt(issue.get("id"))}'),
        ("PARENT", txt(parent_id)),
        ("CHALLENGER", txt(child_id)),
        ("PROMPTS", _prompt_changes(parent, child)),
        ("NODES ADDED", ", ".join(escape(n) for n in added) or DASH),
        ("NODES REMOVED", ", ".join(escape(n) for n in removed) or DASH),
        ("STEP BUDGET", f'{num(parent.get("step_budget"), "{:.0f}")} &rarr; '
                        f'{num(child.get("step_budget"), "{:.0f}")}'),
    )
    body = "".join(
        f'<div class="drow"><span class="dk mono">{label}</span>'
        f'<span class="dv mono">{value}</span></div>'
        for label, value in rows
    )
    return f'<div class="defs">{body}</div><div class="pillrow">{_node_pills(child)}</div>'


GATE_SIDES = (
    ("mean_score", "how often it is right", "{:.3f}"),
    ("pass3_rate", "right 3 times running", "{:.3f}"),
    ("hard_fails", "serious mistakes", "{:.0f}"),
    ("gen_gap", "worse on unseen tasks by", "{:+.3f}"),
)

# The exact binomial paired test, said out loud. p is the probability of seeing a win record
# this good if the change did nothing, so the honest label is what it measures, not its letter.
GATE_STATS = (
    ("p", "chance this was luck", "{:.3f}"),
    ("alpha", "has to be under", "{:.2f}"),
    ("wins", "tasks it did better on", "{:.0f}"),
    ("losses", "tasks it did worse on", "{:.0f}"),
    ("min_discordant_to_promote", "wins needed to pass", "{:.0f}"),
)


def _gate_side(gate: dict[str, Any], key: str) -> dict[str, Any]:
    block = gate.get(key)
    return block if isinstance(block, dict) else {}


# The field names gate.py writes into its reason line, and how the console says them.
GATE_REASON_WORDS = (
    ("hard_fails", "serious mistakes"),
    ("incumbent", "the current best's"),
    ("alpha", "the limit"),
    ("p", "chance it was luck"),
    ("pass3", "right 3 times running"),
    ("gen_gap", "the drop on unseen tasks"),
)


def _gate_reason(gate: dict[str, Any]) -> str:
    """The gate's own reason line, with its two field names read out as words.

    gate.py writes a machine-readable reason ("hard_fails 3 > incumbent 1", "p 1.000 >= alpha
    0.1") because it is also what the report and the tests quote. Showing it verbatim to a
    reader means showing them two identifiers and a comparison operator, so the two terms are
    translated here and the sentence is left otherwise exactly as the gate wrote it.
    """
    reason = str(gate.get("reason") or "")
    if not reason:
        return DASH
    # whole words only: a bare replace would rewrite the "p" inside "promoted"
    for term, phrase in GATE_REASON_WORDS:
        reason = re.sub(rf"(?<![\w-]){re.escape(term)}(?![\w-])", phrase, reason)
    for symbol, phrase in ((">=", "is not below"), ("<=", "is not above"),
                           (">", "is more than"), ("<", "is less than")):
        reason = reason.replace(symbol, phrase)
    return escape(reason)


def render_gate(gate: dict[str, Any]) -> str:
    """Gate's artefact: the verdict, the paired-test grid, and the line it wrote to explain it."""
    if not gate:
        return _note("No gate.json for this iteration — the gate runs once a challenger exists.")
    decision = str(gate.get("decision") or "")
    cls = {"promote": "verdict ok", "reject": "verdict bad"}.get(decision, "verdict")
    candidate, incumbent = _gate_side(gate, "candidate"), _gate_side(gate, "incumbent")
    sides = "".join(
        f'<tr><th>{escape(label)}</th><td class="mono num">{num(incumbent.get(key), fmt)}</td>'
        f'<td class="mono num">{num(candidate.get(key), fmt)}</td></tr>'
        for key, label, fmt in GATE_SIDES
    )
    stats = "".join(
        f'<div class="stat"><span class="k">{escape(label)}</span>'
        f'<span class="v mono">{num(gate.get(key), fmt)}</span></div>'
        for key, label, fmt in GATE_STATS
    )
    if gate.get("underpowered"):
        stats += (
            '<div class="stat"><span class="k">power</span>'
            '<span class="v mono">underpowered</span></div>'
        )
    return (
        f'<p class="{cls}">{escape(vocab.decision(decision)) if decision else DASH}</p>'
        f'<table class="tbl gate"><thead><tr><th></th><th>current best</th>'
        f"<th>this round's try</th></tr>"
        f"</thead><tbody>{sides}</tbody></table>"
        f'<div class="stats">{stats}</div>'
        f'<p class="reason">{_gate_reason(gate)}</p>'
    )


def render_pareto(fronts: dict[str, list[Point]]) -> str:
    """Anneal's artefact: the cost/score front the downshift search produced."""
    if not fronts:
        return _note(
            "No pareto.json yet — <code>anneal anneal</code> walks the winner down the model "
            "ladder and writes the front here."
        )
    blocks = []
    for domain, points in fronts.items():
        front = [p.label for p in points if p.extra.get("on_front")]
        line = (
            f'<div class="sublabel mono">{escape(domain)} · on the front: '
            f'{escape(", ".join(front)) if front else DASH}</div>'
        )
        blocks.append(line + pareto_chart(points))
    return "".join(blocks)


def _curves_for(domain: str, points: list[Point], pending: tuple[str, ...]) -> str:
    charts = "".join(line_chart(points, key, title, fmt, pending) for key, title, fmt in METRICS)
    return (
        f'<div class="pblock"><h3 class="mono">{escape(domain)}</h3>'
        f'<div class="charts">{charts}</div></div>'
    )


def plural_rounds(n: int) -> str:
    """"1 round" / "4 rounds": the loop's iterations, counted the way a reader counts."""
    return f"{n} round" if n == 1 else f"{n} rounds"


def render_curves(
    summaries: dict[str, list[Point]], pending: dict[str, tuple[str, ...]] | None = None
) -> str:
    """The measured curve per metric; iterations we hold no summary for stay hollow."""
    pending = pending or {}
    if not summaries:
        return _panel(
            "curves",
            "Measurements",
            _note(
                "No runs yet — <code>uv run anneal run domains/&lt;name&gt;</code> and this "
                "fills in one point per iteration."
            ),
        )
    body = "".join(
        _curves_for(d, pts, pending.get(d, ())) for d, pts in summaries.items()
    )
    iters = sum(len(pts) for pts in summaries.values())
    return _panel("curves", "Measurements", body, f"{plural_rounds(iters)} so far")


def render_balance(balance: float | None) -> str:
    """The Dodo credit balance, or an em-dash when billing has not cached one."""
    text = DASH if balance is None else f"${balance:,.2f}"
    return (
        '<div id="balance" class="meter"><span class="k">credits</span>'
        f'<span class="v mono">{escape(text)}</span></div>'
    )


def render_topbar(
    domain: str | None,
    domains: list[str],
    it: Iteration | None,
    titles: dict[str, str] | None = None,
) -> str:
    """Wordmark, the agents present in runs/, the budget meter and the credit balance."""
    titles = titles or {}
    if domains:
        links = "".join(
            f'<a class="dom{" on" if d == domain else ""}" href="?domain={escape(d)}">'
            f'<span class="dname">{escape(titles.get(d) or d)}</span>'
            f'<span class="dslug mono">{escape(d)}</span></a>'
            for d in domains
        )
    else:
        links = f'<span class="dom absent mono">{DASH}</span>'
    summary = it.summary if it else {}
    spend, budget = summary.get("spend_usd"), summary.get("budget_usd")
    used = 0.0
    if isinstance(spend, int | float) and isinstance(budget, int | float) and budget:
        used = max(0.0, min(1.0, float(spend) / float(budget)))
    meter = (
        '<div class="meter"><span class="k">budget</span>'
        f'<span class="v mono">{num(spend, "${:.2f}")} / {num(budget, "${:.2f}")}</span>'
        f'<span class="bar wide"><i style="width:{used * 100:.1f}%"></i></span></div>'
    )
    return (
        '<header id="topbar" class="topbar"><a class="mark" href="/">ANNEAL</a>'
        f'<nav class="doms">{links}</nav>'
        f'<div class="meters">{meter}{render_balance(credit_balance())}</div></header>'
    )


CSS = """
/* Tokens from .stitch/DESIGN.md: Paper canvas, one bordered Panel, Ink text, a single orange
   accent (the running step's dot and the measured curve), 1px Rule separation, no shadows. */
:root { --paper:#F7F7F5; --panel:#FFFFFF; --ink:#111214; --graphite:#5B6068; --ash:#8A9099;
  --rule:#E4E4E1; --orange:#F4511E; --green:#2F9E79; --clay:#C4544F; color-scheme: light; }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font:400 13px/1.5 Geist,"Geist Sans",ui-sans-serif,sans-serif; letter-spacing:-0.01em; }
.mono, .mono * { font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums; letter-spacing:0; }
.wrap { max-width:1440px; margin:0 auto; padding:0 48px 64px; }
#flow { scroll-margin-top:72px; }
a { color:inherit; text-decoration:none; }
.absent { color:var(--ash); }
code { font-family:"Geist Mono",ui-monospace,monospace; color:var(--ink); }

.topbar { display:flex; align-items:center; gap:32px; height:72px; }
.mark { font-weight:600; letter-spacing:0.2em; font-size:13px; }
.doms { display:flex; gap:4px; flex:1; }
.dom { display:flex; flex-direction:column; gap:1px; padding:5px 10px;
  border:1px solid transparent; }
/* The agent's plain-English name leads; the slug under it is what you type on the CLI. */
.dname { font-size:12px; color:var(--ash); line-height:1.25; }
.dslug { font-size:10px; color:var(--ash); opacity:.7; }
.dom:hover .dname { color:var(--ink); }
.dom.on { border-color:var(--rule); background:var(--panel); }
.dom.on .dname { color:var(--ink); }
.cname { display:flex; align-items:baseline; gap:8px; font-size:15px; color:var(--ink);
  margin-bottom:16px; }
.cname .dslug { font-size:10px; text-transform:uppercase; letter-spacing:0.12em; }
/* An internal id is never what you read first. It sits under the phrase it belongs to, small
   and grey, because it is still what you type on the command line. */
.cid { display:block; font-size:10px; color:var(--ash); }
.cls .dslug { display:block; font-size:10px; margin-top:2px; }
.meters { display:flex; gap:28px; align-items:center; }
.meter { display:flex; align-items:center; gap:8px; font-size:10px; color:var(--ash);
  text-transform:uppercase; letter-spacing:0.12em;
  font-family:"Geist Mono",ui-monospace,monospace; }
.meter .v { color:var(--ink); font-size:12px; text-transform:none; letter-spacing:0; }
.bar { display:inline-block; height:3px; width:60px; background:var(--rule); }
.bar.wide { width:120px; }
.bar i { display:block; height:100%; background:var(--ink); }

/* the one place elevation exists, and it is a border */
.product { background:var(--panel); border:1px solid var(--rule); }
.strip { display:flex; align-items:center; justify-content:space-between; padding:12px 24px;
  border-bottom:1px solid var(--rule); font-size:11px; color:var(--ash);
  font-family:"Geist Mono",ui-monospace,monospace; }
.strip .right { display:flex; align-items:center; gap:8px; }
.strip .sdot { width:7px; height:7px; border-radius:50%; background:var(--ash); }
.strip .sdot.ok { background:var(--green); } .strip .sdot.bad { background:var(--clay); }

.contract { padding:24px; border-bottom:1px solid var(--rule); }
.contract .meta { font-size:10px; text-transform:uppercase; letter-spacing:0.12em;
  color:var(--ash); margin-bottom:12px; }
.drow { display:grid; grid-template-columns:110px minmax(0,1fr) 200px; gap:16px;
  padding:9px 0; border-top:1px solid var(--rule); align-items:baseline; }
.drow:first-of-type { border-top:0; }
.dk { font-size:10px; letter-spacing:0.12em; color:var(--ash); }
.dv { font-size:12px; color:var(--ink); overflow-wrap:anywhere; }
.dh { font-size:11px; color:var(--ash); text-align:right; }
.defs .drow { grid-template-columns:150px minmax(0,1fr); }
.pillrow { margin-top:14px; }

.panel { padding:24px; border-bottom:1px solid var(--rule); }
.ph { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:16px; }
.ph h2 { margin:0; font-size:11px; font-weight:500; text-transform:uppercase;
  letter-spacing:0.14em; color:var(--ash);
  font-family:"Geist Mono",ui-monospace,monospace; }
.ph .meta { font-size:11px; color:var(--ash); }
.note { color:var(--graphite); font-size:12px; margin:8px 0; max-width:65ch; }
.sublabel { font-size:10px; text-transform:uppercase; letter-spacing:0.12em; color:var(--ash);
  margin:20px 0 8px; }

/* the step timeline is the page's structure: steps left, the open step's artefact right */
.flow { display:grid; grid-template-columns:44fr 56fr; }
.steps { list-style:none; margin:0; padding:16px 0; border-right:1px solid var(--rule); }
.step { position:relative; animation:cascade .3s both; animation-delay:calc(var(--i) * 40ms); }
.step::before { content:""; position:absolute; left:31px; top:0; bottom:0; width:1px;
  background:var(--rule); }
.step:first-child::before { top:50%; } .step:last-child::before { bottom:50%; }
.step a { display:grid; grid-template-columns:24px minmax(0,1fr) auto 16px; gap:12px;
  align-items:center; padding:14px 24px; position:relative; }
.step a:hover { background:var(--paper); }
.step .dot { width:9px; height:9px; border-radius:50%; background:var(--rule);
  justify-self:center; position:relative; z-index:1; box-shadow:0 0 0 4px var(--panel); }
.step .sname { font-size:13px; color:var(--ash); }
.step .sdesc { display:none; font-size:12px; color:var(--ash); grid-column:2; }
.step .sstat { font-size:11px; color:var(--ash); }
.step .chev { width:12px; height:12px; fill:none; stroke:var(--rule); stroke-width:1.5; }
.step[data-state=done] .dot { background:var(--ash); }
.step[data-state=done] .sname { color:var(--ink); }
.step[data-state=active] .dot { background:var(--orange); animation:pulse 2.4s infinite; }
.step[data-state=active] .sname { color:var(--ink); font-weight:500; }
.step[data-open=true] a { background:var(--paper); }
.step[data-open=true] .sname { color:var(--ink); }
.step[data-open=true] .sdesc { display:block; }
.step[data-open=true] .chev { stroke:var(--ink); transform:rotate(90deg); }
.step[data-open=true] a::before { content:""; position:absolute; left:0; top:0; bottom:0;
  width:2px; background:var(--ink); }
.detail { padding:24px; min-width:0; }
@keyframes pulse { 0%,100% { transform:scale(1); opacity:1; }
  50% { transform:scale(1.65); opacity:.5; } }
@keyframes cascade { from { opacity:0; transform:translateY(4px); } to { opacity:1; } }

.tblwrap { overflow-x:auto; }
.tbl { border-collapse:collapse; width:100%; min-width:480px; table-layout:auto; }
.tbl th { text-align:left; font-weight:400; font-size:10px; text-transform:uppercase;
  letter-spacing:0.1em; color:var(--ash); padding:0 8px 8px;
  font-family:"Geist Mono",ui-monospace,monospace; }
.tbl td { padding:8px; border-top:1px solid var(--rule); font-size:12px;
  vertical-align:middle; }
.tbl tbody tr { animation:cascade .3s both; animation-delay:calc(var(--i) * 40ms); }
.tbl tbody tr:hover { background:var(--paper); }
.num { text-align:right; }
.lrow.bad td:first-child { box-shadow:inset 2px 0 0 var(--clay); }
.id { white-space:nowrap; }
.chip { display:inline-block; margin-left:6px; padding:1px 5px; font-size:9px;
  text-transform:uppercase; letter-spacing:0.08em; border:1px solid var(--rule);
  color:var(--ash); font-family:"Geist Mono",ui-monospace,monospace; }
.chip.ok { border-color:var(--green); color:var(--green); }
.chip.bad { border-color:var(--clay); color:var(--clay); }
.chip.warn { border-color:var(--ash); color:var(--graphite); }
.st .chip { margin-left:0; margin-right:4px; }
.pills { display:flex; align-items:center; flex-wrap:wrap; gap:2px; margin-top:6px; }
.pill { border:1px solid var(--rule); padding:2px 6px; font-size:10px; color:var(--ink);
  white-space:nowrap; }
.pill .tier { margin-left:6px; font-weight:400; color:var(--ash); }
.link { display:inline-block; width:10px; height:1px; background:var(--rule); }
.scorecell { white-space:nowrap; }
.scorecell .bar { width:56px; margin-right:8px; }
.scorecell .bar i { background:var(--ink); }
.pips { display:inline-flex; gap:3px; align-items:center; }
.pips i { width:6px; height:6px; background:var(--rule); display:block; }
.pips i.on { background:var(--ink); }
.scroll { max-height:380px; overflow:auto; }
.op { color:var(--ink); }
.tried { display:block; color:var(--ash); font-size:11px; margin-top:3px;
  overflow-wrap:anywhere; }
/* The two operator columns carry the longest strings in the console. Letting them wrap
   keeps all seven columns inside the panel at the width the flow leaves them; .tblwrap
   still scrolls on a genuinely narrow viewport. */
.op { overflow-wrap:anywhere; }

.verdict { margin:0 0 16px; font-size:28px; font-weight:600; letter-spacing:-0.03em;
  color:var(--ink); }
.verdict.ok { color:var(--green); } .verdict.bad { color:var(--clay); }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--rule);
  border:1px solid var(--rule); margin:16px 0; }
.stat { background:var(--panel); padding:10px 12px; display:flex; flex-direction:column;
  gap:2px; }
.stat .k { font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:var(--ash);
  font-family:"Geist Mono",ui-monospace,monospace; }
.stat .v { font-size:13px; }
.reason { color:var(--graphite); font-size:11px; margin:0; }

.pblock { margin-bottom:8px; }
.pblock h3 { margin:0 0 10px; font-size:10px; color:var(--ash); font-weight:400;
  text-transform:uppercase; letter-spacing:0.12em; }
.charts { display:flex; flex-wrap:wrap; gap:32px; }
.chart { width:260px; } .chart.wide { width:100%; }
.chart h4 { margin:0 0 8px; font-size:10px; font-weight:400; color:var(--ash);
  text-transform:uppercase; letter-spacing:0.1em;
  font-family:"Geist Mono",ui-monospace,monospace; }
svg { width:100%; height:auto; }
.axis { stroke:var(--rule); stroke-width:1; }
.series { fill:none; stroke:var(--ink); stroke-width:1.5; }
.chart circle { fill:var(--ink); }
/* the accent marks the measured score curve; iterations we hold no measurement for are
   hollow dots on a dotted Ash line, so the chart cannot imply a number we never took */
.chart.score .series { stroke:var(--orange); stroke-width:2; }
.chart.score circle { fill:var(--orange); }
.pending { fill:none; stroke:var(--ash); stroke-width:1; stroke-dasharray:2 3; }
circle.hollow { fill:var(--panel); stroke:var(--ash); stroke-width:1; }
.tick { fill:var(--ash); font-size:9px;
  font-family:"Geist Mono",ui-monospace,monospace; }
.latest { margin:8px 0 0; color:var(--graphite); font-size:11px; }
.latest b { color:var(--ink); font-weight:500; }
.pt { fill:var(--rule); stroke:var(--ash); }
.pt.front { fill:var(--green); fill-opacity:.85; stroke:var(--green); }
.front-line { fill:none; stroke:var(--green); stroke-width:1; stroke-opacity:.6; }

@media (max-width:768px) {
  .wrap { padding:0 16px 40px; }
  .flow { grid-template-columns:1fr; }
  .steps { border-right:0; border-bottom:1px solid var(--rule); }
  .drow, .defs .drow { grid-template-columns:1fr; gap:4px; }
  .dh { display:none; }
  .stats { grid-template-columns:repeat(2,1fr); }
  .charts { gap:24px; }
}
"""

SCRIPT = """
// Re-fetch each region every 10s and swap it only when its markup actually changed, so the
// cascade plays once per real update instead of flickering the page on a timer. Switching
// steps is a plain link; this only keeps the open step fresh.
const ids=['topbar','contract','timeline','curves'];
const last={};
setInterval(()=>{for(const id of ids){
  fetch('/fragments/'+id+location.search).then(r=>r.text()).then(html=>{
    if(html===last[id]) return;
    last[id]=html;
    const el=document.getElementById(id); if(el) el.outerHTML=html;
  }).catch(()=>{});
}},10000);
"""


@dataclass(frozen=True)
class AppState:
    """Where this dashboard reads from."""

    runs_dir: Path
    ledger_path: Path


def _selected(state: AppState, domain: str | None) -> tuple[str | None, list[str]]:
    domains = list_domains(state.runs_dir)
    if domain in domains:
        return domain, domains
    return (domains[0] if domains else None), domains


def _issues(state: AppState, domain: str | None) -> list[dict[str, Any]]:
    """Ledger rows, narrowed to the selected domain when that domain has any of its own."""
    rows = load_ledgers(state.runs_dir, state.ledger_path)
    mine = [row for row in rows if row.get("domain") == domain] if domain else []
    return mine or rows


def _curve_data(
    state: AppState, domain: str | None
) -> tuple[dict[str, list[Point]], dict[str, tuple[str, ...]]]:
    """Measured points for the selected domain, and which of its iterations are still pending.

    The page and the ``/fragments/curves`` refresh must narrow to the *same* domain. They used
    to disagree: the page resolved a default domain through ``_selected`` while the fragment
    only filtered when a ``?domain=`` was present, so opening ``/`` showed one domain's charts
    and then the ten-second refresh silently replaced them with all four stacked. Resolving the
    selection in one place is what keeps the switcher meaning what it says.
    """
    summaries = load_summaries(state.runs_dir)
    pending = {d: pending_iterations(state.runs_dir, d) for d in summaries}
    selected, _ = _selected(state, domain)
    if selected in summaries:
        summaries = {selected: summaries[selected]}
    return summaries, pending


def _blank(domain: str | None) -> Iteration:
    return Iteration(domain or DASH, None, DASH, {}, {}, {}, None, [], False)


def _iteration(state: AppState, domain: str | None) -> tuple[str | None, Iteration]:
    selected, _ = _selected(state, domain)
    if not selected:
        return None, _blank(None)
    return selected, load_iteration(state.runs_dir, selected)


def _fronts(state: AppState, domain: str | None) -> dict[str, list[Point]]:
    fronts = load_pareto(state.runs_dir)
    return {k: v for k, v in fronts.items() if k == domain} if domain else fronts


def _strip(it: Iteration) -> str:
    """The panel's header strip: the artefact path on the left, the run's state on the right."""
    decision = str(it.summary.get("decision") or "")
    dot = {"promote": "sdot ok", "reject": "sdot bad"}.get(decision, "sdot")
    word = escape(vocab.decision(decision)) if decision else DASH
    return (
        f'<div class="strip"><span class="mono">{escape(str(it.path)) if it.path else DASH}</span>'
        f'<span class="right"><span>{word}</span><span class="{dot}"></span></span></div>'
    )


def render_page(
    app_state: AppState, domain: str | None = None, step: str | None = None
) -> str:
    """The console: the contract, the measurements, and the step timeline with one step open."""
    selected, domains = _selected(app_state, domain)
    it = load_iteration(app_state.runs_dir, selected) if selected else _blank(None)
    issues = _issues(app_state, selected)
    summaries, pending = _curve_data(app_state, selected)
    contract = load_contract(app_state.runs_dir, selected or "", it.summary)
    title = agent_title(app_state.runs_dir, selected or "", it.summary)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Anneal console</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        f"{render_topbar(selected, domains, it, agent_titles(app_state.runs_dir, domains))}"
        f'<main class="product">{_strip(it)}'
        f"{render_contract(contract, selected, title)}"
        f"{render_curves(summaries, pending)}"
        f'<div id="flow">{render_timeline(it, issues, _fronts(app_state, selected), step)}</div>'
        f"</main></div><script>{SCRIPT}</script></body></html>"
    )


def create_app(runs_dir: Path | str = "runs", ledger_path: Path | str = "ledger.json") -> FastAPI:
    """FastAPI app serving the console plus one endpoint per region."""
    state = AppState(Path(runs_dir), Path(ledger_path))
    app = FastAPI(title="Anneal dashboard")
    app.state.anneal = state

    @app.get("/", response_class=HTMLResponse)
    def index(domain: str | None = None, step: str | None = None) -> str:
        return render_page(state, domain, step)

    @app.get("/landing", response_class=HTMLResponse)
    def landing() -> str:
        """The product landing page (also openable directly from landing/index.html)."""
        path = Path(__file__).resolve().parent.parent / "landing" / "index.html"
        if not path.exists():
            return "<html><body><p>landing/index.html is missing; see the repo.</p></body></html>"
        return path.read_text(encoding="utf-8")

    @app.get("/fragments/topbar", response_class=HTMLResponse)
    def topbar(domain: str | None = None) -> str:
        selected, domains = _selected(state, domain)
        return render_topbar(selected, domains, _iteration(state, domain)[1],
                             agent_titles(state.runs_dir, domains))

    @app.get("/fragments/contract", response_class=HTMLResponse)
    def contract(domain: str | None = None) -> str:
        selected, it = _iteration(state, domain)
        return render_contract(
            load_contract(state.runs_dir, selected or "", it.summary), selected,
            agent_title(state.runs_dir, selected or "", it.summary))

    @app.get("/fragments/timeline", response_class=HTMLResponse)
    def timeline(domain: str | None = None, step: str | None = None) -> str:
        selected, it = _iteration(state, domain)
        return render_timeline(it, _issues(state, selected), _fronts(state, selected), step)

    @app.get("/fragments/curves", response_class=HTMLResponse)
    def curves(domain: str | None = None) -> str:
        return render_curves(*_curve_data(state, domain))

    @app.get("/fragments/ledger", response_class=HTMLResponse)
    def ledger(domain: str | None = None) -> str:
        return render_ledger(_issues(state, _selected(state, domain)[0]))

    @app.get("/fragments/pareto", response_class=HTMLResponse)
    def pareto(domain: str | None = None) -> str:
        return render_pareto(_fronts(state, _selected(state, domain)[0]))

    @app.get("/fragments/balance", response_class=HTMLResponse)
    def balance() -> str:
        return render_balance(credit_balance())

    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the console on http://localhost:8000."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="anneal dashboard")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--ledger", default="ledger.json")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run(create_app(args.runs_dir, args.ledger), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
