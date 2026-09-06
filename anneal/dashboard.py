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
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from anneal import launcher, newagent, onboard, vocab

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
    "output",
)


# How much of an agent's answer is held in memory per task. Enough to read what it
# said, far short of the megabytes a full trace runs to.
OUTPUT_CLIP = 400


def load_live_rows(iter_dir: Path, candidate_id: str | None) -> tuple[str | None, list[dict]]:
    """Per-task rows for one candidate's search run: ``(candidate_id, rows)``.

    Only the ``search`` split is ever opened here; the reserved split belongs to the gate and
    this module must never read it. ``trace`` is dropped on the way in because a trace payload
    is megabytes; ``output`` is kept but clipped to :data:`OUTPUT_CLIP`, because the answer the
    agent actually gave is the one thing on this page a person who will never open a trace file
    can read, and dropping it left that column permanently empty.
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
            kept = {key: row.get(key) for key in LIVE_KEYS}
            output = kept.get("output")
            if isinstance(output, str) and len(output) > OUTPUT_CLIP:
                kept["output"] = output[:OUTPUT_CLIP]
            rows.append(kept)
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
            '<p class="note">Nothing has been made cheaper yet. Run <code>anneal anneal</code> to '
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
        '<div class="chart wide"><h4>score against cost, point size is the slowest run, '
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
    count = f"{len(tools)} tool" if len(tools) == 1 else f"{len(tools)} tools"
    if not names:
        return escape(count)
    shown = ", ".join(names[:3])
    more = f" +{len(names) - 3}" if len(names) > 3 else ""
    # the count is the fact; the function names are the machinery behind it
    return f'{escape(count)}<span class="dslug mono">{escape(shown + more)}</span>'


def _scorer_line(path: Path) -> str | None:
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("THRESHOLD"):
            _, _, value = line.partition("=")
            # The plain half is a sentence; the filename and the constant follow it as
            # secondary text that the plain reading level hides.
            share = value.strip()
            said = "Every answer has to be right" if share in {"1.0", "1"} else (
                f"At least {share} of each answer has to be right"
            )
            return (
                f'{escape(said)}<span class="dslug mono">'
                f'eval.py · THRESHOLD {escape(share)}</span>'
            )
    return '<span class="dslug mono">eval.py</span>'


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
        "goal": escape(vocab_strip_goal(_goal_line(root / "goal.md")) or ""),
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
    title = vocab.title_from_goal(root / "goal.md") if root else ""
    return title or domain.replace("_", " ").replace("-", " ")


def agent_titles(runs_dir: Path | str, domains: list[str]) -> dict[str, str]:
    """``{slug: plain-English name}`` for every domain the switcher offers."""
    return {d: agent_title(runs_dir, d, {}) for d in domains}


# Three inputs, named twice: the plain label a person reads and the one a developer expects.
# GOAL / TOOLS / SCORER are the loop's words for these, and they are worth keeping in the
# technical view because they are what the three files are called.
CONTRACT_ROWS = (
    ("goal", "What it should do", "GOAL", "the job you described"),
    ("tools", "What it can use", "TOOLS", "the tools you gave it"),
    ("scorer", "How we mark it", "SCORER", "what counts as getting it right"),
)


def render_contract(
    contract: dict[str, str], domain: str | None, title: str | None = None
) -> str:
    """The three inputs, always visible: mono label left, value right."""
    rows = "".join(
        f'<div class="drow"><span class="dk">{escape(plain)}'
        f'<span class="dslug mono">{escape(technical)}</span></span>'
        # _tools_line and _scorer_line escape their own text and append the machinery in a
        # <span>, so this value is already safe markup and must not be escaped a second time.
        f'<span class="dv">{contract.get(key) or DASH}</span>'
        f'<span class="dh">{escape(hint)}</span></div>'
        for key, plain, technical, hint in CONTRACT_ROWS
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
    href = f"/console?step={key}" + (f"&amp;domain={escape(domain)}" if domain else "") + "#flow"
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
    title, blurb = next((name, desc) for key, name, desc in STEP_META if key == open_key)
    detail = step_detail(open_key, it, issues, fronts)
    return (
        f'<div id="timeline" class="flow" tabindex="-1"><ol class="steps">{rows}</ol>'
        f'<section class="detail" data-step="{escape(open_key)}">'
        f'<header class="ph"><div><h2>{escape(title)}</h2>'
        f'<p class="phdesc">{escape(blurb)}</p></div>'
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
            "No designs for this round yet. "
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


def _secs(value: Any) -> str:
    """Milliseconds as seconds, which is the unit a reader thinks in.

    A latency the runner records as 80185 is eighty seconds; printed as "80185ms" it reads as
    a machine number nobody converts in their head. Anything under a second keeps two decimals
    so a fast run does not round to 0.0s.
    """
    number = value if isinstance(value, int | float) and not isinstance(value, bool) else None
    if number is None:
        return DASH
    seconds = float(number) / 1000
    return f"{seconds:.2f}s" if seconds < 1 else f"{seconds:.1f}s"


SAID_LIMIT = 160


def _plain_answer(output: Any) -> str:
    """The answer itself, without the container the evaluator needed it in.

    A structured domain asks its agent for an object, so the recorded answer is
    ``{'label': 'manager'}``. The reader wants "manager". A single field is shown as its
    value alone; several fields are shown as "field: value" pairs; anything that is not an
    object is shown as it was written.
    """
    if isinstance(output, str):
        stripped = output.strip()
        if stripped.startswith(("{", "[")):
            try:
                output = json.loads(stripped)
            except ValueError:
                return " ".join(stripped.split())
        else:
            return " ".join(stripped.split())
    if isinstance(output, dict):
        if len(output) == 1:
            return " ".join(str(next(iter(output.values()))).split())
        return ", ".join(f"{key}: {value}" for key, value in output.items())
    return " ".join(str(output or "").split())


def _said(row: dict[str, Any]) -> str:
    """What the agent actually replied on this task.

    Every other column is a measurement of the answer. This is the answer, and for someone who
    is not going to open a trace file it is the only place the agent's own words appear. Long
    replies are clipped in the cell and kept whole in the tooltip rather than dropped.
    """
    text = _plain_answer(row.get("output"))
    if not text:
        return f'<span class="mono absent">{DASH}</span>'
    shown = text if len(text) <= SAID_LIMIT else f"{text[:SAID_LIMIT].rstrip()}..."
    return f'<span title="{escape(text)}">{escape(shown)}</span>'


def render_run(it: Iteration) -> str:
    """Run's artefact: what each candidate scored, then the task rows behind that score."""
    scores = "".join(
        f'<tr style="--i:{i}"><td>{escape(vocab.design_name(cid))}'
        f'<span class="cid mono">{escape(cid)}</span>{_roles(it, cid)}</td>'
        f'<td class="scorecell"><span class="bar">'
        f'{_score_bar(_num(m if isinstance(m, dict) else {}, SCORE_KEYS))}</span>'
        f'<span class="mono">{num(_num(m if isinstance(m, dict) else {}, SCORE_KEYS))}</span></td>'
        f'<td class="mono num">{num(_num(m if isinstance(m, dict) else {}, COST_KEYS), "${:.4f}")}'
        "</td>"
        f'<td class="mono num">{_secs(_num(m if isinstance(m, dict) else {}, P95_KEYS))}</td>'
        f'<td class="mono num">'
        f'{num((m if isinstance(m, dict) else {}).get("hard_fails"), "{:.0f}")}</td></tr>'
        for i, (cid, m) in enumerate(sorted(it.search.items()))
    )
    if not scores and not it.live:
        return _note(
            "No scored tasks for this round yet. One line appears per task "
            "as it finishes."
        )
    table = (
        '<div class="tblwrap"><table class="tbl"><thead><tr><th>design</th>'
        "<th>how often it is right</th><th>cost per task</th>"
        f"<th>slowest runs</th><th>serious mistakes</th></tr></thead>"
        f"<tbody>{scores}</tbody></table></div>"
        if scores
        else ""
    )
    if not it.live:
        return table + _note("No per-task rows on disk for this candidate yet.")
    hard = sum(1 for row in it.live if row.get("hard_fail"))
    rows = "".join(
        f'<tr class="lrow{" bad" if row.get("hard_fail") else ""}" style="--i:{i}">'
        f'<td class="mono">{txt(row.get("task_id"))}</td>'
        f'<td class="said">{_said(row)}</td>'
        f"<td>{_pips(row.get('score'))}</td>"
        f'<td class="mono num">{num(row.get("score"))}</td>'
        f'<td class="mono num">{_secs(row.get("latency_ms"))}</td>'
        f'<td class="st">{_status(row)}</td></tr>'
        for i, row in enumerate(it.live[:60])
    )
    def plural(n: int, word: str) -> str:
        return f"{n} {word}" if n == 1 else f"{n} {word}s"

    design = vocab.design_name(it.live_candidate) if it.live_candidate else DASH
    label = (
        f'<div class="sublabel">Each task, for {escape(design)}. '
        f"{escape(plural(len(it.live), 'task'))}, "
        f"{escape(plural(hard, 'serious mistake'))}</div>"
    )
    return (
        table + label + '<div class="scroll"><table class="tbl"><thead><tr><th>task</th>'
        "<th>what it answered</th><th>how well it did</th><th></th>"
        "<th>took</th><th>status</th></tr></thead>"
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
        return _note("Nothing has gone wrong yet, so there is nothing to fix.")
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
            "No repair this round. One is made as soon as a fault names an operator that has "
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


# How the console says each metric the gate can reject on. The keys are gate.Verdict.metric.
GATE_REASON_WORDS = {
    "hard_fails": "serious mistakes",
    "pass3_rate": "right 3 times running",
    "p": "the chance this was luck",
    "gen_gap": "the drop on unseen tasks",
}
# each carries its own connector so the sentence reads naturally either way
GATE_REASON_AGAINST = {"incumbent": "the version in use's", "alpha": "the limit of"}
GATE_REASON_COMPARATOR = {
    "<": "is below", ">": "is above", "<=": "is not above", ">=": "is not below",
}


def _legacy_reason_parts(gate: dict[str, Any]) -> dict[str, Any] | None:
    """Rebuild the verdict's fields for a run recorded before ``reason_parts`` existed.

    Only the metric name and the comparator are taken from the reason line, and both are tokens
    gate.py writes in a format it owns and that ``tests/test_gate.py`` pins. Every number comes
    from the structured metrics the same file already records, so nothing here is parsed out of
    a formatted float or reconstructed by guess. Anything that does not match this exact shape
    returns None and the sentence is shown as it was written.
    """
    tokens = str(gate.get("reason") or "").split()
    if len(tokens) != 5 or tokens[2] not in GATE_REASON_COMPARATOR:
        return None
    metric, comparator, against_label = tokens[0], tokens[2], tokens[3]
    if metric == "p":
        value, against = gate.get("p"), gate.get("alpha")
        fmt, against_fmt = "{:.3f}", "{:g}"
    else:
        value = (gate.get("candidate") or {}).get(metric)
        against = (gate.get("incumbent") or {}).get(metric)
        fmt = "{:.0f}" if metric == "hard_fails" else "{:.3f}"
        against_fmt = fmt
    if not isinstance(value, int | float) or not isinstance(against, int | float):
        return None
    return {"metric": metric, "value": value, "comparator": comparator, "against": against,
            "against_label": against_label, "fmt": fmt, "against_fmt": against_fmt}


def _gate_reason(gate: dict[str, Any]) -> str:
    """The condition that settled this gate, said in words.

    gate.py records its verdict twice: ``reason`` is the line the report and the tests quote
    ("hard_fails 3 > incumbent 1"), and ``reason_parts`` is the same verdict as fields. The
    console renders the fields, because showing a reader two identifiers and a comparison
    operator tells them nothing, and taking the sentence apart again to rewrite it would be
    guessing at our own output.

    Runs made before ``reason_parts`` existed carry only the sentence, and it is shown as it
    was written rather than half-translated.
    """
    parts = gate.get("reason_parts")
    if not isinstance(parts, dict) or not parts.get("metric"):
        parts = _legacy_reason_parts(gate)
    if not parts:
        return txt(gate.get("reason"))
    fmt = str(parts.get("fmt") or "{}")
    metric = str(parts.get("metric"))
    name = GATE_REASON_WORDS.get(metric, vocab.humanize(metric))
    against = GATE_REASON_AGAINST.get(str(parts.get("against_label")),
                                      "the version in use's")
    comparator = GATE_REASON_COMPARATOR.get(str(parts.get("comparator")), "differs from")
    try:
        left = fmt.format(parts.get("value"))
        right = str(parts.get("against_fmt") or fmt).format(parts.get("against"))
    except (TypeError, ValueError):  # a malformed record must not blank the panel
        return txt(gate.get("reason"))
    return escape(f"{name} {left} {comparator} {against} {right}")


def render_gate(gate: dict[str, Any]) -> str:
    """Gate's artefact: the verdict, the paired-test grid, and the line it wrote to explain it."""
    if not gate:
        return _note("Nothing to judge this round. This step runs once a repair has been made.")
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
            "Nothing made cheaper yet. <code>anneal anneal</code> walks the winner down the model "
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
            f'<a class="dom{" on" if d == domain else ""}" href="/console?domain={escape(d)}">'
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
        '<header id="topbar" class="topbar">'
        f'<nav class="doms">{links}</nav>'
        f'<div class="meters">{meter}{render_balance(credit_balance())}</div></header>'
    )


CSS = """





.pagehead { padding:48px 0 28px; }
.pagehead h1 { font-size:26px; font-weight:500; letter-spacing:-0.015em; margin:0 0 6px; }
.pagehead p { color:var(--graphite); font-size:14px; max-width:62ch; margin:0; }
.agents { border:1px solid var(--rule); background:var(--panel); }
.agent { display:grid; grid-template-columns:minmax(200px,2fr) repeat(4, minmax(0,1fr));
  gap:20px; align-items:center; padding:18px 20px; text-decoration:none; color:var(--ink);
  border-bottom:1px solid var(--rule); }
.agent:last-child { border-bottom:0; }
.agent:hover { background:var(--paper); }
.aname { font-size:14px; }
.aname .dslug { display:block; font-size:10px; color:var(--ash); margin-top:2px; }
.afield { display:block; }
.ak { display:block; font-size:10px; text-transform:uppercase; letter-spacing:0.1em;
  color:var(--ash); margin-bottom:3px; }
.av { font-size:13px; }
@media (max-width:860px) { .agent { grid-template-columns:1fr 1fr; }
  .chrome { padding:16px 24px; } }
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
.dk { font-size:13px; color:var(--graphite); }
.dk .dslug { display:block; font-size:10px; letter-spacing:0.12em; color:var(--ash);
  margin-top:2px; }
.dv { font-size:14px; color:var(--ink); overflow-wrap:anywhere; }
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
/* a caption, not a label: it is a sentence now, so it is not shouted in caps */
.sublabel { font-size:12px; color:var(--graphite);
  margin:20px 0 8px; }

/* the step timeline is the page's structure: the six steps across, the open one beneath */
/* The six steps run across the top and the open step's panel takes the whole width beneath.
   They used to sit side by side, 44/56, which left the panel about 700px: every table with
   more than four columns was clipped at the right edge, and the answer to that is width, not
   fewer facts. Reading order is also the loop's order this way -- left to right, then down. */
.flow { display:block; }
.steps { list-style:none; margin:0; padding:0; display:flex; flex-wrap:wrap;
  border-bottom:1px solid var(--rule); }
.step { position:relative; flex:1 1 150px; min-width:0; border-right:1px solid var(--rule);
  animation:cascade .3s both; animation-delay:calc(var(--i) * 40ms); }
.step:last-child { border-right:0; }
/* The vertical spine that joined the steps when they were a column is gone: a row of
   steps is joined by being a row, and the line ran straight through their captions. */
.step a { display:block; padding:14px 16px 16px; text-decoration:none; }
.step .dot { display:inline-block; margin-bottom:8px; }
.step .sname { display:block; }
.step .sstat { display:block; margin-top:4px; overflow-wrap:anywhere; }
.step .chev { display:none; }  /* a row of steps reads as a sequence; a chevron per step does not */
.step a:hover { background:var(--paper); }
.step .dot { width:9px; height:9px; border-radius:50%; background:var(--rule);
  justify-self:center; position:relative; z-index:1; box-shadow:0 0 0 4px var(--panel); }
.step .sname { font-size:13px; color:var(--ash); }
.step .sdesc { display:none; }  /* the open step's blurb belongs in the panel header */
.step .sstat { font-size:11px; color:var(--ash); }
.step .chev { width:12px; height:12px; fill:none; stroke:var(--rule); stroke-width:1.5; }
.step[data-state=done] .dot { background:var(--ash); }
.step[data-state=done] .sname { color:var(--ink); }
.step[data-state=active] .dot { background:var(--orange); animation:pulse 2.4s infinite; }
.step[data-state=active] .sname { color:var(--ink); font-weight:500; }
.step[data-open=true] a { background:var(--paper); }
.step[data-open=true] .sname { color:var(--ink); }


.step[data-open=true] a::after { content:""; position:absolute; left:0; right:0; bottom:-1px;
  height:2px; background:var(--ink); }
.detail { padding:24px 24px 32px; min-width:0; }
.phdesc { margin:4px 0 0; color:var(--graphite); font-size:13px; max-width:70ch; }
@keyframes pulse { 0%,100% { transform:scale(1); opacity:1; }
  50% { transform:scale(1.65); opacity:.5; } }
@keyframes cascade { from { opacity:0; transform:translateY(4px); } to { opacity:1; } }

.tblwrap { overflow-x:auto; }
.said { font-size:12px; color:var(--graphite); max-width:34ch; }
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

/* === overrides: these come last so they win over the base rules above === */
.choice.pick { align-items:flex-start; padding:14px; }
.choice.pick input { margin-top:3px; }
.choice.pick span { display:block; }
.choice.pick b { display:block; font-weight:500; font-size:14px; }
.choice.pick em { display:block; font-style:normal; font-size:12px; color:var(--graphite);
  margin-top:3px; }
.usedby { color:var(--ash) !important; }
.sub-field { margin-top:16px; padding-left:14px; border-left:2px solid var(--rule); }
.sub-field .flabel { font-size:13px; color:var(--graphite); }
.reveal[hidden] { display:none; }

/* --- the new-agent form ---------------------------------------------------------------
   Label above the control, helper text below it, errors above the form and in words. No
   placeholder-as-label anywhere: a placeholder disappears the moment someone starts typing. */
.newform { max-width:760px; padding-bottom:64px; }
.field { display:block; margin-bottom:32px; }
.flabel { display:block; font-size:14px; color:var(--ink); margin-bottom:8px; }
.finput { width:100%; padding:11px 12px; font:inherit; font-size:14px; color:var(--ink);
  background:var(--panel); border:1px solid var(--rule); border-radius:0; }
.finput:focus { outline:2px solid var(--ink); outline-offset:-1px; }
.fhelp { margin:8px 0 0; font-size:12px; color:var(--graphite); max-width:65ch; }
.choices { display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule); }
.choice { display:flex; align-items:center; gap:10px; padding:12px 14px; background:var(--panel);
  font-size:14px; cursor:pointer; }
.choice:hover { background:var(--paper); }
.extable { border:1px solid var(--rule); background:var(--rule); display:grid; gap:1px; }
.exhead { display:grid; grid-template-columns:1fr 1fr; gap:1px; }
.exhead span { background:var(--panel); padding:10px 12px; font-size:11px;
  text-transform:uppercase; letter-spacing:0.1em; color:var(--ash); }
.exrow { display:grid; grid-template-columns:1fr 1fr; gap:1px; }
.exrow textarea { font:inherit; font-size:13px; color:var(--ink); padding:10px 12px;
  border:0; background:var(--panel); resize:vertical; }
.exrow textarea:focus { outline:2px solid var(--ink); outline-offset:-2px; }
.fsubmit { display:inline-block; padding:12px 22px; font:inherit; font-size:14px;
  background:var(--ink); color:var(--panel); border:1px solid var(--ink); cursor:pointer;
  text-decoration:none; }
.fsubmit:active { transform:translateY(1px); }
/* "link" was already taken by the connector between node pills, whose rule sets
   width:10px; a button wearing it collapsed to 46px with the label spilling out. */
.fsubmit.quiet { background:var(--panel); color:var(--ink); }
.formnote { max-width:65ch; margin:0 0 16px; padding:12px 14px; font-size:13px;
  color:var(--ink); background:var(--panel); border-left:2px solid var(--orange); }
.formerror { max-width:65ch; margin:0 0 28px; padding:12px 14px; font-size:14px;
  color:var(--clay); background:var(--panel); border:1px solid var(--clay); }
.donebox { border:1px solid var(--rule); background:var(--panel); padding:24px;
  margin-bottom:64px; }
.filelist { list-style:none; margin:0 0 20px; padding:0; display:flex; flex-wrap:wrap; gap:14px;
  font-size:12px; color:var(--ash); }
.cmd { margin:0 0 20px; padding:16px; background:var(--paper); border:1px solid var(--rule);
  font-family:"Geist Mono",ui-monospace,monospace; font-size:13px; overflow-x:auto; }
.pagehead .fsubmit { margin-top:18px; }
@media (max-width:700px) { .exhead, .exrow { grid-template-columns:1fr; } }

/* --- product chrome: the bar that makes three pages one application ------------------- */
.chrome { display:flex; align-items:center; gap:28px; padding:16px 48px;
  border-bottom:1px solid var(--rule); background:var(--panel); }
.chrome .mark { font-family:"Geist Mono",ui-monospace,monospace; letter-spacing:0.22em;
  font-size:13px; color:var(--ink); text-decoration:none; }
.navlinks { display:flex; gap:20px; }
.navlink { font-size:13px; color:var(--ash); text-decoration:none; }
.navlink:hover, .navlink.on { color:var(--ink); }

/* --- the two reading levels ----------------------------------------------------------
   Plain is the default and hides every internal identifier: run directories, candidate
   ids, domain slugs, tool function names, the evaluator's filename and threshold. None of
   it is removed from the document, so the switch is instant and the technical view is the
   same page with the machinery shown rather than a different page. */
body.plain .dslug,
body.plain .cid,
body.plain .strip,
body.plain .techonly { display:none !important; }
.switch { margin-left:auto; display:inline-flex; align-items:center; gap:8px; font-size:12px;
  color:var(--ash); text-decoration:none; }
.switch:hover { color:var(--ink); }
.switch .knob { width:26px; height:14px; border:1px solid var(--rule); border-radius:999px;
  position:relative; background:var(--paper); }
.switch .knob::after { content:""; position:absolute; top:2px; left:2px; width:8px; height:8px;
  border-radius:50%; background:var(--ash); transition:left .15s, background .15s; }
.switch.on { color:var(--ink); }
.switch.on .knob { border-color:var(--ink); }
.switch.on .knob::after { left:14px; background:var(--ink); }

.agent { grid-template-columns:minmax(180px,2fr) repeat(4, minmax(0,1fr)) auto; }
.aopen { color:var(--ink); text-decoration:none; }
.arun { padding:8px 14px; font:inherit; font-size:13px; background:var(--panel);
  color:var(--ink); border:1px solid var(--ink); cursor:pointer; }
.arun:hover { background:var(--ink); color:var(--panel); }
.arun:active { transform:translateY(1px); }
.agoing, .running { color:var(--orange); }
.failed { color:var(--clay); }
.afield.right { text-align:right; }
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
    launcher: Any = None
    """Runs this server started from the browser. None when nothing can be launched."""


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


def _console_query(domain: str | None, step: str | None) -> str:
    """The console's current query string, so the switch returns you to the same view."""
    parts = [f"domain={domain}" if domain else "", f"step={step}" if step else ""]
    kept = "&".join(part for part in parts if part)
    return f"?{kept}" if kept else ""


def render_page(
    app_state: AppState, domain: str | None = None, step: str | None = None,
    level: str = "plain",
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
        f"<style>{CSS}</style></head>"
        f'<body class="{escape(level)}">'
        f'{render_nav("/console", level, _console_query(selected, step))}'
        f"<div class='wrap'>"
        f"{render_topbar(selected, domains, it, agent_titles(app_state.runs_dir, domains))}"
        f'<main class="product">{_strip(it)}'
        f"{render_contract(contract, selected, title)}"
        f'<div id="flow">{render_timeline(it, issues, _fronts(app_state, selected), step)}</div>'
        f"{render_curves(summaries, pending)}"
        f"</main></div><script>{SCRIPT}</script></body></html>"
    )


DOMAINS_DIR = Path(__file__).resolve().parent.parent / "domains"

NAV = (("/", "Home"), ("/agents", "Agents"), ("/new", "New agent"),
       ("/console", "Console"))

# Two reading levels, one page. Everything the loop needs to name a thing precisely -- run
# directories, candidate ids, domain slugs, evaluator filenames, tool function names -- is
# machinery, and a person who asked for an agent that triages invoices has no use for it. The
# plain view hides that layer with CSS rather than dropping it from the document, so turning
# the switch on is instant and nothing has to be re-fetched or re-rendered to get it back.
TECHNICAL = "technical"


def detail_level(value: str | None) -> str:
    """``"technical"`` only when explicitly asked for; plain otherwise, including on nonsense."""
    return TECHNICAL if (value or "").strip().lower() == TECHNICAL else "plain"


def render_nav(active: str, level: str = "plain", query: str = "") -> str:
    """The bar every page carries, so no page is a dead end.

    Anneal is three surfaces, not one screen: the landing page explains what it is, the agents
    index says which agents exist and how each is doing, and the console follows one agent
    through a round. Each was reachable only by typing its URL, which is why the console read
    as the whole product.
    """
    links = "".join(
        f'<a class="navlink{" on" if href == active else ""}" href="{href}">{escape(label)}</a>'
        for href, label in NAV
    )
    on = level == TECHNICAL
    target = query + ("" if on else ("&" if query else "?") + f"detail={TECHNICAL}")
    if on:
        target = "&".join(
            part for part in query.lstrip("?").split("&")
            if part and not part.startswith("detail=")
        )
        target = f"?{target}" if target else active
    toggle = (
        f'<a class="switch{" on" if on else ""}" href="{escape(target or active)}">'
        f'<span class="knob"></span>{"Hide" if on else "Show"} technical detail</a>'
    )
    return (
        '<header class="chrome"><a class="mark" href="/">ANNEAL</a>'
        f'<nav class="navlinks">{links}</nav>{toggle}</header>'
    )


def page(title: str, active: str, body: str, level: str = "plain") -> str:
    """One document shell for every page that is not the console.

    The agents index shipped once as a bare fragment with no <html> around it and rendered
    completely unstyled. Going through here means a new page cannot repeat that.
    """
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title>"
        f"<style>{CSS}</style></head>"
        f'<body class="{escape(level)}">{render_nav(active, level)}'
        f'<div class="wrap">{body}</div></body></html>'
    )


def known_agents(state: AppState) -> list[str]:
    """Every agent that exists, whether or not it has ever run.

    ``list_domains`` reads the runs directory, so an agent created a minute ago was invisible
    here: the page that is supposed to say "what do I have" could only see what had already
    produced numbers. An agent is a directory under ``domains/``; having run is a property of
    one, not the definition.
    """
    on_disk = []
    if DOMAINS_DIR.is_dir():
        on_disk = [
            d.name for d in sorted(DOMAINS_DIR.iterdir())
            if d.is_dir() and not d.name.startswith(("_", "."))
            and (d / "goal.md").is_file()
        ]
    with_runs = list_domains(state.runs_dir)
    return list(dict.fromkeys([*on_disk, *with_runs]))


def _agent_row(state: AppState, name: str, titles: dict[str, str],
               summaries: dict[str, list[Point]]) -> str:
    """One row of the index: what it is, where it got to, and what you can do about it."""
    points = summaries.get(name) or []
    latest = points[-1] if points else None
    status = state.launcher.status(name) if state.launcher else None
    if status == "running":
        where = '<span class="running">Running now</span>'
    elif not points:
        where = "Never run" if status != "failed" else '<span class="failed">Run failed</span>'
    else:
        it = load_iteration(state.runs_dir, name)
        stages = pipeline_stages(it, _issues(state, name))
        where = escape(next(
            (label for (_, label, _), (_, st, _) in zip(STEP_META, stages, strict=True)
             if st == "active"),
            STEP_META[-1][1],
        ))
    action = (
        '<span class="agoing">working</span>' if status == "running" else
        f'<button class="arun" type="submit" formaction="/agents/{escape(name)}/run">'
        f'{"Run again" if points else "Run it"}</button>'
    )
    return (
        f'<div class="agent"><a class="aopen" href="/console?domain={escape(name)}">'
        f'<span class="aname">{escape(titles.get(name) or name)}'
        f'<span class="dslug mono">{escape(name)}</span></span></a>'
        f'<span class="afield"><span class="ak">rounds</span>'
        f'<span class="av mono">{len(points)}</span></span>'
        f'<span class="afield"><span class="ak">how often it is right</span>'
        f'<span class="av mono">{num(latest.score if latest else None)}</span></span>'
        f'<span class="afield"><span class="ak">cost per task</span>'
        f'<span class="av mono">'
        f'{num(latest.cost_per_task if latest else None, "${:.4f}")}</span></span>'
        f'<span class="afield"><span class="ak">state</span>'
        f'<span class="av">{where}</span></span>'
        f'<span class="afield right">{action}</span></div>'
    )


def render_agents(state: AppState, level: str = "plain") -> str:
    """The agents index: every agent that has run, what it does, and where it got to.

    This is the page a person lands on after the front door. It answers "what do I have and
    which one needs me", which the console cannot answer because the console is always looking
    at exactly one agent.
    """
    domains = known_agents(state)
    titles = agent_titles(state.runs_dir, domains)
    summaries = load_summaries(state.runs_dir)
    rows = [_agent_row(state, name, titles, summaries) for name in domains]
    body = "".join(rows) or _note(
        'No agents yet. <a href="/new">Answer five questions</a> and Anneal writes one, '
        "or run <code>uv run anneal init</code> in a terminal."
    )
    return page(
        "Anneal - your agents", "/agents",
        '<div class="pagehead"><h1>Your agents</h1>'
        '<p>Each one was built from a goal, a set of tools and a way to score it. '
        "Open one to watch the round it is on.</p>"
        '<a class="fsubmit quiet" href="/new">New agent</a></div>'
        f'<form method="post" class="agents">{body}</form>'
        '<p class="note">A run takes a few minutes. This page refreshes itself.</p>'
        '<script>setTimeout(()=>location.reload(),10000)</script>',
        level,
    )


def create_app(runs_dir: Path | str = "runs", ledger_path: Path | str = "ledger.json") -> FastAPI:
    """FastAPI app serving the console plus one endpoint per region."""
    state = AppState(
        Path(runs_dir), Path(ledger_path),
        launcher=launcher.Launcher(Path(runs_dir), DOMAINS_DIR),
    )
    app = FastAPI(title="Anneal dashboard")
    app.state.anneal = state

    def _landing() -> str:
        path = Path(__file__).resolve().parent.parent / "landing" / "index.html"
        if not path.exists():
            return "<html><body><p>landing/index.html is missing; see the repo.</p></body></html>"
        return path.read_text(encoding="utf-8")

    @app.exception_handler(404)
    async def not_found(request: Request, exc: Exception) -> HTMLResponse:
        """A page, not FastAPI's JSON. Someone who mistypes a URL is still a reader."""
        body = (
            '<div class="pagehead"><h1>No such page</h1>'
            "<p>That address does not exist here.</p>"
            '<a class="fsubmit quiet" href="/agents">Go to your agents</a></div>'
        )
        return HTMLResponse(page("Anneal - not found", "", body), status_code=404)

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        """The front door. The console used to live here, which made it the whole product."""
        return _landing()

    @app.get("/landing/img/{name}")
    def landing_image(name: str) -> FileResponse:
        """Serve the landing page's own images. The name is a filename, never a path."""
        path = (Path(__file__).resolve().parent.parent / "landing" / "img" / Path(name).name)
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.get("/landing", response_class=HTMLResponse)
    def landing() -> str:
        """Kept so links written before the console moved still work."""
        return _landing()

    @app.get("/agents", response_class=HTMLResponse)
    def agents(detail: str | None = None) -> str:
        return render_agents(state, detail_level(detail))

    @app.post("/agents/{domain}/run")
    def start_run(domain: str) -> RedirectResponse:
        """Start the loop for one agent and come straight back to the index.

        A redirect rather than a rendered page so a refresh does not start a second run, and
        so the index's own ten-second refresh is what reports progress.
        """
        try:
            state.launcher.start(domain)
        except (RuntimeError, FileNotFoundError, OSError) as exc:
            log.warning("could not start %s: %s", domain, exc)
        return RedirectResponse("/agents", status_code=303)

    @app.post("/agents/{domain}/stop")
    def stop_run(domain: str) -> RedirectResponse:
        state.launcher.stop(domain)
        return RedirectResponse("/agents", status_code=303)

    @app.get("/new", response_class=HTMLResponse)
    def new_agent(detail: str | None = None) -> str:
        return page("Anneal - new agent", "/new", newagent.form_body(), detail_level(detail))

    @app.post("/new", response_class=HTMLResponse)
    async def create_agent(request: Request) -> HTMLResponse:
        """Run the same interview `anneal init` runs, over a posted form.

        Everything the interview refuses -- too few examples, a name already taken, a tool
        module that will not import -- comes back as its own message on the form rather than a
        traceback, because the person filling this in is not reading our logs.
        """
        submission = newagent.parse(dict(await request.form()))
        try:
            path, notes = newagent.create(submission, DOMAINS_DIR)
        except onboard.OnboardError as exc:
            body = newagent.form_body(str(exc))
            return HTMLResponse(page("Anneal - new agent", "/new", body), status_code=400)
        except Exception as exc:  # noqa: BLE001 - any failure is the form's to report
            log.warning("new agent failed: %s", exc)
            body = newagent.form_body(f"Could not create the agent: {exc}")
            return HTMLResponse(page("Anneal - new agent", "/new", body), status_code=400)
        body = newagent.done_body(path, notes)
        return HTMLResponse(page("Anneal - agent created", "/new", body))

    @app.get("/console", response_class=HTMLResponse)
    def console(
        domain: str | None = None, step: str | None = None, detail: str | None = None
    ) -> str:
        return render_page(state, domain, step, detail_level(detail))

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
