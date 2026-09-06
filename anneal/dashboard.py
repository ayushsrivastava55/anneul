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
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

log = logging.getLogger("anneal.dashboard")

# summary.json / pareto.json field aliases, most specific first. cli.py writes mean_score /
# pass3_rate / p95_latency_ms at the top level and the per-candidate search metrics (which is
# where cost_per_task lives) under summary["search"][<candidate id>]; anneal.py's pareto points
# use their own names. Read them all; note top-level "cost_usd" is a split total, not $/task.
SCORE_KEYS = ("score", "mean_score", "accuracy")
PASS3_KEYS = ("pass3", "pass3_rate", "pass_cubed", "pass^3")
COST_KEYS = ("cost_per_task", "usd_per_task", "dollars_per_task", "cost_usd_per_task")
P95_KEYS = ("p95_latency_ms", "latency_p95_ms", "p95_ms", "p95")

METRICS: tuple[tuple[str, str, str], ...] = (
    ("score", "score", "{:.3f}"),
    ("pass3", "pass^3", "{:.3f}"),
    ("cost_per_task", "$/task", "${:.4f}"),
    ("p95_latency_ms", "p95 latency (ms)", "{:.0f}"),
)

# Ledger table: the class, how bad it is, how often it fired, and the operator Mutate would
# reach for next. Severity and the operator ladder come from specs/failure_taxonomy.yaml.
LEDGER_COLUMNS = (
    "domain", "id", "class", "node", "severity", "count", "status", "next operator", "tried",
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
    diagnosed = f"{len(mine)} issues" if mine else (
        str(issue.get("class")) if isinstance(issue, dict) and issue.get("class") else DASH
    )
    search = it.search
    facts: list[tuple[str, bool, str]] = [
        ("Architect", specs_on_disk > 0, f"{specs_on_disk} specs" if specs_on_disk else DASH),
        ("Run", bool(search), f"{len(search)} candidates" if search else DASH),
        ("Diagnose", diagnosed != DASH, diagnosed),
        ("Mutate", bool(it.summary.get("operator")), str(it.summary.get("operator") or DASH)),
        ("Gate", bool(it.gate), str(it.gate.get("decision") or DASH) if it.gate else DASH),
        ("Anneal", it.has_pareto, "pareto.json" if it.has_pareto else DASH),
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


def line_chart(points: list[Point], metric: str, title: str, fmt: str) -> str:
    """A small SVG line chart of ``metric`` over iterations; empty state when nothing is set."""
    pairs = [(p.label, p.get(metric)) for p in points]
    known = [(label, v) for label, v in pairs if v is not None]
    if not known:
        return (
            f'<div class="chart empty"><h4>{escape(title)}</h4>'
            f'<p class="note">no {escape(title)} recorded in these summaries</p></div>'
        )
    w, h, pad = 260.0, 130.0, 26.0
    values = [v for _, v in known]
    lo, hi = min(values), max(values)
    span = (hi - lo) or (abs(hi) or 1.0)
    lo, hi = lo - span * 0.1, hi + span * 0.1
    xs = [_scale(i, 0, max(len(known) - 1, 1), pad, w - pad / 2) for i in range(len(known))]
    ys = [_scale(v, lo, hi, h - pad, pad / 2) for v in values]
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys, strict=True))
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2"><title>{escape(label)}: '
        f"{escape(fmt.format(v))}</title></circle>"
        for (label, v), x, y in zip(known, xs, ys, strict=True)
    )
    return (
        f'<div class="chart"><h4>{escape(title)}</h4>'
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{escape(title)}">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad / 2}" y2="{h - pad}" class="axis"/>'
        f'<line x1="{pad}" y1="{pad / 2}" x2="{pad}" y2="{h - pad}" class="axis"/>'
        f'<polyline points="{path}" class="series"/>{dots}'
        f'<text x="{pad}" y="{pad / 2 - 2}" class="tick">{escape(fmt.format(hi))}</text>'
        f'<text x="{pad}" y="{h - pad + 12}" class="tick">{escape(fmt.format(lo))}</text>'
        f"</svg>"
        f'<p class="latest">latest <b class="mono">{escape(fmt.format(known[-1][1]))}</b> @ iter '
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


# --- panels ---------------------------------------------------------------------------------


def render_rail(stages: list[tuple[str, str, str]]) -> str:
    """The six loop stages. State comes from the artefacts on disk, never from a constant."""
    items = "".join(
        f'<li class="stage" data-state="{state}" style="--i:{i}">'
        f'<span class="dot"></span><span class="sname">{escape(name)}</span>'
        f'<span class="sdetail mono">{escape(detail)}</span></li>'
        for i, (name, state, detail) in enumerate(stages)
    )
    return f'<nav id="rail" class="rail"><ol>{items}</ol></nav>'


def _node_pills(spec: dict[str, Any] | None) -> str:
    """The topology as connected node pills, each carrying its model tier badge."""
    nodes = (spec or {}).get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return f'<span class="pills"><span class="pill mono absent">{DASH}</span></span>'
    pills = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        name = txt(node.get("name") or node.get("role"))
        tier = node.get("model_tier")
        cls = "tier" if tier else "tier absent"
        pills.append(f'<span class="pill mono">{name}<b class="{cls}">{txt(tier)}</b></span>')
    return '<span class="pills">' + '<i class="link"></i>'.join(pills) + "</span>"


def _roles(it: Iteration, cid: str) -> str:
    summary = it.summary
    chips = []
    if cid == summary.get("incumbent_id"):
        chips.append('<span class="chip">incumbent</span>')
    if cid == summary.get("candidate_id"):
        decision = str(summary.get("decision") or "")
        cls = {"promote": "chip ok", "reject": "chip bad"}.get(decision, "chip")
        chips.append('<span class="chip">challenger</span>')
        if decision:
            chips.append(f'<span class="{cls}">{escape(decision)}</span>')
    if cid == summary.get("winner_id"):
        chips.append('<span class="chip ok">winner</span>')
    return "".join(chips)


def _candidate_row(it: Iteration, cid: str, metrics: dict[str, Any], index: int) -> str:
    score = _num(metrics, SCORE_KEYS)
    bar = "" if score is None else f'<i style="width:{max(0.0, min(1.0, score)) * 100:.1f}%"></i>'
    verdict = ""
    if cid == it.summary.get("candidate_id"):
        verdict = {"promote": " promote", "reject": " reject"}.get(
            str(it.summary.get("decision") or ""), ""
        )
    spec = it.specs.get(cid)
    return (
        f'<tr class="crow{verdict}" style="--i:{index}">'
        f'<td class="mono id">{escape(cid)}{_roles(it, cid)}</td>'
        f'<td class="topo"><span class="mono topo-name">{txt((spec or {}).get("topology"))}</span>'
        f"{_node_pills(spec)}</td>"
        f'<td class="scorecell"><span class="bar">{bar}</span>'
        f'<span class="mono">{num(score)}</span></td>'
        f'<td class="mono num">{num(_num(metrics, COST_KEYS), "${:.4f}")}</td>'
        f'<td class="mono num">{num(_num(metrics, P95_KEYS), "{:.0f}ms")}</td>'
        f'<td class="mono num">{num(metrics.get("hard_fails"), "{:.0f}")}</td></tr>'
    )


def render_candidates(it: Iteration) -> str:
    """One row per candidate scored in this iteration, read from summary["search"]."""
    search = it.search
    if not search:
        body = _note(
            "No candidates scored yet — <code>anneal run domains/&lt;name&gt;</code> asks the "
            "architect for specs and fills this in."
        )
    else:
        rows = "".join(
            _candidate_row(it, cid, metrics if isinstance(metrics, dict) else {}, i)
            for i, (cid, metrics) in enumerate(sorted(search.items()))
        )
        body = (
            '<table class="tbl"><thead><tr><th>candidate</th><th>topology</th><th>score</th>'
            "<th>$/task</th><th>p95</th><th>hard fails</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    meta = f"{escape(it.domain)} · iter {escape(it.label)}"
    return _panel("candidates", "Candidates", body, meta)


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


def render_live(it: Iteration) -> str:
    """Per-task rows from the candidate's search jsonl; hard-fail rows carry a clay border."""
    if not it.live:
        body = _note(
            "No task rows on disk for this iteration yet — the runner writes one line per "
            "task as it finishes."
        )
    else:
        rows = "".join(
            f'<tr class="lrow{" bad" if row.get("hard_fail") else ""}" style="--i:{i}">'
            f'<td class="mono">{txt(row.get("task_id"))}</td>'
            f"<td>{_pips(row.get('score'))}</td>"
            f'<td class="mono num">{num(row.get("score"))}</td>'
            f'<td class="mono num">{num(row.get("latency_ms"), "{:.0f}ms")}</td>'
            f'<td class="st">{_status(row)}</td></tr>'
            for i, row in enumerate(it.live[:60])
        )
        body = (
            '<div class="scroll"><table class="tbl"><thead><tr><th>task</th><th>score</th>'
            "<th></th><th>latency</th><th>status</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    hard = sum(1 for row in it.live if row.get("hard_fail"))
    meta = (
        f"{escape(it.live_candidate or DASH)} · {len(it.live)} tasks · {hard} hard fails"
        if it.live
        else escape(DASH)
    )
    return _panel("live", "Live run", body, meta)


def _next_operator(row: dict[str, Any]) -> str:
    """The operator Mutate would reach for next: first in the class ladder not yet tried."""
    entry = taxonomy().get(str(row.get("class")))
    operators = (entry or {}).get("operators")
    tried = row.get("operators_tried")
    tried = tried if isinstance(tried, list) else []
    if not isinstance(operators, list) or not operators:
        return DASH  # a class the taxonomy does not carry; guessing an operator would be a lie
    remaining = [op for op in operators if op not in tried]
    return escape(str(remaining[0])) if remaining else "all tried"


def _severity(row: dict[str, Any]) -> Any:
    return (taxonomy().get(str(row.get("class"))) or {}).get("severity")


def _rank(row: dict[str, Any]) -> float:
    """severity x count, so the class costing the most accuracy sits at the top."""
    count = row.get("count")
    count = float(count) if isinstance(count, int | float) and not isinstance(count, bool) else 0.0
    severity = _severity(row)
    severity = float(severity) if isinstance(severity, int | float) else 0.0
    return severity * count


def render_ledger(ledger: list[dict[str, Any]]) -> str:
    """Failure classes ranked by severity x count, each pointing at its next operator."""
    if not ledger:
        return _panel(
            "ledger",
            "Failure ledger",
            _note("Ledger is empty — no failures have been diagnosed from traces yet."),
        )
    ranked = sorted(ledger, key=_rank, reverse=True)
    head = "".join(f"<th>{escape(c)}</th>" for c in LEDGER_COLUMNS[2:])
    rows = "".join(
        f'<tr class="lgrow" style="--i:{i}" title="{escape(str(row.get("id") or ""))} · '
        f'{escape(str(row.get("domain") or ""))}">'
        f'<td class="mono cls">{txt(row.get("class"))}</td>'
        f'<td class="mono">{txt(row.get("node"))}</td>'
        f'<td class="mono num">{num(_severity(row), "{:.0f}")}</td>'
        f'<td class="mono num">{txt(row.get("count"))}</td>'
        f'<td class="mono">{txt(row.get("status"))}</td>'
        f'<td class="mono op">&rarr; {_next_operator(row)}</td>'
        f'<td class="mono tried">{txt(", ".join(str(o) for o in row.get("operators_tried") or []))}'
        f"</td></tr>"
        for i, row in enumerate(ranked)
    )
    body = f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>'
    return _panel("ledger", "Failure ledger", body, f"{len(ledger)} issues")


GATE_SIDES = (
    ("mean_score", "score", "{:.3f}"),
    ("pass3_rate", "pass^3", "{:.3f}"),
    ("hard_fails", "hard fails", "{:.0f}"),
    ("gen_gap", "gen gap", "{:+.3f}"),
)

GATE_STATS = (
    ("p", "p", "{:.3f}"),
    ("alpha", "alpha", "{:.2f}"),
    ("wins", "wins", "{:.0f}"),
    ("losses", "losses", "{:.0f}"),
    ("min_discordant_to_promote", "need", "{:.0f}"),
)


def _gate_side(gate: dict[str, Any], key: str) -> dict[str, Any]:
    block = gate.get(key)
    return block if isinstance(block, dict) else {}


def render_gate(gate: dict[str, Any]) -> str:
    """The verdict, the paired-test grid and the one line the gate wrote to explain itself."""
    if not gate:
        return _panel(
            "gate",
            "Gate decision",
            _note("No gate.json for this iteration — the gate runs once a mutation exists."),
        )
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
    body = (
        f'<p class="{cls}">{txt(decision).upper()}</p>'
        f'<table class="tbl gate"><thead><tr><th></th><th>incumbent</th><th>candidate</th></tr>'
        f"</thead><tbody>{sides}</tbody></table>"
        f'<div class="stats">{stats}</div>'
        f'<p class="reason mono">{txt(gate.get("reason"))}</p>'
    )
    ids = f'{escape(str(_gate_side(gate, "candidate").get("candidate_id") or DASH))}'
    return _panel("gate", "Gate decision", body, ids)


def render_pareto(fronts: dict[str, list[Point]]) -> str:
    if not fronts:
        return _panel(
            "pareto",
            "Anneal",
            _note(
                "No pareto.json yet — <code>anneal anneal</code> walks the winner down the "
                "model ladder and writes the front here."
            ),
        )
    body = "".join(
        f'<div class="pblock"><h3 class="mono">{escape(d)}</h3>{pareto_chart(pts)}</div>'
        for d, pts in fronts.items()
    )
    return _panel("pareto", "Anneal", body, "cost / score front")


def _curves_for(domain: str, points: list[Point]) -> str:
    charts = "".join(line_chart(points, key, title, fmt) for key, title, fmt in METRICS)
    return (
        f'<div class="pblock"><h3 class="mono">{escape(domain)}</h3>'
        f'<div class="charts">{charts}</div></div>'
    )


def render_curves(summaries: dict[str, list[Point]]) -> str:
    if not summaries:
        return _panel(
            "curves",
            "Iterations",
            _note(
                "No runs yet — <code>uv run anneal run domains/&lt;name&gt;</code> and this "
                "fills in one point per iteration."
            ),
        )
    body = "".join(_curves_for(d, pts) for d, pts in summaries.items())
    iters = sum(len(pts) for pts in summaries.values())
    return _panel("curves", "Iterations", body, f"{iters} summaries")


def render_balance(balance: float | None) -> str:
    """The Dodo credit balance, or an em-dash when billing has not cached one."""
    text = DASH if balance is None else f"${balance:,.2f}"
    return (
        '<div id="balance" class="meter"><span class="k">credits</span>'
        f'<span class="v mono">{escape(text)}</span></div>'
    )


def render_topbar(domain: str | None, domains: list[str], it: Iteration | None) -> str:
    """Wordmark, the domains present in runs/, the budget meter and the credit balance."""
    if domains:
        links = "".join(
            f'<a class="dom mono{" on" if d == domain else ""}" href="?domain={escape(d)}">'
            f"{escape(d)}</a>"
            for d in domains
        )
    else:
        links = f'<span class="dom mono absent">{DASH}</span>'
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
/* Tokens from .stitch/DESIGN.md: Paper canvas, Ink text, one orange accent, 1px Rule
   separation and no shadows anywhere. Orange is spent on the active stage dot and the
   measured chart series only. */
:root { --paper:#F7F7F5; --panel:#FFFFFF; --ink:#111214; --graphite:#5B6068; --ash:#8A9099;
  --rule:#E4E4E1; --orange:#F4511E; --green:#2F9E79; --clay:#C4544F; color-scheme: light; }
* { box-sizing: border-box; }
body { margin:0; background:var(--paper); color:var(--ink);
  font:400 13px/1.45 Geist,"Geist Sans",ui-sans-serif,sans-serif; letter-spacing:-0.01em; }
.mono, .mono * { font-family:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums; letter-spacing:0; }
.wrap { max-width:1440px; margin:0 auto; padding:0 24px 48px; }
a { color:inherit; text-decoration:none; }
.absent { color:var(--ash); }
code { font-family:"Geist Mono",ui-monospace,monospace; color:var(--ink); }

.topbar { display:flex; align-items:center; gap:24px; height:60px;
  border-bottom:1px solid var(--rule); }
.mark { font-weight:600; letter-spacing:0.18em; font-size:13px; }
.doms { display:flex; gap:2px; flex:1; }
.dom { padding:5px 10px; color:var(--ash); border:1px solid transparent; font-size:12px; }
.dom:hover { color:var(--ink); }
.dom.on { color:var(--ink); border-color:var(--rule); background:var(--panel); }
.meters { display:flex; gap:24px; align-items:center; }
.meter { display:flex; align-items:center; gap:8px; font-size:11px; color:var(--ash);
  text-transform:uppercase; letter-spacing:0.1em; }
.meter .v { color:var(--ink); font-size:12px; text-transform:none; letter-spacing:0; }
.bar { display:inline-block; height:4px; width:60px; background:var(--rule); }
.bar.wide { width:120px; }
.bar i { display:block; height:100%; background:var(--ink); }

.rail { border-bottom:1px solid var(--rule); }
.rail ol { display:grid; grid-template-columns:repeat(6,1fr); margin:0; padding:0;
  list-style:none; }
.stage { padding:16px; border-left:1px solid var(--rule); display:flex;
  flex-direction:column; gap:5px; animation:cascade .3s both;
  animation-delay:calc(var(--i) * 40ms); }
.stage:first-child { border-left:0; }
.stage .dot { width:7px; height:7px; border-radius:50%; background:var(--rule); }
.stage .sname { font-size:12px; color:var(--ash); }
.stage .sdetail { font-size:11px; color:var(--rule); }
.stage[data-state=done] .dot { background:var(--ash); }
.stage[data-state=done] .sname { color:var(--ink); }
.stage[data-state=done] .sdetail { color:var(--ash); }
.stage[data-state=active] .dot { background:var(--orange); animation:pulse 2.4s infinite; }
.stage[data-state=active] .sname { color:var(--ink); font-weight:500; }
.stage[data-state=active] .sdetail { color:var(--ash); }
@keyframes pulse { 0%,100% { transform:scale(1); opacity:1; }
  50% { transform:scale(1.7); opacity:.5; } }
@keyframes cascade { from { opacity:0; transform:translateY(4px); } to { opacity:1; } }

.console { display:grid; grid-template-columns:62fr 38fr; }
.col { min-width:0; }
.col.right { border-left:1px solid var(--rule); }
.panel { border-bottom:1px solid var(--rule); padding:16px; background:var(--panel); }
.ph { display:flex; align-items:baseline; justify-content:space-between; margin-bottom:14px; }
.ph h2 { margin:0; font-size:11px; font-weight:500; text-transform:uppercase;
  letter-spacing:0.14em; color:var(--ash); }
.ph .meta { font-size:11px; color:var(--ash); }
.note { color:var(--graphite); font-size:12px; margin:8px 0; max-width:65ch; }

.tbl { border-collapse:collapse; width:100%; }
.tbl th { text-align:left; font-weight:400; font-size:10px; text-transform:uppercase;
  letter-spacing:0.1em; color:var(--ash); padding:0 8px 6px;
  font-family:"Geist Mono",ui-monospace,monospace; }
.tbl td { padding:8px; border-top:1px solid var(--rule); font-size:12px;
  vertical-align:middle; }
.tbl tbody tr { animation:cascade .3s both; animation-delay:calc(var(--i) * 40ms); }
.tbl tbody tr:hover { background:var(--paper); }
.num { text-align:right; }
.crow.promote td:first-child { box-shadow:inset 2px 0 0 var(--green); }
.crow.reject td:first-child { box-shadow:inset 2px 0 0 var(--clay); }
.lrow.bad td:first-child { box-shadow:inset 2px 0 0 var(--clay); }
.id { white-space:nowrap; }
.chip { display:inline-block; margin-left:6px; padding:1px 5px; font-size:9px;
  text-transform:uppercase; letter-spacing:0.08em; border:1px solid var(--rule);
  color:var(--ash); font-family:"Geist Mono",ui-monospace,monospace; }
.chip.ok { border-color:var(--green); color:var(--green); }
.chip.bad { border-color:var(--clay); color:var(--clay); }
.chip.warn { border-color:var(--ash); color:var(--graphite); }
.st .chip { margin-left:0; margin-right:4px; }
.topo-name { color:var(--ash); font-size:11px; margin-right:8px; }
.pills { display:inline-flex; align-items:center; flex-wrap:wrap; }
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
.scroll { max-height:420px; overflow:auto; }
.op { color:var(--ink); }
.tried { color:var(--ash); font-size:11px; }

.verdict { margin:0 0 12px; font-size:26px; font-weight:600; letter-spacing:-0.03em;
  color:var(--ink); }
.verdict.ok { color:var(--green); }
.verdict.bad { color:var(--clay); }
.stats { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--rule);
  border:1px solid var(--rule); margin:14px 0; }
.stat { background:var(--panel); padding:8px 10px; display:flex; flex-direction:column;
  gap:2px; }
.stat .k { font-size:10px; text-transform:uppercase; letter-spacing:0.1em; color:var(--ash);
  font-family:"Geist Mono",ui-monospace,monospace; }
.stat .v { font-size:13px; }
.reason { color:var(--graphite); font-size:11px; margin:0; }

.pblock { margin-bottom:16px; }
.pblock h3 { margin:0 0 8px; font-size:11px; color:var(--ash); font-weight:400; }
.charts { display:flex; flex-wrap:wrap; gap:16px; }
.chart { width:280px; } .chart.wide { width:100%; }
.chart h4 { margin:0 0 6px; font-size:10px; font-weight:400; color:var(--ash);
  text-transform:uppercase; letter-spacing:0.1em;
  font-family:"Geist Mono",ui-monospace,monospace; }
svg { width:100%; height:auto; }
.axis { stroke:var(--rule); stroke-width:1; }
.series { fill:none; stroke:var(--orange); stroke-width:2; }
.chart circle { fill:var(--orange); }
.tick { fill:var(--ash); font-size:9px;
  font-family:"Geist Mono",ui-monospace,monospace; }
.latest { margin:6px 0 0; color:var(--graphite); font-size:11px; }
.latest b { color:var(--ink); font-weight:500; }
.pt { fill:var(--rule); stroke:var(--ash); }
.pt.front { fill:var(--green); fill-opacity:.85; stroke:var(--green); }
.front-line { fill:none; stroke:var(--green); stroke-width:1; stroke-opacity:.6; }

@media (max-width:768px) {
  .console { grid-template-columns:1fr; }
  .col.right { border-left:0; border-top:1px solid var(--rule); }
  .rail ol { grid-template-columns:repeat(2,1fr); }
  .stats { grid-template-columns:repeat(2,1fr); }
}
"""

SCRIPT = """
const ids=['topbar','rail','candidates','live','ledger','gate','pareto','curves'];
setInterval(()=>{for(const id of ids){
  fetch('/fragments/'+id+location.search).then(r=>r.text()).then(html=>{
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


def render_page(app_state: AppState, domain: str | None = None) -> str:
    """The whole console. Every region also has its own endpoint for the refresh script."""
    selected, domains = _selected(app_state, domain)
    it = load_iteration(app_state.runs_dir, selected) if selected else None
    issues = _issues(app_state, selected)
    summaries = load_summaries(app_state.runs_dir)
    fronts = load_pareto(app_state.runs_dir)
    if selected:
        summaries = {k: v for k, v in summaries.items() if k == selected} or summaries
        fronts = {k: v for k, v in fronts.items() if k == selected}
    blank = Iteration(selected or DASH, None, DASH, {}, {}, {}, None, [], False)
    it = it or blank
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Anneal console</title>"
        f"<style>{CSS}</style></head><body><div class='wrap'>"
        f"{render_topbar(selected, domains, it)}"
        f"{render_rail(pipeline_stages(it, issues))}"
        f'<div class="console"><div class="col left">'
        f"{render_candidates(it)}{render_live(it)}</div>"
        f'<div class="col right">{render_ledger(issues)}{render_gate(it.gate)}</div></div>'
        f"{render_pareto(fronts)}{render_curves(summaries)}"
        f"</div><script>{SCRIPT}</script></body></html>"
    )


def create_app(runs_dir: Path | str = "runs", ledger_path: Path | str = "ledger.json") -> FastAPI:
    """FastAPI app serving the console plus one endpoint per region."""
    state = AppState(Path(runs_dir), Path(ledger_path))
    app = FastAPI(title="Anneal dashboard")
    app.state.anneal = state

    def _iteration(domain: str | None) -> Iteration:
        selected, _ = _selected(state, domain)
        if not selected:
            return Iteration(DASH, None, DASH, {}, {}, {}, None, [], False)
        return load_iteration(state.runs_dir, selected)

    @app.get("/", response_class=HTMLResponse)
    def index(domain: str | None = None) -> str:
        return render_page(state, domain)

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
        return render_topbar(selected, domains, _iteration(domain))

    @app.get("/fragments/rail", response_class=HTMLResponse)
    def rail(domain: str | None = None) -> str:
        selected, _ = _selected(state, domain)
        return render_rail(pipeline_stages(_iteration(domain), _issues(state, selected)))

    @app.get("/fragments/candidates", response_class=HTMLResponse)
    def candidates(domain: str | None = None) -> str:
        return render_candidates(_iteration(domain))

    @app.get("/fragments/live", response_class=HTMLResponse)
    def live(domain: str | None = None) -> str:
        return render_live(_iteration(domain))

    @app.get("/fragments/gate", response_class=HTMLResponse)
    def gate(domain: str | None = None) -> str:
        return render_gate(_iteration(domain).gate)

    @app.get("/fragments/curves", response_class=HTMLResponse)
    def curves(domain: str | None = None) -> str:
        summaries = load_summaries(state.runs_dir)
        if domain in summaries:
            summaries = {domain: summaries[domain]}
        return render_curves(summaries)

    @app.get("/fragments/ledger", response_class=HTMLResponse)
    def ledger(domain: str | None = None) -> str:
        return render_ledger(_issues(state, domain) if domain else
                             load_ledgers(state.runs_dir, state.ledger_path))

    @app.get("/fragments/pareto", response_class=HTMLResponse)
    def pareto(domain: str | None = None) -> str:
        fronts = load_pareto(state.runs_dir)
        if domain in fronts:
            fronts = {domain: fronts[domain]}
        return render_pareto(fronts)

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
