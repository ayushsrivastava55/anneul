"""Single-page FastAPI + HTMX dashboard over ``runs/``, ``ledger.json`` and the credit balance.

Everything rendered here is read from disk: ``runs/<domain>/<iter>/summary.json`` for the
iteration curves, ``ledger.json`` for the issue table and ``runs/<domain>/anneal/pareto.json``
for the Pareto scatter. No number is ever hardcoded; a missing file renders an empty state.

Charts are plain inline SVG built server-side, so the page is fully readable with JavaScript
disabled. HTMX is used only to poll the four fragments; without it the page is static.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

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

LEDGER_COLUMNS = ("id", "class", "node", "count", "status", "operators_tried")


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
        return f'<div class="chart empty"><h4>{escape(title)}</h4><p>no data</p></div>'
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
        f'<p class="latest">latest {escape(fmt.format(known[-1][1]))} @ iter '
        f"{escape(known[-1][0])}</p></div>"
    )


def pareto_chart(points: list[Point]) -> str:
    """Score against $/task, point radius by p95 latency."""
    usable = [p for p in points if p.score is not None and p.cost_per_task is not None]
    if not usable:
        return '<div class="chart empty"><h4>Pareto</h4><p>no pareto.json yet</p></div>'
    w, h, pad = 420.0, 240.0, 40.0
    costs = [p.cost_per_task or 0.0 for p in usable]
    scores = [p.score or 0.0 for p in usable]
    p95s = [p.p95_latency_ms for p in usable if p.p95_latency_ms is not None]
    lo_p, hi_p = (min(p95s), max(p95s)) if p95s else (0.0, 0.0)
    marks = []
    for point in usable:
        x = _scale(point.cost_per_task or 0.0, min(costs), max(costs), pad, w - pad)
        y = _scale(point.score or 0.0, min(scores), max(scores), h - pad, pad)
        r = 5.0 if point.p95_latency_ms is None else _scale(point.p95_latency_ms, lo_p, hi_p, 4, 13)
        p95_text = "n/a" if point.p95_latency_ms is None else f"{point.p95_latency_ms:.0f}ms"
        on_front = bool(point.extra.get("on_front"))
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" '
            f'class="pt{" front" if on_front else ""}"><title>'
            f"{escape(point.label)}{' (front)' if on_front else ''}: score {point.score:.3f}, "
            f"${point.cost_per_task:.4f}/task, p95 {escape(p95_text)}</title></circle>"
        )
    return (
        '<div class="chart wide">'
        "<h4>Pareto — score vs $/task (radius = p95, filled = on the front)</h4>"
        f'<svg viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="Pareto scatter">'
        f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" class="axis"/>'
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" class="axis"/>'
        f'<text x="{pad}" y="{pad - 8}" class="tick">score {max(scores):.3f}</text>'
        f'<text x="{pad}" y="{h - pad + 14}" class="tick">${min(costs):.4f}</text>'
        f'<text x="{w - pad - 50}" y="{h - pad + 14}" class="tick">${max(costs):.4f}</text>'
        f"{''.join(marks)}</svg></div>"
    )


# --- fragments ----------------------------------------------------------------------------


def _curves_for(domain: str, points: list[Point]) -> str:
    charts = "".join(line_chart(points, key, title, fmt) for key, title, fmt in METRICS)
    return (
        f'<section class="domain"><h3>{escape(domain)}</h3>'
        f'<div class="row">{charts}</div></section>'
    )


def render_curves(summaries: dict[str, list[Point]]) -> str:
    if not summaries:
        return (
            '<div id="curves" class="card"><h2>Iterations</h2>'
            '<p class="empty">No runs yet. Run <code>uv run anneal run domains/&lt;name&gt;</code>'
            " and this fills in.</p></div>"
        )
    body = "".join(_curves_for(d, pts) for d, pts in summaries.items())
    return f'<div id="curves" class="card"><h2>Iterations</h2>{body}</div>'


def _cell(row: dict[str, Any], column: str) -> str:
    value = row.get(column, "")
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    return escape(str(value))


def render_ledger(ledger: list[dict[str, Any]]) -> str:
    if not ledger:
        return (
            '<div id="ledger" class="card"><h2>Ledger</h2>'
            '<p class="empty">Ledger is empty.</p></div>'
        )
    head = "".join(f"<th>{escape(c)}</th>" for c in LEDGER_COLUMNS)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_cell(row, c)}</td>" for c in LEDGER_COLUMNS) + "</tr>"
        for row in ledger
    )
    return (
        f'<div id="ledger" class="card"><h2>Ledger <span class="muted">{len(ledger)} issues</span>'
        f"</h2><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def render_pareto(fronts: dict[str, list[Point]]) -> str:
    if not fronts:
        return (
            '<div id="pareto" class="card"><h2>Pareto</h2>'
            '<p class="empty">No pareto.json yet — run <code>anneal anneal</code>.</p></div>'
        )
    body = "".join(
        f'<section class="domain"><h3>{escape(d)}</h3>{pareto_chart(pts)}</section>'
        for d, pts in fronts.items()
    )
    return f'<div id="pareto" class="card"><h2>Pareto</h2>{body}</div>'


def render_balance(balance: float | None) -> str:
    text = "—" if balance is None else f"${balance:,.2f}"
    return (
        f'<div id="balance" class="card balance"><h2>Dodo credits</h2>'
        f'<p class="big">{escape(text)}</p></div>'
    )


CSS = """
:root { color-scheme: dark; }
body { margin:0; padding:24px; background:#0f1115; color:#e8e8ea;
  font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
h1 { font-size:20px; margin:0 0 4px; } h2 { font-size:15px; margin:0 0 12px; color:#9fb3ff; }
h3 { font-size:13px; margin:12px 0 6px; color:#cfd3dc; } h4 { font-size:11px; margin:0 0 4px;
  color:#8b93a7; font-weight:normal; }
.card { background:#161923; border:1px solid #242938; border-radius:10px; padding:16px;
  margin-bottom:16px; }
.row { display:flex; flex-wrap:wrap; gap:16px; }
.chart { background:#11141c; border:1px solid #222736; border-radius:8px; padding:8px;
  width:280px; }
.chart.wide { width:440px; } .chart.empty p { color:#5d6478; margin:24px 0; text-align:center; }
svg { width:100%; height:auto; } .axis { stroke:#39405a; stroke-width:1; }
.series { fill:none; stroke:#7aa2ff; stroke-width:2; } circle { fill:#7aa2ff; }
.pt { fill:#2c5f4e; fill-opacity:.8; stroke:#59d3a8; }
.pt.front { fill:#59d3a8; }
.tick { fill:#6b7389; font-size:9px; } .latest { margin:4px 0 0; color:#cfd3dc; font-size:11px; }
table { border-collapse:collapse; width:100%; } th,td { text-align:left; padding:6px 10px;
  border-bottom:1px solid #222736; font-size:12px; } th { color:#8b93a7; font-weight:normal; }
.muted { color:#6b7389; } .empty { color:#6b7389; }
.balance .big { font-size:28px; margin:0; color:#59d3a8; }
.grid { display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap; }
.grid > * { flex:1 1 320px; } code { color:#9fb3ff; }
"""

HTMX_SRC = "https://unpkg.com/htmx.org@1.9.12/dist/htmx.min.js"


@dataclass(frozen=True)
class AppState:
    """Where this dashboard reads from."""

    runs_dir: Path
    ledger_path: Path


def render_page(app_state: AppState) -> str:
    """The whole page. Each fragment also has its own endpoint for HTMX polling."""
    summaries = load_summaries(app_state.runs_dir)
    fragments = (
        render_curves(summaries),
        render_pareto(load_pareto(app_state.runs_dir)),
        render_ledger(load_ledger(app_state.ledger_path)),
    )
    poll = 'hx-trigger="every 10s" hx-swap="outerHTML"'
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>Anneal</title>"
        f"<style>{CSS}</style>"
        f'<script src="{HTMX_SRC}" defer></script></head><body>'
        "<h1>Anneal</h1>"
        f'<p class="muted">runs: {escape(str(app_state.runs_dir))} · '
        f"ledger: {escape(str(app_state.ledger_path))}</p>"
        f'<div class="grid">'
        f'<div hx-get="/fragments/balance" {poll}>{render_balance(credit_balance())}</div>'
        f'<div hx-get="/fragments/pareto" {poll}>{fragments[1]}</div></div>'
        f'<div hx-get="/fragments/curves" {poll}>{fragments[0]}</div>'
        f'<div hx-get="/fragments/ledger" {poll}>{fragments[2]}</div>'
        "</body></html>"
    )


def create_app(runs_dir: Path | str = "runs", ledger_path: Path | str = "ledger.json") -> FastAPI:
    """FastAPI app serving the page plus one endpoint per HTMX fragment."""
    state = AppState(Path(runs_dir), Path(ledger_path))
    app = FastAPI(title="Anneal dashboard")
    app.state.anneal = state

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render_page(state)

    @app.get("/fragments/curves", response_class=HTMLResponse)
    def curves() -> str:
        return render_curves(load_summaries(state.runs_dir))

    @app.get("/fragments/ledger", response_class=HTMLResponse)
    def ledger() -> str:
        return render_ledger(load_ledger(state.ledger_path))

    @app.get("/fragments/pareto", response_class=HTMLResponse)
    def pareto() -> str:
        return render_pareto(load_pareto(state.runs_dir))

    @app.get("/fragments/balance", response_class=HTMLResponse)
    def balance() -> str:
        return render_balance(credit_balance())

    return app


def main(argv: list[str] | None = None) -> int:
    """Serve the dashboard on http://localhost:8000."""
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
