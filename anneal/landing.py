"""The landing page's headline figures, derived from the runs the report is generated from.

CLAUDE.md's fifth non-negotiable is that numbers come from ``runs/*.jsonl`` and are never typed
by hand. The README obeys it through ``anneal report --write-readme``; the landing page did not,
and drifted exactly as far as you would expect -- it advertised "29x cheaper", "$0.0014 per
task" and "5 -> 3 hard fails on airline", none of which appear anywhere in ``runs/final``.

So the page's figures are generated the same way the README's table is: this module turns the
same summaries into a block of HTML, ``anneal report --write-landing`` replaces the marked
region in ``landing/index.html``, and a guard test fails the build if the committed page stops
being what the tool emits.

What it selects is a rule, not an editorial choice: for every domain, compare the iteration-0
baseline against the best stage that exists for it, keep the movements that are real
improvements, rank them by relative size and show the strongest few. A domain that got worse
contributes nothing to this page and is still in the README table, where regressions are
reported rather than hidden.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from anneal import vocab

LANDING_START = "<!-- stats:start -->"
LANDING_END = "<!-- stats:end -->"

MAX_CARDS = 3  # plus the refusal count, which is always shown


@dataclass(frozen=True)
class Movement:
    """One measured improvement: what moved, for which agent, from what to what."""

    domain: str
    metric: str
    heading: str
    before: float
    after: float
    caption: str

    @property
    def ratio(self) -> float:
        """How much better, as a multiple. Every metric here is one where lower is better."""
        return self.before / self.after if self.after else float("inf")


def _fmt(metric: str, value: float) -> str:
    if metric == "cost_per_task":
        return f"${value:.3g}"  # three significant figures, as the report table uses
    if metric == "p95_latency_ms":
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}"


def agent_name(runs_dir: Path, domain: str) -> str:
    """What this agent is for, in words; its directory slug only if the goal file is unreadable."""
    for candidate in (Path(__file__).resolve().parent.parent / "domains" / domain,
                      runs_dir.parent / "domains" / domain):
        title = vocab.title_from_goal(candidate / "goal.md")
        if title:
            return title
    return domain


def _annealed_point(runs_dir: Path, domain: str) -> dict[str, Any] | None:
    """The Pareto point ``anneal anneal`` picked, or None if the stage has not run."""
    path = runs_dir / domain / "anneal" / "pareto.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    winner = data.get("winner")
    return next((p for p in data.get("points", []) if p.get("config_id") == winner), None)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def movements(runs_dir: Path, summaries: list[dict[str, Any]]) -> list[Movement]:
    """Every real improvement in the runs, strongest first.

    Cost and latency are read from the annealed configuration when the downshift stage ran for
    that domain, because that stage is what moves them; hard fails are read from the last
    iteration's gate, because that is what moves those. A metric that did not improve is left
    out entirely rather than shown with a flattering sign.
    """
    found: list[Movement] = []
    for name in dict.fromkeys(b["domain"] for b in summaries):
        got = [b for b in summaries if b["domain"] == name]
        base = got[0]["search"].get(got[0].get("incumbent_id"), {})
        annealed = _annealed_point(runs_dir, name)
        if annealed:
            for metric, heading, caption in (
                ("cost_per_task", "cost per task",
                 "after the winner was moved onto cheaper models"),
                ("p95_latency_ms", "slowest runs",
                 "on the same tasks, after the same downshift"),
            ):
                before, after = _number(base.get(metric)), _number(annealed.get(metric))
                if before and after and after < before:
                    found.append(Movement(name, metric, heading, before, after, caption))
        first_fails = _number((got[0].get("incumbent") or {}).get("hard_fails"))
        last_fails = _number((got[-1].get("incumbent") or {}).get("hard_fails"))
        if first_fails is not None and last_fails is not None and last_fails < first_fails:
            found.append(Movement(
                name, "hard_fails", "serious mistakes", first_fails, last_fails,
                "counted on tasks held back from every repair step",
            ))
    return sorted(found, key=lambda m: m.ratio, reverse=True)


def refusals(summaries: list[dict[str, Any]], gate_of) -> int:
    """How many mutations the gate refused across every domain in the run set."""
    total = 0
    for body in summaries:
        decision = body.get("decision") or (gate_of(body) or {}).get("decision")
        if decision == "reject":
            total += 1
    return total


def _card(number: str, heading: str, body: str) -> str:
    return (
        f'<div class="stat"><p class="n">{number}</p>'
        f"<h3>{escape(heading)}</h3><p>{escape(body)}</p></div>"
    )


def render_stats(runs_dir: Path, summaries: list[dict[str, Any]], gate_of) -> str:
    """The landing page's stat block: the strongest measured movements, then the refusal count."""
    cards = []
    for move in movements(runs_dir, summaries)[:MAX_CARDS]:
        before = escape(_fmt(move.metric, move.before))
        after = escape(_fmt(move.metric, move.after))
        # The metric leads and the agent follows in the body: agent names come from each
        # domain's own goal.md and can be a whole clause long, which made a poor heading.
        cards.append(_card(
            f'<span class="from">{before} &rarr;</span> {after}',
            move.heading.capitalize(),
            f"{agent_name(runs_dir, move.domain)}, {move.caption}.",
        ))
    refused = refusals(summaries, gate_of)
    domains = len({b["domain"] for b in summaries})
    cards.append(_card(
        escape(str(refused)),
        "changes refused" if refused != 1 else "change refused",
        f"Across {domains} agents, every repair that failed the held-back test was thrown "
        "away and is named in the results table, including ones that scored higher.",
    ))
    return '<div class="stats">' + "".join(cards) + "</div>"


def write_landing(path: Path, block: str) -> None:
    """Replace the marked stats block in ``path`` and touch nothing else."""
    text = path.read_text(encoding="utf-8")
    head, marker, rest = text.partition(LANDING_START)
    body, end, tail = rest.partition(LANDING_END)
    if not marker or not end:
        raise ValueError(f"{path} has no {LANDING_START} / {LANDING_END} block")
    path.write_text(f"{head}{marker}\n{block}\n{end}{tail}", encoding="utf-8")
