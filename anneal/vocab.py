"""The words the console shows a reader who has never opened this repo.

Everywhere else in Anneal, a thing is named by the identifier the code routes on:
``planner_executor``, ``unsafe_action``, ``add_cite_or_abstain``, ``cand-02-i1``. Those names
are correct and must not change -- they are keys in specs, ledger rows and run files. They are
also unreadable to the person the product is for, who asked for an agent that triages invoices
and did not sign up to learn our taxonomy.

This module is the single translation layer between the two. It maps identifier to phrase and
nothing else: no rendering, no HTML, no policy. The dashboard is its only caller today, so
there is exactly one place to change a word, and a term that has no entry degrades to a
readable de-slugged form rather than disappearing.

Two rules hold every label here:

* **Say what happened, not what it is called.** "Called the wrong tool", not "wrong tool".
* **No identifier the reader cannot act on.** Internal ids stay available as small secondary
  text next to the phrase, because they are what you type on the command line; they are just
  never the thing you read first.

Failure-class labels live in ``specs/failure_taxonomy.yaml`` beside the class they name, since
that file already owns the taxonomy's meaning. Everything else is defined in ``anneal/`` and so
is named here.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "specs" / "failure_taxonomy.yaml"

# Every goal.md opens "# Goal: <what the agent is for>". Both the switcher label and the
# contract row strip this, because each already says which of the two it is showing.
GOAL_PREFIXES = ("goal:", "goal -", "goal —")

# The six stages of the loop, as a verb the reader can follow rather than our internal noun.
STEPS: dict[str, tuple[str, str]] = {
    "architect": ("Design", "Turns your goal and your tools into a few candidate agent designs."),
    "run": ("Try it", "Runs every design over the practice tasks and scores each one."),
    "diagnose": ("Find the faults",
                 "Reads the failed attempts and groups them by what went wrong."),
    "mutate": ("Fix one", "Applies the repair that fault calls for, making a challenger."),
    "gate": ("Prove it helped",
             "Re-runs challenger and current best on held-back tasks, then decides."),
    "anneal": ("Make it cheaper",
               "Moves the winner onto cheaper models for as long as the score holds."),
}

# How the nodes of an agent are wired together.
TOPOLOGIES: dict[str, str] = {
    "single": "One agent",
    "planner_executor": "Planner and doer",
    "critic_loop": "Doer with a reviewer",
    "tool_router": "Router picks the tool",
}

# What each node in the agent is there to do.
ROLES: dict[str, str] = {
    "planner": "Planner",
    "executor": "Doer",
    "critic": "Reviewer",
    "router": "Router",
    "validator": "Checker",
    "escalate": "Hands off to a human",
}

# The model ladder, named by what it costs rather than by our tier id.
TIERS: dict[str, str] = {
    "frontier": "Best",
    "mid": "Mid",
    "cheap": "Cheap",
    "flash": "Cheaper",
    "nano": "Cheapest",
}

# The repairs the loop knows how to make, phrased as the change the reader would see.
OPERATORS: dict[str, str] = {
    "rewrite_tool_desc": "Rewrite the tool instructions",
    "add_fewshots": "Show it worked examples",
    "add_validator_node": "Add a checker",
    "add_cite_or_abstain": "Make it cite or say it doesn't know",
    "add_step_budget_and_critic": "Cap its steps and add a reviewer",
    "add_memory": "Let it remember past attempts",
    "add_escalation_node": "Let it hand off to a human",
    "switch_topology": "Rewire the agent",
    "synthesize_tool": "Write a new tool",
}

# What the gate decided.
DECISIONS: dict[str, str] = {
    "promote": "Kept the change",
    "reject": "Threw the change away",
    "no_candidate": "Nothing to try",
}


def humanize(slug: Any) -> str:
    """A readable phrase for any identifier with no entry of its own.

    Used as the fallback for every lookup below, so a class or operator added to the taxonomy
    before it is added here still reads as words rather than vanishing or showing raw snake
    case. It is deliberately dumb: underscores and dashes become spaces, the first letter is
    capitalised, and nothing else is guessed.
    """
    text = str(slug or "").replace("_", " ").replace("-", " ").strip()
    return text[:1].upper() + text[1:] if text else ""


def _lookup(table: dict[str, str], key: Any) -> str:
    return table.get(str(key or ""), "") or humanize(key)


def step(key: Any) -> tuple[str, str]:
    """``(name, one-line explanation)`` for a stage of the loop."""
    return STEPS.get(str(key or ""), (humanize(key), ""))


def topology(key: Any) -> str:
    return _lookup(TOPOLOGIES, key)


def role(key: Any) -> str:
    return _lookup(ROLES, key)


def tier(key: Any) -> str:
    return _lookup(TIERS, key)


def operator(key: Any) -> str:
    return _lookup(OPERATORS, key)


def decision(key: Any) -> str:
    return _lookup(DECISIONS, key)


@lru_cache(maxsize=1)
def _failure_labels(path: str = "") -> dict[str, str]:
    """``{class id: label}`` from the taxonomy, which owns what a failure class means."""
    try:
        data = yaml.safe_load(Path(path or TAXONOMY_PATH).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    classes = (data or {}).get("classes")
    if not isinstance(classes, list):
        return {}
    return {
        str(row["id"]): str(row["label"])
        for row in classes
        if isinstance(row, dict) and row.get("id") and row.get("label")
    }


def failure(key: Any) -> str:
    """What this failure class means, in the taxonomy's own words."""
    return _failure_labels().get(str(key or ""), "") or humanize(key)


def design_name(candidate_id: Any) -> str:
    """A readable name for a candidate spec, e.g. "Design 2 + a checker".

    Candidate ids are built by the loop as ``cand-NN`` for an architect proposal and
    ``<parent>-<operator>-i<iteration>`` for each repair applied on top (see
    ``mutate._with_lineage``). That is a precise lineage and a poor label: it is the longest
    string in the console and says nothing to a reader. Reading the lineage back out turns it
    into the sentence it always meant -- which design this is, and what was changed on it.

    An id that does not follow the pattern is returned de-slugged rather than guessed at.
    """
    text = str(candidate_id or "")
    if not text:
        return ""
    parts = text.split("-")
    if len(parts) < 2 or parts[0] != "cand":
        return humanize(text)
    name = f"Design {parts[1].lstrip('0') or parts[1]}"
    rest = parts[2:]
    # trailing "-i<N>" is the iteration the repair was applied on; the round is shown elsewhere
    if rest and rest[-1].startswith("i") and rest[-1][1:].isdigit():
        rest = rest[:-1]
    if rest:
        repair = OPERATORS.get("_".join(rest), "")
        tail = repair[:1].lower() + repair[1:] if repair else humanize("-".join(rest))
        name += f" + {tail}"
    return name


def title_from_goal(goal_md: Path | str) -> str:
    """The plain-English name of an agent, read from the first heading of its goal.md.

    Every goal.md opens "# Goal: <what the agent is for>", including the ones ``anneal init``
    writes from the user's own answer. That line is the only place a domain states its purpose
    in words, so both the console's switcher and the landing page's headline figures name an
    agent from it rather than from its directory slug. A file that cannot be read returns "",
    and the caller falls back to the slug.
    """
    try:
        text = Path(goal_md).read_text(encoding="utf-8")
    except OSError:
        return ""
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if not line:
            continue
        lowered = line.lower()
        for prefix in GOAL_PREFIXES:
            if lowered.startswith(prefix):
                return _clip(line[len(prefix):].strip())
        return _clip(line)
    return ""


def _clip(text: str, limit: int = 60) -> str:
    """Trim a title to fit a switcher tab, on a word boundary rather than mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:")
    return f"{cut or text[:limit]}…"
