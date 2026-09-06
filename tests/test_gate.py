"""Unit tests for anneal.gate on synthetic rows. No LLM calls, no runner, no network."""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest
from scipy.stats import binomtest

from anneal import gate
from anneal.spec import HarnessSpec, Node


def _spec(spec_id: str) -> HarnessSpec:
    return HarnessSpec(
        id=spec_id,
        topology="single",
        step_budget=4,
        nodes=[Node(name="exec", role="executor", model_tier="mid", system_prompt_ref="p@v1")],
    )


def _domain(threshold: float = 1.0) -> Any:
    return types.SimpleNamespace(name="synthetic", eval=types.SimpleNamespace(THRESHOLD=threshold))


def _row(task_id: str, score: float, hard_fail: bool = False) -> dict[str, Any]:
    return {"task_id": task_id, "score": score, "hard_fail": hard_fail}


def _rows(scores: dict[str, list[float]], hard: set[str] = frozenset()) -> list[list[dict]]:
    """scores[task] = [score seed0, seed1, seed2] -> rows grouped by seed."""
    n_runs = len(next(iter(scores.values())))
    return [
        [_row(t, s[i], hard_fail=t in hard and i == 0) for t, s in scores.items()]
        for i in range(n_runs)
    ]


class FakeRunner:
    """Scripted stand-in for runner.run keyed by candidate_id, returns rows per seed."""

    def __init__(self, by_spec: dict[str, list[list[dict]]]):
        self.by_spec = by_spec
        self.calls: list[tuple[str, str, int, int]] = []

    def __call__(self, spec, domain, split, *, seed=0, iteration=0, **kw):
        self.calls.append((spec.id, split, seed, iteration))
        rows = self.by_spec[spec.id][seed]
        return [dict(r, candidate_id=spec.id) for r in rows]


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)
    gate.clear_cache()
    yield
    gate.clear_cache()


# --- pass^3 math ---------------------------------------------------------------------------


def test_pass3_requires_all_runs_at_or_above_threshold():
    rows = _rows({"a": [1.0, 1.0, 1.0], "b": [1.0, 0.0, 1.0], "c": [0.8, 0.8, 0.8]})
    assert gate.pass3_by_task(rows, threshold=1.0) == {"a": True, "b": False, "c": False}
    assert gate.pass3_by_task(rows, threshold=0.8) == {"a": True, "b": False, "c": True}


def test_spec_metrics_mean_pass3_hard_fails_and_gen_gap():
    rows = _rows({"a": [1.0, 1.0, 1.0], "b": [0.0, 0.0, 0.0]}, hard={"b"})
    m = gate.spec_metrics(rows, threshold=1.0, search_rows=[_row("s1", 1.0), _row("s2", 0.5)])
    assert m.mean_score == pytest.approx(0.5)
    assert m.pass3_rate == pytest.approx(0.5)
    assert m.hard_fails == 1
    assert m.gen_gap == pytest.approx(0.75 - 0.5)
    assert m.n_tasks == 2 and m.n_runs == 3


def test_gen_gap_is_none_without_search_rows():
    rows = _rows({"a": [1.0, 1.0, 1.0]})
    assert gate.spec_metrics(rows, threshold=1.0, search_rows=[]).gen_gap is None


# --- binomial paired test ------------------------------------------------------------------


@pytest.mark.parametrize("wins,losses", [(5, 0), (4, 1), (3, 3), (0, 2), (7, 1)])
def test_paired_test_matches_scipy_exact_one_sided(wins: int, losses: int):
    """One-sided by design: the gate only ever asks whether the candidate is better."""
    tasks = [f"w{i}" for i in range(wins)] + [f"l{i}" for i in range(losses)] + ["tie"]
    cand = {t: t.startswith("w") or t == "tie" for t in tasks}
    inc = {t: t.startswith("l") or t == "tie" for t in tasks}
    result = gate.paired_test(cand, inc)
    assert (result.wins, result.losses) == (wins, losses)
    expected = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue
    assert result.p == pytest.approx(expected)


def test_a_lone_clean_win_is_reported_as_underpowered_not_as_no_effect():
    """The airline regression: 1 win, 0 losses, rejected. The rejection is arithmetic.

    No candidate can clear alpha on a single discordant pair, so this must be legible as
    "too little data to tell" rather than "the change did not help".
    """
    result = gate.paired_test({"a": True, "b": True}, {"a": False, "b": True})
    assert (result.wins, result.losses) == (1, 0)
    assert result.p >= gate.ALPHA
    assert gate.underpowered(result)
    # Enough same-direction pairs and the very same test does clear alpha.
    many = gate.paired_test({f"w{i}": True for i in range(4)}, {f"w{i}": False for i in range(4)})
    assert (many.wins, many.losses) == (4, 0)
    assert many.p < gate.ALPHA
    assert not gate.underpowered(many)


def test_pass_counts_see_partial_movement_that_pass3_booleans_hide():
    """The bugfix regression, with its real shape: 3 runs, threshold 1.0, 4 tasks that moved.

    Under pass^3 booleans none of these tasks changed state, so the test saw zero discordant
    pairs and reported p=1.000 -- indistinguishable from "we have no data". Replayed on the
    archived bugfix rows, this is exactly what happened twice, and the pass-count statistic
    recovers 2-2 and 3-1 respectively. It must recover the pairs WITHOUT promoting: the
    candidate here is a wash, and a wash has to stay rejected.
    """
    runs_inc = [
        [{"task_id": "a", "score": 0.0}, {"task_id": "b", "score": 1.0}],
        [{"task_id": "a", "score": 0.0}, {"task_id": "b", "score": 0.0}],
        [{"task_id": "a", "score": 0.0}, {"task_id": "b", "score": 0.0}],
    ]
    runs_cand = [
        [{"task_id": "a", "score": 1.0}, {"task_id": "b", "score": 0.0}],
        [{"task_id": "a", "score": 1.0}, {"task_id": "b", "score": 0.0}],
        [{"task_id": "a", "score": 0.0}, {"task_id": "b", "score": 0.0}],
    ]
    # Neither task is a clean sweep either side, so pass^3 calls both False: no pairs at all.
    assert gate.pass3_by_task(runs_cand, 1.0) == {"a": False, "b": False}
    assert gate.pass3_by_task(runs_inc, 1.0) == {"a": False, "b": False}
    blind = gate.paired_test(gate.pass3_by_task(runs_cand, 1.0), gate.pass3_by_task(runs_inc, 1.0))
    assert (blind.wins, blind.losses, blind.p) == (0, 0, 1.0)

    # Pass counts see a moved 0->2 and b moved 1->0: one win, one loss. A wash, but visible.
    assert gate.passes_by_task(runs_cand, 1.0) == {"a": 2, "b": 0}
    assert gate.passes_by_task(runs_inc, 1.0) == {"a": 0, "b": 1}
    seeing = gate.paired_test(
        gate.passes_by_task(runs_cand, 1.0), gate.passes_by_task(runs_inc, 1.0)
    )
    assert (seeing.wins, seeing.losses) == (1, 1)
    assert seeing.p >= gate.ALPHA, "a wash must still be rejected"


def test_min_discordant_to_promote_matches_the_exact_binomial_floor():
    floor = gate._min_discordant_to_promote()
    assert 0.5**floor < gate.ALPHA <= 0.5 ** (floor - 1)


def test_paired_test_no_discordant_pairs_has_p_one():
    result = gate.paired_test({"a": True}, {"a": True})
    assert (result.wins, result.losses, result.p) == (0, 0, 1.0)


def test_paired_test_5_wins_0_losses_is_below_alpha():
    cand = {str(i): True for i in range(5)}
    inc = {str(i): False for i in range(5)}
    assert gate.paired_test(cand, inc).p < 0.1


# --- decision ------------------------------------------------------------------------------


def _metrics(pass3_rate: float, hard_fails: int) -> gate.SpecMetrics:
    return gate.SpecMetrics(
        candidate_id="x", mean_score=pass3_rate, pass3_rate=pass3_rate, hard_fails=hard_fails,
        gen_gap=None, pass3={}, n_tasks=10, n_runs=3,
    )


def test_decide_rejects_when_pass3_below_incumbent():
    ok, verdict = gate.decide(_metrics(0.5, 0), _metrics(0.6, 0), p=0.01)
    assert not ok and verdict.metric == "pass3_rate"
    assert str(verdict) == "pass3_rate 0.500 < incumbent 0.600"


def test_decide_rejects_when_more_hard_fails():
    ok, verdict = gate.decide(_metrics(0.9, 2), _metrics(0.6, 1), p=0.01)
    assert not ok and verdict.metric == "hard_fails"
    assert str(verdict) == "hard_fails 2 > incumbent 1"


def test_decide_rejects_when_not_significant():
    ok, verdict = gate.decide(_metrics(0.7, 0), _metrics(0.6, 0), p=0.25)
    assert not ok and verdict.metric == "p"
    # the sentence is byte-identical to the one the gate wrote before Verdict existed:
    # three decimals on the p-value, alpha bare
    assert str(verdict) == f"p 0.250 >= alpha {gate.ALPHA:g}"


def test_decide_promotes_when_all_conditions_hold():
    ok, verdict = gate.decide(_metrics(0.9, 0), _metrics(0.6, 0), p=0.05)
    assert ok and str(verdict) == "promoted"
    assert verdict.parts() is None  # a promotion has no failing condition to record


def test_the_verdict_is_recorded_as_both_a_sentence_and_its_fields():
    """gate.json keeps the line the report quotes, plus the fields the console reads.

    Parsing the sentence back apart to render it would be guessing at our own output, and a
    console that shows "hard_fails 3 > incumbent 1" to a reader has told them nothing.
    """
    _, verdict = gate.decide(_metrics(0.9, 3), _metrics(0.6, 1), p=0.01)
    assert verdict.parts() == {
        "metric": "hard_fails", "value": 3, "comparator": ">", "against": 1,
        "against_label": "incumbent", "fmt": "{:.0f}", "against_fmt": "",
    }


# --- end to end with injected runner ----------------------------------------------------


def _scenario() -> dict[str, list[list[dict]]]:
    tasks = [f"t{i}" for i in range(8)]
    inc = _rows({t: [1.0, 1.0, 1.0] if i < 2 else [0.0, 0.0, 0.0] for i, t in enumerate(tasks)},
                hard={"t7"})
    cand = _rows({t: [1.0, 1.0, 1.0] for t in tasks})
    return {"inc": inc, "cand": cand}


def test_gate_promote_path_writes_gate_json_and_flips_labels(tmp_path: Path, monkeypatch):
    runner = FakeRunner(_scenario())
    flipped: list[str] = []
    monkeypatch.setattr(gate, "promote_prompts", lambda spec: flipped.append(spec.id) or [])
    result = gate.gate(
        _spec("inc"), _spec("cand"), _domain(), iteration=2, runs_dir=tmp_path,
        search_rows=[_row("s", 1.0)], run=runner,
    )
    assert result.promoted and result.reason == "promoted"
    assert result.wins == 6 and result.losses == 0
    assert result.p == pytest.approx(binomtest(6, 6, 0.5, alternative="greater").pvalue)
    assert flipped == ["cand"]
    path = tmp_path / "synthetic" / "2" / "gate.json"
    data = json.loads(path.read_text())
    assert data["decision"] == "promote" and data["reason"] == "promoted"
    assert data["candidate"]["pass3_rate"] == 1.0 and data["incumbent"]["pass3_rate"] == 0.25
    assert data["incumbent"]["hard_fails"] == 1 and data["candidate"]["hard_fails"] == 0
    assert data["candidate"]["gen_gap"] == pytest.approx(0.0)
    assert data["incumbent"]["gen_gap"] is None
    assert data["p"] == pytest.approx(result.p) and data["iteration"] == 2
    assert {c[1] for c in runner.calls} == {"holdout"}
    assert sorted(c[2] for c in runner.calls if c[0] == "cand") == [0, 1, 2]


def test_underpowered_but_winning_candidate_earns_a_second_block_and_promotes(
    tmp_path: Path, monkeypatch
):
    """2-0 in the first block cannot clear alpha (floor is 4); the escalated gate can.

    Six seeds are scripted around a flaky incumbent: in the first block only t1/t2 are
    discordant (2-0, no verdict reachable), and the extension surfaces the incumbent's
    flakiness on t3/t4, ending 4-0 with p=0.0625 < 0.1. The gate must buy the second
    block itself, extend the incumbent too, and record `escalated`.
    """
    inc_scores = {
        "t1": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "t2": [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "t3": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        "t4": [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    }
    cand_scores = {t: [1.0] * 6 for t in inc_scores}
    runner = FakeRunner({"inc": _rows(inc_scores), "cand": _rows(cand_scores)})
    monkeypatch.setattr(gate, "promote_prompts", lambda spec: [])
    result = gate.gate(
        _spec("inc"), _spec("cand"), _domain(), iteration=0, runs_dir=tmp_path, run=runner
    )
    assert result.escalated
    assert result.promoted, result.reason
    assert (result.wins, result.losses) == (4, 0)
    assert result.p == pytest.approx(0.0625)
    data = json.loads((tmp_path / "synthetic" / "0" / "gate.json").read_text())
    assert data["escalated"] is True
    # both specs ran seeds 0..5 on holdout
    assert sorted(c[2] for c in runner.calls if c[0] == "cand") == [0, 1, 2, 3, 4, 5]
    assert sorted(c[2] for c in runner.calls if c[0] == "inc") == [0, 1, 2, 3, 4, 5]


def test_a_regression_rejection_never_escalates(tmp_path: Path, monkeypatch):
    """pass^3 or hard-fail regressions are verdicts; more data is not bought for them."""
    scen = _scenario()
    scen["cand"], scen["inc"] = scen["inc"], scen["cand"]  # candidate is the worse one
    runner = FakeRunner(scen)
    monkeypatch.setattr(gate, "promote_prompts", lambda spec: [])
    result = gate.gate(
        _spec("inc"), _spec("cand"), _domain(), iteration=0, runs_dir=tmp_path, run=runner
    )
    assert not result.promoted and not result.escalated
    assert max(c[2] for c in runner.calls) == 2  # never went past the first block


def test_escalation_can_be_disabled_by_env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("ANNEAL_GATE_ESCALATION", "0")
    inc_scores = {"t1": [0.0] * 3, "t2": [0.0] * 3, "t3": [1.0] * 3, "t4": [1.0] * 3}
    cand_scores = {"t1": [1.0] * 3, "t2": [1.0] * 3, "t3": [1.0] * 3, "t4": [1.0] * 3}
    runner = FakeRunner({"inc": _rows(inc_scores), "cand": _rows(cand_scores)})
    monkeypatch.setattr(gate, "promote_prompts", lambda spec: [])
    result = gate.gate(
        _spec("inc"), _spec("cand"), _domain(), iteration=0, runs_dir=tmp_path, run=runner
    )
    assert not result.promoted and not result.escalated
    assert max(c[2] for c in runner.calls) == 2


def test_gate_reject_does_not_flip_labels(tmp_path: Path, monkeypatch):
    scen = _scenario()
    scen["cand"], scen["inc"] = scen["inc"], scen["cand"]  # candidate is the worse one
    runner = FakeRunner(scen)
    flipped: list[str] = []
    monkeypatch.setattr(gate, "promote_prompts", lambda spec: flipped.append(spec.id) or [])
    result = gate.gate(
        _spec("inc"), _spec("cand"), _domain(), iteration=0, runs_dir=tmp_path, run=runner
    )
    assert not result.promoted and result.reason.startswith("pass3_rate")
    assert flipped == []
    data = json.loads((tmp_path / "synthetic" / "0" / "gate.json").read_text())
    assert data["decision"] == "reject"


def test_incumbent_holdout_rows_cached_per_iteration(tmp_path: Path):
    runner = FakeRunner(_scenario())
    gate.gate(_spec("inc"), _spec("cand"), _domain(), iteration=1, runs_dir=tmp_path, run=runner)
    gate.gate(_spec("inc"), _spec("cand"), _domain(), iteration=1, runs_dir=tmp_path, run=runner)
    inc_calls = [c for c in runner.calls if c[0] == "inc"]
    cand_calls = [c for c in runner.calls if c[0] == "cand"]
    assert len(inc_calls) == 3 and len(cand_calls) == 6
    gate.gate(_spec("inc"), _spec("cand"), _domain(), iteration=2, runs_dir=tmp_path, run=runner)
    assert len([c for c in runner.calls if c[0] == "inc"]) == 6


def test_promote_prompts_is_noop_without_key():
    assert gate.promote_prompts(_spec("cand")) == []
