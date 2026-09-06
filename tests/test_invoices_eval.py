"""Acceptance tests for domains/invoices/eval.py. No LLM calls; never reads holdout content."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import yaml

from domains.invoices import eval as invoices_eval
from domains.invoices.fixtures import tools as invoices_tools

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ROOT / "domains" / "invoices"


def _task(task_id: str) -> invoices_eval.Task:
    for task in invoices_eval.load_tasks("train") + invoices_eval.load_tasks("search"):
        if task.id == task_id:
            return task
    raise AssertionError(f"no non-holdout task {task_id}")


def _tagged(*tags: str) -> invoices_eval.Task:
    for task in invoices_eval.load_tasks("train") + invoices_eval.load_tasks("search"):
        if set(tags) <= set(task.tags):
            return task
    raise AssertionError(f"no non-holdout task tagged {tags}")


def _good_output(task: invoices_eval.Task) -> dict:
    return {name: task.expected[name] for name in invoices_eval.FIELDS} | {
        "decision": task.expected["decision"],
        "reason": task.expected["rule"] or "all checks passed",
    }


# --------------------------------------------------------------------------- scoring


def test_known_good_output_scores_one() -> None:
    task = _tagged("clean")
    invoices_eval.setup(task)
    assert invoices_eval.score(task, _good_output(task)) == 1.0


def test_known_good_output_as_json_string_scores_one() -> None:
    task = _tagged("messy")
    assert invoices_eval.score(task, json.dumps(_good_output(task))) == 1.0


def test_known_good_output_in_a_fenced_block_scores_one() -> None:
    task = _tagged("clean")
    fenced = f"Here you go:\n```json\n{json.dumps(_good_output(task))}\n```\n"
    assert invoices_eval.score(task, fenced) == 1.0


def test_wrong_decision_scores_half() -> None:
    task = _tagged("messy", "duplicate_invoice")
    output = _good_output(task) | {"decision": "approve"}
    assert invoices_eval.score(task, output) == 0.5


def test_wrong_field_scores_half() -> None:
    task = _tagged("clean")
    output = _good_output(task) | {"total": 0.01}
    assert invoices_eval.score(task, output) == 0.5


def test_unparseable_output_scores_zero() -> None:
    task = _tagged("clean")
    assert invoices_eval.score(task, "I was unable to read this invoice.") == 0.0
    assert invoices_eval.score(task, None) == 0.0


def test_absent_po_is_matched_by_any_nullish_spelling() -> None:
    task = _tagged("messy", "po_absent")
    assert task.expected["po_number"] is None
    for spelling in (None, "", "null", "N/A", "not provided"):
        assert invoices_eval.score(task, _good_output(task) | {"po_number": spelling}) == 1.0


def test_total_accepts_numeric_strings() -> None:
    task = _tagged("clean")
    printed = f"{task.expected['total']:,.2f}"
    assert invoices_eval.score(task, _good_output(task) | {"total": printed}) == 1.0


# --------------------------------------------------------------------------- hard fail


def test_posting_a_messy_invoice_is_hard_fail() -> None:
    task = _tagged("messy", "duplicate_invoice")
    assert task.expected["decision"] == "escalate"
    trace = [
        {"tool": "lookup_po", "args": {"po_number": task.expected["po_number"]}},
        {
            "tool": "post_entry",
            "args": {
                "invoice_number": task.expected["invoice_number"],
                "vendor": task.expected["vendor"],
                "total": task.expected["total"],
                "currency": task.expected["currency"],
            },
        },
    ]
    assert invoices_eval.is_hard_fail(task, trace) is True


def test_escalating_a_messy_invoice_is_not_hard_fail() -> None:
    task = _tagged("messy")
    trace = [{"tool": "escalate", "args": {"invoice_number": "x", "reason": "no PO"}}]
    assert invoices_eval.is_hard_fail(task, trace) is False


def test_posting_a_clean_invoice_once_is_not_hard_fail() -> None:
    task = _tagged("clean")
    trace = [{"tool": "post_entry", "args": {}}]
    assert invoices_eval.is_hard_fail(task, trace) is False


def test_posting_a_clean_invoice_twice_is_hard_fail() -> None:
    task = _tagged("clean")
    trace = [{"tool": "post_entry", "args": {}}, {"tool": "post_entry", "args": {}}]
    assert invoices_eval.is_hard_fail(task, trace) is True


def test_escalating_a_clean_invoice_scores_half_and_is_not_hard_fail() -> None:
    task = _tagged("clean")
    output = _good_output(task) | {"decision": "escalate"}
    assert invoices_eval.score(task, output) == 0.5
    assert invoices_eval.is_hard_fail(task, [{"tool": "escalate", "args": {}}]) is False


def test_reads_are_never_hard_fail() -> None:
    task = _tagged("messy")
    trace = [
        {"tool": "lookup_po", "args": {"po_number": "PO-4000"}},
        {"tool": "lookup_receipt", "args": {"po_number": "PO-4000"}},
    ]
    assert invoices_eval.is_hard_fail(task, trace) is False


# --------------------------------------------------------------------------- dataset


def test_split_sizes_and_threshold() -> None:
    assert invoices_eval.THRESHOLD == 1.0
    lines = [json.loads(line) for line in (DOMAIN / "tasks.jsonl").read_text().splitlines()]
    assert len(lines) == 60
    counts: dict[str, int] = {}
    for raw in lines:  # counted from the file so no holdout row is ever materialised
        counts[raw["split"]] = counts.get(raw["split"], 0) + 1
    assert counts == {"train": 30, "search": 15, "holdout": 15}
    assert len(invoices_eval.load_tasks("train")) == 30
    assert len(invoices_eval.load_tasks("search")) == 15


def test_non_trivial_cases_are_forty_percent_and_evenly_spread() -> None:
    """Dataset v2: 18 escalate rows + 6 approve traps, uniform 40% per split."""
    lines = [json.loads(line) for line in (DOMAIN / "tasks.jsonl").read_text().splitlines()]
    messy = [raw for raw in lines if "messy" in raw["tags"]]
    traps = [raw for raw in lines if "trap" in raw["tags"]]
    assert len(messy) == 18
    assert len(traps) == 6
    by_rule: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for raw in messy + traps:
        by_rule[raw["expected"]["rule"]] = by_rule.get(raw["expected"]["rule"], 0) + 1
        by_split[raw["split"]] = by_split.get(raw["split"], 0) + 1
    assert by_rule == {
        "duplicate_invoice": 3,
        "currency_mismatch": 3,
        "missing_po": 2,
        "tolerance_breach": 6,  # 2 plain + 4 total_match_breach
        "missing_receipt": 4,  # 2 absent + 2 short_receipt
        None: 6,  # approve traps break no rule
    }
    assert by_split == {"train": 12, "search": 6, "holdout": 6}
    for raw in traps:
        assert raw["expected"]["decision"] == "approve", raw["id"]


def test_every_task_carries_the_fields_the_evaluator_needs() -> None:
    for task in invoices_eval.load_tasks("train") + invoices_eval.load_tasks("search"):
        assert task.input["invoice_text"].startswith("INVOICE")
        assert task.expected["decision"] in {"approve", "escalate"}
        for name in invoices_eval.FIELDS:
            assert name in task.expected
        assert (task.expected["decision"] == "escalate") == ("messy" in task.tags)


def test_tools_yaml_impls_resolve() -> None:
    spec = yaml.safe_load((DOMAIN / "tools.yaml").read_text())
    names = {tool["name"] for tool in spec["tools"]}
    assert names == {"lookup_po", "lookup_receipt", "post_entry", "escalate"}
    for tool in spec["tools"]:
        module_path, _, fn_name = tool["impl"].removeprefix("python:").rpartition(".")
        fn = getattr(importlib.import_module(module_path), fn_name)
        assert callable(fn)
        assert tool["name"] == fn_name
        assert set(tool["args"]["required"]) <= set(tool["args"]["properties"])


# --------------------------------------------------------------------------- tools


def test_lookup_po_returns_null_for_an_unknown_po() -> None:
    invoices_tools.reset()
    assert json.loads(invoices_tools.lookup_po("PO-0000")) is None


def test_lookup_receipt_returns_empty_list_when_nothing_was_received() -> None:
    task = _tagged("messy", "missing_receipt")
    invoices_eval.setup(task)
    assert json.loads(invoices_tools.lookup_receipt(task.expected["po_number"])) == []


def test_lookup_po_exposes_prior_postings_for_the_vendor() -> None:
    task = _tagged("messy", "duplicate_invoice")
    invoices_eval.setup(task)
    po = json.loads(invoices_tools.lookup_po(task.expected["po_number"]))
    assert task.expected["invoice_number"] in po["posted_invoices"]

    clean = _tagged("clean")
    clean_po = json.loads(invoices_tools.lookup_po(clean.expected["po_number"]))
    assert clean.expected["invoice_number"] not in clean_po["posted_invoices"]


def test_post_entry_appends_once_and_refuses_a_repeat() -> None:
    task = _tagged("clean")
    invoices_eval.setup(task)
    before = len(invoices_tools.posted_entries())
    first = json.loads(invoices_tools.post_entry(**_post_args(task)))
    assert first["status"] == "posted"
    assert len(invoices_tools.posted_entries()) == before + 1
    again = invoices_tools.post_entry(**_post_args(task))
    assert again.startswith("Error")
    assert len(invoices_tools.posted_entries()) == before + 1


def test_setup_resets_the_ledger_between_runs() -> None:
    task = _tagged("clean")
    invoices_eval.setup(task)
    before = invoices_tools.posted_entries()
    invoices_tools.post_entry(**_post_args(task))
    invoices_tools.escalate(invoice_number="INV-1", reason="test")
    assert invoices_tools.posted_entries() != before
    invoices_eval.setup(task)
    assert invoices_tools.posted_entries() == before
    assert invoices_tools.escalated_entries() == []


def _post_args(task: invoices_eval.Task) -> dict:
    return {
        "invoice_number": task.expected["invoice_number"],
        "vendor": task.expected["vendor"],
        "total": task.expected["total"],
        "currency": task.expected["currency"],
    }


@pytest.mark.parametrize("fixture", ["pos.json", "receipts.json", "posted.json"])
def test_fixtures_are_committed_and_parse(fixture: str) -> None:
    assert json.loads((DOMAIN / "fixtures" / fixture).read_text())
