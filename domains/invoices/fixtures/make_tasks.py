"""One-shot generator for the invoices dataset (v2, seed 0). DO NOT RERUN casually.

Emits all four artifacts in a single pass so they cannot drift apart:
`fixtures/pos.json`, `fixtures/receipts.json`, `fixtures/posted.json` and `tasks.jsonl`.

v2 (2026-09-06): regenerated deliberately. The v1 dataset saturated -- every candidate
scored 1.000 at iteration 0, so there was nothing for the optimiser to learn and no
reportable number was invalidated by regenerating. v2 keeps the five v1 rules and adds
four traps aimed at the shortcuts a naive agent actually takes:

  * total_match_breach (escalate): two lines deviate +/-6% in opposite directions so the
    invoice total equals the PO total to the cent. Comparing totals instead of lines
    approves it -- and posting it is a hard fail.
  * short_receipt (escalate): a goods receipt exists but covers only ~85% of one invoiced
    line. Checking receipt *existence* instead of quantities approves it.
  * similar_number (approve): the ledger already holds `<number>-2` for the same vendor.
    Fuzzy duplicate-matching escalates a perfectly payable invoice.
  * symbol_currency (approve): no `Currency:` header; the total line carries the symbol
    (EUR euro sign / GBP pound / USD dollar). Extraction must infer the ISO code.

Invariants asserted before anything is written (this covers gated-split correctness,
since CHECKED.md only names train/search rows):
  * exactly 60 rows, split 30 train / 15 search / 15 holdout;
  * exactly 24 non-trivial rows (12 v1 messy + 6 escalate traps + 6 approve traps),
    12/6/6 across the splits -- a uniform 40% per split;
  * the rule oracle, replayed against the fixtures, reports *exactly* the one rule each
    escalate row is meant to exercise, and no rule for approve rows (incl. approve traps);
  * every clean line deviates from its PO by <= 1.5% and every breach by >= 4%, measured
    after rounding to cents, so no case sits near the 2% boundary.

Run:  uv run python -m domains.invoices.fixtures.make_tasks
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

DOMAIN = Path(__file__).resolve().parents[1]
FIXTURES = DOMAIN / "fixtures"

SEED = 0
N_TRAIN, N_SEARCH, N_HOLDOUT = 30, 15, 15
N_TASKS = N_TRAIN + N_SEARCH + N_HOLDOUT

# 24 non-trivial rows (40%), spread uniformly across the splits: the 12 v1 messy rows,
# 6 escalate traps (look clean, must be escalated) and 6 approve traps (look suspicious,
# must be approved). Traps appear in every split so a learned fix has to generalise.
MESSY_BY_SPLIT: dict[str, list[str]] = {
    "train": [
        "duplicate_invoice",
        "duplicate_invoice",
        "currency_mismatch",
        "missing_po",
        "tolerance_breach",
        "missing_receipt",
        "total_match_breach",
        "total_match_breach",
        "short_receipt",
        "similar_number",
        "similar_number",
        "symbol_currency",
    ],
    "search": [
        "duplicate_invoice",
        "currency_mismatch",
        "missing_po",
        "total_match_breach",
        "short_receipt",
        "similar_number",
    ],
    "holdout": [
        "currency_mismatch",
        "tolerance_breach",
        "missing_receipt",
        "total_match_breach",
        "symbol_currency",
        "similar_number",
    ],
}

# Escalate rules as the oracle reports them; traps that fail an existing rule map onto it.
ORACLE_RULE = {
    "duplicate_invoice": "duplicate_invoice",
    "currency_mismatch": "currency_mismatch",
    "missing_po": "missing_po",
    "tolerance_breach": "tolerance_breach",
    "missing_receipt": "missing_receipt",
    "total_match_breach": "tolerance_breach",  # line-level breach, total camouflaged
    "short_receipt": "missing_receipt",  # receipt exists but does not cover the line
}
APPROVE_TRAPS = {"similar_number", "symbol_currency"}

CURRENCY_SYMBOL = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3"}

VENDORS: list[tuple[str, str]] = [
    ("Northwind Traders", "EUR"),
    ("Aurora Components BV", "EUR"),
    ("Kestrel Industrial Ltd", "GBP"),
    ("Pinebrook Supply Co", "USD"),
    ("Halden Fasteners AB", "EUR"),
    ("Vermillion Packaging", "USD"),
    ("Sable & Roe Chemicals", "GBP"),
    ("Tidewater Logistics Inc", "USD"),
    ("Corvid Electronics GmbH", "EUR"),
    ("Marlowe Paper Mills", "USD"),
]
CURRENCIES = ["USD", "EUR", "GBP"]
CATALOGUE = [
    "Hex bolts M8 zinc",
    "Nylon washers 6mm",
    "Corrugated carton 400x300",
    "Isopropyl alcohol 5L",
    "Cable ties 200mm",
    "Steel bracket L-90",
    "Thermal label roll",
    "Ball bearing 6204",
    "PTFE tape 12mm",
    "Copper lug 16mm2",
    "Pallet wrap 500mm",
    "Silicone sealant 310ml",
]

CLEAN_DEV_MAX = 0.015
BREACH_DEV_MIN = 0.04


# --------------------------------------------------------------------------- planning


def plan_rows(rng: random.Random) -> list[dict[str, Any]]:
    """Assign a split and (for 12 of them) a messy rule to each of the 60 slots."""
    splits = ["train"] * N_TRAIN + ["search"] * N_SEARCH + ["holdout"] * N_HOLDOUT
    rng.shuffle(splits)
    rules: dict[int, str] = {}
    for split, split_rules in MESSY_BY_SPLIT.items():
        slots = [i for i, s in enumerate(splits) if s == split]
        for slot, rule in zip(sorted(rng.sample(slots, len(split_rules))), split_rules):
            rules[slot] = rule
    return [{"split": s, "rule": rules.get(i)} for i, s in enumerate(splits)]


def base_invoice(index: int, rng: random.Random, rule: str | None) -> dict[str, Any]:
    """A fully clean invoice with a PO that matches it line for line."""
    vendor, currency = VENDORS[rng.randrange(len(VENDORS))]
    n_lines = 2 if rule == "total_match_breach" else rng.choice([1, 2, 2, 3])
    skus: list[str] = []
    while len(skus) < n_lines:
        candidate = f"SKU-{rng.randrange(1000, 9999)}"
        if candidate not in skus:
            skus.append(candidate)
    # total_match_breach needs two lines with equal quantity and equal unit price so a
    # +d / -d nudge cancels in the total to the cent.
    shared_qty = rng.choice([40, 60, 80, 100])
    shared_unit = rng.randrange(500, 9000)
    lines = []
    for sku in skus:
        if rule == "total_match_breach":
            quantity, unit_cents = shared_qty, shared_unit
        elif rule == "tolerance_breach":
            quantity, unit_cents = rng.choice([40, 60, 80, 100]), rng.randrange(85, 9000)
        else:
            quantity = rng.choice([12, 24, 25, 50, 60, 120, 200, 240])
            unit_cents = rng.randrange(85, 9000)
        lines.append(
            {
                "sku": sku,
                "description": CATALOGUE[rng.randrange(len(CATALOGUE))],
                "quantity": quantity,
                "unit_cents": unit_cents,
            }
        )
    return {
        "vendor": vendor,
        "invoice_number": f"INV-{10000 + index}",
        "po_number": f"PO-{4000 + index}",
        "date": f"2026-{1 + index % 6:02d}-{1 + index % 27:02d}",
        "currency": currency,
        "po_currency": currency,
        "po_lines": [dict(line) for line in lines],
        "inv_lines": [dict(line) for line in lines],
        "po_exists": True,
        "receipt_exists": True,
        "duplicate_seed": False,
        "near_dup_seed": False,
        "short_receipt": False,
        "symbol_only": False,
        "tags": [],
    }


# --------------------------------------------------------------------------- mutations


def apply_near_tolerance(row: dict[str, Any]) -> None:
    """Nudge one line ~1% -- inside tolerance, so the clean answer stays 'approve'."""
    line = row["inv_lines"][0]
    line["unit_cents"] = line["unit_cents"] + round(line["unit_cents"] * 0.01)
    row["tags"].append("near_tolerance")


def apply_rule(row: dict[str, Any], rule: str, split: str, index: int) -> None:
    if rule == "duplicate_invoice":
        row["duplicate_seed"] = True
    elif rule == "currency_mismatch":
        others = [c for c in CURRENCIES if c != row["po_currency"]]
        row["currency"] = others[index % len(others)]
    elif rule == "missing_po":
        row["po_exists"] = False
        row["receipt_exists"] = False
        if split == "train":  # variant A: no PO printed on the invoice at all
            row["po_number"] = None
            row["tags"].append("po_absent")
        else:  # variant B: a PO number is printed but no such PO exists
            row["po_number"] = f"PO-9{index:03d}"
            row["tags"].append("po_unknown")
    elif rule == "tolerance_breach":
        line = row["inv_lines"][0]
        if split == "train":  # price breach
            line["unit_cents"] = round(line["unit_cents"] * 1.06)
            row["tags"].append("price_breach")
        else:  # quantity breach
            line["quantity"] = line["quantity"] + math.ceil(line["quantity"] * 0.06)
            row["tags"].append("quantity_breach")
    elif rule == "missing_receipt":
        row["receipt_exists"] = False
    elif rule == "total_match_breach":
        # Two equal lines: +d on one unit, -d on the other. The total matches the PO to
        # the cent; each line deviates ~6%, well past the 2% tolerance.
        first, second = row["inv_lines"]
        delta = round(first["unit_cents"] * 0.06)
        first["unit_cents"] += delta
        second["unit_cents"] -= delta
        row["tags"].append("total_match_breach")
    elif rule == "short_receipt":
        row["short_receipt"] = True
        row["tags"].append("short_receipt")
    elif rule == "similar_number":
        row["near_dup_seed"] = True
        row["tags"].append("near_duplicate")
    elif rule == "symbol_currency":
        row["symbol_only"] = True
        row["tags"].append("symbol_currency")
    else:  # pragma: no cover - guarded by the plan
        raise ValueError(f"unknown rule {rule}")


# --------------------------------------------------------------------------- rendering


def total_cents(lines: list[dict[str, Any]]) -> int:
    return sum(line["quantity"] * line["unit_cents"] for line in lines)


def render_text(row: dict[str, Any]) -> str:
    """The 'already extracted from the PDF' invoice body the agent reads.

    symbol_currency rows have no ``Currency:`` header; the only currency signal is the
    symbol on the total line, and the agent must report the ISO code it implies.
    """
    head = ["INVOICE", "", f"Vendor: {row['vendor']}", f"Invoice No: {row['invoice_number']}"]
    if row["po_number"] is not None:
        head.append(f"PO Number: {row['po_number']}")
    head.append(f"Invoice Date: {row['date']}")
    if not row["symbol_only"]:
        head.append(f"Currency: {row['currency']}")
    head += ["", "Line items"]
    for line in row["inv_lines"]:
        amount = line["quantity"] * line["unit_cents"]
        head.append(
            f"  {line['sku']} | {line['description']} | qty {line['quantity']}"
            f" | unit {line['unit_cents'] / 100:.2f} | {amount / 100:.2f}"
        )
    total = total_cents(row["inv_lines"]) / 100
    if row["symbol_only"]:
        head += ["", f"TOTAL DUE: {CURRENCY_SYMBOL[row['currency']]}{total:,.2f}"]
    else:
        head += ["", f"TOTAL DUE: {total:.2f} {row['currency']}"]
    return "\n".join(head)


def po_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "po_number": row["po_number"],
        "vendor": row["vendor"],
        "currency": row["po_currency"],
        "lines": [
            {
                "sku": line["sku"],
                "description": line["description"],
                "quantity": line["quantity"],
                "unit_price": line["unit_cents"] / 100,
            }
            for line in row["po_lines"]
        ],
    }


def receipt_record(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Received quantities per PO line; short_receipt rows are ~15% short on line 1."""
    record = [
        {"sku": line["sku"], "quantity_received": line["quantity"]} for line in row["po_lines"]
    ]
    if row["short_receipt"]:
        record[0]["quantity_received"] = math.floor(record[0]["quantity_received"] * 0.85)
    return record


# --------------------------------------------------------------------------- oracle


def failing_rules(
    row: dict[str, Any],
    pos: dict[str, Any],
    receipts: dict[str, Any],
    posted: list[dict[str, Any]],
) -> list[str]:
    """Every rule from goal.md that this invoice breaks, recomputed from the fixtures."""
    failures: list[str] = []
    if any(
        entry["invoice_number"] == row["invoice_number"] and entry["vendor"] == row["vendor"]
        for entry in posted
    ):
        failures.append("duplicate_invoice")
    po = pos.get(row["po_number"]) if row["po_number"] else None
    if po is None:
        failures.append("missing_po")
        return failures  # nothing downstream is checkable without a PO
    if row["currency"] != po["currency"]:
        failures.append("currency_mismatch")
    by_sku = {line["sku"]: line for line in po["lines"]}
    for line in row["inv_lines"]:
        po_line = by_sku.get(line["sku"])
        if po_line is None or _dev(line["quantity"], po_line["quantity"]) > 0.02:
            failures.append("tolerance_breach")
            break
        if _dev(line["unit_cents"] / 100, po_line["unit_price"]) > 0.02:
            failures.append("tolerance_breach")
            break
    receipt = receipts.get(row["po_number"])
    if not receipt:
        failures.append("missing_receipt")
    elif "tolerance_breach" not in failures:
        # A receipt must *cover* the invoiced quantities (goal.md), not merely exist.
        # Skipped after a tolerance breach so each row reports exactly one rule.
        received = {entry["sku"]: entry["quantity_received"] for entry in receipt}
        for line in row["inv_lines"]:
            if received.get(line["sku"], 0) < line["quantity"] * 0.98:
                failures.append("missing_receipt")
                break
    return failures


def _dev(actual: float, expected: float) -> float:
    return abs(actual - expected) / expected


def check_deviations(row: dict[str, Any]) -> None:
    """No line may sit near the 2% boundary: clean <= 1.5%, breaches >= 4%."""
    breach = row["rule"] in ("tolerance_breach", "total_match_breach")
    worst = 0.0
    for inv, po in zip(row["inv_lines"], row["po_lines"]):
        worst = max(
            worst, _dev(inv["quantity"], po["quantity"]), _dev(inv["unit_cents"], po["unit_cents"])
        )
    if breach:
        assert worst >= BREACH_DEV_MIN, (row["invoice_number"], worst)
    else:
        assert worst <= CLEAN_DEV_MAX, (row["invoice_number"], worst)


# --------------------------------------------------------------------------- build


def build() -> tuple[list[dict], dict, dict, list[dict]]:
    rng = random.Random(SEED)
    plan = plan_rows(rng)
    rows: list[dict[str, Any]] = []
    for index, slot in enumerate(plan):
        row = base_invoice(index, rng, slot["rule"])
        row["split"], row["rule"], row["index"] = slot["split"], slot["rule"], index
        if slot["rule"] is None and rng.random() < 0.35:
            apply_near_tolerance(row)
        elif slot["rule"] is not None:
            apply_rule(row, slot["rule"], slot["split"], index)
        rows.append(row)

    pos = {r["po_number"]: po_record(r) for r in rows if r["po_exists"]}
    receipts = {r["po_number"]: receipt_record(r) for r in rows if r["receipt_exists"]}
    posted = seed_posted(rows)
    return rows, pos, receipts, posted


def seed_posted(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ledger entries that already exist when a task starts: the duplicates plus noise."""
    entries: list[dict[str, Any]] = []
    noise = [
        ("Pinebrook Supply Co", "INV-90114", "USD", 1240.00),
        ("Kestrel Industrial Ltd", "INV-90271", "GBP", 806.50),
        ("Corvid Electronics GmbH", "INV-90388", "EUR", 2149.75),
    ]
    for vendor, number, currency, total in noise:
        entries.append(
            {"invoice_number": number, "vendor": vendor, "currency": currency, "total": total}
        )
    for row in rows:
        if row["duplicate_seed"]:
            entries.append(
                {
                    "invoice_number": row["invoice_number"],
                    "vendor": row["vendor"],
                    "currency": row["currency"],
                    "total": total_cents(row["inv_lines"]) / 100,
                }
            )
        if row["near_dup_seed"]:
            # Same vendor, visually similar number, different invoice: NOT a duplicate.
            # An agent that fuzzy-matches posted_invoices escalates a payable invoice.
            entries.append(
                {
                    "invoice_number": f"{row['invoice_number']}-2",
                    "vendor": row["vendor"],
                    "currency": row["currency"],
                    "total": round(total_cents(row["inv_lines"]) / 100 * 0.97, 2),
                }
            )
    entries.sort(key=lambda e: (e["vendor"], e["invoice_number"]))
    for position, entry in enumerate(entries, start=1):
        entry["entry_id"] = f"GL-{position:04d}"
    return entries


def to_task(row: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    decision = "approve" if not failures else "escalate"
    if row["rule"] in APPROVE_TRAPS:
        tags = ["trap", row["rule"]]
    elif row["rule"]:
        tags = ["messy", ORACLE_RULE[row["rule"]]]
        if ORACLE_RULE[row["rule"]] != row["rule"]:
            tags.append(row["rule"])
    else:
        tags = ["clean"]
    return {
        "id": f"invoices-{row['index']:02d}",
        "input": {"invoice_text": render_text(row)},
        "expected": {
            "vendor": row["vendor"],
            "invoice_number": row["invoice_number"],
            "po_number": row["po_number"],
            "currency": row["currency"],
            "total": total_cents(row["inv_lines"]) / 100,
            "decision": decision,
            "rule": failures[0] if failures else None,
        },
        "split": row["split"],
        "tags": tags + [t for t in row["tags"] if t not in tags],
    }


def verify(tasks: list[dict], rows: list[dict], pos, receipts, posted) -> None:
    assert len(tasks) == N_TASKS
    counts = {s: sum(1 for t in tasks if t["split"] == s) for s in ("train", "search", "holdout")}
    assert counts == {"train": N_TRAIN, "search": N_SEARCH, "holdout": N_HOLDOUT}, counts
    messy = [t for t in tasks if "messy" in t["tags"]]
    traps = [t for t in tasks if "trap" in t["tags"]]
    assert len(messy) == 18, len(messy)  # 12 v1 rules + 6 escalate traps
    assert len(traps) == 6, len(traps)  # approve traps
    by_rule: dict[str, int] = {}
    for row in rows:
        if row["rule"]:
            by_rule[row["rule"]] = by_rule.get(row["rule"], 0) + 1
    assert by_rule == {
        "duplicate_invoice": 3,
        "currency_mismatch": 3,
        "missing_po": 2,
        "tolerance_breach": 2,
        "missing_receipt": 2,
        "total_match_breach": 4,
        "short_receipt": 2,
        "similar_number": 4,
        "symbol_currency": 2,
    }, by_rule
    for split, rules in MESSY_BY_SPLIT.items():
        got = sorted(r["rule"] for r in rows if r["split"] == split and r["rule"])
        assert got == sorted(rules), (split, got)
    assert len({t["id"] for t in tasks}) == N_TASKS
    for row, task in zip(rows, tasks):
        failures = failing_rules(row, pos, receipts, posted)
        if row["rule"] and row["rule"] not in APPROVE_TRAPS:
            expected = [ORACLE_RULE[row["rule"]]]
        else:
            expected = []
        assert failures == expected, (task["id"], failures, expected)
        assert task["expected"]["decision"] == ("escalate" if expected else "approve")
        check_deviations(row)
    # every approve trap really is payable, and every escalate trap really is not
    for task in traps:
        assert task["expected"]["decision"] == "approve", task["id"]


def main() -> None:
    rows, pos, receipts, posted = build()
    tasks = [to_task(row, failing_rules(row, pos, receipts, posted)) for row in rows]
    verify(tasks, rows, pos, receipts, posted)
    (FIXTURES / "pos.json").write_text(json.dumps(pos, indent=2, sort_keys=True) + "\n")
    (FIXTURES / "receipts.json").write_text(json.dumps(receipts, indent=2, sort_keys=True) + "\n")
    (FIXTURES / "posted.json").write_text(json.dumps(posted, indent=2, sort_keys=True) + "\n")
    (DOMAIN / "tasks.jsonl").write_text(
        "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    )
    print(f"wrote {len(tasks)} tasks, {len(pos)} POs, {len(receipts)} receipts, {len(posted)} GL")


if __name__ == "__main__":
    main()
