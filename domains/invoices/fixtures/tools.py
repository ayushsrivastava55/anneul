"""Module-level AP store plus the `python:` tool impls referenced by tools.yaml.

State lives in `_STORE`, loaded once at import from `pos.json`, `receipts.json` and
`posted.json`. `eval.setup(task)` calls `reset()` before every run so each task starts from
the same pristine ledger. The JSON files on disk are read-only seed data -- `post_entry` and
`escalate` mutate only the in-memory copy.

Every wrapper returns a string (airline-style), so the runtime never sees an exception:
JSON on success, `"null"` where tools.yaml promises null, `"Error: ..."` on refusal.

Note on `lookup_po`: alongside the PO fields it returns `posted_invoices`, the invoice
numbers already posted to the ledger *for that PO's vendor*. That is the only channel an
agent has for the "duplicate invoice number for the same vendor" rule -- probing with
`post_entry` is a hard failure by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _load() -> dict[str, Any]:
    return {
        "pos": json.loads((_HERE / "pos.json").read_text()),
        "receipts": json.loads((_HERE / "receipts.json").read_text()),
        "posted": json.loads((_HERE / "posted.json").read_text()),
        "escalated": [],
    }


_STORE: dict[str, Any] = _load()


def reset() -> None:
    """Reload the pristine ledger from the JSON fixtures. Called by eval.setup()."""
    _STORE.clear()
    _STORE.update(_load())


def posted_entries() -> list[dict[str, Any]]:
    """Live ledger contents (read-only view for evaluators and tests)."""
    return list(_STORE["posted"])


def escalated_entries() -> list[dict[str, Any]]:
    """Invoices handed to a human during this run."""
    return list(_STORE["escalated"])


def lookup_po(po_number: str) -> str:
    """Fetch a purchase order, or the JSON literal `null` when no such PO exists."""
    po = _STORE["pos"].get(po_number)
    if po is None:
        return json.dumps(None)
    record = dict(po)
    record["posted_invoices"] = [
        entry["invoice_number"] for entry in _STORE["posted"] if entry["vendor"] == po["vendor"]
    ]
    return json.dumps(record)


def lookup_receipt(po_number: str) -> str:
    """Goods receipts for a PO. An empty list means nothing was ever received."""
    return json.dumps(_STORE["receipts"].get(po_number, []))


def post_entry(invoice_number: str, vendor: str, total: float, currency: str) -> str:
    """Post an approved invoice to the ledger. Refuses an invoice already posted."""
    for entry in _STORE["posted"]:
        if entry["invoice_number"] == invoice_number and entry["vendor"] == vendor:
            return (
                f"Error: invoice {invoice_number} is already posted for {vendor}"
                f" (entry {entry['entry_id']})"
            )
    entry_id = f"GL-{len(_STORE['posted']) + 1:04d}"
    _STORE["posted"].append(
        {
            "entry_id": entry_id,
            "invoice_number": invoice_number,
            "vendor": vendor,
            "currency": currency,
            "total": total,
        }
    )
    return json.dumps({"entry_id": entry_id, "status": "posted"})


def escalate(invoice_number: str, reason: str) -> str:
    """Hand the invoice to a human reviewer with the rule that failed."""
    _STORE["escalated"].append({"invoice_number": invoice_number, "reason": reason})
    return json.dumps({"status": "escalated", "invoice_number": invoice_number})
