# Goal: AP invoice triage

You receive one supplier invoice at a time (text fields already extracted from the PDF, may
contain errors). Decide whether it can be posted to the ledger or must be escalated to a human.

## What done means
Return JSON: `{"vendor": str, "invoice_number": str, "po_number": str|null, "currency": str,
"total": number, "decision": "approve"|"escalate", "reason": str}`.

## Rules
- Approve only if a matching PO exists, currency matches the PO, and each line's quantity and
  unit price are within 2% of the PO. A goods receipt must exist for the PO.
- Escalate on: no PO, duplicate invoice number for the same vendor, currency mismatch,
  tolerance breach, missing receipt, or any field you cannot read with confidence.
- Never call `post_entry` for an invoice you decided to escalate. Never post twice.
- When escalating, the reason must name the specific rule that failed.

## Tools
`lookup_po`, `lookup_receipt`, `post_entry`, `escalate` (see tools.yaml). Posting or
escalating is the last action.
