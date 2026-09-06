# Hand-checked cases — domains/invoices (dataset v2, 2026-09-06)

Ten rows read end to end against `fixtures/pos.json`, `fixtures/receipts.json` and
`fixtures/posted.json`, with the line arithmetic recomputed by hand. All named rows live in
`train`/`search`; the gated split's rows are deliberately **not** described here.

Gated-split correctness is covered instead by the generator's oracle
(`fixtures/make_tasks.py:failing_rules` + `verify`), which replays every rule in `goal.md`
against the fixtures for all 60 rows and asserts that each escalate row breaks *exactly one*
rule, each approve row (including the approve traps) breaks none, and no line sits near the
2% boundary (clean ≤ 1.5%, breaches ≥ 4%).

## What changed in v2

The v1 dataset saturated — every candidate scored 1.000 at iteration 0 — so v2 adds four
trap classes aimed at the shortcuts a naive agent takes. 24 of 60 rows are non-trivial
(12 v1 rules, 6 escalate traps, 6 approve traps), 12/6/6 across the splits.

## Reading the `expected` block

`expected` records **what is printed on the invoice**, not what the PO says. On a currency
mismatch the expected currency is the invoice's currency; on a tolerance breach the expected
total is the printed total. On a symbol-currency row the expected currency is the ISO code
the printed symbol implies. Only `expected.decision` reflects the rules in `goal.md`.

## The ten

| # | Task | Split | Case | What was checked by hand |
|---|---|---|---|---|
| 1 | `invoices-02` | train | *none* — approve, just inside tolerance | Unit price ~1% over PO (inside 2%), receipts cover both SKUs, `INV-10057-2` on the vendor's ledger is a *different* invoice number. **approve**. |
| 2 | `invoices-03` | search | `duplicate_invoice` | PO matches line for line, currency and receipts fine — but the ledger already holds `INV-10003` for Pinebrook Supply Co. **escalate**. |
| 3 | `invoices-16` | train | `missing_po` (variant A: no PO printed) | The invoice body has **no `PO Number:` line**; `expected.po_number` is `null`. Nothing to look up. **escalate**. |
| 4 | `invoices-18` | search | `currency_mismatch` | Invoice currency differs from the PO's; everything else matches. **escalate**; `expected.currency` stays the printed value. |
| 5 | `invoices-30` | train | `tolerance_breach` via **total_match_breach** | Lines invoiced at 33.74 and 29.92 against PO unit 31.83 on both (**±6%**), quantities equal, so the printed total 2546.40 equals the PO total *to the cent* (2 × 40 × 31.83). An agent that only reconciles totals posts it — a hard fail. **escalate**. |
| 6 | `invoices-33` | train | `missing_receipt` via **short_receipt** | PO matches, currency matches — but the receipt shows `quantity_received: 20` against an invoiced qty 24 on SKU-1331 (≈83%). A receipt *exists*; it does not *cover*. **escalate**. |
| 7 | `invoices-37` | search | `missing_po` (variant B: PO printed but unknown) | The invoice prints a `PO-9xxx` number; `lookup_po` returns the JSON literal `null`. `expected.po_number` is the printed value. **escalate**. |
| 8 | `invoices-47` | train | **similar_number** (approve trap) | All three lines match PO-4047 exactly, receipts complete — and the ledger holds `INV-10047-2` for the same vendor. That is a *different* invoice number; the duplicate rule is exact-match on (number, vendor). **approve**. |
| 9 | `invoices-51` | search | `missing_receipt` via **short_receipt** | Single line qty 25; receipt shows 21 received (84%). **escalate**. |
| 10 | `invoices-55` | train | **symbol_currency** (approve trap) | No `Currency:` header; the total line reads `TOTAL DUE: $1,045.92`. The `$` implies USD, the PO is USD, line matches exactly, receipt complete. `expected.currency` is `USD`. **approve**. |

## Notes for whoever reads a failing trace

- The only channel for the duplicate rule is `lookup_po`'s extra `posted_invoices` field —
  the invoice numbers already on the ledger for that PO's vendor. Probing with `post_entry`
  is a hard failure by construction (`eval.is_hard_fail`).
- The traps are symmetric on purpose: `total_match_breach` and `short_receipt` punish
  under-checking (wrong approve → hard fail), `similar_number` and `symbol_currency` punish
  over-caution (wrong escalate → accuracy loss). A candidate can only pass both kinds by
  actually applying the rules as written.
- Duplicate rows can share a vendor, so a `posted_invoices` view may list several numbers,
  including `-2` suffixed near-duplicates from other rows. Only an exact match of the
  invoice's own number counts.
