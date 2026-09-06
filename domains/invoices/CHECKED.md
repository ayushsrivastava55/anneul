# Hand-checked cases — domains/invoices

Ten rows read end to end against `fixtures/pos.json`, `fixtures/receipts.json` and
`fixtures/posted.json`, with the line arithmetic recomputed by hand. Nine of the twelve messy
rows live in `train`/`search` and all nine are listed here, plus one clean row; the remaining
three messy rows are in `holdout` and are deliberately **not** described here.

Holdout correctness is covered instead by the generator's oracle
(`fixtures/make_tasks.py:failing_rules` + `verify`), which replays every rule in `goal.md`
against the fixtures for all 60 rows and asserts that each messy row breaks *exactly one*
rule, each clean row breaks none, and no line sits near the 2% boundary (clean ≤ 1.5%,
breaches ≥ 4%).

## Reading the `expected` block

`expected` records **what is printed on the invoice**, not what the PO says. On a currency
mismatch the expected currency is the invoice's currency; on a tolerance breach the expected
total is the printed total. Only `expected.decision` reflects the rules in `goal.md`.

## The ten

| # | Task | Split | Rule exercised | What was checked by hand |
|---|---|---|---|---|
| 1 | `invoices-02` | train | *none* — approve path, just inside tolerance | Lines 24 × 90.31 = 2167.44 and 200 × 86.79 = 17358.00 sum to the printed 19525.44. PO-4002 unit price 89.42 vs invoiced 90.31 = **0.99% over**, inside 2%; second line identical. Currency EUR on both, receipts cover both SKUs in full, ledger empty for Aurora Components BV → **approve**. |
| 2 | `invoices-03` | search | `duplicate_invoice` | PO-4003 matches line for line, currency USD on both, receipts complete — the only defect is that GL-0003 already holds INV-10003 for Pinebrook Supply Co. **escalate**. |
| 3 | `invoices-12` | train | `duplicate_invoice` | Three lines total 6723.00 and match PO-4012 exactly; GL-0006 already holds INV-10012 for Vermillion Packaging. **escalate**. |
| 4 | `invoices-16` | train | `duplicate_invoice` | Single line 240 × 51.97 = 12472.80, matches PO-4016; GL-0005 already holds INV-10016 for Tidewater Logistics Inc. **escalate**. |
| 5 | `invoices-19` | search | `currency_mismatch` | Invoice says GBP, PO-4019 is denominated in EUR. Quantity and unit price match exactly, receipt present, no prior posting for Northwind Traders. **escalate**; `expected.currency` stays `GBP` (the printed value). |
| 6 | `invoices-29` | train | `currency_mismatch` | Invoice GBP vs PO-4029 USD. Deliberate near-miss: the ledger *does* hold another Tidewater Logistics Inc invoice (INV-10016), so an agent that escalates on "vendor has prior postings" rather than on a matching invoice number gets the right answer for the wrong reason — the reason must name the currency rule. **escalate**. |
| 7 | `invoices-33` | train | `missing_po` (variant A: no PO printed) | The invoice body has **no `PO Number:` line at all**; `expected.po_number` is `null`. Lines total 23858.30. Nothing to look up, so no downstream rule is even checkable. **escalate**. |
| 8 | `invoices-50` | train | `tolerance_breach` (unit price) | Line 1 invoiced at 83.69 against PO-4050's 78.95 = **6.00% over**, well past 2%. Line 2 identical, currency EUR on both, receipts complete, no prior posting. **escalate**. |
| 9 | `invoices-52` | search | `missing_po` (variant B: PO printed but unknown) | The invoice prints `PO-9052`; `lookup_po("PO-9052")` returns the JSON literal `null` — no such PO exists. `expected.po_number` is the printed `"PO-9052"`, since field extraction scores what is on the page. **escalate**. |
| 10 | `invoices-55` | train | `missing_receipt` | PO-4055 exists and matches all three lines exactly, currency EUR on both, no prior posting — but `receipts.json` has no entry for PO-4055, so `lookup_receipt` returns `[]`. Goods were never received. **escalate**. |

## Notes for whoever reads a failing trace

- The only channel for the duplicate rule is `lookup_po`'s extra `posted_invoices` field —
  the invoice numbers already on the ledger for that PO's vendor. Probing with `post_entry`
  is a hard failure by construction (`eval.is_hard_fail`).
- Case 7 has no PO number, so `lookup_po` cannot be called at all; the correct trace is a
  single `escalate`.
- Cases 2, 3, 4 and 10 are the ones most likely to be wrongly approved: everything the agent
  usually checks lines up, and the defect lives in a field agents tend to skip.
