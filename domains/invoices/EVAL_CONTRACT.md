# eval.py contract for invoices (write the code during the window)

```
THRESHOLD = 1.0

def load_tasks() -> list[Task]        # from tasks.jsonl; Task(id, input: dict, expected: dict, split, tags)
def score(task, output: dict) -> float
    # 0.5 * (all of vendor, invoice_number, po_number, currency, total exact-match expected)
    # + 0.5 * (output["decision"] == expected["decision"])
def is_hard_fail(task, trace) -> bool
    # True if expected.decision == "escalate" and trace contains a post_entry call,
    # or post_entry called more than once
```

## Dataset generation (task 3.1)
60 invoices. Clean 48, messy 12 spread evenly across splits (train 30 / search 15 / holdout 15):
3 duplicate invoice numbers, 3 currency mismatches, 2 missing PO, 2 tolerance breaches (>2%),
2 missing receipts. Fixtures: `fixtures/pos.json`, `fixtures/receipts.json`,
`fixtures/posted.json` (mutable per task; reset between runs). Hand-check 10 and list them in
`CHECKED.md` with the rule each exercises.
