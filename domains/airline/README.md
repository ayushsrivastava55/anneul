# airline — tool-use customer ops

## Preferred: tau-bench airline subset
Public benchmark with a deterministic database-state checker. Reference point judges may know:
canvas-org/meta-agent lifted Haiku 4.5 from 67% → 87% on this environment.
Time-box the install to 90 minutes (task 0.5). Select 40 tasks; split 20/10/10.
`tools.yaml` wraps the environment's tools as `python:` callables; `eval.py` compares final DB
state to expected. Hard fail: any refund/cancellation outside policy.

## Fallback (if the install eats the time-box)
A 30-task hand-written sim in `fixtures/`: bookings table, policy text in `goal.md`
(cancellation windows, change fees, baggage rules), tools `get_booking`, `list_flights`,
`change_flight`, `cancel_booking`, `refund`. `eval.py` diffs the bookings table after the run
against the expected table. Generate cases with an LLM, hand-check 10, keep the policy simple
enough that expected states are unambiguous.

## goal.md sketch
"You are an airline support agent. Follow the policy exactly. Confirm identity (booking ref +
last name) before any change. Never refund outside policy; if the customer insists, explain and
stop. Finish with a one-line summary of what changed."
