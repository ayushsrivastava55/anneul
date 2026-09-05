# DEMO_SCRIPT — 3 minutes, one take, screen + voice

Record at 1080p. Pre-open every window. No live LLM calls on camera except the one run at 0:20
(pre-warm it). Speak the bold lines.

| Time | Screen | Say |
|---|---|---|
| 0:00 | AO kanban: orchestrator + worker sessions incl. terminated ones | **"Everything you're about to see was built and run through AO. Twenty-something sessions, one per task, each on its own branch."** |
| 0:20 | Terminal: `domains/invoices/` three files; `uv run anneal run domains/invoices --iterations 4 --budget 2` | **"Anneal only ever sees three files: a goal, a tool list, an evaluator. It never sees the holdout tasks."** |
| 0:40 | AO board: candidate sessions appear as cards; dashboard iteration 0 row | **"It proposes three architectures and scores them. Iteration zero: 61% accuracy, 48% pass-three, and it auto-approved a mismatched invoice. That's a hard fail."** |
| 0:55 | Neatlogs: the failing trace, span tree, the `post_entry` call on the currency mismatch | **"Here's the trace. Anneal reads this through Neatlogs' MCP server and classifies it: unsafe action."** |
| 1:10 | `ledger.json` entry; dashboard ledger table | **"Every failure class has a typed fix. Unsafe action maps to an escalation node."** |
| 1:20 | Neatlogs prompt registry: v1 → v2 diff, label `staging` | **"The prompt change is versioned. Then the gate runs the holdout three times."** |
| 1:35 | `gate.json`: pass^3 up, hard fails 0, p-value; label flips to `production` | **"Promoted. And here's one it rejected: the few-shot fix looked better on the search split and worse on holdout. The gate caught it."** |
| 1:50 | Dashboard: domains B and C curves, same loop | **"Same code, no domain logic, on airline customer ops and Python bug repair. In bug repair it noticed the agent couldn't see test output, so it asked an AO worker to build a run_tests tool."** |
| 2:05 | AO session `tool-run-tests` and the generated file + passing test | (let it breathe 3 s) |
| 2:20 | Pareto chart; TensorMux `/metrics`; per-node model tiers | **"Then Anneal anneals: it walks each node down to a cheaper model behind TensorMux while the holdout score holds. Cost per task fell 70%, p95 latency fell 40%."** |
| 2:40 | Dodo dashboard: credit balance moving; checkout link for the shipped agent | **"Every run debits a Dodo credit ledger, so the optimiser has a budget, and the agent it ships is billable from day one."** |
| 2:50 | README before/after table, repo URL | **"Three domains, four metrics, all traceable. Anneal: the agent that engineers agents."** |

Replace the numbers with real ones from `runs/final/`. If a number is worse, say so and show the
gate rejecting it; judges score reliability, not perfection.
