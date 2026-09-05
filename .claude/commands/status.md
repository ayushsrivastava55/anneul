Produce a status report for the orchestrator.

1. Read STATUS.md, docs/PLAN.md and `git log --oneline -30`.
2. Compute the current phase from the checkpoints in PLAN.md and the time now (IST). Say whether we are ahead, on time, or behind, in minutes.
3. List open tasks from docs/TASKS.md in the current phase, and any blocked ones with the blocker.
4. If `runs/` exists, print the latest per-domain summary: iteration, holdout accuracy, pass^3, hard fails, $/task, p95 latency. Never type numbers by hand; read them from summary.json files.
5. If behind by more than 60 minutes, propose the next item from the PLAN.md cut list and stop.
6. Update STATUS.md with this report under a timestamped heading.
