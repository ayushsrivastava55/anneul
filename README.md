# Anneal

> The agent that engineers agents, then makes them cheap.

**Track 1 — Automated Agent Engineering · Syndicate by Maximor, Sep 5–6 2026**

Give Anneal a goal, a tool manifest and an evaluator. It proposes agent architectures, runs
them, reads its own traces to classify failures, applies typed fixes to prompts, tools, memory
and orchestration, promotes only changes that survive a held-out test three times over, then
walks each node down to the cheapest model that still passes. Same code, three unrelated
domains, four numbers.

## Results

<!-- results:start -->
| Domain | Stage | Holdout acc | pass^3 | Gen gap | Hard fails | $/task | p95 s | p (gate) |
|---|---|---|---|---|---|---|---|---|
| invoices | iteration 0 | 0.189 | 0.000 | 0.211 | 3 | 0.000169 | 148.3 | 1.000 |
| invoices | final | 0.178 | 0.000 | 0.222 | 1 | 0.000169 | 148.3 | 1.000 |

**Rejected mutations** — the gate refusing to promote, and which condition failed.

| Domain | Iteration | Operator | Gate condition that failed |
|---|---|---|---|
| invoices | 0 | rewrite_tool_desc | p 1.000 >= alpha 0.1 |
| invoices | 1 | add_fewshots | hard_fails 2 > incumbent 1 |

`—` means the value does not exist in the runs (no gate ran at that iteration, or the
spec was never scored on that split) — it is never a zero and never a rounded-away number.
Holdout accuracy, pass^3, hard fails and p come from the gate's `gate.json`; `$/task` and p95
are measured on the search split. Gen gap is the search mean minus the gated mean, recomputed
from those two recorded means when the gate stored it only for the candidate.

Inference is **local** (Ollama, qwen2.5 3b / 1.5b / 0.5b), so these runs cost $0 in real money.
Tokens and latency are measured. USD is those measured tokens priced at the reference rates in
`specs/models.yaml`, where each tier carries a `price_source` (`published` or `scaled`); the
sub-7B rates are scaled from a published 7B rate, not quoted. Do not read `$/task` as the cost
of a hosted provider.
<!-- results:end -->

Every cell above is emitted by `uv run anneal report runs/final --write-readme`, which reads
`runs/<domain>/<iteration>/summary.json` and `gate.json`. None of them are typed by hand; a
domain appears here only once it has actually run.

## How it works

```
goal.md + tools.yaml + eval.py
   → Architect (3–4 harness specs: single | planner_executor | critic_loop | tool_router)
   → Run (parallel, every call traced to Neatlogs, every call through TensorMux)
   → Diagnose (traces → failure taxonomy → persistent issue ledger)
   → Mutate (one typed operator per iteration; code-level ones run as AO worker sessions)
   → Gate (holdout ×3, pass^3, paired binomial test; promote or reject)
   → Anneal (per-node model downshift on a score / $ / latency Pareto)
   → Ship (Dodo-metered agent)
```

Details: `docs/ARCHITECTURE.md`. Failure classes and operators: `specs/failure_taxonomy.yaml`.

## Run it

```
uv sync
cp .env.example .env   # keys: TensorMux, Neatlogs, Dodo (test), AI Grants India, optional frontier
uv run anneal run domains/invoices --iterations 4 --budget 2.00
uv run anneal anneal runs/latest
uv run anneal dashboard   # http://localhost:8000
```

## Domains

| Domain | Task | Evaluator | Hard fail |
|---|---|---|---|
| `domains/invoices` | AP invoice triage: approve or escalate | field exact match + decision | auto-approving a mismatched invoice |
| `domains/airline` | tau-bench airline customer ops | final DB state | refund outside policy |
| `domains/bugfix` | fix a failing Python function | pytest | writing outside the sandbox |

No domain-specific code exists in `anneal/`; the core sees only the three input files.

## How we used AO

AO is not a wrapper we bolted on at the end. It is both how Anneal was built and how Anneal
executes code-level mutations at runtime.

**As the build system.** Every task in `docs/TASKS.md` ran as its own AO worker session on its
own branch and git worktree, driven by one orchestrator session. The orchestrator wrote each
worker's prompt from the task's Deliverable and Accept cells, ran the Accept check itself
against the branch, and merged only when `uv run ruff check . && uv run pytest` was green.
Nothing was merged on a worker's say-so. Shared interfaces were pinned in `docs/CONTRACTS.md`
before the parallel phases so that six to seven workers could build disjoint modules at once
without stepping on each other. Commit messages carry the session name, so `git log` reads as
the session history.

**As a runtime executor.** `anneal/ao.py` drives the same daemon from inside the product. When
Diagnose classifies a failure as `missing_capability`, the `synthesize_tool` operator asks for a
tool specification, spawns an AO worker on `ao/tool-<name>`, and waits. The branch is accepted
only when `pytest` passes on the generated tool's own test file, checked in a throwaway
worktree. A rejected branch leaves the candidate spec untouched and marks the ledger issue
attempted. This path is verified end to end: `uv run python -m anneal.ao --selftest` spawned a
real session that wrote a tool, committed it, and passed the gate.

Two things we learned about AO and worked around, both documented in `anneal/ao.py`: `ao spawn`
has no `--json` flag despite what the architecture notes assumed, so the REST body was recovered
by probing; and a project needs a remote with a resolved default branch before it will create
worktrees.

## Sponsor usage

- **AO**: build orchestration for every task, and the executor for code-level mutations via the daemon API.
- **Neatlogs**: spans for every node/tool/LLM call; MCP trace reads drive Diagnose; prompt registry versions every mutation (`staging` → `production`); detections flag token spikes.
- **TensorMux**: single OpenAI-compatible endpoint; Anneal's downshift is a config change; per-request backend and latency give the cost/speed metrics.
- **AI Grants India**: the cheap tier behind TensorMux.
- **Dodo Payments**: credit ledger debited per run (budget-aware optimisation, `balance_low` halts); shipped agents get a usage meter.
- **Maximor**: the invoices domain, with escalation on low confidence and an audit trail of every change the optimiser made.

## Team

_Names here._
