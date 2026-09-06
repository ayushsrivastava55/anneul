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
| airline | iteration 0 | 0.800 | 0.800 | -0.100 | 0 | 0.363 | 421.7 | 1.000 |
| airline | final | 0.767 | 0.700 | -0.067 | 1 | 0.363 | 421.7 | 1.000 |
| bugfix | iteration 0 | 0.333 | 0.100 | 0.067 | 0 | 0.0899 | 37.9 | 1.000 |
| bugfix | final | 0.133 | 0.100 | 0.267 | 0 | 0.0899 | 37.9 | 1.000 |
| bugfix | annealed | 0.233 | 0.100 | — | 0 | 0.026 | 13.5 | — |
| filesystem | iteration 0 | 0.111 | 0.000 | 0.222 | 7 | 0.00027 | 31.6 | 0.750 |
| filesystem | final | 0.222 | 0.000 | 0.111 | 3 | 0.00027 | 31.6 | 0.875 |
| invoices | iteration 0 | — | — | — | — | 0.035 | 8.9 | — |
| invoices | final | — | — | — | — | 0.035 | 8.9 | — |
| invoices | annealed | 1.000 | 1.000 | — | 0 | 0.0111 | 6.4 | — |

**Rejected mutations** — the gate refusing to promote, and which condition failed.

| Domain | Iteration | Operator | Gate condition that failed |
|---|---|---|---|
| airline | 0 | add_escalation_node | hard_fails 1 > incumbent 0 |
| airline | 1 | add_validator_node | hard_fails 3 > incumbent 1 |
| bugfix | 0 | add_step_budget_and_critic | p 1.000 >= alpha 0.1 |
| bugfix | 1 | switch_topology | p 1.000 >= alpha 0.1 |
| filesystem | 0 | switch_topology | p 0.750 >= alpha 0.1 |
| filesystem | 1 | add_escalation_node | hard_fails 4 > incumbent 3 |

`—` means the value does not exist in the runs (no gate ran at that iteration, or the
spec was never scored on that split) — it is never a zero and never a rounded-away number.
Holdout accuracy, pass^3, hard fails and p come from the gate's `gate.json`; `$/task` and p95
are measured on the search split. Gen gap is the search mean minus the gated mean, recomputed
from those two recorded means when the gate stored it only for the candidate.

Tokens and latency are measured. USD is those tokens priced at the rates the run's ladder declares; that ladder is listed below, never assumed. `price_source` is what each tier recorded: `published` is the provider's list price, `scaled` is derived from a published rate for a different model size, and `unrecorded` means the tier carries no provenance. A local provider costs nothing in money; the column is still what those tokens would cost at the listed rate, so rows stay comparable across ladders.

Ladder `specs/models.yaml`:

| Tier | Provider | Model | $/1M in | $/1M out | price_source |
|---|---|---|---|---|---|
| frontier | frontier | `claude-opus-5` | 5 | 25 | published |
| mid | frontier | `claude-sonnet-5` | 3 | 15 | published |
| cheap | frontier | `claude-haiku-4-5-20251001` | 1 | 5 | published |
| flash | tensormux | `glm-4-7-flash` | 0.06 | 0.4 | published |
| nano | aigi | `gpt-5-nano` | 0.05 | 0.4 | published |
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

Two ladders ship with the repo and `--models` chooses between them. `specs/models.local.yaml`
is three local Ollama models and needs no key; `specs/models.yaml` is the hosted ladder and
needs the provider keys in `.env`. Everything below uses the local one, so it runs on a clean
clone with no account anywhere.

```
uv sync
cp .env.example .env            # every key may stay empty on the local ladder

brew install ollama && ollama serve &
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:0.5b-instruct

uv run anneal run domains/invoices --models specs/models.local.yaml --iterations 2 --budget 2.00
uv run anneal report runs/final           # the results table below, straight from the runs
uv run anneal dashboard --runs-dir runs/final  # http://localhost:8000
```

Drop `--models` to use the hosted ladder instead; then `FRONTIER_API_KEY` and its siblings in
`.env` must be set, or the run stops on the first call with an unset-key error.

On a machine with 8 GB of RAM, hold one model in memory at a time or the run will be
OOM-killed: `OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_NUM_PARALLEL=1 ollama serve`. Point the tiers at
any OpenAI-compatible provider by editing `specs/models.yaml` and the matching `*_BASE_URL` /
`*_API_KEY` variables; nothing else changes.

## Domains

| Domain | Task | Evaluator | Hard fail |
|---|---|---|---|
| `domains/invoices` | AP invoice triage: approve or escalate | field exact match + decision | auto-approving a mismatched invoice |
| `domains/airline` | tau-bench airline customer ops | final DB state hash | an action outside the expected set |
| `domains/bugfix` | fix a failing Python function | pytest in a sandbox | writing outside the sandbox |
| `domains/filesystem` | file tasks via a **third-party MCP server** | final directory state | touching a path outside the task dir |

No domain-specific code exists in `anneal/`; the core sees only the three input files. All four
load through the same `anneal.domain.load_domain`, and a regression test fails the build if the
core ever repeats a heading from any domain's `goal.md`.

`domains/bugfix` ships a deliberately incomplete tool manifest: it has `read_file` and
`write_file` but **no test runner**. That is the setup for the tool-synthesis path, where the
system classifies the failure as `missing_capability` and commissions an AO worker to write the
missing tool. Expect a low score there until that operator has run.

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

This is not hypothetical: in a live bugfix run the loop diagnosed `missing_capability` (the
domain deliberately withholds `run_tests`), spawned session `tool-run_pytest-9b51`, and the
worker wrote `run_pytest.py` with its test, which the gate then evaluated on the reserved
split like any other mutation. A separate worker session (`readme-results`) built the README
results splicer used below — the optimiser and its own build pipeline share the same executor.

Two things we learned about AO and worked around, both documented in `anneal/ao.py`: `ao spawn`
has no `--json` flag despite what the architecture notes assumed, so the REST body was recovered
by probing; and a project needs a remote with a resolved default branch before it will create
worktrees.

## Sponsor usage

Split honestly into what actually ran and what is wired but unexercised, because we will
not claim otherwise.

**Exercised end to end**

- **AO** — every task built as its own worker session on its own branch, and `anneal/ao.py`
  spawns AO workers *at runtime* to write missing tools, accepted only when their tests pass.
  Verified: `uv run python -m anneal.ao --selftest`. On the bugfix domain the optimiser
  itself spawned a worker that wrote `run_pytest` plus its acceptance test.
- **Neatlogs** — every node, tool and LLM call is a live span (workflow `anneal`); Diagnose
  reads traces; the prompt registry versions every operator edit and the gate flips
  `staging` → `production` on promotion. Two SDK issues found and worked around: an
  unbounded serializer recursion (bounded replacement patched in `anneal/tracing.py`) and a
  401 on `save_as_version` (`create_prompt` versions instead).
- **TensorMux** — the `flash` tier, glm-4-7-flash, reached by the downshift and used as a
  learning-run baseline. Its 60 requests/min cap is survived with exponential backoff.
- **AI Grants India** — the `nano` tier (gpt-5-nano): the cheapest rung of the ladder, and
  on invoices it holds a perfect 1.000 at ~29× below the mid tier's cost.
- **Model Context Protocol** — `anneal/mcp.py` speaks real MCP over stdio and streamable HTTP.
  `domains/filesystem` is served by the official `@modelcontextprotocol/server-filesystem`
  through `npx`; the agent discovers that server's 14 tools from its own `tools/list` rather
  than from anything we hand-wrote.
- **Maximor's problem space** — `domains/invoices` is AP triage with an escalation hard-fail
  rule, plus the issue ledger and versioned prompts as an audit trail of every change the
  optimiser made to itself.

**Built and tested, but not exercised against a live account**

- **Dodo Payments** — `anneal/billing.py` implements credit entitlement, deterministic
  per-run event ids, batched ingestion, a budget guard that halts the loop, and a shipped-agent
  product with a usage meter. Without `DODO_API_KEY` it runs as a local ledger, which still
  enforces `--budget`. No live Dodo call was made (no key was ever obtained).
- **Local inference (Ollama)** — a fully supported provider (`ollama` in `specs/models.yaml`);
  the fork lineage ran the whole loop on qwen2.5 3b/1.5b/0.5b at $0 real cost. The shipped
  config uses the live five-tier ladder because every provider in it was exercised.
- **smallest.ai** — voice credits received but unused: no voice domain, by scope decision.

## Team

Ayush Srivastava · Tanush Singhal
