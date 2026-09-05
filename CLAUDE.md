# Anneal — project rules for Claude Code / AO agents

Anneal is a system that, given only `goal.md + tools.yaml + eval.py`, generates an agent
architecture, runs it, diagnoses failures from traces, applies typed fixes, promotes only
changes that survive a holdout gate, then "anneals" the winner down to the cheapest model mix
that still passes. Built in 30 hours for Syndicate by Maximor, Track 1 (Automated Agent
Engineering). Read `docs/PRD.md` for what we're building and `docs/PLAN.md` for when.

## Non-negotiables (read before touching anything)

1. **Holdout is sacred.** Nothing in `anneal/architect.py`, `anneal/diagnose.py`, or
   `anneal/mutate.py` may read tasks tagged `split: holdout`. Only `anneal/gate.py` may.
   If you need more signal, use the `search` split. Violating this invalidates our results.
2. **Every LLM call goes through the gateway.** Use `anneal/llm.py` (`get_client()`), which
   points the OpenAI SDK at `$TENSORMUX_BASE_URL`. Never instantiate `openai.OpenAI()` directly.
   Never hardcode a model name outside `specs/models.yaml`.
3. **Every node, tool call and LLM call is a Neatlogs span.** Use the decorators in
   `anneal/tracing.py`. Tag spans with `candidate_id`, `iteration`, `domain`, `split`.
4. **Domain code never leaks into the core.** `anneal/` must not import from `domains/`.
   The core only ever sees the three input files. If a fix only works for one domain, it's wrong.
5. **Report, don't claim.** Numbers in README/dashboard come from `runs/*.jsonl`, never typed
   by hand. A regression that Gate rejected is reported, not hidden.
6. **AO is how we build.** Every unit of work is an AO session on its own branch. Commit
   messages reference the session name. Do not work outside AO; judges review session history.
7. **No code before the window opens** (Sat 5 Sep 2026, 9:30 PM IST). This kit is docs only.

## Stack

- Python 3.12, `uv` for env and deps (`uv sync`, `uv run ...`).
- `openai` SDK (OpenAI-compatible endpoint = TensorMux), `pydantic` for specs, `pyyaml`,
  `neatlogs` SDK, `dodopayments` SDK, `httpx`, `rich` for CLI, `scipy` for the paired test.
- Dashboard: single-file FastAPI + HTMX page reading `runs/`. No React, no build step.
- Tests: `pytest`. Unit tests for spec parsing, taxonomy classifier, gate math, operators.

## Layout (target; create as you go)

```
anneal/
  cli.py          # `anneal run <domain_dir> --budget 5.00 --iterations 6`
  spec.py         # pydantic models for harness spec (see specs/harness.example.yaml)
  runtime.py      # generic executor for any harness spec (topologies: single, planner_executor, critic_loop, tool_router)
  llm.py          # get_client(), model tiers, cost table (specs/models.yaml)
  tracing.py      # neatlogs init + span decorators
  architect.py    # goal+tools -> N candidate specs
  runner.py       # run candidate over a split, parallel, write runs/*.jsonl
  diagnose.py     # neatlogs MCP client -> failure taxonomy -> ledger.json
  mutate.py       # operators keyed by failure class (see specs/failure_taxonomy.yaml)
  gate.py         # holdout, pass^3, paired test, promote/reject
  anneal.py       # per-node model downshift, Pareto front
  billing.py      # dodo credit ledger, usage events, balance_low halt
  ao.py           # spawn AO worker sessions for code-level mutations (tool synthesis)
  dashboard.py
domains/<name>/   # goal.md, tools.yaml, eval.py, tasks.jsonl (with split tags)
specs/            # harness.example.yaml, failure_taxonomy.yaml, models.yaml
runs/             # jsonl per run (gitignored except final/)
ledger.json       # persistent issue ledger
```

## Conventions

- Branch per AO session: `ao/<session-name>`. Small PRs, merged by the orchestrator.
- Type everything; `ruff` clean; functions under 60 lines; no clever metaprogramming.
- Logs are structured (`json`), one line per event; humans read the dashboard, not logs.
- Costs: use `specs/models.yaml` prices; attribute per node using the `x-tensormux-backend`
  response header when present, else the requested model.
- Secrets only via `.env` (see `.env.example`). Never commit keys. Never print keys.
- When unsure between "more features" and "cleaner numbers", choose cleaner numbers.

## How to run (once implemented)

```
uv sync
cp .env.example .env            # fill keys
uv run anneal run domains/airline --iterations 4 --budget 3.00
uv run anneal gate runs/latest   # re-run holdout on the incumbent
uv run anneal anneal runs/latest # cost/latency downshift
uv run anneal dashboard          # http://localhost:8000
```

## Definition of done for the hackathon

- Three domains run through the identical loop with zero domain-specific code in `anneal/`.
- README table: iteration 0 vs final vs annealed, for accuracy, pass^3, $/task, p95 latency.
- Demo video shows AO sessions, a failing Neatlogs trace, the ledger, a promoted mutation,
  the Pareto chart, and the Dodo balance moving.
- Devpost submitted before Sun 6 Sep 6:00 PM EDT (Mon 7 Sep 3:30 AM IST).
