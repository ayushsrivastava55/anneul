# STATUS

Phase: 1 (core loop, H2–H8) — six workers running; Phase 0 fully merged (49 tests). H2 checkpoint (traced call through TensorMux visible in Neatlogs) is blocked on keys.

## AO sessions (project `anneal`)
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-10 | 0.1 scaffold | ao/scaffold | merged to main (3 tests) |
| anneal-6 | 0.2 spec-models | ao/spec-models | merged to main (+ mutates follow-up) |
| anneal-7 | 0.3 llm-gateway | ao/llm-gateway | merged to main (9 pass, live skipped: no key) |
| anneal-8 | 0.4 tracing | ao/tracing | merged to main (offline tests; Neatlogs UI check pending key) |
| anneal-9 | 0.5 domain-airline | ao/domain-airline | merged to main (tau-bench, 40 tasks 20/10/10) |

## Phase 1 AO sessions
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-11 | 1.1 runtime-single | ao/runtime-single | working |
| anneal-12 | 1.2 runner | ao/runner | working |
| anneal-13 | 1.3 architect (+ prompts.py) | ao/architect | working |
| anneal-16 | 1.4 diagnose-v1 | ao/diagnose-v1 | working |
| anneal-14 | 1.5 mutate-v1 | ao/mutate-v1 | working |
| anneal-15 | 1.6 gate | ao/gate | working |
| — | 1.7 cli-run | — | held until runtime + runner merge |
| anneal-7 | llm-gateway follow-up: complete() + tests/fakes.py | ao/llm-gateway | merged to main (57 tests) |

Shared interfaces for Phase 1 are pinned in docs/CONTRACTS.md.

## Done
- [x] anneal-1..5 killed (stalled on permission prompts); project switched to bypass-permissions; respawned as anneal-6..10
- [x] orchestrator: git init, kit imported, AO desktop installed, daemon up, project registered
- [x] orchestrator: tau-bench airline env vendored (MIT) into domains/airline/fixtures/tau_airline
- [x] 0.4 anneal-8 - anneal/tracing.py: init_tracing (once, no-op without key), wrap_client, node_span/tool_span/llm_span (sync+async, offline pass-through), run-context tags via contextvar, flush/shutdown, current_trace_id; tests/test_tracing.py (15 tests); smoke `uv run python -m anneal.tracing`

## Blockers
- Clock: machine time was 6:50 PM IST when Phase 1 spawned; docs say the window opens 9:30 PM IST. User to confirm which is right.
- keys: `.env` exists but every value is empty (TensorMux, Neatlogs, Dodo, AIGI). `specs/models.yaml` still REPLACE_ME.
- GitHub remote: `gh repo create` was blocked in the orchestrator session; `origin` is a local bare repo at ~/.ao/data/anneal-origin.git. Swap to GitHub before submission.
- Docker daemon down (only needed for the optional TensorMux OSS gateway).

## Decisions
- Airline: no user simulator; task instruction is the single customer message. eval.py exposes setup(task) to reset DB state. 40 of 50 tasks, split 20/10/10, seed 0.
- Merge order: scaffold first (owns pyproject), then rebase the other four onto it.

## Latest numbers
none
- [x] 0.1 scaffold - uv project (pyproject, hatchling, uv.lock), anneal/ package with __version__, config.env() via python-dotenv, argparse+rich cli stub (`uv run anneal`), ruff (100 cols, py312, fixtures excluded), pytest (3 tests). `uv run ruff check . && uv run pytest` green.
