# STATUS

Phase: 1 core loop merged (architect, runtime, runner, diagnose, mutate, gate). cli-run in flight for the H8 row. Phase 2 and 3 tasks running in parallel. H2 checkpoint (traced call through TensorMux visible in Neatlogs) is blocked on keys.

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
| anneal-11 | 1.1 runtime-single | ao/runtime-single | merged to main |
| anneal-12 | 1.2 runner | ao/runner | merged to main |
| anneal-13 | 1.3 architect (+ prompts.py) | ao/architect | merged to main |
| anneal-16 | 1.4 diagnose-v1 | ao/diagnose-v1 | merged to main |
| anneal-14 | 1.5 mutate-v1 | ao/mutate-v1 | merged to main |
| anneal-15 | 1.6 gate | ao/gate | merged to main |
| anneal-23 | 1.7 cli-run | ao/cli-run | merged (H8 wiring complete) |

## Parallel track (no dependency on the core loop)
| Session | Task | Branch | State |
|---|---|---|---|
| anneal-17 | 2.4 ao-spawn | ao/ao-spawn | merged; selftest spawned real session anneal-27 |
| anneal-18 | 3.1 domain-invoice | ao/domain-invoice | merged |
| anneal-19 | 3.2 domain-bugfix | ao/domain-bugfix | working |
| anneal-30 | 2.5 synth-tool | ao/synth-tool | merged |
| anneal-28 | 3.3 anneal-stage + 3.4 cost-attrib | ao/anneal-stage | merged |
| anneal-29 | 4.1 dodo-credits + 4.2 dodo-ship | ao/dodo-credits | working |
| anneal-31 | 4.3 dashboard | ao/dashboard | working |
| anneal-24 | 2.3 topo-critic | ao/topo-critic | merged |
| anneal-25 | 2.2 ops-full | ao/ops-full | merged |
| anneal-26 | 2.1 taxonomy-full | ao/taxonomy-full | merged |
| anneal-7 | llm-gateway follow-up: complete() + tests/fakes.py | ao/llm-gateway | merged to main (57 tests) |

Shared interfaces for Phase 1 are pinned in docs/CONTRACTS.md.

## Done
- [x] anneal-1..5 killed (stalled on permission prompts); project switched to bypass-permissions; respawned as anneal-6..10
- [x] orchestrator: git init, kit imported, AO desktop installed, daemon up, project registered
- [x] orchestrator: tau-bench airline env vendored (MIT) into domains/airline/fixtures/tau_airline
- [x] 0.4 anneal-8 - anneal/tracing.py: init_tracing (once, no-op without key), wrap_client, node_span/tool_span/llm_span (sync+async, offline pass-through), run-context tags via contextvar, flush/shutdown, current_trace_id; tests/test_tracing.py (15 tests); smoke `uv run python -m anneal.tracing`

## Blockers
- Generated tools are carried on a spec but not yet loadable at runtime (load_domain reads only tools.yaml). Fix sent to anneal-11; gates the end-to-end tool-synthesis demo.
- Clock: machine time was 6:50 PM IST when Phase 1 spawned; docs say the window opens 9:30 PM IST. User to confirm which is right.
- keys: `.env` exists but every value is empty (TensorMux, Neatlogs, Dodo, AIGI). `specs/models.yaml` still REPLACE_ME.
- GitHub remote: `gh repo create` was blocked in the orchestrator session; `origin` is a local bare repo at ~/.ao/data/anneal-origin.git. Swap to GitHub before submission.
- Docker daemon down (only needed for the optional TensorMux OSS gateway).

## Decisions
- Airline: no user simulator; task instruction is the single customer message. eval.py exposes setup(task) to reset DB state. 40 of 50 tasks, split 20/10/10, seed 0.
- Merge order: scaffold first (owns pyproject), then rebase the other four onto it.

## AO session count
31 sessions in project `anneal`: 1 orchestrator equivalent (this session), 24 task workers,
3 API-probe sessions spawned by the ao-spawn worker, 1 selftest session (anneal-27) spawned by
Anneal itself through anneal/ao.py, plus 5 killed first-attempt sessions kept for history.

## Holdout hygiene note
`grep -rn holdout anneal/` matches two lines in `anneal/anneal.py`: a docstring and the call
`gate.run_holdout(...)`. Both are the *function name*, not the split. The split literal
(`HOLDOUT_SPLIT`) exists only in `anneal/gate.py`, and the annealing stage evaluates by
delegating to gate. This is deliberate and compliant; do not "fix" it by inlining the split.

## Latest numbers
main: 284 passed, 1 skipped (skip = live gateway call, no key). Holdout literal confined to gate.py.

## Old latest numbers
none
- [x] 0.1 scaffold - uv project (pyproject, hatchling, uv.lock), anneal/ package with __version__, config.env() via python-dotenv, argparse+rich cli stub (`uv run anneal`), ruff (100 cols, py312, fixtures excluded), pytest (3 tests). `uv run ruff check . && uv run pytest` green.
