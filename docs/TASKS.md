# TASKS — ordered, one AO worker session each

Name ≤ 20 chars (AO rule). Branch `ao/<name>`. Each task lists its acceptance check so the
orchestrator can verify before merging. Order matters inside a phase; phases match PLAN.md.

## Phase 0 · Foundations (H0–H2)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 0.1 | `scaffold` | `uv` project, `anneal/` package, `ruff`, `pytest`, `.env` loading, `runs/` gitignore | `uv run pytest` passes (1 smoke test) |
| 0.2 | `spec-models` | `anneal/spec.py` pydantic models; loads `specs/harness.example.yaml` | tests: parse example, reject bad topology |
| 0.3 | `llm-gateway` | `anneal/llm.py` client via TensorMux; `specs/models.yaml` tiers+prices; cost calc | traced hello-world call returns, cost computed |
| 0.4 | `tracing` | `anneal/tracing.py` neatlogs init + span decorators + tags | span visible in Neatlogs UI |
| 0.5 | `domain-airline` | `domains/airline/{goal.md,tools.yaml,eval.py,tasks.jsonl}` with splits | `eval.py` scores a known-good and known-bad output |

## Phase 1 · Core loop (H2–H8)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 1.1 | `runtime-single` | `runtime.py` single + planner_executor, tool dispatch, step budget | runs 3 airline tasks end to end |
| 1.2 | `runner` | parallel run over split → jsonl with per-node tokens/backend/latency/trace id | jsonl schema test |
| 1.3 | `architect` | 3 candidate specs from goal+tools | specs validate; ≥2 distinct topologies |
| 1.4 | `diagnose-v1` | MCP client, 2 failure classes, ledger upsert | ledger has entries after a run with failures |
| 1.5 | `mutate-v1` | `rewrite_tool_desc`, `add_fewshots`; prompt version to Neatlogs | mutated spec validates; prompt v2 exists |
| 1.6 | `gate` | holdout 3×, pass^3, binomial test, promote/reject, label flip | unit tests on gate math with synthetic results |
| 1.7 | `cli-run` | `anneal run <dir> --iterations N --budget X` wiring | README row for airline iter 0 vs iter 3 |

## Phase 2 · Full operators + AO (H8–H14)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 2.1 | `taxonomy-full` | all classes in `specs/failure_taxonomy.yaml` + deterministic pre-checks | classifier tests |
| 2.2 | `ops-full` | remaining operators incl. `add_escalation_node`, `switch_topology` | each operator has a unit test |
| 2.3 | `topo-critic` | `critic_loop`, `tool_router` topologies | runs on airline |
| 2.4 | `ao-spawn` | `anneal/ao.py`: spawn worker, poll, acceptance by pytest on branch | spawns a trivial tool and imports it |
| 2.5 | `synth-tool` | `synthesize_tool` operator using `ao.py` | promoted candidate uses a generated tool |
| 2.6 | `detections` | token-spike detection wired into diagnose | ledger entry from a detection |

## Phase 3 · Domains + Anneal (H14–H20)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 3.1 | `domain-invoice` | dataset gen (60, 20% messy), `eval.py` with hard-fail rule | 10 hand-checked; hard-fail fires on auto-approve mismatch |
| 3.2 | `domain-bugfix` | 40 buggy functions + tests; `eval.py` runs pytest | scores baseline candidate |
| 3.3 | `anneal-stage` | `anneal.py` per-node downshift, Pareto output | Pareto json with ≥3 points |
| 3.4 | `cost-attrib` | backend header → per-node cost | $/task per node in summary |

## Phase 4 · Billing + dashboard (H20–H24)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 4.1 | `dodo-credits` | entitlement, debit per run, balance poll → halt | loop halts at budget |
| 4.2 | `dodo-ship` | product + usage meter for shipped agent; checkout link | link printed |
| 4.3 | `dashboard` | FastAPI+HTMX page: curves, ledger, Pareto, balance | opens with real runs |
| 4.4 | `readme-final` | README from template with tables from `runs/final/` | no hand-typed numbers |

## Phase 5 · Ship (H24–H30)
| # | Session | Deliverable | Accept |
|---|---|---|---|
| 5.1 | `repro-run` | clean-clone run, 3 domains, `runs/final/` | tables regenerate |
| 5.2 | `video` | 3-min video per DEMO_SCRIPT.md | uploaded, link in README |
| 5.3 | `submit` | Devpost form complete | confirmation email |
