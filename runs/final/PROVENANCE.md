# What `runs/final` is

The reported run set. `README.md`'s results block is generated from exactly this directory by
`uv run anneal report runs/final --write-readme`, and
`tests/test_report.py::test_the_repo_readme_block_is_what_the_tool_emits` fails the build if the
two ever drift. It is the only run directory git tracks (`.gitignore`: `runs/*`, `!runs/final/`).

This file exists because the directory was **assembled from two separate runs**, and a table
whose rows came from different machines and different model ladders is misleading unless that is
stated. Nothing here is regenerated in place; if you re-run, replace a whole domain directory and
regenerate the README rather than editing cells.

## Where each domain's rows came from

| Domain | Run | Machine | Model ladder | Inference |
|---|---|---|---|---|
| airline | `runs-pregate-1442` | teammate's | `specs/models.yaml` (Anthropic + TensorMux + AIGI) | hosted, real spend |
| bugfix | `runs-pregate-1442` | teammate's | same | hosted, real spend |
| invoices | `runs-pregate-1442` | teammate's | same | hosted, real spend |
| filesystem | run 6 Sep on the orchestrator's machine | 8 GB laptop | `specs/models.local.yaml` (qwen2.5 3b/1.5b/0.5b) | local Ollama, $0 real spend |

The other committed run directories (`runs-archive/`, `runs-learning*/`, `runs-pregate-1442/`)
are kept deliberately as the full experimental record. They are **not** the reported set and the
README is not generated from them.

## Known limitations of this set — read before quoting a number

1. **The `$/task` column mixes two ladders.** Three domains are priced on hosted models at their
   published rates; `filesystem` is priced on the local ladder, whose sub-7B rates are *scaled*
   from a published 7B rate rather than quoted. Compare `$/task` **within** a domain (iteration 0
   against annealed), never across domains in this table.
2. **These runs predate `summary.json` recording `models_path`.** That field was added on 6 Sep so
   a report can state each row's ladder itself; runs made before it carry `null`, which is why the
   mapping above is written here by hand instead of being derived. Any run made after that commit
   records its own ladder and needs no entry here.
3. **`domain_path` in the hosted summaries points at the teammate's checkout.** It is the absolute
   path of the machine that produced the run and means nothing on any other machine. Nothing reads
   it; the report resolves everything relative to the runs directory it is given.
4. **`invoices` has no gated iteration rows.** Its iteration 0 and final rows are em-dashes for
   accuracy, pass^3 and p because no gate ran at that iteration. Only the annealed row carries
   scores. An em-dash in this table always means "this value does not exist in the runs", never a
   zero and never a rounded-away number.
5. **Absolute scores are modest and are not the claim.** The defensible results here are the cost
   and latency reductions from the anneal stage, and the gate's refusal to promote regressions.
   Accuracy moved in both directions across these domains; the rejected-mutations table under the
   results block lists every refusal and the condition that caused it.
