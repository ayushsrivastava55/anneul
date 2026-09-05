# bugfix — Python bug repair

40 small functions (15–40 lines) each with a single injected bug and a pytest file that fails
on the bug and passes on the fix. Generate with an LLM from a list of classic bug types
(off-by-one, wrong comparison, mutable default, missing edge case, wrong return type, swapped
args, bad regex), then verify each pair by running the tests against both versions. Split 20/10/10.

## Tools (start deliberately incomplete)
`read_file(path)`, `write_file(path, content)` only. Do NOT provide `run_tests` at the start.
The intended demo moment is Anneal classifying failures as `missing_capability` and asking an
AO worker to synthesise `run_tests` under `generated_tools/`, after which pass^3 jumps.

## eval.py
`score` = 1.0 if `pytest tests/test_<id>.py` passes in a sandboxed copy, else 0.0.
`is_hard_fail` = any `write_file` outside the task's sandbox directory. `THRESHOLD = 1.0`.

## Fallback
If dataset generation runs over 90 minutes, switch to text-to-SQL on a Spider subset
(evaluator = result-set equality against SQLite). Keep the same split sizes.

## As built (task 3.2)

40 cases, `bugfix_01` .. `bugfix_40`, seven bug types: off-by-one (6), wrong comparison (6),
mutable default (5), missing edge case (6), wrong return type (5), swapped args (6), bad regex
(6). Every pair was verified by actually running pytest twice — it must fail on the buggy
module and pass on the fixed one — via
`uv run python -m domains.bugfix.fixtures.make_tasks --verify` (40/40).

```
fixtures/sources/<id>/          authored once: the CORRECT module, its pytest file, bug.json
fixtures/cases/<id>/            generated + committed: what the agent sees (buggy module,
                                tests/, empty conftest.py so the module is importable)
fixtures/make_tasks.py          injects each bug, deals the split, writes ../tasks.jsonl
```

`bug.json` records the single defect as an `old` -> `new` string replacement that must match
exactly once, so every case is a one-line change from a known-good module and the diff is
auditable. The fixed source never ships inside a task directory; `eval.fixed_source(task)`
reads it from `sources/` for the tests only.

Split: 20 train / 10 search / 10 holdout, `Random(0)`, stratified — cases are grouped by bug
type, shuffled within the group and dealt train, train, search, holdout, so all seven types
appear in all three splits rather than a whole type being stranded in the holdout.

Scoring runs `pytest tests/test_<id>.py` in a subprocess (30 s timeout) inside a throwaway copy
of the task sandbox, never in the repo tree, and restores the pristine `tests/` and
`conftest.py` over that copy first — rewriting the test file cannot buy a pass.
