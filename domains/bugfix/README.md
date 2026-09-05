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
