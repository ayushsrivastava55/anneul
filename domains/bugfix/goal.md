# Goal: repair one bug in a small Python module

Each task hands you a tiny Python project in a working directory. Exactly **one** function in
the named module has a single injected bug: an off-by-one, a wrong comparison, a mutable
default argument, a missing edge case, a wrong return type, swapped arguments, or a bad
regex. A pytest file under `tests/` states the intended behaviour precisely. Your job is to
make that test file pass by fixing the module.

## Definition of done
- The named module is repaired so that `tests/test_<task id>.py` passes.
- The repair is a real fix to the logic, not a rewrite of the behaviour around it: every other
  function in the module keeps working exactly as before.
- Nothing under `tests/` is changed. The tests are the specification; editing them is
  cheating and scores zero.
- You finish with a one-line summary naming the bug you fixed.

## Working rules
- All paths you pass to tools are relative to the task directory, e.g. `chunk_list.py` or
  `tests/test_bugfix_01.py`. Never use an absolute path and never use `..`: writing outside
  the task directory is a hard failure that ends the run.
- Read the module and the test file before you write anything. The test file tells you the
  exact expected values; do not guess them.
- Write the **whole** file back with `write_file` — it replaces the file's contents, it does
  not patch it. Keep every line you did not intend to change.
- Change as little as possible. These bugs are one-line defects; a large rewrite usually
  breaks a case the test also checks.
- Do not add new files, new dependencies, or imports that are not already used.
