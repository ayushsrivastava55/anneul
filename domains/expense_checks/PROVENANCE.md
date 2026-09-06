# How domains/expense_checks was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**Name this domain.**

> expense checks

**Describe the job in one or two sentences.**

> An expense claim arrives as one line. Say whether it can be paid or needs a manager.

**Choose what the agent can use.**

> Nothing yet — the agent works from the task text alone

**Choose what makes a run right.**

> A decision or label matches

## Tools discovered

- none

## Examples and splits

- 6 examples given, split seeded with seed 0: holdout 2, search 2, train 2
- success criterion: `label` (evaluator template used for eval.py)
- goal.md source: `template` (`template` = deterministic only; `elaborated` = model-written rules appended)

> Only 6 examples: enough to run, too few for the reserved evaluation split to mean anything. Add more before quoting a number (see PROVENANCE.md).
