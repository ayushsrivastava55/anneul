# How domains/refund_checks was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**Name this domain.**

> refund checks

**Describe the job in one or two sentences.**

> Decide whether a refund request can be approved without a manager.

**Choose what the agent can use.**

> Nothing yet. It works from the task text alone

**Choose what makes a run right.**

> A decision or label matches

## Tools discovered

- none

## Examples and splits

- 3 examples given, split seeded with seed 0: holdout 1, search 1, train 1
- success criterion: `label` (evaluator template used for eval.py)
- goal.md source: `template` (`template` = deterministic only; `elaborated` = model-written rules appended)

> Only 3 examples: enough to run, too few for the reserved evaluation split to mean anything. Add more before quoting a number (see PROVENANCE.md).
