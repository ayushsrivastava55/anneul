# How domains/triage was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**What should we call this? (a short name for the folder)**

> triage

**In one or two sentences, what should the agent do?**

> You receive a raw bug report written by a user. Decide how severe it is, which component it belongs to, and whether a reproduction case is still needed.

**What can the agent use to do it?**

> Nothing yet — it answers from the task text alone

**How do we know a run was right?**

> Specific fields in the answer match

## Tools discovered

- none

## Examples and splits

- 9 examples given, split seeded with seed 0: holdout 2, search 2, train 5
- success criterion: `fields` (evaluator template used for eval.py)
- goal.md source: `elaborated` (`template` = deterministic only; `elaborated` = model-written rules appended)
