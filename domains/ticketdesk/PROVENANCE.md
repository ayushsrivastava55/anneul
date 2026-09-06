# How domains/ticketdesk was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**Name this domain.**

> ticketdesk

**Describe the job in one or two sentences.**

> A helpdesk ticket arrives as one sentence. Decide which team should handle it and answer with just the team name.

**Choose what the agent can use.**

> Python functions in a module

**Name the Python module that holds them.**

> usercode.ticketdesk_tools

**Choose what makes a run right.**

> A decision or label matches

## Tools discovered

- `list_teams` (`python:usercode.ticketdesk_tools.list_teams`) — List every team you can route a ticket to, and what each one handles.
- `lookup_team` (`python:usercode.ticketdesk_tools.lookup_team`) — Find which team handles a single keyword. Returns the team name, or 'unknown'.

## Examples and splits

- 12 examples given, split seeded with seed 0: holdout 3, search 3, train 6
- success criterion: `label` (evaluator template used for eval.py)
- goal.md source: `elaborated` (`template` = deterministic only; `elaborated` = model-written rules appended)
