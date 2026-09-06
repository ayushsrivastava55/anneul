# How domains/roomdesk was generated

`anneal init` wrote every file in this directory from the interview below. Nothing
here was hand-written; nothing here is hidden.

## The interview

**Name this domain.**

> roomdesk

**Describe the job in one or two sentences.**

> Someone asks for a meeting room in one message. Book the smallest room that fits everyone and is free for the whole slot, using the tools. If nothing fits, book nothing and say so.

**Choose what the agent can use.**

> Python functions in a module

**Name the Python module that holds them.**

> usercode.roomdesk_tools

**Choose what makes a run right.**

> The final state of the system matches

## Tools discovered

- `reset` (`python:usercode.roomdesk_tools.reset`) — Restore the calendar to the start of the day. Called by the evaluator per task.
- `list_rooms` (`python:usercode.roomdesk_tools.list_rooms`) — List every meeting room with how many people it seats and whether it has a screen.
- `check_availability` (`python:usercode.roomdesk_tools.check_availability`) — Say whether one room is free for a whole slot. Hours are 24h, end is exclusive.
- `book_room` (`python:usercode.roomdesk_tools.book_room`) — Book a room for a slot. Fails if the room is too small or the slot is already taken.
- `confirmed` (`python:usercode.roomdesk_tools.confirmed`) — Every booking made so far today. The evaluator reads this, the agent does not need it.

## Examples and splits

- 10 examples given, split seeded with seed 0: holdout 2, search 2, train 6
- success criterion: `state` (evaluator template used for eval.py)
- goal.md source: `elaborated` (`template` = deterministic only; `elaborated` = model-written rules appended)
