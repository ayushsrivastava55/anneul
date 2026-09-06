# Your own tools

Anneal looks in this folder, and only this folder, for Python functions an agent can call. It
is listed in the New agent page's tool picker, so a file you drop in here shows up there with
no further step.

Nothing in `anneal/` imports from here, and nothing here is part of Anneal. The two modules
present belong to two of the example agents (`roomdesk` and `ticketdesk`); delete them if you
do not want them.

## Writing one

One function per thing the agent should be able to do. The first line of each docstring is the
description the agent reads when deciding whether to call it, so write it for the agent:

```python
def check_availability(room: str, start_hour: int, end_hour: int) -> str:
    """Say whether one room is free for a whole slot. Hours are 24h, end is exclusive."""
```

The argument names and type annotations become the tool's schema. Functions whose names start
with an underscore are skipped, as is any file whose name starts with one.

Anneal marks a function as a write action when its name contains a word like `book`, `send`,
`delete` or `create`, which is what makes the agent leave it until last. Name accordingly.
