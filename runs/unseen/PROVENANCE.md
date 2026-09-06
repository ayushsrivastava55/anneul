# The unseen-domain test

Track 1 asks for a system that designs, tests and improves agents **for tasks it has never seen
before**. `runs/final` cannot show that: those four domains were written by hand while the loop
was being built, so the loop's authors had seen them. This directory is the answer to that
objection.

Both agents here were created on 7 Sep by answering `anneal init`'s five questions as a user
would, on a laptop with no API key of any kind. Nothing in `anneal/` was changed for either of
them; nobody wrote a `goal.md`, a `tools.yaml` or an `eval.py`. The tools are two small Python
modules under `usercode/`, which is what a person bringing their own code would have.

## ticketdesk — helpdesk routing

*A helpdesk ticket arrives as one sentence. Decide which team should handle it.* Two tools
(`list_teams`, `lookup_team`), twelve examples, label scoring.

| Design | How it is wired | How often it is right | $/task | Slowest run |
|---|---|---|---|---|
| Design 1 | one agent | 0.000 | $0.00010 | 7.2s |
| **Design 2** | **planner and doer** | **1.000** | $0.00047 | 34.4s |
| Design 3 | doer with a reviewer | 0.333 | $0.00018 | 19.0s |

The architecture search is what moved accuracy: the same goal, tools and scorer, wired three
ways, scored 0.000, 1.000 and 0.333. The loop then stopped with `no_candidate` because the
winner had no failing tasks left to diagnose, which is the correct thing to do and not a
crash.

The downshift stage then tried three cheaper model mixes on the winner. All three lost
accuracy (1.000 → 0.556 and 0.111), so it kept the original. A cost reduction was available
and was refused because it was not free.

## roomdesk — meeting-room booking

*Book the smallest room that fits everyone and is free for the whole slot.* Five tools, ten
examples, end-state scoring.

Every design scored 0.000. The 3B local model answered with a clarifying question instead of
calling a tool. Two repairs were proposed, applied and gated; both were rejected at p = 1.000
with **zero discordant tasks**, and the gate said so rather than reporting "the change did not
help" about a sample that could not have shown anything.

This one is kept deliberately. It is what the system does when the model underneath it is not
capable of the task: it still runs the whole loop, and it declines to claim an improvement.

## Read this before quoting a number

1. **Three tasks in the reserved split.** Twelve examples split three ways is enough to run and
   far too few for the paired test to have power. `anneal init` says so on screen and in each
   domain's own PROVENANCE.md.
2. **`$/task` is local tokens at the local ladder's rates**, which are scaled from a published
   7B rate. The real spend was $0.
3. **roomdesk's tool module is not thread-safe.** It keeps the calendar in module globals, so
   the first run of it under the default concurrency of 6 raced itself. Both runs here used
   `--concurrency 1`. That is a bug in the tools a user brought, not in the loop, and it is
   recorded here because the first numbers it produced were meaningless.
