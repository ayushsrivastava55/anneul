# Describe your contribution

We built Anneal: a system that takes a job description in plain words and produces an agent
that does it, then keeps improving that agent only when it can prove the improvement.

You answer five questions in a browser. Anneal writes the goal file, the tool list and the
scorer, so nothing is hand-written. It then wires your goal and tools into several different
agents, runs all of them on your examples, groups the failures by what actually went wrong,
applies the repair that fault calls for, and re-tests on tasks it was never allowed to see
while it was repairing. If the change does not win there, the change is thrown away. Whatever
survives is then walked down to the cheapest model mix that still holds the score.

The part we would defend hardest is what it refuses. Across four agents the gate rejected six
changes. One of them scored higher than the version it was replacing, 0.800 against 0.767, and
we threw it away because it made three serious mistakes where the old one made one. A system
that only reports its wins is not measuring anything, so every refusal is listed in the
repository next to the wins.

Measured results, all read out of the committed runs by the reporting tool rather than typed
in by hand, with a test that fails the build if the two ever disagree:

- Bug fixing: $0.0899 to $0.026 per task, slowest run 37.9s to 13.5s
- Invoice triage: $0.035 to $0.0111 per task
- Filesystem assistant: serious mistakes 7 to 3

To show it works on something it had never seen, we built two more agents through the same
five questions, on a laptop with no API key and nothing written by hand. On helpdesk routing
the same goal and tools wired three ways scored 0.000, 1.000 and 0.333: choosing the
architecture is what moved accuracy. The second agent scored zero throughout, because the small
local model could not do the task, and the gate refused both repairs rather than claiming an
improvement it could not demonstrate. We kept that run in the repository too.

It runs with no account anywhere. Three local models through Ollama are the default ladder, and
the same code points at hosted providers by changing one file. Tools come from any Model
Context Protocol server over stdio or streamable HTTP, or from Python functions you write; the
interview asks what the agent should be able to do rather than for a launch command, and
assembles the command itself.

Built with Agent Orchestrator: 29 sessions, each on its own branch, each taking one piece of
the system. 214 commits, 641 tests, about 13,400 lines of Python. One of those sessions was not
started by us. Anneal spawned it itself, to write a tool one of its agents was missing.

## What we did not do

Accuracy moved in both directions. On two of the four benchmark agents it fell, and those rows
are in the results table rather than removed from it. The reserved evaluation splits are small,
so the statistical test has little power and we report that per run instead of quoting the
p-values as though they settled something. Neatlogs tracing is wired and covered by tests but
was switched off for every run, so there is no populated trace project to look at.
