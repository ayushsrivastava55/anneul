"""The browser form that creates an agent: the interview, with a web front end.

``anneal init`` already asks five questions and writes a runnable domain. Until now that was
the only way in, so the product had no answer to "how do I make one" that did not start with a
terminal. This module is the third :class:`anneal.onboard.Transport` the interview's docstring
anticipated, and it reimplements none of it: the form's fields are read straight off
:data:`anneal.onboard.SCRIPT`, the answers are handed to ``ScriptedTransport`` in the order the
script asks for them, and ``run_interview`` plus ``generate_domain`` do the work exactly as they
do in the terminal.

It is one page rather than five, and that is deliberate. A five-request wizard needs server-side
session state, which is a database this product does not have and a source of half-finished
agents. One form, one POST, one domain, and the interview's own validation reports what is
wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from anneal import onboard

# How many example pairs the form offers. The interview needs three and accepts fifty; six rows
# is enough to start honestly and few enough that the page is not a wall of boxes.
EXAMPLE_ROWS = 6


@dataclass(frozen=True)
class Submission:
    """One filled-in form, before the interview has looked at it."""

    name: str
    job: str
    tools: str
    tools_target: str
    success: str
    examples: list[tuple[str, str]]

    def queue(self) -> list[str]:
        """The answers in the order :data:`onboard.SCRIPT` asks for them.

        ``tools_target`` and ``tools_module`` are follow-ups gated on the tools answer, so
        exactly one of them is asked and the queue carries exactly one value for them.
        """
        answers = [self.name, self.job, self.tools]
        if self.tools in {"mcp", "python"}:
            answers.append(self.tools_target)
        answers.append(self.success)
        for given, expected in self.examples:
            answers += [given, expected]
        # Two blanks, not one. The interview treats a first blank below the minimum as a nudge
        # and asks once more; a single blank would exhaust the queue there and report "scripted
        # answers exhausted" instead of the reason a person can act on, which is that they gave
        # too few examples.
        answers += ["", ""]
        return answers


def parse(form: dict[str, Any]) -> Submission:
    """Read a posted form into a :class:`Submission`, keeping only complete example pairs."""
    examples = []
    for index in range(EXAMPLE_ROWS):
        given = str(form.get(f"given_{index}") or "").strip()
        expected = str(form.get(f"expected_{index}") or "").strip()
        if given and expected:
            examples.append((given, expected))
    return Submission(
        name=str(form.get("name") or "").strip(),
        job=str(form.get("job") or "").strip(),
        tools=str(form.get("tools") or "none").strip(),
        tools_target=str(form.get("tools_target") or "").strip(),
        success=str(form.get("success") or "label").strip(),
        examples=examples,
    )


def create(submission: Submission, domains_dir: Path) -> Path:
    """Run the interview over this submission and write the domain. Raises OnboardError."""
    transport = onboard.ScriptedTransport(submission.queue())
    interview = onboard.run_interview(transport, domains_dir=domains_dir)
    return onboard.generate_domain(interview, domains_dir=domains_dir)


# --- the form ------------------------------------------------------------------------------


def _question(qid: str) -> onboard.Question:
    """The script's own question, so the form's wording cannot drift from the terminal's."""
    return next(q for q in onboard.SCRIPT if q.id == qid)


def _field(qid: str, control: str, label: str = "") -> str:
    """Label above, control, helper text below: the form pattern the whole product uses."""
    question = _question(qid)
    helper = (
        f'<p class="fhelp">{escape(question.help)}</p>' if question.help else ""
    )
    return (
        f'<div class="field"><label class="flabel" for="{escape(qid)}">'
        f"{escape(label or question.prompt)}</label>{control}{helper}</div>"
    )


def _choices(qid: str, checked: str) -> str:
    rows = "".join(
        f'<label class="choice"><input type="radio" name="{escape(qid)}" '
        f'value="{escape(value)}"{" checked" if value == checked else ""}>'
        f"<span>{escape(text)}</span></label>"
        for value, text in _question(qid).choices
    )
    return f'<div class="choices">{rows}</div>'


def _examples_grid() -> str:
    rows = "".join(
        '<div class="exrow">'
        f'<textarea name="given_{i}" rows="2" placeholder=""></textarea>'
        f'<textarea name="expected_{i}" rows="2" placeholder=""></textarea>'
        "</div>"
        for i in range(EXAMPLE_ROWS)
    )
    return (
        '<div class="extable"><div class="exhead"><span>What the agent receives</span>'
        "<span>What a correct answer looks like</span></div>"
        f"{rows}</div>"
    )


def form_body(error: str = "") -> str:
    """The whole form. Every question comes from the interview script, none is written here."""
    banner = (
        f'<p class="formerror" role="alert">{escape(error)}</p>' if error else ""
    )
    return (
        '<div class="pagehead"><h1>New agent</h1>'
        "<p>Five questions. Anneal writes the goal, the tool list and the scorer, then you "
        "can run it.</p></div>"
        f'{banner}<form class="newform" method="post" action="/new">'
        + _field("name", '<input class="finput" id="name" name="name" required>', "Name it")
        + _field(
            "job",
            '<textarea class="finput" id="job" name="job" rows="3" required></textarea>',
            "What should it do?",
        )
        + _field("tools", _choices("tools", "none"), "What can it use?")
        + '<div class="field"><label class="flabel" for="tools_target">'
        "Where are they?</label>"
        '<input class="finput" id="tools_target" name="tools_target" '
        'placeholder="npx -y @modelcontextprotocol/server-filesystem /tmp/box">'
        '<p class="fhelp">An MCP server command or URL, or a Python module path such as '
        "usercode.my_tools. Leave blank if it uses nothing.</p></div>"
        + _field("success", _choices("success", "label"), "What makes an answer right?")
        + '<div class="field"><label class="flabel">Show it some examples</label>'
        f"{_examples_grid()}"
        '<p class="fhelp">Three or more, filled in pairs. These become the tasks it is '
        "scored on, split so that some are held back from every repair step.</p></div>"
        '<button class="fsubmit" type="submit">Create the agent</button>'
        "</form>"
    )


def done_body(path: Path) -> str:
    """What was written, and the one command that runs it."""
    files = "".join(
        f'<li class="mono">{escape(p.name)}</li>' for p in sorted(path.iterdir())
    )
    return (
        '<div class="pagehead"><h1>Your agent is ready</h1>'
        f"<p>Anneal wrote it to <span class=\"mono\">{escape(str(path))}</span>. "
        "Run it, and it will design a few versions, score them, and keep only what survives "
        "the held-back tasks.</p></div>"
        f'<div class="donebox"><ul class="filelist">{files}</ul>'
        f'<pre class="cmd">uv run anneal run {escape(str(path))} \\\n'
        "  --models specs/models.local.yaml</pre>"
        '<a class="fsubmit link" href="/agents">Back to your agents</a></div>'
    )
