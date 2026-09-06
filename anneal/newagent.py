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

ROOT = Path(__file__).resolve().parent.parent

# --- what the agent can use, offered as things rather than as commands ----------------------

# Asking a non-developer to "give the server's launch command" is asking them to already know
# the answer. These four are the servers published under the Model Context Protocol's own npm
# scope, checked against the registry rather than remembered, and each is named for what it
# lets an agent do. "Something else" is still there for anyone who does have a command.
CATALOGUE: tuple[tuple[str, str, str, str], ...] = (
    (
        "files",
        "Work with files in a folder",
        "Read, write and list files under one folder you choose. Nothing outside it.",
        "npx -y @modelcontextprotocol/server-filesystem {path}",
    ),
    (
        "memory",
        "Remember things between tasks",
        "A place to store facts it learns and look them up again later.",
        "npx -y @modelcontextprotocol/server-memory",
    ),
    (
        "thinking",
        "Think a problem through in steps",
        "Lets it break a hard task into steps and revise them as it goes.",
        "npx -y @modelcontextprotocol/server-sequential-thinking",
    ),
    (
        "everything",
        "A sampler, for trying Anneal out",
        "The protocol's own demonstration server. Useful for a first run, not for real work.",
        "npx -y @modelcontextprotocol/server-everything",
    ),
)

# Where the {path} in the files server is filled from when the person does not say.
DEFAULT_FILES_PATH = "/tmp/anneal-box"


def catalogue_command(choice: str, path: str = "") -> str:
    """The launch command for a catalogue entry, or "" when the choice is not one."""
    for key, _, _, command in CATALOGUE:
        if key == choice:
            return command.replace("{path}", (path or DEFAULT_FILES_PATH).strip())
    return ""


# The one folder a person's own tool modules live in. Everything else in this repository is
# ours: the benchmark domains under domains/ carry fixtures that exist to make our own test
# agents work, and offering those to somebody building their own agent is offering them our
# internals. This picker looks here and nowhere else.
USERCODE_DIR = "usercode"


def python_modules(root: Path | None = None) -> list[tuple[str, str]]:
    """``(dotted path, what is in it)`` for each tool module in ``usercode/``.

    The list is scanned, never written down, so a file dropped into that folder appears here
    with no further step. It used to also sweep ``*/*/fixtures/tools.py``, which meant a person
    creating an agent was shown ``domains.airline.fixtures.tools`` and three others like it:
    the plumbing of our benchmark domains, which is neither theirs nor usable by them.
    """
    root = root or ROOT
    found: list[tuple[str, str]] = []
    for path in sorted((root / USERCODE_DIR).glob("*.py")):
        if path.name.startswith("_"):
            continue
        dotted = f"{USERCODE_DIR}.{path.stem}"
        names = _public_functions(path)
        if names:
            found.append((dotted, ", ".join(names[:4]) + (" and more" if len(names) > 4 else "")))
    return found


def _public_functions(path: Path) -> list[str]:
    """Function names a module defines, read from its source without importing it."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    return [
        node.name for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    ]



@dataclass(frozen=True)
class Submission:
    """One filled-in form, before the interview has looked at it."""

    name: str
    job: str
    tools: str
    tools_target: str
    success: str
    examples: list[tuple[str, str]]

    def prepare(self) -> None:
        """Make the world match the answer before the interview looks at it.

        Someone who picks "work with files in a folder" has said the agent should have one.
        The folder not existing yet is not a decision they made; it is a step they should not
        have to take, and without it the server refuses to start and the agent is created with
        no tools at all.
        """
        if self.tools != "mcp" or not self.tools_target.startswith("npx "):
            return
        folder = self.tools_target.rsplit(" ", 1)[-1]
        if folder.startswith(("/", ".")):
            Path(folder).mkdir(parents=True, exist_ok=True)

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
    tools = str(form.get("tools") or "none").strip()
    return Submission(
        name=str(form.get("name") or "").strip(),
        job=str(form.get("job") or "").strip(),
        tools=tools,
        tools_target=_target(tools, form),
        success=str(form.get("success") or "label").strip(),
        examples=examples,
    )


def _target(tools: str, form: dict[str, Any]) -> str:
    """Where the tools are, from whichever picker the choice revealed.

    A catalogue pick becomes its launch command here rather than in the browser, so the page
    never has to carry a command around and a person never has to see one.
    """
    if tools == "mcp":
        pick = str(form.get("mcp_choice") or "").strip()
        if pick and pick != "other":
            return catalogue_command(pick, str(form.get("files_path") or ""))
        return str(form.get("mcp_other") or "").strip()
    if tools == "python":
        pick = str(form.get("python_choice") or "").strip()
        if pick and pick != "other":
            return pick
        return str(form.get("python_other") or "").strip()
    return ""


def create(submission: Submission, domains_dir: Path) -> tuple[Path, list[str]]:
    """Run the interview over this submission and write the domain. Raises OnboardError.

    Returns the path and whatever the interview wants said out loud, so the page can report a
    server that would not start rather than announcing an agent with nothing to call.
    """
    submission.prepare()
    transport = onboard.ScriptedTransport(submission.queue())
    interview = onboard.run_interview(transport, domains_dir=domains_dir)
    path = onboard.generate_domain(interview, domains_dir=domains_dir)
    return path, list(interview.notes)


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




def _mcp_picker() -> str:
    """The catalogue, as things the agent could do. Revealed only if MCP was chosen."""
    rows = "".join(
        f'<label class="choice pick"><input type="radio" name="mcp_choice" '
        f'value="{escape(key)}"{" checked" if index == 0 else ""}>'
        f'<span><b>{escape(title)}</b><em>{escape(blurb)}</em></span></label>'
        for index, (key, title, blurb, _) in enumerate(CATALOGUE)
    )
    return (
        '<div class="field reveal" data-when="mcp">'
        '<label class="flabel">What should it be able to do?</label>'
        f'<div class="choices">{rows}'
        '<label class="choice pick"><input type="radio" name="mcp_choice" value="other">'
        "<span><b>Something else</b><em>You have a server command or a URL of your own"
        ".</em></span></label></div>"
        '<div class="sub-field reveal" data-pick="mcp_choice:files">'
        '<label class="flabel" for="files_path">Which folder may it use?</label>'
        f'<input class="finput" id="files_path" name="files_path" '
        f'value="{escape(DEFAULT_FILES_PATH)}">'
        '<p class="fhelp">It can read and write here and nowhere else.</p></div>'
        '<div class="sub-field reveal" data-pick="mcp_choice:other">'
        '<label class="flabel" for="mcp_other">Your own server</label>'
        '<input class="finput" id="mcp_other" name="mcp_other">'
        '<p class="fhelp">A command that starts the server, or its web address.</p>'
        "</div></div>"
    )


def _python_picker() -> str:
    """The tool modules in ``usercode/``, as a list. Revealed only if Python was chosen."""
    modules = python_modules()
    if not modules:
        return (
            '<div class="field reveal" data-when="python">'
            '<label class="flabel">Which functions?</label>'
            '<p class="fhelp">There are no tool modules yet. Put a Python file in the '
            f'<span class="mono">{USERCODE_DIR}/</span> folder of this project, with one '
            "function per thing the agent should be able to do, and it will be listed here. "
            "Each function's first docstring line becomes the description the agent reads."
            "</p>"
            '<div class="sub-field"><label class="flabel" for="python_other">'
            "Or name one yourself</label>"
            '<input class="finput" id="python_other" name="python_other" '
            'placeholder="usercode.my_tools">'
            '<input type="hidden" name="python_choice" value="other">'
            "</div></div>"
        )
    rows = "".join(
        f'<label class="choice pick"><input type="radio" name="python_choice" '
        f'value="{escape(dotted)}"{" checked" if index == 0 else ""}>'
        f'<span><b class="mono">{escape(dotted)}</b><em>{escape(contains)}</em></span></label>'
        for index, (dotted, contains) in enumerate(modules)
    )
    return (
        '<div class="field reveal" data-when="python">'
        '<label class="flabel">Which functions?</label>'
        f'<div class="choices">{rows}'
        '<label class="choice pick"><input type="radio" name="python_choice" value="other"'
        f'{"" if modules else " checked"}>'
        "<span><b>Another module</b><em>Something importable from this project"
        ".</em></span></label></div>"
        '<div class="sub-field reveal" data-pick="python_choice:other">'
        '<label class="flabel" for="python_other">Its import path</label>'
        '<input class="finput" id="python_other" name="python_other" '
        'placeholder="usercode.my_tools">'
        '<p class="fhelp">Anything importable from this project.</p></div></div>'
    )


# Progressive disclosure, in the smallest amount of script that does it: the two pickers are in
# the page already and only one is ever shown, so nothing has to be fetched when the choice
# changes and the form still submits without any script at all.
REVEAL_SCRIPT = """
// Everything is in the page already; only one branch is ever shown. Nothing is fetched when
// the choice changes, and with no script at all the form still submits, because the server
// reads the answer belonging to whichever tools choice was made.
const shown = (el) => {
  const outer = el.dataset.when;
  if (outer) {
    const picked = document.querySelector('input[name=tools]:checked');
    return outer === (picked ? picked.value : 'none');
  }
  const [group, value] = (el.dataset.pick || '').split(':');
  const inner = document.querySelector('input[name=' + group + ']:checked');
  return Boolean(inner) && inner.value === value && shown(inner.closest('.reveal[data-when]'));
};
const sync = () => document.querySelectorAll('.reveal').forEach(el => {
  el.hidden = !shown(el);
});
document.querySelectorAll('input[type=radio]').forEach(el =>
  el.addEventListener('change', sync));
sync();
"""


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
        + _mcp_picker() + _python_picker()
        + _field("success", _choices("success", "label"), "What makes an answer right?")
        + '<div class="field"><label class="flabel">Show it some examples</label>'
        f"{_examples_grid()}"
        '<p class="fhelp">Three or more, filled in pairs. These become the tasks it is '
        "scored on, split so that some are held back from every repair step.</p></div>"
        '<button class="fsubmit" type="submit">Create the agent</button>'
        "</form>"
        f"<script>{REVEAL_SCRIPT}</script>"
    )


def done_body(path: Path, notes: list[str] | None = None) -> str:
    """What was written, anything that did not go to plan, and how to run it.

    The interview records a server that would not start or a module that would not import and
    then carries on with no tools. Those notes were dropped here, so an agent could be created
    with nothing to call and the page would say only that it was ready.
    """
    files = "".join(
        f'<li class="mono">{escape(p.name)}</li>' for p in sorted(path.iterdir())
    )
    said = "".join(f'<p class="formnote">{escape(note)}</p>' for note in (notes or []))
    return (
        said
        + '<div class="pagehead"><h1>Your agent is ready</h1>'
        f"<p>Anneal wrote it to <span class=\"mono\">{escape(str(path))}</span>. "
        "Run it, and it will design a few versions, score them, and keep only what survives "
        "the held-back tasks.</p></div>"
        f'<div class="donebox"><ul class="filelist">{files}</ul>'
        f'<pre class="cmd">uv run anneal run {escape(str(path))} \\\n'
        "  --models specs/models.local.yaml</pre>"
        '<a class="fsubmit quiet" href="/agents">Back to your agents</a></div>'
    )
