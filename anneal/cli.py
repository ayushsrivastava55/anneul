"""`anneal` command line entry point. Subcommands are stubs until their modules land."""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from anneal import __version__

COMMANDS: dict[str, str] = {
    "run": "generate, run, diagnose, mutate and gate candidates for a domain",
    "gate": "re-run the holdout gate on the incumbent",
    "anneal": "downshift node models along the cost/latency Pareto front",
    "dashboard": "serve the runs/ dashboard on http://localhost:8000",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anneal", description="Anneal agent harnesses.")
    parser.add_argument("--version", action="version", version=f"anneal {__version__}")
    sub = parser.add_subparsers(dest="command")
    for name, help_text in COMMANDS.items():
        sub.add_parser(name, help=help_text)
    return parser


def print_usage(console: Console) -> None:
    console.print(f"[bold]anneal[/bold] {__version__}")
    console.print(r"usage: anneal <command> \[options]", end="\n\n")
    for name, help_text in COMMANDS.items():
        console.print(f"  [cyan]{name:<10}[/cyan] {help_text}")


def main(argv: list[str] | None = None) -> int:
    console = Console()
    args = build_parser().parse_args(argv)
    if args.command is None:
        print_usage(console)
        return 0
    console.print(f"[yellow]anneal {args.command}[/yellow]: not implemented yet")
    return 2


if __name__ == "__main__":
    sys.exit(main())
