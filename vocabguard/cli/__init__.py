"""`vocabguard` command line entry point. argparse only; the package has no CLI framework dependency."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic_ai.exceptions import UserError

from . import _baseline, _check, _contrast, _evaluate, _report, _rewrite, _score, _scrape
from ._common import Command, CommandRunner

__all__ = ('main',)

COMMANDS: tuple[Command, ...] = (
    _score.COMMAND,
    _check.COMMAND,
    _scrape.COMMAND,
    _baseline.COMMAND,
    _rewrite.COMMAND,
    _contrast.COMMAND,
    _evaluate.COMMAND,
    _report.COMMAND,
)


def _implies_score(argv: list[str]) -> bool:
    """Bare text, or piped stdin with no arguments, means `score`."""
    if not argv:
        return not sys.stdin.isatty()
    first = argv[0]
    return not first.startswith('-') and first not in {command.name for command in COMMANDS}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='vocabguard', description='Catch vocabulary drift in agent-written prose.')
    subparsers = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')
    for command in COMMANDS:
        subparser = subparsers.add_parser(command.name, help=command.help, description=command.help)
        command.configure(subparser)
        subparser.set_defaults(run=command.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _implies_score(arguments):
        arguments.insert(0, 'score')
    args = build_parser().parse_args(arguments)
    run: CommandRunner = args.run
    try:
        return run(args)
    except UserError as error:
        print(f'vocabguard: {error}', file=sys.stderr)
        return 2
