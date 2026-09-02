"""`vocabguard` command line entry point. argparse only; the package has no CLI framework dependency."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from pydantic_ai.exceptions import UserError

from . import _baseline, _check, _contrast, _report, _rewrite, _scrape
from ._common import Command, CommandRunner

__all__ = ('main',)

COMMANDS: tuple[Command, ...] = (
    _scrape.COMMAND,
    _baseline.COMMAND,
    _rewrite.COMMAND,
    _contrast.COMMAND,
    _check.COMMAND,
    _report.COMMAND,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='vocabguard', description='Catch vocabulary drift in agent-written prose.')
    subparsers = parser.add_subparsers(dest='command', required=True, metavar='COMMAND')
    for command in COMMANDS:
        subparser = subparsers.add_parser(command.name, help=command.help, description=command.help)
        command.configure(subparser)
        subparser.set_defaults(run=command.run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run: CommandRunner = args.run
    try:
        return run(args)
    except UserError as error:
        print(f'vocabguard: {error}', file=sys.stderr)
        return 2
