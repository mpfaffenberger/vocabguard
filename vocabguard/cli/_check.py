from __future__ import annotations

import argparse
import os
from pathlib import Path

from .. import _git
from ..scoring import score_file
from ._common import Command, add_watchlist_option, iter_prose_files, load_watchlist


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('files', nargs='+', type=Path, help='Files or directories to check.')
    add_watchlist_option(parser)
    parser.add_argument('--diff-base', default=None, metavar='REF', help='Score only lines added since this ref.')
    parser.add_argument('--threshold', type=float, default=0.0, help='Fail when the score is above this.')
    parser.add_argument('--min-tokens', type=int, default=20, help='Skip scoring below this many tokens.')


def run(args: argparse.Namespace) -> int:
    files: list[Path] = args.files
    diff_base: str | None = args.diff_base
    threshold: float = args.threshold
    min_tokens: int = args.min_tokens
    watchlist = load_watchlist(args.watchlist)
    failed = False
    for path in iter_prose_files(files):
        if not path.is_file():
            continue
        previous = _git.read_file(diff_base, f'./{os.path.relpath(path)}') if diff_base else None
        report = score_file(
            path=str(path),
            content=path.read_text(encoding='utf-8'),
            previous=previous,
            watchlist=watchlist,
            min_tokens=min_tokens,
        )
        if report is None or not report.hit:
            continue
        print(report.describe())
        failed = failed or report.exceeds(threshold)
    return 1 if failed else 0


COMMAND = Command(
    name='check', help='Score files against a watchlist; exit 1 on hits above threshold.', configure=configure, run=run
)
