"""Drift over time. This is the monitor; the guard never computes divergence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import _git
from ..normalize import lemmatize_words, normalize
from ..stats import CorpusCounts, jensen_shannon
from ..watchlist import Watchlist
from ._common import Command, add_watchlist_option, extract_prose, load_watchlist


@dataclass(kw_only=True)
class Row:
    sha: str
    date: str
    subject: str
    words: int
    js_divergence: float
    hits_per_thousand_words: float


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--baseline', type=Path, required=True, help='Counts written by `vocabguard baseline`.')
    add_watchlist_option(parser)
    parser.add_argument('--since', required=True, metavar='REF', help='Report every commit after this ref.')
    parser.add_argument('--format', choices=('csv', 'json'), default='csv')


def run(args: argparse.Namespace) -> int:
    baseline = CorpusCounts.load(args.baseline)
    watchlist = load_watchlist(args.watchlist)
    since: str = args.since
    output_format: str = args.format
    rows = [row for commit in _git.commits_since(since) if (row := _row_for(commit, baseline, watchlist)) is not None]
    if output_format == 'json':
        json.dump([asdict(row) for row in rows], sys.stdout, indent=2)
        print()
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=list(Row.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return 0


def _row_for(commit: _git.Commit, baseline: CorpusCounts, watchlist: Watchlist) -> Row | None:
    prose = '\n\n'.join(
        extract_prose(path, added) for path, added in _git.added_lines_by_file(commit.sha).items()
    ).strip()
    if not prose:
        return None
    words = lemmatize_words(prose)
    tokens = normalize(prose)
    hits = sum(1 for token in tokens if token in watchlist.terms)
    return Row(
        sha=commit.sha[:12],
        date=commit.date,
        subject=commit.subject,
        words=len(words),
        js_divergence=round(jensen_shannon(CorpusCounts.from_documents([prose]), baseline), 4),
        hits_per_thousand_words=round(1000 * hits / len(words), 2) if words else 0.0,
    )


COMMAND = Command(
    name='report',
    help='Per-commit divergence from the baseline and watchlist hits per thousand words.',
    configure=configure,
    run=run,
)
