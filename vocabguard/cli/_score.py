"""Score a piece of text with the bundled classifier. The quickest way to see what the guard would do.

`vocabguard "some text"` and `echo text | vocabguard` both land here; `main` routes to this command
when the first argument is not a command name.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from pydantic_ai.exceptions import UserError

from ..scoring import score_text
from ._common import Command, add_threshold_option, add_watchlist_option, load_watchlist, resolve_threshold


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('text', nargs='*', help='Text to score. Omit it, or pass -, to read stdin.')
    add_watchlist_option(parser)
    add_threshold_option(parser)
    parser.add_argument('--min-tokens', type=int, default=20, help='Below this many tokens only banned patterns fire.')
    parser.add_argument('--json', action='store_true', help='Print the report as JSON instead of text.')


def run(args: argparse.Namespace) -> int:
    pieces: list[str] = args.text
    min_tokens: int = args.min_tokens
    as_json: bool = args.json
    text = sys.stdin.read() if not pieces or pieces == ['-'] else ' '.join(pieces)
    if not text.strip():
        raise UserError('nothing to score: pass text as an argument or on stdin')
    watchlist = load_watchlist(args.watchlist)
    threshold = resolve_threshold(watchlist, args.threshold)
    report = score_text(text, watchlist, min_tokens=min_tokens, source='text')
    drifted = report.exceeds(threshold)
    if as_json:
        print(json.dumps({**asdict(report), 'threshold': threshold, 'drifted': drifted}, ensure_ascii=False))
    else:
        print(report.describe())
        print(f'score {report.score:.3f} vs threshold {threshold:.2f}: {"drifted" if drifted else "ok"}')
    return 1 if drifted else 0


COMMAND = Command(name='score', help='Score text and exit 1 if it drifts.', configure=configure, run=run)
