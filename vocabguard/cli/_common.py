from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from ..extract import extractor_for
from ..watchlist import Watchlist

CommandRunner: TypeAlias = Callable[[argparse.Namespace], int]


@dataclass(kw_only=True)
class Command:
    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    run: CommandRunner


def status(message: str) -> None:
    """Progress goes to stderr so stdout stays clean for csv and json output."""
    print(message, file=sys.stderr)


def add_watchlist_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--watchlist', type=Path, default=None, help='Watchlist JSON. Defaults to the bundled starter list.'
    )


def load_watchlist(path: Path | None) -> Watchlist:
    return Watchlist.starter() if path is None else Watchlist.load(path)


def add_path_filter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--path', action='append', default=[], metavar='PATH', help='Limit to these paths. Repeatable.')


def add_output_option(parser: argparse.ArgumentParser, *, default: str, help: str) -> None:
    parser.add_argument('-o', '--output', type=Path, default=Path(default), help=help)


def iter_prose_files(roots: list[Path]) -> Iterator[Path]:
    """Expand directories to the prose files inside them; files are yielded as given."""
    for root in roots:
        if root.is_dir():
            yield from (path for path in sorted(root.rglob('*')) if path.is_file() and extractor_for(str(path)))
        else:
            yield root


def extract_prose(path: str, text: str) -> str:
    extractor = extractor_for(path)
    return '\n\n'.join(extractor.extract(text, path=path)) if extractor else ''
