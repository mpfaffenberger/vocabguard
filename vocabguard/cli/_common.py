from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pydantic_ai.exceptions import UserError

from .. import _git
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


def add_source_options(parser: argparse.ArgumentParser) -> None:
    """Prose comes from a git ref or a plain directory (for example one built by `vocabguard scrape`)."""
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--ref', help='Git ref to read prose files from.')
    source.add_argument('--dir', type=Path, help='Directory to read prose files from.')
    parser.add_argument('--path', action='append', default=[], metavar='PATH', help='Limit to these paths. Repeatable.')


def load_documents(args: argparse.Namespace) -> list[tuple[str, str]]:
    """(relative path, text) for every prose file at the chosen source."""
    ref: str | None = args.ref
    directory: Path | None = args.dir
    paths: list[str] = args.path
    if ref is not None:
        files = _git.prose_files(ref, paths)
        return [(path, text) for path in files if (text := _git.read_file(ref, path)) is not None]
    if directory is None or not directory.is_dir():
        raise UserError(f'{directory} is not a directory')
    roots = [directory / path for path in paths] or [directory]
    return [
        (str(path.relative_to(directory)), path.read_text(encoding='utf-8', errors='replace'))
        for path in iter_prose_files(roots)
    ]


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
