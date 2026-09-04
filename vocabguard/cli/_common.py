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
from ..stats import CorpusCounts, evenness, log_odds_z
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
        '--watchlist', type=Path, default=None, help='Watchlist JSON. Defaults to the bundled measured classifier.'
    )


def add_threshold_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--threshold',
        type=float,
        default=None,
        help='Score above which prose counts as drifted. Defaults to the threshold stored in the watchlist.',
    )


def load_watchlist(path: Path | None) -> Watchlist:
    return Watchlist.default() if path is None else Watchlist.load(path)


def resolve_threshold(watchlist: Watchlist, override: float | None) -> float:
    if override is not None and override < 0:
        raise UserError('--threshold must be zero or greater')
    return watchlist.threshold if override is None else override


def add_source_options(parser: argparse.ArgumentParser) -> None:
    """Prose comes from a git ref or a plain directory of files."""
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


def add_contrast_options(parser: argparse.ArgumentParser) -> None:
    """The knobs that turn two corpora into a ranked term list; shared by `contrast` and `evaluate`."""
    parser.add_argument('--alpha0', type=float, default=500.0, help='Total prior mass for the Dirichlet prior.')
    parser.add_argument('--z', type=float, default=2.5, help='Minimum z toward the model corpus to be watched.')
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        metavar='N',
        help='Keep only the N highest scores. With corpora of millions of tokens nearly every n-gram clears --z.',
    )
    parser.add_argument(
        '--evenness',
        type=float,
        default=1.0,
        metavar='GAMMA',
        help='Weight z by how evenly a term spreads across documents, to the power GAMMA. '
        'Voice spreads and topic bursts, so 1.0 favours voice; 0 disables the weighting.',
    )


def rank_terms(args: argparse.Namespace, *, model: CorpusCounts, baseline: CorpusCounts) -> list[tuple[str, float]]:
    """Weighted z per term, highest first, after the --z floor and the --top cap."""
    alpha0: float = args.alpha0
    z_min: float = args.z
    top: int | None = args.top
    gamma: float = args.evenness
    if top is not None and top < 1:
        raise UserError('--top must be at least 1')
    if gamma < 0:
        raise UserError('--evenness must be zero or greater')
    # The z floor is a significance test, so it applies to the raw z; the weighting only reorders survivors.
    scores = {term: z for term, z in log_odds_z(model=model, baseline=baseline, alpha0=alpha0).items() if z > z_min}
    if gamma:
        spread = evenness(model=model, baseline=baseline)
        scores = {term: z * spread[term] ** gamma for term, z in scores.items()}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:top]


def describe_ranking(args: argparse.Namespace, count: int) -> str:
    cap = f', top {args.top}' if args.top is not None else ''
    weighting = f', evenness^{args.evenness:g}' if args.evenness else ''
    return f'{count} watched n-grams (z > {args.z}{cap}{weighting})'


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
