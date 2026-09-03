from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai.exceptions import UserError

from ..stats import CorpusCounts, log_odds_z
from ..watchlist import CuratedFile, Watchlist
from ._common import Command, add_output_option, extract_prose, iter_prose_files, status


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--baseline', type=Path, required=True, help='Counts written by `vocabguard baseline`.')
    parser.add_argument('--corpus', type=Path, required=True, help='Directory written by `vocabguard rewrite`.')
    add_output_option(parser, default='watchlist.json', help='Where to write the watchlist.')
    parser.add_argument('--alpha0', type=float, default=500.0, help='Total prior mass for the Dirichlet prior.')
    parser.add_argument('--z', type=float, default=2.5, help='Minimum z toward the model corpus to be watched.')
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        metavar='N',
        help='Keep only the N highest z. With corpora of millions of tokens nearly every n-gram clears --z.',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Score above which prose counts as drifted, stored in the watchlist. Take it from `evaluate`.',
    )
    parser.add_argument(
        '--curated',
        type=Path,
        default=None,
        help='JSON with hand-maintained `replacements` and `banned_patterns` to merge into the output.',
    )


def run(args: argparse.Namespace) -> int:
    baseline_path: Path = args.baseline
    corpus_dir: Path = args.corpus
    output: Path = args.output
    alpha0: float = args.alpha0
    z_min: float = args.z
    top: int | None = args.top
    threshold: float = args.threshold
    curated_path: Path | None = args.curated
    if top is not None and top < 1:
        raise UserError('--top must be at least 1')
    if not corpus_dir.is_dir():
        raise UserError(f'{corpus_dir} is not a directory')
    baseline = CorpusCounts.load(baseline_path)
    documents = (extract_prose(str(path), path.read_text(encoding='utf-8')) for path in iter_prose_files([corpus_dir]))
    model = CorpusCounts.from_documents(documents, source=f'rewrite corpus {corpus_dir}')
    if model.total == 0:
        raise UserError(f'no prose found under {corpus_dir}')
    scores = log_odds_z(model=model, baseline=baseline, alpha0=alpha0)
    curated = CuratedFile.load(curated_path) if curated_path else CuratedFile()
    ranked = sorted(((term, z) for term, z in scores.items() if z > z_min), key=lambda item: (-item[1], item[0]))
    watchlist = Watchlist.from_parts(
        terms=dict(ranked[:top]),
        replacements=curated.replacements,
        banned_patterns=curated.banned_patterns,
        threshold=threshold,
    )
    watchlist.save(output)
    cap = f', top {top}' if top is not None else ''
    status(f'{len(watchlist.terms)} watched n-grams (z > {z_min}{cap}) -> {output}')
    return 0


COMMAND = Command(
    name='contrast',
    help='Compare the model corpus against the baseline and write the watchlist.',
    configure=configure,
    run=run,
)
