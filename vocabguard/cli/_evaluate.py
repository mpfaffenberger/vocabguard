"""Held-out grading of the contrast pipeline.

Builds a watchlist from part of each corpus and scores the rest with the same scorer the guard
uses. This is how the recommended threshold for a shipped watchlist is chosen, and how a user
with their own corpora can tell whether a watchlist is worth wiring in before it blocks writes.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from pydantic_ai.exceptions import UserError

from ..normalize import normalize
from ..scoring import score_prose
from ..stats import CorpusCounts, log_odds_z, separation
from ..watchlist import Watchlist
from ._common import Command, extract_prose, iter_prose_files


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--baseline-dir', type=Path, required=True, help='Directory of baseline (human) prose files.')
    parser.add_argument('--corpus-dir', type=Path, required=True, help='Directory of model prose files.')
    parser.add_argument('--holdout', type=float, default=0.2, help='Fraction of each corpus kept for scoring.')
    parser.add_argument('--seed', type=int, default=0, help='Shuffle seed, so runs are comparable.')
    parser.add_argument('--alpha0', type=float, default=500.0, help='Total prior mass for the Dirichlet prior.')
    parser.add_argument('--z', type=float, default=2.5, help='Minimum z toward the model corpus to be watched.')
    parser.add_argument('--top', type=int, default=None, metavar='N', help='Keep only the N highest z.')
    parser.add_argument('--min-tokens', type=int, default=20, help='Documents shorter than this are dropped.')


def _load(directory: Path, min_tokens: int) -> list[str]:
    if not directory.is_dir():
        raise UserError(f'{directory} is not a directory')
    documents: list[str] = []
    for path in iter_prose_files([directory]):
        prose = extract_prose(str(path), path.read_text(encoding='utf-8', errors='replace'))
        if len(normalize(prose)) >= min_tokens:
            documents.append(prose)
    return documents


def _split(documents: list[str], holdout: float, seed: int) -> tuple[list[str], list[str]]:
    shuffled = documents[:]
    random.Random(seed).shuffle(shuffled)
    cut = len(shuffled) - max(1, int(len(shuffled) * holdout))
    return shuffled[:cut], shuffled[cut:]


def run(args: argparse.Namespace) -> int:
    baseline_dir: Path = args.baseline_dir
    corpus_dir: Path = args.corpus_dir
    holdout: float = args.holdout
    seed: int = args.seed
    alpha0: float = args.alpha0
    z_min: float = args.z
    top: int | None = args.top
    min_tokens: int = args.min_tokens
    if not 0 < holdout < 1:
        raise UserError('--holdout must be between 0 and 1')

    baseline_docs = _load(baseline_dir, min_tokens)
    model_docs = _load(corpus_dir, min_tokens)
    if len(baseline_docs) < 2 or len(model_docs) < 2:
        raise UserError('evaluate needs at least two documents on each side')
    baseline_train, baseline_test = _split(baseline_docs, holdout, seed)
    model_train, model_test = _split(model_docs, holdout, seed)

    scores = log_odds_z(
        model=CorpusCounts.from_documents(model_train),
        baseline=CorpusCounts.from_documents(baseline_train),
        alpha0=alpha0,
    )
    ranked = sorted(((term, z) for term, z in scores.items() if z > z_min), key=lambda item: item[1], reverse=True)
    watchlist = Watchlist.from_parts(terms=dict(ranked[:top]), replacements={}, banned_patterns=[])
    if not watchlist.terms:
        raise UserError(f'no n-grams cleared z > {z_min}; lower --z or use larger corpora')
    result = separation(
        baseline=[score_prose([doc], watchlist, min_tokens=min_tokens).score for doc in baseline_test],
        model=[score_prose([doc], watchlist, min_tokens=min_tokens).score for doc in model_test],
    )

    cap = f', top {top}' if top is not None else ''
    print(f'train: {len(baseline_train)} baseline / {len(model_train)} model documents')
    print(f'held out: {len(baseline_test)} baseline / {len(model_test)} model documents')
    print(f'watchlist: {len(watchlist.terms)} terms (z > {z_min}{cap})')
    print(f'AUC: {result.auc:.3f}')
    print(
        f'threshold {result.threshold:.2f}: flags {result.true_positive_rate:.0%} of model documents '
        f'and {result.false_positive_rate:.0%} of baseline documents '
        f'(balanced accuracy {result.balanced_accuracy:.3f})'
    )
    return 0


COMMAND = Command(
    name='evaluate',
    help='Build a watchlist on part of each corpus and grade it on the rest.',
    configure=configure,
    run=run,
)
