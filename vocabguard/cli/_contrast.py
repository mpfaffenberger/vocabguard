from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai.exceptions import UserError

from ..stats import CorpusCounts
from ..watchlist import CuratedFile, Watchlist
from ._common import (
    Command,
    add_contrast_options,
    add_output_option,
    describe_ranking,
    extract_prose,
    iter_prose_files,
    rank_terms,
    status,
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--baseline', type=Path, required=True, help='Counts written by `vocabguard baseline`.')
    parser.add_argument('--corpus', type=Path, required=True, help='Directory written by `vocabguard rewrite`.')
    add_output_option(parser, default='watchlist.json', help='Where to write the watchlist.')
    add_contrast_options(parser)
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
    threshold: float = args.threshold
    curated_path: Path | None = args.curated
    if not corpus_dir.is_dir():
        raise UserError(f'{corpus_dir} is not a directory')
    baseline = CorpusCounts.load(baseline_path)
    documents = (extract_prose(str(path), path.read_text(encoding='utf-8')) for path in iter_prose_files([corpus_dir]))
    model = CorpusCounts.from_documents(documents, source=f'rewrite corpus {corpus_dir}')
    if model.total == 0:
        raise UserError(f'no prose found under {corpus_dir}')
    curated = CuratedFile.load(curated_path) if curated_path else CuratedFile()
    watchlist = Watchlist.from_parts(
        terms=dict(rank_terms(args, model=model, baseline=baseline)),
        replacements=curated.replacements,
        banned_patterns=curated.banned_patterns,
        threshold=threshold,
    )
    watchlist.save(output)
    status(f'{describe_ranking(args, len(watchlist.terms))} -> {output}')
    return 0


COMMAND = Command(
    name='contrast',
    help='Compare the model corpus against the baseline and write the watchlist.',
    configure=configure,
    run=run,
)
