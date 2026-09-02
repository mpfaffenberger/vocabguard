from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai.exceptions import UserError

from ..stats import CorpusCounts
from ._common import Command, add_output_option, add_source_options, extract_prose, load_documents, status


def configure(parser: argparse.ArgumentParser) -> None:
    add_source_options(parser)
    add_output_option(parser, default='baseline.json', help='Where to write the n-gram counts.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing baseline.')


def run(args: argparse.Namespace) -> int:
    output: Path = args.output
    force: bool = args.force
    if output.exists() and not force:
        raise UserError(f'{output} already exists; a baseline is frozen once written. Pass --force to replace it.')
    documents = load_documents(args)
    source = f'baseline {args.ref or args.dir}'
    counts = CorpusCounts.from_documents((extract_prose(path, text) for path, text in documents), source=source)
    counts.save(output)
    status(f'{len(documents)} files, {counts.total} tokens, {len(counts.counts)} distinct n-grams -> {output}')
    return 0


COMMAND = Command(
    name='baseline', help='Count n-grams in the prose at a git ref or directory.', configure=configure, run=run
)
