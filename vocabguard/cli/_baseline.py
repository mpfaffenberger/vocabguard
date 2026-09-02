from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai.exceptions import UserError

from .. import _git
from ..stats import CorpusCounts
from ._common import Command, add_output_option, add_path_filter, extract_prose, status


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--ref', required=True, help='Git ref whose prose defines the baseline.')
    add_path_filter(parser)
    add_output_option(parser, default='baseline.json', help='Where to write the n-gram counts.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing baseline.')


def run(args: argparse.Namespace) -> int:
    ref: str = args.ref
    paths: list[str] = args.path
    output: Path = args.output
    force: bool = args.force
    if output.exists() and not force:
        raise UserError(f'{output} already exists; a baseline is frozen once written. Pass --force to replace it.')
    files = _git.prose_files(ref, paths)
    documents = (extract_prose(path, text) for path in files if (text := _git.read_file(ref, path)) is not None)
    counts = CorpusCounts.from_documents(documents, source=f'baseline {ref}')
    counts.save(output)
    status(f'{len(files)} files, {counts.total} tokens, {len(counts.counts)} distinct n-grams -> {output}')
    return 0


COMMAND = Command(name='baseline', help='Count n-grams in the prose at a git ref.', configure=configure, run=run)
