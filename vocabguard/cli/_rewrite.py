"""The only place vocabguard makes model requests. Keep it that way."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai import Agent

from .. import _git
from ._common import Command, add_output_option, add_path_filter, extract_prose, status

INSTRUCTIONS = (
    'Rewrite this document in your own words. Keep the same content and the same length. '
    'Reply with the rewritten document only.'
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--ref', required=True, help='Git ref whose prose gets rewritten.')
    parser.add_argument('--model', required=True, help='Pydantic AI model name, for example openai:gpt-5.')
    add_path_filter(parser)
    add_output_option(parser, default='corpus', help='Directory for the rewritten documents.')


def run(args: argparse.Namespace) -> int:
    ref: str = args.ref
    model: str = args.model
    paths: list[str] = args.path
    output: Path = args.output
    agent: Agent[None, str] = Agent(model, instructions=INSTRUCTIONS)
    files = _git.prose_files(ref, paths)
    for index, path in enumerate(files, start=1):
        # Rewritten prose is markdown whatever the source was, so contrast can extract it the same way.
        target = output / f'{path}.md'
        if target.exists():
            status(f'[{index}/{len(files)}] {path}: already rewritten, skipping')
            continue
        text = _git.read_file(ref, path)
        prose = extract_prose(path, text) if text is not None else ''
        if not prose.strip():
            status(f'[{index}/{len(files)}] {path}: no prose, skipping')
            continue
        status(f'[{index}/{len(files)}] {path}')
        rewritten = agent.run_sync(prose).output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewritten, encoding='utf-8')
    return 0


COMMAND = Command(
    name='rewrite',
    help='Have a model rewrite each prose document at a ref, building the model corpus.',
    configure=configure,
    run=run,
)
