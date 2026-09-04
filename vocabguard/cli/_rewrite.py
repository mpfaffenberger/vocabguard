"""Model requests happen here and in the optional rewriter agent, nowhere else in the package."""

from __future__ import annotations

import argparse
from pathlib import Path

from pydantic_ai import Agent

from ._common import Command, add_output_option, add_source_options, extract_prose, load_documents, status

INSTRUCTIONS = (
    'Rewrite this document in your own words. Keep the same content and the same length. '
    'Reply with the rewritten document only.'
)


def configure(parser: argparse.ArgumentParser) -> None:
    add_source_options(parser)
    parser.add_argument('--model', required=True, help='Pydantic AI model name, for example openai:gpt-5.')
    add_output_option(parser, default='corpus', help='Directory for the rewritten documents.')


def run(args: argparse.Namespace) -> int:
    model: str = args.model
    output: Path = args.output
    agent: Agent[None, str] = Agent(model, instructions=INSTRUCTIONS)
    documents = load_documents(args)
    for index, (path, text) in enumerate(documents, start=1):
        # Rewritten prose is markdown whatever the source was, so contrast can extract it the same way.
        target = output / f'{path}.md'
        if target.exists():
            status(f'[{index}/{len(documents)}] {path}: already rewritten, skipping')
            continue
        prose = extract_prose(path, text)
        if not prose.strip():
            status(f'[{index}/{len(documents)}] {path}: no prose, skipping')
            continue
        status(f'[{index}/{len(documents)}] {path}')
        rewritten = agent.run_sync(prose).output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rewritten, encoding='utf-8')
    return 0


COMMAND = Command(
    name='rewrite',
    help='Have a model rewrite each prose document at a ref or directory, building the model corpus.',
    configure=configure,
    run=run,
)
