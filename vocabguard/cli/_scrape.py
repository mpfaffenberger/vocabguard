"""Interactive corpus gathering. Model requests happen here and in `rewrite`, nowhere else."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model

from .. import openrouter
from ..display import make_display
from ..scrape import ScrapeDeps, build_agent, chat, make_client
from ._common import Command, add_output_option, status

DEFAULT_MODEL = 'openrouter:openrouter/free'
_OPENROUTER_PREFIX = 'openrouter:'


def configure(parser: argparse.ArgumentParser) -> None:
    add_output_option(parser, default='corpus_reference', help='Directory the agent saves documents into.')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'Pydantic AI model name. Default {DEFAULT_MODEL}, which signs you in to OpenRouter on first use.',
    )
    parser.add_argument('--target', type=int, default=50, help='How many documents to ask the agent for.')


def run(args: argparse.Namespace) -> int:
    output: Path = args.output
    model_name: str = args.model
    target: int = args.target
    agent = build_agent(resolve_model(model_name))
    with make_client() as client:
        deps = ScrapeDeps(output=output, client=client, target=target)
        status(f'Saving documents to {output}. Type quit to end the session.')
        asyncio.run(chat(agent, deps, make_display()))
    status(f'{len(deps.saved)} documents saved this session.')
    return 0


def resolve_model(model_name: str) -> Model | str:
    """OpenRouter names get the OAuth key; anything else is left to pydantic-ai's own inference."""
    if not model_name.startswith(_OPENROUTER_PREFIX):
        return model_name
    try:
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider
    except ImportError as error:
        raise UserError('OpenRouter support needs the scrape extra: pip install "vocabguard[scrape]"') from error
    key = openrouter.api_key()
    return OpenRouterModel(model_name.removeprefix(_OPENROUTER_PREFIX), provider=OpenRouterProvider(api_key=key))


COMMAND = Command(
    name='scrape',
    help='Chat with an agent that gathers a reference corpus of desirable prose.',
    configure=configure,
    run=run,
)
