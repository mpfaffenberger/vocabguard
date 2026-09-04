"""The optional second agent. It rewrites flagged text so the primary model does not have to retry.

This is the one place the capability makes model requests, and only when a rewriter is configured.
"""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models import Model

from .scoring import HitReport

__all__ = ('REWRITE_INSTRUCTIONS', 'build_rewriter', 'rewrite')

REWRITE_INSTRUCTIONS = """\
You rewrite text so it reads like prose written before 2026.

Keep the meaning, facts, names, numbers, code, links, and markup exactly; keep the length close.
Write complete sentences addressed to a reader. Use articles and pronouns where they belong.
Prefer one plain sentence over a stack of noun phrases or a list of fragments.
Do not add commentary, headings, or explanations. Return only the rewritten text.\
"""


def build_rewriter(spec: Agent[None, str] | Model | str) -> Agent[None, str]:
    """Your own agent is used as given; a model name or instance gets the default instructions."""
    if isinstance(spec, Agent):
        return spec
    return Agent(spec, output_type=str, instructions=REWRITE_INSTRUCTIONS)


async def rewrite(agent: Agent[None, str], text: str, report: HitReport) -> str:
    avoid = ', '.join(
        f'{hit.term} (prefer: {hit.replacement})' if hit.replacement else hit.term for hit in report.terms[:25]
    )
    lines = ['Rewrite the text below.']
    if avoid:
        lines.append(f'Avoid these words and phrases: {avoid}.')
    if report.banned:
        lines.append('Remove: ' + ', '.join(repr(match.text) for match in report.banned) + '.')
    lines.append(f'\n<text>\n{text}\n</text>')
    result = await agent.run('\n'.join(lines))
    return result.output.strip()
