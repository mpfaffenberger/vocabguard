"""The `VocabularyGuard` capability: intercept file writes and push back before the prose lands."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import Literal

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .scoring import HitReport, score_file
from .watchlist import Watchlist

__all__ = ('VocabularyGuard',)

_INSTRUCTION_TERMS = 15


@dataclass
class VocabularyGuard(AbstractCapability[AgentDepsT]):
    """Score the prose in file-write tool calls and ask the model to rephrase when watched terms appear.

    The guard never makes model requests and never reads files; the watchlist is loaded at construction.
    Only added lines are scored when the tool call carries the previous text, so edits to a file that
    already contains watched terms still go through.
    """

    watchlist: Watchlist
    _: KW_ONLY
    tools: Sequence[str] | Callable[[str], bool] = ('edit_file', 'write_file', 'create_file')
    """Tool names to watch, or a predicate over the tool name."""
    path_keys: Sequence[str] = ('path', 'file_path')
    content_keys: Sequence[str] = ('content', 'new_string', 'new_str')
    previous_keys: Sequence[str] = ('old_string', 'old_str')
    threshold: float = 0.0
    """Score above which the guard fires. Zero means any watched term fires."""
    min_tokens: int = 20
    """Below this many tokens the score is skipped; banned patterns still fire."""
    mode: Literal['retry', 'warn'] = 'retry'
    """`retry` raises `ModelRetry`; `warn` lets the write through and only calls `on_hit`."""
    instruct: bool = True
    """Tell the model the top watched terms up front so the guard fires less often."""
    on_hit: Callable[[HitReport], None] | None = None
    """Called on every hit in both modes, for logging or metrics."""

    def __post_init__(self) -> None:
        if self.mode not in ('retry', 'warn'):
            raise UserError(f'VocabularyGuard mode must be "retry" or "warn", not {self.mode!r}')
        if not callable(self.tools) and not self.tools:
            raise UserError('VocabularyGuard needs at least one tool name to watch')
        if self.threshold < 0:
            raise UserError('VocabularyGuard threshold must be zero or greater')
        if self.min_tokens < 0:
            raise UserError('VocabularyGuard min_tokens must be zero or greater')

    def get_instructions(self) -> str | None:
        if not self.instruct or not self.watchlist.terms:
            return None
        lines = ['When writing prose, avoid these words and phrases:']
        for term, _ in self.watchlist.top_terms(_INSTRUCTION_TERMS):
            replacement = self.watchlist.replacements.get(term)
            lines.append(f'- {self.watchlist.label(term)}' + (f' (prefer: {replacement})' if replacement else ''))
        return '\n'.join(lines)

    async def before_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        if not self._watches(call.tool_name):
            return args
        path = _first_str(args, self.path_keys)
        content = _first_str(args, self.content_keys)
        if path is None or content is None:
            return args
        report = score_file(
            path=path,
            content=content,
            previous=_first_str(args, self.previous_keys),
            watchlist=self.watchlist,
            min_tokens=self.min_tokens,
        )
        if report is None or not report.hit:
            return args
        if self.on_hit is not None:
            self.on_hit(report)
        if self.mode == 'retry' and report.exceeds(self.threshold):
            raise ModelRetry(
                f'{report.describe()}\n\nRephrase the prose without these terms and resubmit the same '
                f'{call.tool_name} call with the corrected text.'
            )
        return args

    def _watches(self, tool_name: str) -> bool:
        return self.tools(tool_name) if callable(self.tools) else tool_name in self.tools


def _first_str(args: Mapping[str, object], keys: Sequence[str]) -> str | None:
    """Tool args arrive untyped; only string values are usable as paths or content."""
    for key in keys:
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None
