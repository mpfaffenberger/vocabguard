"""The `VocabularyGuard` capability: score tool arguments and model text, and push back before they land."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic_ai.capabilities import AbstractCapability, ValidatedToolArgs
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .extract import extractor_for
from .scoring import HitReport, score_file, score_text
from .watchlist import Watchlist

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ('VocabularyGuard',)

_INSTRUCTION_TERMS = 15


@dataclass
class VocabularyGuard(AbstractCapability[AgentDepsT]):
    """Score the prose in tool calls and model responses; ask the model to rephrase when watched terms appear.

    Tool calls that carry a path are scored as that file type, and only the added lines when the
    previous text is present, so edits to a file that already contains watched terms still go
    through. Tool calls without a path have every string argument scored as plain text. Model
    responses are scored on their text parts. The guard never makes model requests or reads files.

    `VocabularyGuard()` with no arguments is the bundled classifier at the threshold measured for it.
    """

    watchlist: Watchlist = field(default_factory=Watchlist.default)
    _: KW_ONLY
    tools: Sequence[str] | Callable[[str], bool] | None = None
    """Tool names to watch, or a predicate over the tool name. None watches every tool."""
    output: bool = True
    """Score the text of every model response, mid-run and final."""
    path_keys: Sequence[str] = ('path', 'file_path')
    content_keys: Sequence[str] = ('content', 'new_string', 'new_str')
    previous_keys: Sequence[str] = ('old_string', 'old_str')
    threshold: float | None = None
    """Score above which the guard fires. None uses the watchlist's own threshold; zero means any watched term fires."""
    min_tokens: int = 20
    """Below this many tokens the score is skipped; banned patterns still fire."""
    mode: Literal['retry', 'warn'] = 'retry'
    """`retry` raises `ModelRetry`; `warn` lets everything through and only calls `on_hit`."""
    instruct: bool = True
    """Tell the model the top watched terms up front so the guard fires less often."""
    on_hit: Callable[[HitReport], None] | None = None
    """Called on every hit in both modes, for logging or metrics."""

    def __post_init__(self) -> None:
        if self.mode not in ('retry', 'warn'):
            raise UserError(f'VocabularyGuard mode must be "retry" or "warn", not {self.mode!r}')
        if self.tools is not None and not callable(self.tools) and not self.tools:
            raise UserError('VocabularyGuard tools must name at least one tool, or be None to watch every tool')
        if self.effective_threshold < 0:
            raise UserError('VocabularyGuard threshold must be zero or greater')
        if self.min_tokens < 0:
            raise UserError('VocabularyGuard min_tokens must be zero or greater')

    @property
    def effective_threshold(self) -> float:
        """The threshold in force: an explicit one, else the watchlist's."""
        return self.watchlist.threshold if self.threshold is None else self.threshold

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
        report = self._score_tool_call(call.tool_name, args)
        if report is not None:
            self._handle(
                report, f'Rephrase the prose and resubmit the same {call.tool_name} call with the corrected text.'
            )
        return args

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if not self.output:
            return response
        text = '\n\n'.join(part.content for part in response.parts if isinstance(part, TextPart))
        if text.strip():
            report = score_text(text, self.watchlist, min_tokens=self.min_tokens, source='model response')
            self._handle(report, 'Rephrase your response without these terms and reply again.')
        return response

    def _score_tool_call(self, tool_name: str, args: Mapping[str, object]) -> HitReport | None:
        path = _first_str(args, self.path_keys)
        if path is not None:
            # A path names a file type; a type with no extractor (json, yaml) is not prose.
            content = _first_str(args, self.content_keys)
            if content is None or extractor_for(path) is None:
                return None
            return score_file(
                path=path,
                content=content,
                previous=_first_str(args, self.previous_keys),
                watchlist=self.watchlist,
                min_tokens=self.min_tokens,
                source=f'{tool_name}({path})',
            )
        text = '\n\n'.join(value for value in args.values() if isinstance(value, str))
        return score_text(text, self.watchlist, min_tokens=self.min_tokens, source=tool_name) if text else None

    def _handle(self, report: HitReport, instruction: str) -> None:
        if not report.hit:
            return
        if self.on_hit is not None:
            self.on_hit(report)
        if self.mode == 'retry' and report.exceeds(self.effective_threshold):
            raise ModelRetry(f'{report.describe()}\n\n{instruction}')

    def _watches(self, tool_name: str) -> bool:
        if self.tools is None:
            return True
        return self.tools(tool_name) if callable(self.tools) else tool_name in self.tools


def _first_str(args: Mapping[str, object], keys: Sequence[str]) -> str | None:
    """Tool args arrive untyped; only string values are usable as paths or content."""
    for key in keys:
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None
