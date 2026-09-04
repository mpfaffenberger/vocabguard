"""The `VocabularyGuard` capability: score tool arguments and model text, and push back before they land."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, OutputContext, ValidatedToolArgs
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturn, UserContent
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .extract import PythonExtractor, extractor_for
from .rewriter import build_rewriter, rewrite
from .scoring import HitReport, score_file, score_text
from .targets import OutputField, Target, TextOutput, ToolArgument
from .watchlist import Watchlist

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

__all__ = ('VocabularyGuard',)

_INSTRUCTION_TERMS = 15
_RESUBMIT = 'Rephrase the prose and resubmit the same {tool} call with the corrected text.'
_REPLY_AGAIN = 'Rephrase your response without these terms and reply again.'
_REGENERATE = 'Produce the output again with the flagged text rephrased.'


@dataclass
class VocabularyGuard(AbstractCapability[AgentDepsT]):
    """Score the prose in tool calls and model output; ask the model to rephrase when watched terms appear.

    Without `targets`, every tool call and every model response is scored. Tool calls that carry a
    path are scored as that file type, and only the added lines when the previous text is present,
    so edits to a file that already contains watched terms still go through. Tool calls without a
    path have every string argument scored as plain text.

    With `targets`, only the named tool arguments, structured output fields, and final text are
    scored. Add a `rewriter` to have a second agent rewrite what fires instead of retrying.

    `VocabularyGuard()` with no arguments is the bundled classifier at the threshold measured for it.
    """

    watchlist: Watchlist = field(default_factory=Watchlist.default)
    _: KW_ONLY
    tools: Sequence[str] | Callable[[str], bool] | None = None
    """Tool names to watch, or a predicate over the tool name. None watches every tool. Ignored with `targets`."""
    output: bool = True
    """Score the text of every model response, mid-run and final. Ignored with `targets`."""
    targets: Sequence[Target] | None = None
    """Exactly where to look: `ToolArgument`, `OutputField`, and `TextOutput` entries."""
    rewriter: Agent[None, str] | Model | str | None = None
    """A second agent that rewrites flagged text at `targets`. A model name or instance gets default instructions."""
    path_keys: Sequence[str] = ('path', 'file_path')
    content_keys: Sequence[str] = ('content', 'new_string', 'new_str')
    previous_keys: Sequence[str] = ('old_string', 'old_str')
    threshold: float | None = None
    """Score above which the guard fires. None uses the watchlist's own threshold; zero means any watched term fires."""
    min_tokens: int = 20
    """Below this many tokens the score is skipped; banned patterns still fire."""
    mode: Literal['retry', 'nudge', 'warn'] = 'retry'
    """What happens on a hit above threshold that a rewriter did not fix.

    `retry` raises `ModelRetry` before the tool runs, so the model must rephrase. `nudge` lets the
    tool run and appends the report to its result with a suggestion to rewrite, so the model decides.
    `warn` lets everything through and only calls `on_hit`. Model output is never nudged: there
    is no result to attach to, so `nudge` treats it like `warn`.
    """
    instruct: bool = True
    """Tell the model the top watched terms up front so the guard fires less often."""
    on_hit: Callable[[HitReport], None] | None = None
    """Called on every hit in every mode, for logging or metrics."""
    _rewriter: Agent[None, str] | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self.mode not in ('retry', 'nudge', 'warn'):
            raise UserError(f'VocabularyGuard mode must be "retry", "nudge", or "warn", not {self.mode!r}')
        if self.tools is not None and not callable(self.tools) and not self.tools:
            raise UserError('VocabularyGuard tools must name at least one tool, or be None to watch every tool')
        if self.targets is not None and not self.targets:
            raise UserError('VocabularyGuard targets must name at least one target, or be None to watch everything')
        if self.rewriter is not None and self.targets is None:
            raise UserError(
                'VocabularyGuard rewriter needs targets: name the tool arguments or output fields to rewrite'
            )
        if self.effective_threshold < 0:
            raise UserError('VocabularyGuard threshold must be zero or greater')
        if self.min_tokens < 0:
            raise UserError('VocabularyGuard min_tokens must be zero or greater')
        self._rewriter = build_rewriter(self.rewriter) if self.rewriter is not None else None

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
        if self.targets is not None:
            return await self._guard_tool_targets(call.tool_name, args)
        if not self._watches(call.tool_name):
            return args
        report = self._score_tool_call(call.tool_name, args)
        if report is not None:
            await self._resolve(report, None, instruction=_RESUBMIT.format(tool=call.tool_name))
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: object,
    ) -> object:
        """In `nudge` mode, hand the model its result plus the report, so it can choose to rewrite."""
        if self.mode != 'nudge':
            return result
        if self.targets is not None:
            reports = [
                report
                for target in self._tool_targets(call.tool_name)
                if (report := self._score_argument(target, args)) is not None
            ]
        elif self._watches(call.tool_name):
            report = self._score_tool_call(call.tool_name, args)
            reports = [report] if report is not None else []
        else:
            return result
        reports = [report for report in reports if report.exceeds(self.effective_threshold)]
        if not reports:
            return result
        note = '\n\n'.join(report.describe() for report in reports) + (
            f'\n\nThe call succeeded as written. If this text is meant to be read by people, '
            f'consider rewriting the flagged passages with another {call.tool_name} call.'
        )
        return _attach(result, note)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        if not self.output or self.targets is not None:
            return response
        text = '\n\n'.join(part.content for part in response.parts if isinstance(part, TextPart))
        if text.strip():
            report = score_text(text, self.watchlist, min_tokens=self.min_tokens, source='model response')
            await self._resolve(report, None, instruction=_REPLY_AGAIN)
        return response

    async def after_output_process(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        output_context: OutputContext,
        output: object,
    ) -> object:
        """Final text and structured output fields named in `targets`; rewritten in place when a rewriter is set."""
        if self.targets is None or ctx.partial_output:
            return output
        if isinstance(output, str):
            if not any(isinstance(target, TextOutput) for target in self.targets):
                return output
            report = score_text(output, self.watchlist, min_tokens=self.min_tokens, source='output')
            rewritten = await self._resolve(report, output, instruction=_REPLY_AGAIN)
            return output if rewritten is None else rewritten
        for target in self.targets:
            if not isinstance(target, OutputField) or not target.matches(output):
                continue
            value = target.get(output)
            if not isinstance(value, str):
                continue
            report = score_text(value, self.watchlist, min_tokens=self.min_tokens, source=target.source)
            rewritten = await self._resolve(report, value, instruction=_REGENERATE)
            if rewritten is not None:
                output = target.replace(output, rewritten)
        return output

    async def _guard_tool_targets(self, tool_name: str, args: ValidatedToolArgs) -> ValidatedToolArgs:
        updated = args
        for target in self._tool_targets(tool_name):
            report = self._score_argument(target, args)
            if report is None:
                continue
            value = args[target.argument]
            path = _first_str(args, self.path_keys)
            rewritable = path is None or not isinstance(extractor_for(path), PythonExtractor)
            rewritten = await self._resolve(
                report, value if rewritable else None, instruction=_RESUBMIT.format(tool=tool_name)
            )
            if rewritten is not None:
                updated = {**updated, target.argument: rewritten}
        return updated

    def _tool_targets(self, tool_name: str) -> list[ToolArgument]:
        return [t for t in self.targets or () if isinstance(t, ToolArgument) and t.name == tool_name]

    def _score_argument(self, target: ToolArgument, args: Mapping[str, object]) -> HitReport | None:
        value = args.get(target.argument)
        if not isinstance(value, str):
            return None
        path = _first_str(args, self.path_keys)
        if path is not None and extractor_for(path) is not None:
            return score_file(
                path=path,
                content=value,
                previous=_first_str(args, self.previous_keys),
                watchlist=self.watchlist,
                min_tokens=self.min_tokens,
                source=f'{target.source} ({path})',
            )
        return score_text(value, self.watchlist, min_tokens=self.min_tokens, source=target.source)

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

    async def _resolve(self, report: HitReport, text: str | None, *, instruction: str) -> str | None:
        """Apply the policy to a hit. Returns replacement text when a rewriter fixed it, else None.

        `text` is the rewritable original, or None when rewriting is not possible at this site.
        A rewrite that still drifts falls through to `mode` with the rewritten report.
        """
        if not report.hit:
            return None
        if self.on_hit is not None:
            self.on_hit(report)
        if not report.exceeds(self.effective_threshold):
            return None
        if self._rewriter is not None and text is not None:
            rewritten = await rewrite(self._rewriter, text, report)
            check = score_text(
                rewritten, self.watchlist, min_tokens=self.min_tokens, source=f'{report.source} (rewritten)'
            )
            if not check.exceeds(self.effective_threshold):
                return rewritten
            if self.on_hit is not None:
                self.on_hit(check)
            if self.mode == 'retry':
                raise ModelRetry(f'{check.describe()}\n\n{instruction}')
            return rewritten
        if self.mode == 'retry':
            raise ModelRetry(f'{report.describe()}\n\n{instruction}')
        return None

    def _watches(self, tool_name: str) -> bool:
        if self.tools is None:
            return True
        return self.tools(tool_name) if callable(self.tools) else tool_name in self.tools


def _attach(result: object, note: str) -> object:
    """Append a note to a tool result without changing its shape."""
    if isinstance(result, str):
        return f'{result}\n\n{note}'
    if isinstance(result, ToolReturn):
        structured = cast(ToolReturn[object], result)
        existing = structured.content
        if existing is None:
            content: Sequence[UserContent] = [note]
        elif isinstance(existing, str):
            content = [existing, note]
        else:
            content = [*existing, note]
        return replace(structured, content=content)
    return ToolReturn(return_value=result, content=note)


def _first_str(args: Mapping[str, object], keys: Sequence[str]) -> str | None:
    """Tool args arrive untyped; only string values are usable as paths or content."""
    for key in keys:
        value = args.get(key)
        if isinstance(value, str):
            return value
    return None
