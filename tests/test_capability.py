from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, FunctionModel

from vocabguard import HitReport, VocabularyGuard, Watchlist

pytestmark = pytest.mark.anyio

WATCHLIST = Watchlist.from_parts(
    terms={'leverage': 4.0, 'robust': 3.0, 'delve': 5.0},
    replacements={'leverage': 'use'},
    banned_patterns=['\u2014'],
)

CONTAMINATED = (
    'We leverage a robust pipeline here, and this section will delve into why the robust design '
    'lets us leverage every stage without extra work from the caller or the reader.'
)
CLEAN = (
    'The parser reads each line, splits it on tabs, and hands the fields to the renderer, which '
    'writes one row per record and skips blank lines without complaint.'
)


def edit(content: str, *, path: str = 'notes.md', old_string: str | None = None) -> ModelResponse:
    args: dict[str, str] = {'path': path, 'content': content}
    if old_string is not None:
        args['old_string'] = old_string
    return ModelResponse(parts=[ToolCallPart('edit_file', args)])


def send(message: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('send_message', {'channel': 'general', 'message': message})])


def say(text: str, *more: ModelResponsePart) -> ModelResponse:
    return ModelResponse(parts=[TextPart(text), *more])


@dataclass
class Harness:
    """Scripted model plus recording tools, so a test declares the responses and inspects what ran."""

    guard: VocabularyGuard[object]
    responses: list[ModelResponse]
    executed: list[tuple[str, dict[str, str]]] = field(default_factory=list[tuple[str, dict[str, str]]])
    requests: list[ModelRequest] = field(default_factory=list[ModelRequest])

    def model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Each call carries the full history, so the last snapshot is the complete one.
        self.requests = [message for message in messages if isinstance(message, ModelRequest)]
        return self.responses.pop(0) if self.responses else say('done')

    def agent(self) -> Agent[object, str]:
        agent: Agent[object, str] = Agent(FunctionModel(self.model), capabilities=[self.guard], retries=3)

        @agent.tool_plain
        def edit_file(path: str, content: str, old_string: str | None = None) -> str:
            self.executed.append(('edit_file', {'path': path, 'content': content}))
            return 'ok'

        @agent.tool_plain
        def send_message(channel: str, message: str) -> str:
            self.executed.append(('send_message', {'channel': channel, 'message': message}))
            return 'sent'

        @agent.tool_plain
        def run_shell(command: str) -> str:
            self.executed.append(('run_shell', {'command': command}))
            return ''

        return agent

    def retries(self) -> list[RetryPromptPart]:
        return [part for request in self.requests for part in request.parts if isinstance(part, RetryPromptPart)]

    def tool_returns(self) -> list[ToolReturnPart]:
        return [part for request in self.requests for part in request.parts if isinstance(part, ToolReturnPart)]


async def test_file_write_retry_round_trip() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [edit(CONTAMINATED), edit(CLEAN)])
    result = await harness.agent().run('write')
    assert result.output == 'done'
    assert harness.executed == [('edit_file', {'path': 'notes.md', 'content': CLEAN})]
    (retry,) = harness.retries()
    assert retry.tool_name == 'edit_file'
    assert 'edit_file(notes.md): vocabulary score' in retry.model_response()
    assert "'leverage' (z=4.0, x2, prefer: use)" in retry.model_response()
    assert 'resubmit the same edit_file call' in retry.model_response()


async def test_clean_write_passes() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [edit(CLEAN)])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_edit_to_contaminated_file_scores_only_added_lines() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [edit(f'{CONTAMINATED}\n\n{CLEAN}', old_string=CONTAMINATED)])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_python_identifiers_are_not_prose() -> None:
    source = 'def leverage_robust(delve: int) -> int:\n    return delve + leverage_robust(delve - 1) if delve else 0\n'
    harness = Harness(VocabularyGuard(WATCHLIST, min_tokens=0), [edit(source, path='mod.py')])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_python_docstrings_are_prose() -> None:
    harness = Harness(
        VocabularyGuard(WATCHLIST), [edit(f'def f() -> None:\n    """{CONTAMINATED}"""\n', path='mod.py')]
    )
    await harness.agent().run('write')
    assert harness.executed == []
    assert len(harness.retries()) == 1


async def test_file_type_without_extractor_is_skipped() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [edit(f'{{"note": "{CONTAMINATED}"}}', path='config.json')])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_tool_without_path_scores_every_string_argument() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [send(CONTAMINATED), send(CLEAN)])
    await harness.agent().run('post')
    assert harness.executed == [('send_message', {'channel': 'general', 'message': CLEAN})]
    (retry,) = harness.retries()
    assert retry.tool_name == 'send_message'
    assert 'send_message: vocabulary score' in retry.model_response()


async def test_short_arguments_fall_below_min_tokens() -> None:
    response = ModelResponse(parts=[ToolCallPart('run_shell', {'command': 'git log --robust'})])
    harness = Harness(VocabularyGuard(WATCHLIST), [response])
    await harness.agent().run('run')
    assert harness.executed == [('run_shell', {'command': 'git log --robust'})]
    assert harness.retries() == []


async def test_final_output_retry_round_trip() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [say(CONTAMINATED), say(CLEAN)])
    result = await harness.agent().run('explain')
    assert result.output == CLEAN
    (retry,) = harness.retries()
    assert retry.tool_name is None
    assert 'model response: vocabulary score' in retry.model_response()
    assert 'reply again' in retry.model_response()


async def test_text_alongside_a_tool_call_is_scored() -> None:
    harness = Harness(
        VocabularyGuard(WATCHLIST), [say(CONTAMINATED, *edit(CLEAN).parts), say('ok', *edit(CLEAN).parts)]
    )
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert len(harness.retries()) == 1


async def test_output_scoring_can_be_disabled() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST, output=False), [say(CONTAMINATED)])
    result = await harness.agent().run('explain')
    assert result.output == CONTAMINATED
    assert harness.retries() == []


async def test_warn_mode_calls_on_hit_and_still_executes() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, mode='warn', on_hit=hits.append)
    harness = Harness(guard, [edit(CONTAMINATED), say(CONTAMINATED)])
    result = await harness.agent().run('write')
    assert result.output == CONTAMINATED
    assert len(harness.executed) == 1
    assert harness.retries() == []
    assert [hit.source for hit in hits] == ['edit_file(notes.md)', 'model response']
    assert {term.term for term in hits[0].terms} == {'leverage', 'robust', 'delve'}


async def test_retry_mode_also_calls_on_hit() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, on_hit=hits.append)
    harness = Harness(guard, [edit(CONTAMINATED), edit(CLEAN)])
    await harness.agent().run('write')
    assert len(hits) == 1


async def test_banned_pattern_fires_below_min_tokens() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [edit('short \u2014 text'), edit('short text')])
    await harness.agent().run('write')
    assert harness.executed == [('edit_file', {'path': 'notes.md', 'content': 'short text'})]
    (retry,) = harness.retries()
    assert 'banned pattern' in retry.model_response()


async def test_nudge_mode_executes_and_hands_the_model_the_report() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, mode='nudge', on_hit=hits.append)
    harness = Harness(guard, [edit(CONTAMINATED), edit(CLEAN)])
    await harness.agent().run('write')
    # Both writes landed: nothing was blocked.
    assert [content for _, args in harness.executed for content in [args['content']]] == [CONTAMINATED, CLEAN]
    assert harness.retries() == []
    first, second = harness.tool_returns()
    assert isinstance(first.content, str)
    assert first.content.startswith('ok\n\nedit_file(notes.md): vocabulary score')
    assert "'leverage' (z=4.0, x2, prefer: use)" in first.content
    assert 'consider rewriting the flagged passages with another edit_file call' in first.content
    assert second.content == 'ok'
    assert len(hits) == 1


async def test_nudge_mode_leaves_model_responses_alone() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, mode='nudge', on_hit=hits.append)
    harness = Harness(guard, [say(CONTAMINATED)])
    result = await harness.agent().run('write')
    assert result.output == CONTAMINATED
    assert harness.retries() == []
    assert len(hits) == 1


async def test_default_guard_is_the_bundled_classifier_at_its_threshold() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard()
    assert guard.watchlist.terms == Watchlist.default().terms
    assert guard.threshold is None
    assert guard.effective_threshold == Watchlist.default().threshold
    explicit: VocabularyGuard[object] = VocabularyGuard(threshold=0.5)
    assert explicit.effective_threshold == 0.5
    listed: VocabularyGuard[object] = VocabularyGuard(WATCHLIST)
    assert listed.effective_threshold == 0.0


async def test_threshold_lets_low_scores_through() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, threshold=10.0)
    harness = Harness(guard, [edit(CONTAMINATED)])
    await harness.agent().run('write')
    assert len(harness.executed) == 1


async def test_tools_filter_restricts_which_tools_are_watched() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, tools=('edit_file',))
    harness = Harness(guard, [send(CONTAMINATED)])
    await harness.agent().run('post')
    assert len(harness.executed) == 1

    guard = VocabularyGuard(WATCHLIST, tools=lambda name: name.startswith('send'))
    harness = Harness(guard, [edit(CONTAMINATED)])
    await harness.agent().run('write')
    assert len(harness.executed) == 1


async def test_instructions_list_top_terms() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), [])
    await harness.agent().run('write')
    instructions = harness.requests[0].instructions or ''
    assert '- delve' in instructions
    assert '- leverage (prefer: use)' in instructions

    harness = Harness(VocabularyGuard(WATCHLIST, instruct=False), [])
    await harness.agent().run('write')
    assert not harness.requests[0].instructions


def test_misconfiguration_raises_at_construction() -> None:
    with pytest.raises(UserError, match='mode'):
        VocabularyGuard(WATCHLIST, mode='block')  # type: ignore[arg-type]
    with pytest.raises(UserError, match='tools'):
        VocabularyGuard(WATCHLIST, tools=())
    with pytest.raises(UserError, match='threshold'):
        VocabularyGuard(WATCHLIST, threshold=-0.1)
    with pytest.raises(UserError, match='min_tokens'):
        VocabularyGuard(WATCHLIST, min_tokens=-1)
