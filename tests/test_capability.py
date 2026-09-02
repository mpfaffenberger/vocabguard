from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
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


@dataclass
class Harness:
    """Scripted model plus a recording tool, so a test declares the tool calls and inspects what ran."""

    guard: VocabularyGuard[object]
    calls: list[dict[str, str]]
    executed: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    requests: list[ModelRequest] = field(default_factory=list[ModelRequest])

    def model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Each call carries the full history, so the last snapshot is the complete one.
        self.requests = [message for message in messages if isinstance(message, ModelRequest)]
        if self.calls:
            return ModelResponse(parts=[ToolCallPart('edit_file', self.calls.pop(0))])
        return ModelResponse(parts=[TextPart('done')])

    def agent(self) -> Agent[object, str]:
        agent: Agent[object, str] = Agent(FunctionModel(self.model), capabilities=[self.guard], retries=3)

        @agent.tool_plain
        def edit_file(path: str, content: str, old_string: str | None = None) -> str:
            self.executed.append({'path': path, 'content': content})
            return 'ok'

        return agent

    def retries(self) -> list[RetryPromptPart]:
        return [part for request in self.requests for part in request.parts if isinstance(part, RetryPromptPart)]


async def test_retry_round_trip() -> None:
    harness = Harness(
        VocabularyGuard(WATCHLIST),
        calls=[{'path': 'notes.md', 'content': CONTAMINATED}, {'path': 'notes.md', 'content': CLEAN}],
    )
    result = await harness.agent().run('write')
    assert result.output == 'done'
    assert harness.executed == [{'path': 'notes.md', 'content': CLEAN}]
    (retry,) = harness.retries()
    assert retry.tool_name == 'edit_file'
    assert "'leverage' (z=4.0, x2, prefer: use)" in retry.model_response()
    assert 'resubmit the same edit_file call' in retry.model_response()


async def test_clean_write_passes() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), calls=[{'path': 'notes.md', 'content': CLEAN}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_edit_to_contaminated_file_scores_only_added_lines() -> None:
    harness = Harness(
        VocabularyGuard(WATCHLIST),
        calls=[{'path': 'notes.md', 'old_string': CONTAMINATED, 'content': f'{CONTAMINATED}\n\n{CLEAN}'}],
    )
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_python_identifiers_are_not_prose() -> None:
    source = 'def leverage_robust(delve: int) -> int:\n    return delve + leverage_robust(delve - 1) if delve else 0\n'
    harness = Harness(VocabularyGuard(WATCHLIST, min_tokens=0), calls=[{'path': 'mod.py', 'content': source}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []


async def test_python_docstrings_are_prose() -> None:
    source = f'def f() -> None:\n    """{CONTAMINATED}"""\n'
    harness = Harness(VocabularyGuard(WATCHLIST), calls=[{'path': 'mod.py', 'content': source}])
    await harness.agent().run('write')
    assert harness.executed == []
    assert len(harness.retries()) == 1


async def test_warn_mode_calls_on_hit_and_still_executes() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, mode='warn', on_hit=hits.append)
    harness = Harness(guard, calls=[{'path': 'notes.md', 'content': CONTAMINATED}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1
    assert harness.retries() == []
    (hit,) = hits
    assert hit.path == 'notes.md'
    assert {term.term for term in hit.terms} == {'leverage', 'robust', 'delve'}


async def test_retry_mode_also_calls_on_hit() -> None:
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, on_hit=hits.append)
    harness = Harness(
        guard, calls=[{'path': 'notes.md', 'content': CONTAMINATED}, {'path': 'notes.md', 'content': CLEAN}]
    )
    await harness.agent().run('write')
    assert len(hits) == 1


async def test_banned_pattern_fires_below_min_tokens() -> None:
    harness = Harness(
        VocabularyGuard(WATCHLIST),
        calls=[{'path': 'notes.md', 'content': 'short \u2014 text'}, {'path': 'notes.md', 'content': 'short text'}],
    )
    await harness.agent().run('write')
    assert harness.executed == [{'path': 'notes.md', 'content': 'short text'}]
    (retry,) = harness.retries()
    assert 'banned pattern' in retry.model_response()


async def test_threshold_lets_low_scores_through() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, threshold=10.0)
    harness = Harness(guard, calls=[{'path': 'notes.md', 'content': CONTAMINATED}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1


async def test_unhandled_extension_and_unwatched_tool_are_ignored() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), calls=[{'path': 'data.json', 'content': CONTAMINATED}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1

    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, tools=lambda name: name.startswith('write'))
    harness = Harness(guard, calls=[{'path': 'notes.md', 'content': CONTAMINATED}])
    await harness.agent().run('write')
    assert len(harness.executed) == 1


async def test_instructions_list_top_terms() -> None:
    harness = Harness(VocabularyGuard(WATCHLIST), calls=[])
    await harness.agent().run('write')
    instructions = harness.requests[0].instructions or ''
    assert '- delve' in instructions
    assert '- leverage (prefer: use)' in instructions

    harness = Harness(VocabularyGuard(WATCHLIST, instruct=False), calls=[])
    await harness.agent().run('write')
    assert not harness.requests[0].instructions


def test_misconfiguration_raises_at_construction() -> None:
    with pytest.raises(UserError, match='mode'):
        VocabularyGuard(WATCHLIST, mode='block')  # type: ignore[arg-type]
    with pytest.raises(UserError, match='tool name'):
        VocabularyGuard(WATCHLIST, tools=())
    with pytest.raises(UserError, match='threshold'):
        VocabularyGuard(WATCHLIST, threshold=-0.1)
    with pytest.raises(UserError, match='min_tokens'):
        VocabularyGuard(WATCHLIST, min_tokens=-1)
