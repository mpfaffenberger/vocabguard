"""Targets name where the guard looks; a rewriter fixes what fires there instead of retrying."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, FunctionModel

from vocabguard import HitReport, OutputField, TextOutput, ToolArgument, VocabularyGuard, Watchlist

pytestmark = pytest.mark.anyio

WATCHLIST = Watchlist.from_parts(
    terms={'leverage': 4.0, 'robust': 3.0, 'delve': 5.0}, replacements={'leverage': 'use'}, banned_patterns=[]
)
CONTAMINATED = (
    'We leverage a robust pipeline here, and this section will delve into why the robust design '
    'lets us leverage every stage without extra work from the caller or the reader.'
)
CLEAN = (
    'The parser reads each line, splits it on tabs, and hands the fields to the renderer, which '
    'writes one row per record and skips blank lines without complaint.'
)


class Ticket(BaseModel):
    summary: str
    priority: int


@dataclass
class Rewriter:
    """A scripted second agent that records what it was asked and answers with fixed text."""

    reply: str = CLEAN
    prompts: list[str] = field(default_factory=list[str])

    def agent(self) -> Agent[None, str]:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for part in messages[-1].parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    self.prompts.append(part.content)
            return ModelResponse(parts=[TextPart(self.reply)])

        return Agent(FunctionModel(model), output_type=str)


@dataclass
class Primary:
    """A scripted primary model plus a recording edit tool."""

    responses: list[ModelResponse]
    written: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    retries: list[RetryPromptPart] = field(default_factory=list[RetryPromptPart])

    def model(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.retries = [p for m in messages for p in m.parts if isinstance(p, RetryPromptPart)]
        return self.responses.pop(0) if self.responses else ModelResponse(parts=[TextPart('done')])

    def edit_tool(self, agent: Agent[object, object]) -> None:
        @agent.tool_plain
        def edit_file(path: str, content: str) -> str:
            self.written.append({'path': path, 'content': content})
            return 'ok'


def edit(content: str, path: str = 'notes.md') -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('edit_file', {'path': path, 'content': content})])


def ticket(summary: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('final_result', {'summary': summary, 'priority': 2})])


async def test_tool_argument_is_rewritten_before_the_write_lands() -> None:
    rewriter = Rewriter()
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(
        WATCHLIST, targets=[ToolArgument('edit_file', 'content')], rewriter=rewriter.agent(), on_hit=hits.append
    )
    primary = Primary([edit(CONTAMINATED)])
    agent: Agent[object, str] = Agent(FunctionModel(primary.model), capabilities=[guard])
    primary.edit_tool(agent)
    await agent.run('write')
    assert primary.written == [{'path': 'notes.md', 'content': CLEAN}]
    assert primary.retries == []
    (prompt,) = rewriter.prompts
    assert 'leverage (prefer: use)' in prompt and CONTAMINATED in prompt
    assert [hit.source for hit in hits] == ['edit_file.content (notes.md)']


async def test_output_field_is_rewritten_and_the_rest_untouched() -> None:
    rewriter = Rewriter()
    guard: VocabularyGuard[object] = VocabularyGuard(
        WATCHLIST, targets=[OutputField(Ticket, 'summary')], rewriter=rewriter.agent()
    )
    primary = Primary([ticket(CONTAMINATED)])
    agent: Agent[object, Ticket] = Agent(FunctionModel(primary.model), output_type=Ticket, capabilities=[guard])
    result = await agent.run('summarize')
    assert result.output == Ticket(summary=CLEAN, priority=2)
    assert len(rewriter.prompts) == 1


async def test_text_output_is_rewritten() -> None:
    rewriter = Rewriter()
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, targets=[TextOutput()], rewriter=rewriter.agent())
    primary = Primary([ModelResponse(parts=[TextPart(CONTAMINATED)])])
    agent: Agent[object, str] = Agent(FunctionModel(primary.model), capabilities=[guard])
    result = await agent.run('explain')
    assert result.output == CLEAN


async def test_targets_without_a_rewriter_fall_back_to_retry() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, targets=[OutputField(Ticket, 'summary')])
    primary = Primary([ticket(CONTAMINATED), ticket(CLEAN)])
    agent: Agent[object, Ticket] = Agent(
        FunctionModel(primary.model), output_type=Ticket, capabilities=[guard], retries=2
    )
    result = await agent.run('summarize')
    assert result.output.summary == CLEAN
    (retry,) = primary.retries
    assert 'Ticket.summary: vocabulary score' in retry.model_response()
    assert 'Produce the output again' in retry.model_response()


async def test_untargeted_sites_are_ignored() -> None:
    guard: VocabularyGuard[object] = VocabularyGuard(WATCHLIST, targets=[OutputField(Ticket, 'summary')])
    primary = Primary([edit(CONTAMINATED)])
    agent: Agent[object, str] = Agent(FunctionModel(primary.model), capabilities=[guard])
    primary.edit_tool(agent)
    result = await agent.run('write')
    # The contaminated write went through and the contaminated final text was not scored either.
    assert primary.written == [{'path': 'notes.md', 'content': CONTAMINATED}]
    assert result.output == 'done'
    assert primary.retries == []


async def test_python_content_is_never_sent_to_the_rewriter() -> None:
    rewriter = Rewriter()
    guard: VocabularyGuard[object] = VocabularyGuard(
        WATCHLIST, targets=[ToolArgument('edit_file', 'content')], rewriter=rewriter.agent(), mode='warn'
    )
    source = f'def f() -> None:\n    """{CONTAMINATED}"""\n'
    primary = Primary([edit(source, path='mod.py')])
    agent: Agent[object, str] = Agent(FunctionModel(primary.model), capabilities=[guard])
    primary.edit_tool(agent)
    await agent.run('write')
    assert rewriter.prompts == []
    assert primary.written == [{'path': 'mod.py', 'content': source}]


async def test_a_rewrite_that_still_drifts_falls_back_to_mode() -> None:
    rewriter = Rewriter(reply=CONTAMINATED)
    hits: list[HitReport] = []
    guard: VocabularyGuard[object] = VocabularyGuard(
        WATCHLIST, targets=[TextOutput()], rewriter=rewriter.agent(), on_hit=hits.append
    )
    primary = Primary([ModelResponse(parts=[TextPart(CONTAMINATED)]), ModelResponse(parts=[TextPart(CLEAN)])])
    agent: Agent[object, str] = Agent(FunctionModel(primary.model), capabilities=[guard], retries=2)
    result = await agent.run('explain')
    assert result.output == CLEAN
    (retry,) = primary.retries
    assert 'output (rewritten): vocabulary score' in retry.model_response()
    assert [hit.source for hit in hits] == ['output', 'output (rewritten)']


def test_misconfiguration_is_a_user_error() -> None:
    with pytest.raises(UserError, match='rewriter needs targets'):
        VocabularyGuard(WATCHLIST, rewriter='test')
    with pytest.raises(UserError, match='at least one target'):
        VocabularyGuard(WATCHLIST, targets=[])
    with pytest.raises(UserError, match="no field named 'nope'"):
        OutputField(Ticket, 'nope')
    with pytest.raises(UserError, match='pydantic model or a dataclass'):
        OutputField(int, 'real')
