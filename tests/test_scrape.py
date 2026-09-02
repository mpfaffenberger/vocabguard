from __future__ import annotations

import io
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from vocabguard.display import PlainDisplay
from vocabguard.scrape import ScrapeDeps, build_agent, chat

pytestmark = pytest.mark.anyio

ARTICLE = ' '.join(f'word{i}' for i in range(80))


def fake_web(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if 'wikipedia.org' in url and 'list=search' in url:
        return httpx.Response(200, json={'query': {'search': [{'title': 'Aviation'}, {'title': 'Airfoil'}]}})
    if 'wikipedia.org' in url and 'prop=extracts' in url:
        return httpx.Response(200, json={'query': {'pages': {'1': {'extract': ARTICLE}}}})
    if url.startswith('https://api.github.com/search/repositories'):
        assert 'pushed%3A%3C2026-01-01' in url or 'pushed:<2026-01-01' in url
        return httpx.Response(200, json={'items': [{'full_name': 'octo/parser'}]})
    if url == 'https://api.github.com/repos/octo/parser/readme':
        assert request.headers['accept'] == 'application/vnd.github.raw+json'
        return httpx.Response(200, text='# parser\n\n' + ARTICLE)
    if url == 'https://example.com/page':
        html = f'<html><head><style>p{{}}</style></head><body><nav>menu</nav><p>{ARTICLE}</p></body></html>'
        return httpx.Response(200, text=html, headers={'content-type': 'text/html'})
    return httpx.Response(404, text='nope')


def scripted(responses: list[ModelResponse]) -> FunctionModel:
    """A model that plays back responses in order, streamed, then says done."""
    queue = list(responses)

    def next_response() -> ModelResponse:
        return queue.pop(0) if queue else ModelResponse(parts=[TextPart('done')])

    def function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return next_response()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(next_response().parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    return FunctionModel(function, stream_function=stream)


def run_calls(tmp_path: Path, calls: list[tuple[str, dict[str, object]]]) -> tuple[list[ToolReturnPart], ScrapeDeps]:
    """Drive the agent through a scripted list of tool calls and collect what the tools returned."""
    model = scripted([ModelResponse(parts=[ToolCallPart(name, args)]) for name, args in calls])
    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path / 'corpus', client=client, target=2)
        result = build_agent(model).run_sync('go', deps=deps)
    returns = [part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)]
    return returns, deps


def test_wikipedia_tools(tmp_path: Path) -> None:
    returns, deps = run_calls(
        tmp_path,
        [
            ('wikipedia_search', {'query': 'aviation'}),
            ('wikipedia_article', {'title': 'Aviation'}),
            ('save_document', {'name': 'Aviation', 'text': ARTICLE}),
            ('corpus_status', {}),
        ],
    )
    assert returns[0].content == ['Aviation', 'Airfoil']
    assert returns[1].content == ARTICLE
    assert returns[2].content == 'Saved Aviation.md. 1 of 2 documents so far.'
    assert returns[3].content == '1 documents, 80 words, target 2 documents.'
    assert (deps.output / 'Aviation.md').read_text() == ARTICLE


def test_github_tools(tmp_path: Path) -> None:
    returns, _ = run_calls(
        tmp_path,
        [
            ('github_search_repositories', {'query': 'language:python pushed:<2026-01-01', 'limit': 5}),
            ('github_readme', {'repository': 'octo/parser'}),
        ],
    )
    assert returns[0].content == ['octo/parser']
    assert str(returns[1].content).startswith('# parser')


def test_fetch_page_strips_markup(tmp_path: Path) -> None:
    returns, _ = run_calls(tmp_path, [('fetch_page', {'url': 'https://example.com/page'})])
    text = str(returns[0].content)
    assert text.startswith('word0 word1')
    assert 'menu' not in text
    assert '<p>' not in text


def test_save_document_sanitizes_names_and_rejects_short_text(tmp_path: Path) -> None:
    returns, deps = run_calls(
        tmp_path,
        [
            ('save_document', {'name': '../evil/name: with spaces', 'text': ARTICLE}),
            ('save_document', {'name': '../evil/name: with spaces', 'text': ARTICLE}),
            ('save_document', {'name': 'tiny', 'text': 'too short'}),
        ],
    )
    assert sorted(path.name for path in deps.saved) == ['evil-name-with-spaces-2.md', 'evil-name-with-spaces.md']
    assert all(path.parent == deps.output for path in deps.saved)
    # The rejected save shows up as a retry prompt, not a tool return.
    assert len(returns) == 2


def test_http_errors_become_retries(tmp_path: Path) -> None:
    returns, _ = run_calls(
        tmp_path, [('github_readme', {'repository': 'octo/missing'}), ('fetch_page', {'url': 'ftp://x'})]
    )
    assert returns == []


class ScriptedDisplay(PlainDisplay):
    """Answers from a script instead of stdin; records what was shown."""

    def __init__(self, answers: list[str]) -> None:
        super().__init__(io.StringIO())
        self.answers = iter(answers)
        self.turns: list[str] = []
        self.activities: list[str] = []

    def begin_turn(self) -> None:
        self.turns.append('')

    def text(self, delta: str) -> None:
        self.turns[-1] += delta

    def activity(self, message: str) -> None:
        self.activities.append(message)

    def read(self, prompt: str) -> str:
        try:
            return next(self.answers)
        except StopIteration:
            raise EOFError from None


async def test_chat_loop_streams_text_and_tool_calls_then_stops_on_quit(tmp_path: Path) -> None:
    model = scripted(
        [
            ModelResponse(parts=[TextPart('What should the corpus represent?')]),
            ModelResponse(parts=[TextPart('Checking. '), ToolCallPart('corpus_status', {})]),
            ModelResponse(parts=[TextPart('Empty so far.')]),
        ]
    )
    display = ScriptedDisplay(['Wikipedia articles about aviation', 'quit'])
    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path, client=client)
        await chat(build_agent(model), deps, display)
    # Turn two shows the text written next to the tool call, the call itself, and the follow-up.
    assert display.turns == ['What should the corpus represent?', 'Checking. Empty so far.']
    assert display.activities == ['corpus_status {}']


async def test_chat_loop_stops_on_eof(tmp_path: Path) -> None:
    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path, client=client)
        await chat(build_agent(scripted([])), deps, ScriptedDisplay([]))
