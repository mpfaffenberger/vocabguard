from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
from pydantic_ai import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from vocabguard.scrape import ScrapeDeps, build_agent, chat

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


def run_calls(tmp_path: Path, calls: list[tuple[str, dict[str, object]]]) -> tuple[list[ToolReturnPart], ScrapeDeps]:
    """Drive the agent through a scripted list of tool calls and collect what the tools returned."""
    queue = list(calls)

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if queue:
            name, args = queue.pop(0)
            return ModelResponse(parts=[ToolCallPart(name, args)])
        return ModelResponse(parts=[TextPart('done')])

    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path / 'corpus', client=client, target=2)
        result = build_agent(FunctionModel(model)).run_sync('go', deps=deps)
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


def test_chat_loop_opens_with_a_question_and_stops_on_quit(tmp_path: Path) -> None:
    turns: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        turns.append(json.dumps(len(messages)))
        return ModelResponse(parts=[TextPart(f'turn {len(turns)}')])

    answers: Iterator[str] = iter(['Wikipedia articles about aviation', 'quit'])
    written: list[str] = []
    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path, client=client)
        chat(build_agent(FunctionModel(model)), deps, read=lambda prompt: next(answers), write=written.append)
    assert written == ['turn 1', 'turn 2']
    # The second call carries the whole conversation, not a fresh one.
    assert turns == ['1', '3']


def test_chat_loop_stops_on_eof(tmp_path: Path) -> None:
    def read(prompt: str) -> str:
        raise EOFError

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('hello')])

    with httpx.Client(transport=httpx.MockTransport(fake_web)) as client:
        deps = ScrapeDeps(output=tmp_path, client=client)
        chat(build_agent(FunctionModel(model)), deps, read=read, write=lambda text: None)
