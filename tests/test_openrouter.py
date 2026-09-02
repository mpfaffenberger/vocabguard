from __future__ import annotations

import threading
import urllib.parse
from pathlib import Path

import httpx
import pytest
from pydantic_ai.exceptions import UserError

from vocabguard import openrouter


def callback_url_of(auth_url: str) -> str:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)
    assert query['code_challenge_method'] == ['S256']
    assert len(query['code_challenge'][0]) == 43
    return query['callback_url'][0]


def hit_callback(auth_url: str, query: str) -> None:
    """Stand in for the browser: follow the redirect OpenRouter would send, on a thread like a browser would."""
    threading.Thread(target=lambda: httpx.get(f'{callback_url_of(auth_url)}?{query}'), daemon=True).start()


def test_login_exchanges_the_code_for_a_key() -> None:
    seen: list[tuple[str, str]] = []

    def exchange(code: str, verifier: str) -> str:
        seen.append((code, verifier))
        return 'sk-or-test'

    key = openrouter.login(open_url=lambda url: hit_callback(url, 'code=abc123'), exchange=exchange, timeout=5)
    assert key == 'sk-or-test'
    (call,) = seen
    assert call[0] == 'abc123'
    assert len(call[1]) > 40


def test_login_reports_denial() -> None:
    with pytest.raises(UserError, match='user said no'):
        openrouter.login(
            open_url=lambda url: hit_callback(url, 'error=access_denied&error_description=user+said+no'),
            exchange=lambda code, verifier: 'unreachable',
            timeout=5,
        )


def test_login_times_out_without_a_callback() -> None:
    with pytest.raises(UserError, match='Timed out'):
        openrouter.login(open_url=lambda url: None, exchange=lambda code, verifier: 'unreachable', timeout=0.2)


def test_exchange_code_parses_the_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == openrouter.KEYS_URL
        assert b'"code_challenge_method":"S256"' in request.content
        return httpx.Response(200, json={'key': 'sk-or-live'})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        assert openrouter.exchange_code('code', 'verifier', client=client) == 'sk-or-live'

    denied = httpx.MockTransport(lambda request: httpx.Response(403, text='denied'))
    with httpx.Client(transport=denied) as client, pytest.raises(UserError, match='HTTP 403'):
        openrouter.exchange_code('code', 'verifier', client=client)


def test_api_key_prefers_env_then_file_then_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key_file = tmp_path / 'nested' / 'openrouter_key'
    monkeypatch.setenv(openrouter.ENV_VAR, 'from-env')
    assert openrouter.api_key(key_file=key_file, login_flow=lambda: 'from-login') == 'from-env'

    monkeypatch.delenv(openrouter.ENV_VAR)
    assert openrouter.api_key(key_file=key_file, login_flow=lambda: 'from-login') == 'from-login'
    assert key_file.read_text().strip() == 'from-login'
    assert key_file.stat().st_mode & 0o777 == 0o600

    assert openrouter.api_key(key_file=key_file, login_flow=lambda: 'should-not-run') == 'from-login'
