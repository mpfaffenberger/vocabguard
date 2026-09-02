"""OpenRouter API keys via the PKCE browser flow (https://openrouter.ai/docs/use-cases/oauth-pkce).

The flow has no client id, scopes, or refresh tokens: the browser lands on a localhost callback
with a one-time code, and the code plus PKCE verifier are exchanged for a user-controlled key.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import urllib.parse
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import httpx
from pydantic_ai.exceptions import UserError

__all__ = ('ENV_VAR', 'KEY_FILE', 'api_key', 'exchange_code', 'login')

AUTH_URL = 'https://openrouter.ai/auth'
KEYS_URL = 'https://openrouter.ai/api/v1/auth/keys'
ENV_VAR = 'OPENROUTER_API_KEY'
KEY_FILE = Path.home() / '.config' / 'vocabguard' / 'openrouter_key'
# Authorization codes expire after ten minutes; leave most of that window for the user.
CALLBACK_TIMEOUT = 540.0

Exchange = Callable[[str, str], str]


def api_key(*, key_file: Path = KEY_FILE, login_flow: Callable[[], str] | None = None) -> str:
    """Environment variable, then the saved key file, then a browser login that saves the key."""
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return from_env
    if key_file.is_file():
        saved = key_file.read_text(encoding='utf-8').strip()
        if saved:
            return saved
    key = (login_flow or login)()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text(key + '\n', encoding='utf-8')
    key_file.chmod(0o600)
    return key


def exchange_code(code: str, verifier: str, *, client: httpx.Client | None = None) -> str:
    payload = {'code': code, 'code_verifier': verifier, 'code_challenge_method': 'S256'}
    response = client.post(KEYS_URL, json=payload) if client else httpx.post(KEYS_URL, json=payload, timeout=30)
    if response.status_code != 200:
        raise UserError(f'OpenRouter key exchange failed with HTTP {response.status_code}: {response.text[:200]}')
    key = response.json().get('key')
    if not isinstance(key, str) or not key:
        raise UserError('OpenRouter key exchange returned no key')
    return key


def login(
    *,
    open_url: Callable[[str], object] = webbrowser.open,
    exchange: Exchange = exchange_code,
    timeout: float = CALLBACK_TIMEOUT,
) -> str:
    """Run the browser flow and return the new key. Raises `UserError` on denial or timeout."""
    server = _CallbackServer(exchange)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print(f'Opening OpenRouter in your browser. If nothing happens, open this URL:\n{server.auth_url}')
        open_url(server.auth_url)
        if not server.done.wait(timeout):
            raise UserError('Timed out waiting for the OpenRouter callback')
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    if server.key is None:
        raise UserError(server.error or 'OpenRouter authorization did not complete')
    return server.key


class _CallbackServer(HTTPServer):
    def __init__(self, exchange: Exchange) -> None:
        super().__init__(('localhost', 0), _CallbackHandler)
        self.exchange = exchange
        self.verifier = secrets.token_urlsafe(64)
        digest = hashlib.sha256(self.verifier.encode('ascii')).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
        callback = f'http://localhost:{self.server_address[1]}/callback'
        query = urllib.parse.urlencode(
            {'callback_url': callback, 'code_challenge': challenge, 'code_challenge_method': 'S256'}
        )
        self.auth_url = f'{AUTH_URL}?{query}'
        self.key: str | None = None
        self.error: str | None = None
        self.done = threading.Event()


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        # http.server types `self.server` as the base class; this handler only ever runs on ours.
        server = self.server
        assert isinstance(server, _CallbackServer)
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if parsed.path != '/callback':
            self._reply(404, 'Not the OpenRouter callback path.')
            return
        code = params.get('code', [''])[0]
        if not code:
            server.error = params.get('error_description', params.get('error', ['no code in callback']))[0]
            self._reply(400, f'OpenRouter authorization failed: {server.error}')
        else:
            try:
                server.key = server.exchange(code, server.verifier)
            except UserError as error:
                server.error = str(error)
                self._reply(500, server.error)
            else:
                self._reply(200, 'OpenRouter key saved. You can close this tab and go back to vocabguard.')
        server.done.set()

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _reply(self, status: int, message: str) -> None:
        body = message.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
