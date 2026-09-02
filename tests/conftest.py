from __future__ import annotations

import asyncio
import warnings
from collections.abc import Iterator

import pytest
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(autouse=True)
def close_leftover_event_loop() -> Iterator[None]:
    # Agent.run_sync installs an event loop and leaves it set; a later asyncio.run() drops that
    # reference and the loop is collected unclosed, which filterwarnings=error turns into a failure.
    yield
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return
    if not loop.is_closed() and not loop.is_running():
        loop.close()
