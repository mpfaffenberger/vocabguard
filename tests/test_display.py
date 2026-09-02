from __future__ import annotations

import io

import pytest

from vocabguard.display import PlainDisplay, TermflowDisplay

pytestmark = pytest.mark.anyio


async def test_plain_display_writes_deltas_and_activity() -> None:
    output = io.StringIO()
    display = PlainDisplay(output)
    display.begin_turn()
    display.text('Hel')
    display.text('lo')
    display.activity('fetch_page {"url": "x"}')
    await display.end_turn()
    assert output.getvalue() == 'Hello\n  [fetch_page {"url": "x"}]\n\n'


async def test_termflow_display_renders_markdown_across_line_splits() -> None:
    output = io.StringIO()
    display = TermflowDisplay(output, width=60)
    display.begin_turn()
    display.text('# Plan\n\nFetch **fifty** arti')
    display.text('cles.\n- one\n- two')
    display.activity('wikipedia_search {"query": "aviation"}')
    display.text('Saved.')
    await display.end_turn()
    rendered = output.getvalue()
    assert 'Plan' in rendered
    assert 'fifty' in rendered and '**' not in rendered
    assert 'wikipedia_search' in rendered
    assert rendered.index('two') < rendered.index('wikipedia_search') < rendered.index('Saved.')
    assert '\x1b[' in rendered

    # A second turn starts clean rather than continuing the previous list.
    display.begin_turn()
    display.text('plain')
    await display.end_turn()
    assert output.getvalue().endswith('plain\n\n')
