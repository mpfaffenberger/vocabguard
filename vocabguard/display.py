"""How a scrape session shows model text and tool activity. termflow when available, plain print otherwise."""

from __future__ import annotations

import importlib
import sys
from typing import IO, TYPE_CHECKING, Protocol, TextIO, cast

if TYPE_CHECKING:
    from termflow import Parser, Renderer
    from termflow.stream import SmoothWriter

__all__ = ('Display', 'PlainDisplay', 'TermflowDisplay', 'make_display')


class Display(Protocol):
    """One model turn is `begin_turn`, any number of `text` and `activity` calls, then `end_turn`."""

    def begin_turn(self) -> None: ...
    def text(self, delta: str) -> None: ...
    def activity(self, message: str) -> None: ...
    async def end_turn(self) -> None: ...
    def read(self, prompt: str) -> str: ...


class PlainDisplay(Display):
    """No rendering. Used when termflow is not installed, when output is piped, and in tests."""

    def __init__(self, output: IO[str] = sys.stdout) -> None:
        self.output = output

    def begin_turn(self) -> None:
        pass

    def text(self, delta: str) -> None:
        self.output.write(delta)
        self.output.flush()

    def activity(self, message: str) -> None:
        self.output.write(f'\n  [{message}]\n')
        self.output.flush()

    async def end_turn(self) -> None:
        self.output.write('\n')
        self.output.flush()

    def read(self, prompt: str) -> str:
        return input(prompt)


class TermflowDisplay(Display):
    """Streams markdown through termflow's parser and renderer with smooth output pacing.

    The pacer is one-shot, so each turn gets a fresh writer, renderer, and parser. The parser is
    line based, so deltas are buffered until a newline arrives.
    """

    def __init__(self, output: IO[str] = sys.stdout, *, width: int | None = None) -> None:
        # Fail at construction, not mid-session, when the scrape extra is missing.
        importlib.import_module('termflow')

        self._output = output
        self._width = width
        self._writer: SmoothWriter | None = None
        self._renderer: Renderer | None = None
        self._parser: Parser | None = None
        self._pending = ''

    def begin_turn(self) -> None:
        from termflow import Parser, Renderer
        from termflow.stream import SmoothWriter

        self._writer = SmoothWriter(self._output)
        self._writer.start()
        # SmoothWriter is file-like by contract but termflow annotates the renderer's output as TextIO.
        self._renderer = Renderer(output=cast(TextIO, self._writer), width=self._width)
        self._parser = Parser()
        self._pending = ''

    def text(self, delta: str) -> None:
        self._pending += delta
        *lines, self._pending = self._pending.split('\n')
        for line in lines:
            self._render_line(line)

    def activity(self, message: str) -> None:
        from termflow.ansi.codes import DIM_ON, RESET

        self._flush_pending()
        assert self._writer is not None, 'activity() outside a turn'
        self._writer.write(f'{DIM_ON}  [{message}]{RESET}\n')

    async def end_turn(self) -> None:
        self._flush_pending()
        assert self._renderer is not None and self._parser is not None and self._writer is not None
        self._renderer.render_all(self._parser.finalize())
        self._writer.write('\n')
        await self._writer.close()
        self._writer = self._renderer = self._parser = None

    def read(self, prompt: str) -> str:
        return input(prompt)

    def _flush_pending(self) -> None:
        if self._pending:
            self._render_line(self._pending)
            self._pending = ''

    def _render_line(self, line: str) -> None:
        assert self._renderer is not None and self._parser is not None, 'text() outside a turn'
        self._renderer.render_all(self._parser.parse_line(line))


def make_display() -> Display:
    """termflow on a terminal; plain text when piped or when the scrape extra is not installed."""
    if not sys.stdout.isatty():
        return PlainDisplay()
    try:
        return TermflowDisplay()
    except ImportError:
        return PlainDisplay()
