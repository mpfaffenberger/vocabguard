"""Targets accept references where Python allows them, and check names at construction where it does not."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

from vocabguard import OutputField, ToolArgument


class Ticket(BaseModel):
    summary: str
    priority: int


@dataclass
class Note:
    body: str
    tags: list[str]


def edit_file(path: str, content: str, previous: str | None = None) -> str:
    return path + content + (previous or '')


def bump(count: int) -> int:
    return count + 1


def test_output_field_from_a_selector_resolves_the_name() -> None:
    assert OutputField(Ticket, lambda t: t.summary).field == 'summary'
    assert OutputField(Note, lambda n: n.body).field == 'body'
    assert OutputField(Ticket, 'summary').field == 'summary'
    assert OutputField(Ticket, lambda t: t.summary).source == 'Ticket.summary'


def test_output_field_selector_mistakes_are_user_errors() -> None:
    # Written with getattr so pyright cannot catch it here; in real code `t.sumary` is a type error.
    with pytest.raises(UserError, match="no field named 'sumary'"):
        OutputField(Ticket, lambda t: getattr(t, 'sumary'))  # noqa: B009
    with pytest.raises(UserError, match='must only read one attribute'):
        OutputField(Ticket, lambda t: t.summary.upper())
    with pytest.raises(UserError, match='exactly one top-level attribute, got nothing'):
        OutputField(Ticket, lambda t: t)
    with pytest.raises(UserError, match='must only read one attribute'):
        OutputField(Ticket, lambda t: t.summary + 'x')


def test_output_field_replaces_models_and_dataclasses() -> None:
    ticket = OutputField(Ticket, lambda t: t.summary).replace(Ticket(summary='a', priority=1), 'b')
    assert ticket == Ticket(summary='b', priority=1)
    note = OutputField(Note, lambda n: n.body).replace(Note(body='a', tags=['x']), 'b')
    assert note == Note(body='b', tags=['x'])


def test_tool_argument_from_a_function_checks_the_signature() -> None:
    target = ToolArgument(edit_file, 'content')
    assert target.name == 'edit_file'
    assert target.source == 'edit_file.content'
    assert ToolArgument(edit_file, 'previous').name == 'edit_file'
    assert ToolArgument('mcp_write', 'text').name == 'mcp_write'
    with pytest.raises(UserError, match="no parameter named 'body'"):
        ToolArgument(edit_file, 'body')
    with pytest.raises(UserError, match='not str'):
        ToolArgument(bump, 'count')
