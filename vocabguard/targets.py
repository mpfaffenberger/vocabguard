"""Where the guard looks: a tool argument, a field of a structured output, or the final text."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TypeAlias

from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

__all__ = ('OutputField', 'Target', 'TextOutput', 'ToolArgument')


@dataclass(frozen=True)
class ToolArgument:
    """A string argument of a tool call, scored before the tool runs. `ToolArgument('edit_file', 'content')`."""

    tool: str
    argument: str

    @property
    def source(self) -> str:
        return f'{self.tool}.{self.argument}'


@dataclass(frozen=True)
class OutputField:
    """A string field of a structured output, scored after validation. `OutputField(CaseTicket, 'summary')`."""

    type: type[object]
    field: str

    def __post_init__(self) -> None:
        if self.field not in _field_names(self.type):
            raise UserError(f'{self.type.__name__} has no field named {self.field!r}')

    @property
    def source(self) -> str:
        return f'{self.type.__name__}.{self.field}'

    def matches(self, output: object) -> bool:
        return isinstance(output, self.type)

    def get(self, output: object) -> object:
        return getattr(output, self.field)

    def replace(self, output: object, value: str) -> object:
        """A copy of `output` with the field swapped; the original is left alone."""
        if isinstance(output, BaseModel):
            return output.model_copy(update={self.field: value})
        if dataclasses.is_dataclass(output) and not isinstance(output, type):
            return dataclasses.replace(output, **{self.field: value})
        raise UserError(f'Cannot replace {self.source} on {type(output).__name__}: not a pydantic model or dataclass')


@dataclass(frozen=True)
class TextOutput:
    """The final text output of the run."""

    @property
    def source(self) -> str:
        return 'output'


Target: TypeAlias = ToolArgument | OutputField | TextOutput


def _field_names(cls: type[object]) -> set[str]:
    if issubclass(cls, BaseModel):
        return set(cls.model_fields)
    if dataclasses.is_dataclass(cls):
        return {field.name for field in dataclasses.fields(cls)}
    raise UserError(f'OutputField needs a pydantic model or a dataclass, not {cls.__name__}')
