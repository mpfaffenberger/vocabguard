"""Where the guard looks: a tool argument, a field of a structured output, or the final text."""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias, TypeVar

from pydantic import BaseModel
from pydantic_ai.exceptions import UserError

__all__ = ('OutputField', 'Target', 'TextOutput', 'ToolArgument')

OutputT = TypeVar('OutputT')


@dataclass(frozen=True)
class ToolArgument:
    """A string argument of a tool call, scored before the tool runs.

    `ToolArgument(edit_file, 'content')` checks the argument against the function's signature at
    construction; `ToolArgument('edit_file', 'content')` is for tools with no Python function behind them.
    """

    tool: str | Callable[..., object]
    argument: str
    name: str = dataclasses.field(init=False)
    """The tool name the guard matches against."""

    def __post_init__(self) -> None:
        if isinstance(self.tool, str):
            name = self.tool
        else:
            name = self.tool.__name__
            _check_parameter(self.tool, self.argument)
        object.__setattr__(self, 'name', name)

    @property
    def source(self) -> str:
        return f'{self.name}.{self.argument}'


@dataclass(frozen=True, init=False)
class OutputField:
    """A string field of a structured output, scored after validation.

    `OutputField(CaseTicket, lambda t: t.summary)` lets the type checker verify the field exists;
    `OutputField(CaseTicket, 'summary')` is checked at construction instead.
    """

    type: type[object]
    field: str
    """The resolved field name."""

    def __init__(self, type: type[OutputT], selector: str | Callable[[OutputT], object]) -> None:
        # The type variable lives on the constructor, not the class, so a list of mixed targets does
        # not force the selector's parameter to `object` and lose the attribute check.
        name = selector if isinstance(selector, str) else _selected_attribute(selector, type)
        if name not in _field_names(type):
            raise UserError(f'{type.__name__} has no field named {name!r}')
        object.__setattr__(self, 'type', type)
        object.__setattr__(self, 'field', name)

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


class _AttributeProbe:
    """Stands in for the output so a selector lambda reveals which attribute it reads."""

    def __init__(self) -> None:
        self.path: list[str] = []

    def __getattr__(self, name: str) -> _AttributeProbe:
        self.path.append(name)
        return self


def _selected_attribute(selector: Callable[[OutputT], object], cls: type[OutputT]) -> str:
    probe = _AttributeProbe()
    try:
        selector(typing.cast(OutputT, probe))
    except Exception as error:  # a selector that does more than read one attribute
        raise UserError(f'OutputField selector for {cls.__name__} must only read one attribute: {error}') from error
    if len(probe.path) != 1:
        raise UserError(
            f'OutputField selector for {cls.__name__} must read exactly one top-level attribute, '
            f'got {".".join(probe.path) or "nothing"}'
        )
    return probe.path[0]


def _field_names(cls: type[object]) -> set[str]:
    if issubclass(cls, BaseModel):
        return set(cls.model_fields)
    if dataclasses.is_dataclass(cls):
        return {field.name for field in dataclasses.fields(cls)}
    raise UserError(f'OutputField needs a pydantic model or a dataclass, not {cls.__name__}')


def _check_parameter(function: Callable[..., object], argument: str) -> None:
    parameters = inspect.signature(function).parameters
    if argument not in parameters:
        raise UserError(f'{function.__name__}() has no parameter named {argument!r}')
    try:
        hint = typing.get_type_hints(function).get(argument)
    except (NameError, TypeError):  # forward references we cannot resolve; the runtime isinstance check remains
        return
    if hint is not None and hint is not str and str not in typing.get_args(hint):
        raise UserError(f'{function.__name__}() parameter {argument!r} is {hint!r}, not str')
