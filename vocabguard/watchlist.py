"""The watchlist file: watched n-grams with their z scores, curated replacements, and banned regexes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UserError

from .normalize import normalize_term

__all__ = ('CuratedFile', 'Watchlist')


class CuratedFile(BaseModel):
    """The hand-maintained half of a watchlist. `contrast` merges this in and never writes to it."""

    replacements: dict[str, str] = {}
    banned_patterns: list[str] = []

    @classmethod
    def load(cls, path: str | Path) -> CuratedFile:
        try:
            return cls.model_validate_json(Path(path).read_text(encoding='utf-8'))
        except ValidationError as error:
            raise UserError(f'Invalid curated file {path}: {error}') from error


class _WatchlistFile(CuratedFile):
    terms: dict[str, float]


@dataclass(kw_only=True)
class Watchlist:
    terms: dict[str, float]
    """Normalized n-gram to its log-odds z score toward the model corpus."""
    replacements: dict[str, str] = field(default_factory=dict[str, str])
    """Normalized n-gram to a preferred alternative. Hand-maintained; `contrast` merges, never overwrites."""
    banned_patterns: list[re.Pattern[str]] = field(default_factory=list[re.Pattern[str]])
    """Regexes that count as a hit regardless of score or token count."""
    labels: dict[str, str] = field(default_factory=dict[str, str])
    """Normalized n-gram to the spelling it was written with, so messages say `noting` rather than `note`."""

    def label(self, term: str) -> str:
        return self.labels.get(term, term)

    @classmethod
    def load(cls, path: str | Path) -> Watchlist:
        try:
            parsed = _WatchlistFile.model_validate_json(Path(path).read_text(encoding='utf-8'))
        except ValidationError as error:
            raise UserError(f'Invalid watchlist file {path}: {error}') from error
        return cls.from_parts(
            terms=parsed.terms, replacements=parsed.replacements, banned_patterns=parsed.banned_patterns
        )

    @classmethod
    def starter(cls) -> Watchlist:
        """The hand-curated list shipped with the package, for use before you have built corpora."""
        return cls.load(str(files('vocabguard.data').joinpath('starter_watchlist.json')))

    @classmethod
    def from_parts(
        cls, *, terms: dict[str, float], replacements: dict[str, str], banned_patterns: list[str]
    ) -> Watchlist:
        """Build from raw strings, normalizing keys so `noting` and `note` are the same watched term."""
        try:
            compiled = [re.compile(pattern) for pattern in banned_patterns]
        except re.error as error:
            raise UserError(f'Invalid banned pattern in watchlist: {error}') from error
        labels = {normalize_term(term): term for term in (*terms, *replacements)}
        return cls(
            terms={normalize_term(term): z for term, z in terms.items()},
            replacements={normalize_term(term): replacement for term, replacement in replacements.items()},
            banned_patterns=compiled,
            labels=labels,
        )

    def top_terms(self, limit: int) -> list[tuple[str, float]]:
        return sorted(self.terms.items(), key=lambda item: item[1], reverse=True)[:limit]

    def save(self, path: str | Path) -> None:
        payload = {
            'terms': {self.label(term): z for term, z in self.top_terms(len(self.terms))},
            'replacements': {self.label(term): replacement for term, replacement in self.replacements.items()},
            'banned_patterns': [pattern.pattern for pattern in self.banned_patterns],
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
