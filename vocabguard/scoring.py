"""Score prose against a watchlist. Shared by the capability and the `check` command."""

from __future__ import annotations

import difflib
from collections import Counter
from dataclasses import dataclass, field

from .extract import MarkdownExtractor, extractor_for
from .normalize import normalize
from .watchlist import Watchlist

__all__ = ('BannedMatch', 'HitReport', 'TermHit', 'added_lines', 'score_file', 'score_prose', 'score_text')

_PLAIN_TEXT = MarkdownExtractor()


@dataclass(kw_only=True)
class TermHit:
    term: str
    z: float
    count: int
    replacement: str | None


@dataclass(kw_only=True)
class BannedMatch:
    pattern: str
    text: str


@dataclass(kw_only=True)
class HitReport:
    source: str = 'prose'
    """What was scored: a file path, a tool call, or a model response."""
    score: float
    token_count: int
    terms: list[TermHit] = field(default_factory=list[TermHit])
    banned: list[BannedMatch] = field(default_factory=list[BannedMatch])

    @property
    def hit(self) -> bool:
        return bool(self.terms or self.banned)

    def exceeds(self, threshold: float) -> bool:
        """Banned patterns ignore the threshold; that is what makes them banned."""
        return self.score > threshold or bool(self.banned)

    def describe(self) -> str:
        """Human and model readable summary; the retry message and the CLI print the same text."""
        lines = [f'{self.source}: vocabulary score {self.score:.3f} over {self.token_count} tokens']
        for hit in self.terms:
            suffix = f', prefer: {hit.replacement}' if hit.replacement else ''
            lines.append(f'  {hit.term!r} (z={hit.z:.1f}, x{hit.count}{suffix})')
        for match in self.banned:
            lines.append(f'  banned pattern {match.pattern!r} matched {match.text!r}')
        return '\n'.join(lines)


def score_prose(spans: list[str], watchlist: Watchlist, *, min_tokens: int = 20, source: str = 'prose') -> HitReport:
    """Score extracted prose. Below `min_tokens` only banned patterns can fire."""
    text = '\n\n'.join(spans)
    tokens = normalize(text)
    report = HitReport(source=source, score=0.0, token_count=len(tokens))
    for pattern in watchlist.banned_patterns:
        report.banned.extend(
            BannedMatch(pattern=pattern.pattern, text=match.group(0)) for match in pattern.finditer(text)
        )
    if len(tokens) < min_tokens or not tokens:
        return report
    counts = Counter(token for token in tokens if token in watchlist.terms)
    total = 0.0
    for term, count in counts.most_common():
        z = watchlist.terms[term]
        total += max(0.0, z) * count
        report.terms.append(
            TermHit(term=watchlist.label(term), z=z, count=count, replacement=watchlist.replacements.get(term))
        )
    report.score = total / len(tokens)
    return report


def added_lines(previous: str, current: str) -> str:
    """Only the lines an edit introduces. Scoring the whole file would block every edit to a contaminated one."""
    diff = difflib.unified_diff(previous.splitlines(), current.splitlines(), lineterm='', n=0)
    return '\n'.join(line[1:] for line in diff if line.startswith('+') and not line.startswith('+++'))


def score_file(
    *,
    path: str,
    content: str,
    previous: str | None = None,
    watchlist: Watchlist,
    min_tokens: int = 20,
    source: str | None = None,
) -> HitReport | None:
    """Score a write to `path`, or None when the extension carries no prose. Pass `previous` to score only additions."""
    extractor = extractor_for(path)
    if extractor is None:
        return None
    text = content if previous is None else added_lines(previous, content)
    return score_prose(extractor.extract(text, path=path), watchlist, min_tokens=min_tokens, source=source or path)


def score_text(text: str, watchlist: Watchlist, *, min_tokens: int = 20, source: str = 'text') -> HitReport:
    """Score text with no file type: tool arguments and model responses. Markdown markup is stripped."""
    return score_prose(_PLAIN_TEXT.extract(text, path=''), watchlist, min_tokens=min_tokens, source=source)
