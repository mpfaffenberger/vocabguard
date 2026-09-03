"""Corpus counts and the statistics built on them.

Log-odds with an informative Dirichlet prior (Monroe, Colaresi, Quinn 2008) builds the
watchlist. Jensen-Shannon divergence feeds the `report` command only; the guard never computes it.
`separation` grades a watchlist on held-out documents for the `evaluate` command.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UserError

from .normalize import normalize

__all__ = ('CorpusCounts', 'Separation', 'jensen_shannon', 'log_odds_z', 'separation')


class _CorpusFile(BaseModel):
    total: int
    counts: dict[str, int]
    source: str = ''


@dataclass(kw_only=True)
class CorpusCounts:
    total: int
    """Number of tokens (unigrams plus bigrams) in the corpus, including unwatched ones."""
    counts: Counter[str] = field(default_factory=Counter[str])
    source: str = ''
    """Where the counts came from, for the human reading the file later."""

    @classmethod
    def from_documents(cls, documents: Iterable[str], *, source: str = '') -> CorpusCounts:
        counts: Counter[str] = Counter()
        for document in documents:
            counts.update(normalize(document))
        return cls(total=sum(counts.values()), counts=counts, source=source)

    @classmethod
    def load(cls, path: str | Path) -> CorpusCounts:
        try:
            parsed = _CorpusFile.model_validate_json(Path(path).read_text(encoding='utf-8'))
        except ValidationError as error:
            raise UserError(f'Invalid corpus counts file {path}: {error}') from error
        return cls(total=parsed.total, counts=Counter(parsed.counts), source=parsed.source)

    def save(self, path: str | Path) -> None:
        payload = _CorpusFile(total=self.total, counts=dict(self.counts.most_common()), source=self.source)
        Path(path).write_text(payload.model_dump_json(indent=2) + '\n', encoding='utf-8')

    def frequencies(self) -> dict[str, float]:
        return {term: count / self.total for term, count in self.counts.items()} if self.total else {}


def log_odds_z(*, model: CorpusCounts, baseline: CorpusCounts, alpha0: float) -> dict[str, float]:
    """z-scored log-odds of each n-gram toward `model`, with prior mass `alpha0` spread by pooled frequency."""
    pooled_total = model.total + baseline.total
    if pooled_total == 0:
        return {}
    scores: dict[str, float] = {}
    for term in model.counts.keys() | baseline.counts.keys():
        y_i = model.counts[term]
        y_j = baseline.counts[term]
        a_w = alpha0 * (y_i + y_j) / pooled_total
        delta = math.log((y_i + a_w) / (model.total + alpha0 - y_i - a_w)) - math.log(
            (y_j + a_w) / (baseline.total + alpha0 - y_j - a_w)
        )
        variance = 1 / (y_i + a_w) + 1 / (y_j + a_w)
        scores[term] = delta / math.sqrt(variance)
    return scores


def jensen_shannon(p: CorpusCounts, q: CorpusCounts) -> float:
    """Base-2 divergence in [0, 1] between two token distributions. 1.0 when either side is empty."""
    if p.total == 0 or q.total == 0:
        return 1.0
    p_freq = p.frequencies()
    q_freq = q.frequencies()
    total = 0.0
    for term in p_freq.keys() | q_freq.keys():
        p_w = p_freq.get(term, 0.0)
        q_w = q_freq.get(term, 0.0)
        m_w = (p_w + q_w) / 2
        if p_w:
            total += 0.5 * p_w * math.log2(p_w / m_w)
        if q_w:
            total += 0.5 * q_w * math.log2(q_w / m_w)
    return total


@dataclass(kw_only=True, frozen=True)
class Separation:
    """How well scores split held-out baseline documents from held-out model documents."""

    auc: float
    """Probability that a random model document outscores a random baseline document; 0.5 is chance."""
    threshold: float
    """Cut that maximizes balanced accuracy under the rule `score > threshold` means drifted."""
    true_positive_rate: float
    """Share of model documents above the threshold."""
    false_positive_rate: float
    """Share of baseline documents above the threshold."""

    @property
    def balanced_accuracy(self) -> float:
        return (self.true_positive_rate + 1 - self.false_positive_rate) / 2


def separation(*, baseline: Sequence[float], model: Sequence[float]) -> Separation:
    """Rank-based AUC (Mann-Whitney, ties get half credit) and the best threshold over observed scores."""
    if not baseline or not model:
        raise ValueError('separation needs at least one score on each side')
    ranked = sorted([(score, False) for score in baseline] + [(score, True) for score in model])
    model_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        model_rank_sum += average_rank * sum(1 for _, is_model in ranked[index:end] if is_model)
        index = end
    auc = (model_rank_sum - len(model) * (len(model) + 1) / 2) / (len(model) * len(baseline))

    baseline_sorted, model_sorted = sorted(baseline), sorted(model)
    best = Separation(auc=auc, threshold=math.inf, true_positive_rate=0.0, false_positive_rate=0.0)
    for candidate in sorted({score for score, _ in ranked}):
        tpr = 1 - bisect_right(model_sorted, candidate) / len(model_sorted)
        fpr = 1 - bisect_right(baseline_sorted, candidate) / len(baseline_sorted)
        current = Separation(auc=auc, threshold=candidate, true_positive_rate=tpr, false_positive_rate=fpr)
        if current.balanced_accuracy > best.balanced_accuracy:
            best = current
    return best
