from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UserError

from vocabguard.stats import CorpusCounts, jensen_shannon, log_odds_z, separation


def test_log_odds_matches_hand_computation() -> None:
    # Model corpus: 'delve' x4, 'the' x6 (n=10). Baseline: 'the' x10 (n=10). alpha0 = 10.
    #
    # delve: pooled frequency 4/20 = 0.2, so a_w = 2.
    #   delta = ln((4 + 2) / (10 + 10 - 4 - 2)) - ln((0 + 2) / (10 + 10 - 0 - 2))
    #         = ln(6/14) - ln(2/18) = -0.847298 + 2.197225 = 1.349927
    #   var   = 1/6 + 1/2 = 0.666667, so z = 1.349927 / 0.816497 = 1.653316
    #
    # the: pooled frequency 16/20 = 0.8, so a_w = 8.
    #   delta = ln(14/6) - ln(18/2) = 0.847298 - 2.197225 = -1.349927
    #   var   = 1/14 + 1/18 = 0.126984, so z = -1.349927 / 0.356348 = -3.788222
    model = CorpusCounts(total=10, counts=Counter({'delve': 4, 'the': 6}))
    baseline = CorpusCounts(total=10, counts=Counter({'the': 10}))
    z = log_odds_z(model=model, baseline=baseline, alpha0=10)
    assert z['delve'] == pytest.approx(1.653316, abs=1e-6)
    assert z['the'] == pytest.approx(-3.788222, abs=1e-6)


def test_log_odds_is_antisymmetric() -> None:
    a = CorpusCounts(total=30, counts=Counter({'x': 20, 'y': 10}))
    b = CorpusCounts(total=30, counts=Counter({'x': 5, 'y': 25}))
    forward = log_odds_z(model=a, baseline=b, alpha0=50)
    backward = log_odds_z(model=b, baseline=a, alpha0=50)
    assert forward['x'] == pytest.approx(-backward['x'])
    assert forward['x'] > 0 > forward['y']


def test_log_odds_of_empty_corpora_is_empty() -> None:
    empty = CorpusCounts(total=0)
    assert log_odds_z(model=empty, baseline=empty, alpha0=10) == {}


def test_jensen_shannon_bounds_and_symmetry() -> None:
    p = CorpusCounts.from_documents(['the cat sat on the mat'])
    q = CorpusCounts.from_documents(['a dog ran across a field'])
    assert jensen_shannon(p, p) == pytest.approx(0.0)
    assert jensen_shannon(p, q) == pytest.approx(1.0)
    r = CorpusCounts.from_documents(['the cat ran across the field'])
    assert 0.0 < jensen_shannon(p, r) < 1.0
    assert jensen_shannon(p, r) == pytest.approx(jensen_shannon(r, p))
    assert jensen_shannon(p, CorpusCounts(total=0)) == 1.0


def test_corpus_counts_round_trip(tmp_path: Path) -> None:
    target = tmp_path / 'baseline.json'
    counts = CorpusCounts.from_documents(['the cat', 'the dog'], source='test')
    counts.save(target)
    loaded = CorpusCounts.load(target)
    assert loaded == counts
    assert loaded.total == 6


def test_separation_matches_hand_computation() -> None:
    # Model scores beat baseline scores in 10 of the 12 pairs: 0.25 beats {0.1, 0.2}; 0.4 and 0.5 beat all four.
    # Best cut is 0.35: every baseline document sits at or below it, two of three model documents sit above.
    result = separation(baseline=[0.1, 0.2, 0.3, 0.35], model=[0.25, 0.4, 0.5])
    assert result.auc == pytest.approx(10 / 12)
    assert result.threshold == 0.35
    assert result.true_positive_rate == pytest.approx(2 / 3)
    assert result.false_positive_rate == 0.0
    assert result.balanced_accuracy == pytest.approx(5 / 6)


def test_separation_ties_are_chance() -> None:
    result = separation(baseline=[1.0, 1.0], model=[1.0, 1.0])
    assert result.auc == pytest.approx(0.5)
    assert result.balanced_accuracy == pytest.approx(0.5)


def test_separation_needs_both_sides() -> None:
    with pytest.raises(ValueError, match='each side'):
        separation(baseline=[], model=[1.0])


def test_corpus_counts_rejects_bad_file(tmp_path: Path) -> None:
    target = tmp_path / 'bad.json'
    target.write_text('{"counts": {}}')
    with pytest.raises(UserError, match='Invalid corpus counts file'):
        CorpusCounts.load(target)
