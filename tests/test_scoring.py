from __future__ import annotations

import pytest

from vocabguard import Watchlist
from vocabguard.scoring import added_lines, score_file, score_prose

WATCHLIST = Watchlist.from_parts(
    terms={'leverage': 4.0, 'robust': 2.0, 'worth noting': 3.0, 'negative': -5.0},
    replacements={'leverage': 'use'},
    banned_patterns=['\u2014'],
)


def test_score_is_z_mass_over_token_count() -> None:
    # 10 words -> 10 unigrams + 9 bigrams = 19 tokens. Hits: leverage (4.0) x2, robust (2.0) x1.
    text = 'we leverage this and leverage that with a robust parser'
    report = score_prose([text], WATCHLIST, min_tokens=0)
    assert report.token_count == 19
    assert report.score == pytest.approx((4.0 * 2 + 2.0) / 19)
    assert [(hit.term, hit.count, hit.replacement) for hit in report.terms] == [
        ('leverage', 2, 'use'),
        ('robust', 1, None),
    ]


def test_negative_z_contributes_nothing_but_is_still_a_hit() -> None:
    report = score_prose(['a negative word'], WATCHLIST, min_tokens=0)
    assert report.score == 0.0
    assert report.hit
    assert not report.exceeds(0.0)


def test_bigram_terms_match_with_original_spelling_reported() -> None:
    report = score_prose(["it's worth noting that this matters"], WATCHLIST, min_tokens=0)
    assert [hit.term for hit in report.terms] == ['worth noting']


def test_below_min_tokens_skips_scoring_but_not_banned_patterns() -> None:
    report = score_prose(['leverage \u2014 short'], WATCHLIST, min_tokens=20)
    assert report.score == 0.0
    assert report.terms == []
    assert [match.text for match in report.banned] == ['\u2014']
    assert report.exceeds(1000.0)


def test_added_lines_only_reports_insertions() -> None:
    previous = 'one\ntwo\nthree\n'
    current = 'one\ntwo changed\nthree\nfour\n'
    assert added_lines(previous, current) == 'two changed\nfour'


def test_score_file_dispatches_and_diffs() -> None:
    assert score_file(path='data.json', content='leverage ' * 30, watchlist=WATCHLIST) is None
    contaminated = ('leverage robust ' * 15).strip()
    clean_addition = 'a plain sentence about the parser and how it reads input lines'
    report = score_file(
        path='doc.md',
        content=f'{contaminated}\n{clean_addition}\n',
        previous=f'{contaminated}\n',
        watchlist=WATCHLIST,
        min_tokens=5,
    )
    assert report is not None
    assert report.path == 'doc.md'
    assert not report.hit


def test_describe_lists_terms_and_banned_matches() -> None:
    report = score_prose(['we leverage the \u2014 thing'], WATCHLIST, min_tokens=0)
    text = report.describe()
    assert "'leverage' (z=4.0, x1, prefer: use)" in text
    assert 'banned pattern' in text
