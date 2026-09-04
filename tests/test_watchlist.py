from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.exceptions import UserError

from vocabguard import Watchlist
from vocabguard.watchlist import CuratedFile


def test_starter_watchlist_loads_with_replacements() -> None:
    starter = Watchlist.starter()
    assert starter.terms['delve'] > 0
    assert starter.replacements['leverage'] == 'use'
    assert starter.banned_patterns == []


def test_keys_are_normalized_and_labels_kept() -> None:
    watchlist = Watchlist.from_parts(terms={'Delving': 1.0, 'worth noting': 2.0}, replacements={}, banned_patterns=[])
    assert watchlist.terms == {'delve': 1.0, 'worth note': 2.0}
    assert watchlist.label('worth note') == 'worth noting'


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    target = tmp_path / 'watchlist.json'
    original = Watchlist.from_parts(
        terms={'leverage': 3.0, 'worth noting': 2.5},
        replacements={'leverage': 'use'},
        banned_patterns=['--'],
        threshold=1.25,
    )
    original.save(target)
    written = json.loads(target.read_text())
    assert list(written['terms']) == ['leverage', 'worth noting']
    assert written['threshold'] == 1.25
    assert Watchlist.load(target) == original


def test_threshold_defaults_to_zero_and_rejects_negatives(tmp_path: Path) -> None:
    legacy = tmp_path / 'legacy.json'
    legacy.write_text('{"terms": {"delve": 2.0}}')
    assert Watchlist.load(legacy).threshold == 0.0
    with pytest.raises(UserError, match='threshold'):
        Watchlist.from_parts(terms={}, replacements={}, banned_patterns=[], threshold=-1.0)


def test_default_is_the_measured_classifier() -> None:
    default = Watchlist.default()
    assert default.threshold == 0.41
    assert default.terms == Watchlist.bundled('readme_2026_watchlist').terms
    assert Watchlist.starter().threshold == 0.0


def test_top_terms_orders_by_z() -> None:
    watchlist = Watchlist.from_parts(terms={'a': 1.0, 'b': 3.0, 'c': 2.0}, replacements={}, banned_patterns=[])
    assert watchlist.top_terms(2) == [('b', 3.0), ('c', 2.0)]


def test_invalid_files_raise_user_error(tmp_path: Path) -> None:
    missing_terms = tmp_path / 'missing.json'
    missing_terms.write_text('{"replacements": {}}')
    with pytest.raises(UserError, match='Invalid watchlist file'):
        Watchlist.load(missing_terms)
    bad_pattern = tmp_path / 'pattern.json'
    bad_pattern.write_text('{"terms": {}, "banned_patterns": ["("]}')
    with pytest.raises(UserError, match='Invalid banned pattern'):
        Watchlist.load(bad_pattern)
    bad_curated = tmp_path / 'curated.json'
    bad_curated.write_text('{"replacements": []}')
    with pytest.raises(UserError, match='Invalid curated file'):
        CuratedFile.load(bad_curated)
