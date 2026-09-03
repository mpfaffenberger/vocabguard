"""The bundled README watchlist measured from real corpora, and the threshold `evaluate` chose for it.

The corpora themselves are not in the repository, so these tests pin behaviour on synthetic samples
in the two styles the contrast separated: nominal, list-shaped agent prose versus sentences aimed at
a reader. The third sample documents the known weakness rather than hiding it.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import UserError

from vocabguard.scoring import score_text
from vocabguard.watchlist import Watchlist

RECOMMENDED_THRESHOLD = 1.6
"""From `vocabguard evaluate --top 1000` on the corpora described in the README."""

DRIFTED = (
    'Agent runtime with persistent memory across sessions. Every task runs in an isolated worker with full '
    'context. Skills and hooks register via the CLI; no manual configuration. Security review on every route, '
    'integration tests for every provider, local vector search with token budgets. Session state, task '
    'patterns, and workflow descriptions live in JSON.'
)
CLEAN = (
    'This is a small library that you can use if you want to parse config files without much fuss. It will '
    'read the file, and then you get a dictionary back. We wrote it because the alternatives were too heavy '
    'for what we needed. If you find a bug, please open an issue and we will take a look when we can.'
)
HUMAN_VOICE_AGENT_TOPIC = (
    'This is an agent that you can run on your laptop. It will remember what you told it earlier, and if you '
    'want it to use a tool you can add one. We built it because we wanted something simple that we could '
    'understand, and the big frameworks were more than we needed. Let us know if it is useful to you.'
)


@pytest.fixture(scope='module')
def measured() -> Watchlist:
    return Watchlist.bundled('readme_2026_watchlist')


def test_bundled_list_is_large_and_positive(measured: Watchlist) -> None:
    assert len(measured.terms) > 900
    assert all(z > 0 for z in measured.terms.values())
    assert measured.top_terms(1)[0][0] == 'agent'
    assert not measured.replacements and not measured.banned_patterns


def test_drifted_prose_clears_the_threshold_and_clean_prose_does_not(measured: Watchlist) -> None:
    drifted = score_text(DRIFTED, measured)
    clean = score_text(CLEAN, measured)
    assert drifted.score > RECOMMENDED_THRESHOLD * 5
    assert clean.score < RECOMMENDED_THRESHOLD
    assert {hit.term for hit in drifted.terms} >= {'agent', 'session', 'every'}


def test_topic_alone_scores_between_the_two(measured: Watchlist) -> None:
    # A human-voiced README about an agent lands just above the cut. The measured list is part voice and
    # part topic, and this is the topic part; the README says so. It must stay far below the drifted sample.
    topical = score_text(HUMAN_VOICE_AGENT_TOPIC, measured).score
    assert score_text(CLEAN, measured).score < topical < score_text(DRIFTED, measured).score / 4


def test_unknown_bundled_name_is_a_user_error() -> None:
    with pytest.raises(UserError, match='No bundled watchlist'):
        Watchlist.bundled('nope')
