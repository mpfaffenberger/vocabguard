"""Turn prose into the n-grams that baselines, watchlists, and the scorer all agree on.

Stopwords stay in on purpose: the Dirichlet prior in `vocabguard.stats` already
discounts terms that are frequent in both corpora.
"""

from __future__ import annotations

import re
from itertools import pairwise

import simplemma

__all__ = ('lemmatize_words', 'normalize', 'normalize_term')

# Hyphenated compounds and contractions stay whole; punctuation and underscores split.
_WORD = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*")
_LANG = 'en'


def lemmatize_words(text: str) -> list[str]:
    """Lowercased, lemmatized unigrams in document order."""
    return [simplemma.lemmatize(match.group(0).lower(), lang=_LANG) for match in _WORD.finditer(text)]


def normalize(text: str) -> list[str]:
    """Unigrams followed by space-joined bigrams. This is the token list `T` used for scoring."""
    words = lemmatize_words(text)
    return words + [f'{a} {b}' for a, b in pairwise(words)]


def normalize_term(term: str) -> str:
    """Normalize a watchlist key so hand-written terms match model-generated tokens."""
    return ' '.join(lemmatize_words(term))
