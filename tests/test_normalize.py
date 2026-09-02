from __future__ import annotations

from vocabguard.normalize import lemmatize_words, normalize, normalize_term


def test_lemmatizes_and_lowercases() -> None:
    assert lemmatize_words('Leveraging Delves noting') == ['leverage', 'delve', 'note']


def test_hyphenated_compounds_stay_whole() -> None:
    assert lemmatize_words('a state-of-the-art thing') == ['a', 'state-of-the-art', 'thing']


def test_unigrams_then_bigrams() -> None:
    assert normalize('The cats sat.') == ['the', 'cat', 'sit', 'the cat', 'cat sit']


def test_stopwords_are_kept() -> None:
    assert 'the' in normalize('the parser')


def test_term_normalization_matches_text_tokens() -> None:
    assert normalize_term('Worth Noting') in normalize("it's worth noting that")
