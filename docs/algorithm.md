# Algorithm notes

This page gives the formulas behind the README summary, for readers tuning `alpha0` and `z` or
reviewing what a score means.

## Prose extraction

Extraction runs before anything is counted, so code and markup never enter a corpus.

- Markdown, reStructuredText, plain text: fenced code blocks, inline code spans, URLs, HTML tags,
  link targets, and reference definitions are removed. Heading text and body text remain.
- Python: docstrings and comments only. When the input parses, `ast` finds docstrings; edit tools
  often send fragments that do not parse, so `tokenize` is the fallback and any tokenizer error
  ends the scan without failing it. Identifiers are never included.

## Normalization

Text is lowercased and split on word boundaries, keeping hyphenated compounds and contractions as
single tokens. Each token is lemmatized with `simplemma`. The token list is the unigrams followed
by the bigrams. Stopwords stay in; the prior below handles them.

## Log-odds with an informative Dirichlet prior

For an n-gram `w`, with count `y` and corpus size `n` in the model corpus `i` and the baseline `j`:

```
a_w     = alpha0 * (y_w^i + y_w^j) / (n^i + n^j)
delta_w = log((y_w^i + a_w) / (n^i + alpha0 - y_w^i - a_w))
        - log((y_w^j + a_w) / (n^j + alpha0 - y_w^j - a_w))
var_w   = 1 / (y_w^i + a_w) + 1 / (y_w^j + a_w)
z_w     = delta_w / sqrt(var_w)
```

The watchlist is every `w` with `z_w` above the cutoff. `alpha0` is the total prior mass; a
larger value shrinks estimates for rare terms harder, so raising it trims the long tail of
one-off words from the watchlist. The default of 500 suits corpora of a few tens of thousands of
tokens. A small baseline with a large `alpha0` produces very few watched terms; lower `alpha0`
before lowering `z`.

## Scoring

Given the token list `T` of the prose under test and watchlist `W`:

```
score = sum(max(0, z_w) for w in T if w in W) / len(T)
```

Terms with negative z can appear in a watchlist only if it was edited by hand; they are reported
but contribute nothing. When `len(T)` is below `min_tokens` the score is zero and only banned
patterns can fire.

## Drift report

For each commit after the given ref, the added lines of prose files are extracted and normalized
into a token distribution `P`. With the baseline distribution `Q` and `M = (P + Q) / 2`, the
Jensen-Shannon divergence in base 2 is:

```
JS = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
```

It lies in `[0, 1]`, is symmetric, and is `1.0` when either side is empty. Watchlist hits per
thousand words are reported next to it as the term-level view of the same commit.
