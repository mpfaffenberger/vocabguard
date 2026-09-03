# vocabguard

A [Pydantic AI](https://ai.pydantic.dev) capability that watches an agent's tool calls and
responses, scores the prose in them against a per-repo watchlist of model-favored terms, and
returns a `ModelRetry` naming the terms and preferred replacements so the model rephrases before
the text lands anywhere.

It ships with a CLI that builds the watchlist from your repository's own history, runs the same
check as a pre-commit hook, and reports drift over time.

## Constraints

Read these before the examples; they shape how the guard behaves.

- **Prose only.** Markdown, reStructuredText, and plain text are scored after fenced code, inline
  code, URLs, HTML tags, and link targets are stripped. Python files contribute docstrings and
  comments only; identifiers are never scored. A tool call that names a file of any other type
  (`.json`, `.yaml`) is not prose and goes through untouched.
- **Three places text is scored.** Tool calls that carry a path are scored as that file type.
  Tool calls without a path (a chat message, a search query, a shell command) have every string
  argument scored as plain text. Every model response has its text scored, whether it is the final
  answer or a sentence written next to a tool call. Each can be narrowed with `tools` and
  `output`.
- **Term substitution only.** The guard catches words and two-word phrases. It does not see
  sentence structure, hedging, or list-heavy layouts. That is the job of the `[structural]` extra,
  which is not part of this release.
- **Added lines only, when it can tell.** If a tool call carries the previous text
  (`old_string`), only the lines the edit introduces are scored. Scoring the whole file would block
  every edit to a file that already contains watched terms.
- **A minimum size.** Below `min_tokens` (default 20 unigrams plus bigrams) the score is skipped,
  because a five-word edit cannot be judged by frequency. Banned patterns still fire.
- **One tokenizer everywhere.** Baseline, corpus, watchlist keys, and the text under test all go
  through the same lowercase, lemmatize, unigram-plus-bigram pipeline. Hand-written watchlist keys
  are normalized on load, so `noting` and `note` are the same term.
- **No model requests from the guard.** Two commands talk to a model: `vocabguard rewrite` and
  `vocabguard scrape`. The capability itself does no I/O.
- **git is required for the CLI.** `baseline`, `rewrite`, `report`, and `check --diff-base` shell
  out to `git`; there is no git library dependency.
- **Python 3.11 or later.** Dependencies are `pydantic-ai-slim`, `httpx`, and `simplemma`, a
  pure-Python lemmatizer with no model download.

## Install

```bash
pip install vocabguard
# or
uv add vocabguard
```

The `scrape` command needs OpenRouter support from pydantic-ai and the
[termflow](https://pypi.org/project/termflow-md/) renderer, which together are an extra:

```bash
pip install "vocabguard[scrape]"
```

## How the watchlist is built

Two corpora are compared: the prose already in your repository at some ref (the baseline) and a
model's rewrite of that same prose (the model corpus). For every unigram and bigram, the log-odds
ratio between the two corpora is computed with an informative Dirichlet prior (Monroe, Colaresi,
and Quinn, 2008), where the prior mass for each term is `alpha0` times its pooled frequency. The
prior pulls common words toward zero, which is why stopwords are not removed. Each log-odds value
is divided by its standard error to give a z score; terms with z above a cutoff toward the model
corpus become the watchlist. At write time, the score of a piece of prose is the sum of the z
scores of the watched terms it contains divided by its token count, and any score above the
threshold (default zero) asks the model to rephrase.

## Workflow

### 1. Decide what "our voice" is

The baseline is prose you would be happy to have more of. Two ways to get it:

**Your own history.** Pick a ref from before agents started writing in the repo, and restrict to
prose directories if the repo has vendored text you do not want in the sample:

```bash
vocabguard baseline --ref v1.0.0 --path docs --path README.md -o baseline.json
```

**A gathered corpus.** If the repo is new, or its history is already model-written, have an agent
collect reference prose for you:

```bash
vocabguard scrape -o reference/ --target 50
```

The agent opens by asking what the corpus should represent (Wikipedia articles about aviation;
README files of Python repositories on GitHub not updated since before 2026; a docs site you
admire), proposes a plan, and saves documents into the folder as it goes. Replies stream as
rendered markdown and each tool call is shown as it happens. Type `quit` to end the session. It runs on OpenRouter's free router by default; the first run opens your browser to sign
in with OpenRouter and stores the resulting key at `~/.config/vocabguard/openrouter_key`. Set
`OPENROUTER_API_KEY` to skip the sign-in, or pass any other `--model provider:name`. Then:

```bash
vocabguard baseline --dir reference/ -o baseline.json
```

Either way the file is frozen once written; pass `--force` to replace it. Changing the baseline
silently would change what every later report means.

### 2. Build the model corpus

```bash
vocabguard rewrite --ref v1.0.0 --path docs --path README.md --model openai:gpt-5 -o corpus/
```

`--dir reference/` works here too when the baseline came from a gathered corpus.

Each prose document at the ref is sent to the model with the instruction to rewrite it in its own
words at the same length. Output lands at `corpus/<original path>.md`. Files that already exist
are skipped, so an interrupted run can be resumed. This is the only step that costs tokens.

### 3. Contrast the two

```bash
vocabguard contrast --baseline baseline.json --corpus corpus/ -o watchlist.json --alpha0 500 --z 2.5
```

Replacements and banned patterns are hand-maintained in a separate file so re-running `contrast`
never loses them:

```bash
vocabguard contrast --baseline baseline.json --corpus corpus/ -o watchlist.json --curated curated.json
```

```json
{
  "replacements": {"leverage": "use"},
  "banned_patterns": ["\u2014"]
}
```

With corpora of millions of tokens nearly every n-gram clears `--z 2.5`, so cap the list with
`--top 1000` and let `evaluate` pick the threshold.

### 3b. Grade the watchlist before wiring it in

```bash
vocabguard evaluate --baseline-dir human/ --corpus-dir model/ --top 1000
```

`evaluate` builds the watchlist on 80% of each corpus, scores the other 20% with the same scorer the
guard uses, and prints the AUC plus the threshold that best balances catches against false alarms.
Use that threshold in the capability and in `check`; a threshold of `0.0` with a thousand-term list
would fire on almost anything.

### 4. Wire the capability

```python
from pydantic_ai import Agent

from vocabguard import VocabularyGuard, Watchlist

guard = VocabularyGuard(Watchlist.load('watchlist.json'))
agent = Agent('openai:gpt-5', capabilities=[guard])
```

By default every tool call and every model response is scored. A tool call is treated as a file
write when it has a `path` or `file_path` argument; the new text is read from `content`,
`new_string`, or `new_str`, and the previous text from `old_string` or `old_str`. All of these
key names are configurable. With `instruct=True` (the default) the top 15 watched terms and their
replacements are added to the system prompt so the model avoids them before the guard has to fire.

```python
guard = VocabularyGuard(
    Watchlist.load('watchlist.json'),
    tools=lambda name: name.endswith('_file'),  # or a sequence of names; None watches every tool
    output=False,  # leave model responses alone, score tool calls only
    threshold=0.05,
    mode='warn',
    on_hit=lambda report: print(report.describe()),
)
```

In `retry` mode a hit above threshold raises `ModelRetry`. For a tool call, the model is asked to
resubmit the same call with the wording fixed; for a response, to reply again. The message lists
the terms, their z scores, and replacements. In `warn` mode everything goes through and only
`on_hit` is called. `on_hit` is called on every hit in both modes, so you can wire it to your own
logging or metrics; `HitReport.source` says which tool call or response it came from.

Misconfiguration (an unknown mode, an empty `tools` sequence, a negative threshold) raises
`UserError` at construction, not when the first tool call arrives.

### 5. Run the same check in pre-commit

```yaml
repos:
  - repo: https://github.com/mpfaffenberger/vocabguard
    rev: v0.1.0
    hooks:
      - id: vocabguard
        args: [--watchlist, watchlist.json, --diff-base, origin/main]
```

`vocabguard check` accepts files and directories, exits nonzero on hits above threshold, and prints
the same report the guard sends to the model. Without `--watchlist` it uses the bundled starter
list.

### 6. Watch drift over time

```bash
vocabguard report --baseline baseline.json --watchlist watchlist.json --since v1.0.0 --format csv
```

For every commit after the ref, the report gives the Jensen-Shannon divergence between the
commit's added prose and the baseline, plus watchlist hits per thousand words. The guard never
computes divergence; this is the monitor, not the gate. Small commits have few tokens and noisy
divergence, so read the `words` column alongside it.

## Starter watchlist

If you have not built corpora yet, `Watchlist.starter()` (or `vocabguard check` with no
`--watchlist`) loads a short hand-curated list of commonly model-favored terms such as `delve`,
`leverage`, `robust`, `seamless`, and `worth noting`, with modest z values and replacements. It is
a starting point, not a measurement; `contrast` on your own history will disagree with it in both
directions.

## Measured README watchlist

`Watchlist.bundled('readme_2026_watchlist')` is a measurement. It was built with the commands above
from two corpora of GitHub READMEs: 4,319 from repositories with at least 500 stars whose last push
was before 2025 (22 languages, weighted toward TypeScript, Python, Java, Rust, Go, JavaScript, and
C; 6.4M prose tokens), and 1,047 from repositories with Claude Code commits during 2026 (2.1M prose
tokens). Markup, URLs, and code were stripped before counting.

```bash
vocabguard contrast --baseline human.json --corpus claude/ -o readme_2026_watchlist.json --top 1000
vocabguard evaluate --baseline-dir human/ --corpus-dir claude/ --top 1000
```

```text
AUC: 0.872
threshold 1.60: flags 75% of model documents and 8% of baseline documents
```

Use it with `threshold=1.6`. What it measures, and what it does not:

- **Voice**: the 2026 corpus over-uses `every`, `no`, `across`, `via`, `full`, `with`, and `what`,
  and under-uses `the`, `of`, `to`, `you`, `can`, `will`, `be`, `if`. Nominal, list-shaped
  fragments instead of sentences aimed at a reader.
- **Topic**: `agent`, `memory`, `session`, `skill`, `hook`, `tool`, `llm`, `vector` lead the list
  because that is what the 2026 repositories are about. A human writing about an agent in plain
  sentences scores around the threshold; the test suite pins that case rather than hiding it.
- **Attribution is per repository, not per document.** Some READMEs in the 2026 corpus were typed
  by people; some pre-2025 READMEs were not. The 8% false alarm rate was measured on pre-2025
  READMEs, which rarely discuss agents.
- **English only.** The 2026 corpus has more non-English READMEs, so `de` and `si` carry high z.
  Non-English prose scores high for the wrong reason.

The corpora are not in the repository. With `instruct=True` the capability lists the top terms,
which for this list are topic words; consider `instruct=False` or a pruned copy for agent projects.

## Watchlist file

```json
{
  "terms": {"delve": 6.0, "worth noting": 5.0},
  "replacements": {"delve": "look at, examine"},
  "banned_patterns": ["\u2014"]
}
```

`terms` maps an n-gram to its z score. `replacements` is optional and hand-maintained.
`banned_patterns` is an optional list of regular expressions that always count as a hit, whatever
the score or token count.

## Development

```bash
uv sync
uv run ruff format --check . && uv run ruff check . && uv run pyright && uv run pytest
uv run vocabguard check README.md docs/ --watchlist vocabguard/data/starter_watchlist.json
```

The last line is the package checking its own prose, and CI runs it.
