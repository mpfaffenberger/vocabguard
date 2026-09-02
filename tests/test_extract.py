from __future__ import annotations

from vocabguard.extract import MarkdownExtractor, ProseExtractor, PythonExtractor, extractor_for

MARKDOWN = """\
# Getting started

Read the [installation guide](https://example.com/leverage) before you `leverage` anything.

```python
def leverage():  # robust
    return 'seamless'
```

<div class="robust">Visible text</div> and a bare link https://robust.example.com here.

[ref]: https://example.com/comprehensive
"""


def test_markdown_keeps_prose_and_drops_markup() -> None:
    spans = MarkdownExtractor().extract(MARKDOWN, path='doc.md')
    joined = '\n'.join(spans)
    assert 'Getting started' in joined
    assert 'installation guide' in joined
    assert 'Visible text' in joined
    for markup in ('leverage', 'robust', 'seamless', 'comprehensive', 'https://', '<div', '```', '#'):
        assert markup not in joined


PYTHON = '''\
"""Module docstring mentions the parser."""


class Widget:
    """Class docstring about widgets."""

    def leverage_robust_seamless(self, comprehensive: int) -> int:
        # comment explaining the return value
        return comprehensive + streamline_delve
'''


def test_python_takes_docstrings_and_comments_only() -> None:
    spans = PythonExtractor().extract(PYTHON, path='widget.py')
    assert spans == [
        'Module docstring mentions the parser.',
        'Class docstring about widgets.',
        'comment explaining the return value',
    ]


def test_python_fragment_falls_back_to_tokenize() -> None:
    fragment = '        """Docstring inside an unparseable fragment."""\n        x = leverage(  # trailing comment\n'
    spans = PythonExtractor().extract(fragment, path='snippet.py')
    assert spans == ['Docstring inside an unparseable fragment.', 'trailing comment']


def test_python_tolerates_tokenize_errors() -> None:
    assert PythonExtractor().extract('# only a comment\nx = "unterminated', path='broken.py') == ['only a comment']


def test_dispatch_by_extension() -> None:
    assert isinstance(extractor_for('README.md'), MarkdownExtractor)
    assert isinstance(extractor_for('docs/index.rst'), MarkdownExtractor)
    assert isinstance(extractor_for('NOTES.TXT'), MarkdownExtractor)
    assert isinstance(extractor_for('pkg/mod.py'), PythonExtractor)
    assert extractor_for('data.json') is None
    assert extractor_for('Makefile') is None


def test_extractors_satisfy_the_protocol() -> None:
    assert isinstance(MarkdownExtractor(), ProseExtractor)
    assert isinstance(PythonExtractor(), ProseExtractor)
