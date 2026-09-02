"""Pull prose out of files so that code, URLs, and identifiers never reach the scorer."""

from __future__ import annotations

import ast
import io
import re
import textwrap
import tokenize
from pathlib import PurePath
from typing import Protocol, runtime_checkable

__all__ = ('MarkdownExtractor', 'ProseExtractor', 'PythonExtractor', 'extractor_for')


@runtime_checkable
class ProseExtractor(Protocol):
    """Extension point for new file types. Subclass it explicitly so pyright checks the signature."""

    def extract(self, text: str, *, path: str) -> list[str]:
        """Return the prose spans found in `text`; `path` is available for format-specific decisions."""
        ...


_FENCE = re.compile(r'^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$', re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r'`+[^`]*`+')
_URL = re.compile(r'(?:https?|ftp)://\S+|www\.\S+')
_HTML_TAG = re.compile(r'</?[A-Za-z][^>]*>|<!--.*?-->', re.DOTALL)
_IMAGE_OR_LINK = re.compile(r'!?\[([^\]]*)\]\([^)]*\)')
_REF_LINK = re.compile(r'\[([^\]]+)\]\[[^\]]*\]')
_REF_DEFINITION = re.compile(r'^[ \t]*\[[^\]]+\]:[ \t]+\S+.*$', re.MULTILINE)
_HEADING_MARKS = re.compile(r'^[ \t]*#{1,6}[ \t]+', re.MULTILINE)
_RST_DIRECTIVE = re.compile(r'^[ \t]*\.\.[ \t]+\S+::.*$', re.MULTILINE)
_BLANK_LINES = re.compile(r'\n[ \t]*\n+')


class MarkdownExtractor(ProseExtractor):
    """Markdown, reStructuredText, and plain text. Headings and body text survive; markup does not."""

    def extract(self, text: str, *, path: str) -> list[str]:
        text = _FENCE.sub('\n', text)
        text = _INLINE_CODE.sub(' ', text)
        text = _HTML_TAG.sub(' ', text)
        text = _IMAGE_OR_LINK.sub(r'\1', text)
        text = _REF_LINK.sub(r'\1', text)
        text = _REF_DEFINITION.sub('', text)
        text = _RST_DIRECTIVE.sub('', text)
        text = _URL.sub(' ', text)
        text = _HEADING_MARKS.sub('', text)
        return [span.strip() for span in _BLANK_LINES.split(text) if span.strip()]


class PythonExtractor(ProseExtractor):
    """Docstrings and comments only. Identifiers are never prose, even the descriptive ones."""

    def extract(self, text: str, *, path: str) -> list[str]:
        spans = _docstrings_via_ast(text)
        # Edit tools hand us fragments; ast rejects those, so tokenize decides what a docstring is.
        docstrings_from_tokens = spans is None
        spans = list(spans or [])
        spans.extend(_comments_and_fragment_docstrings(text, include_docstrings=docstrings_from_tokens))
        return [span for span in spans if span]


def _docstrings_via_ast(text: str) -> list[str] | None:
    for candidate in (text, textwrap.dedent(text)):
        try:
            tree = ast.parse(candidate)
        except (SyntaxError, ValueError):
            continue
        docstrings: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    docstrings.append(docstring)
        return docstrings
    return None


_LINE_START = {tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}


def _comments_and_fragment_docstrings(text: str, *, include_docstrings: bool) -> list[str]:
    spans: list[str] = []
    previous_type = tokenize.NEWLINE
    try:
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                spans.append(token.string.lstrip('#').strip())
            elif include_docstrings and token.type == tokenize.STRING and previous_type in _LINE_START:
                spans.append(_string_literal_text(token.string))
            if token.type != tokenize.COMMENT:
                previous_type = token.type
    except (tokenize.TokenError, SyntaxError):
        # Partial input truncates the stream; whatever was collected before the error is still prose.
        pass
    return spans


def _string_literal_text(literal: str) -> str:
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return ''
    return textwrap.dedent(value).strip() if isinstance(value, str) else ''


_EXTRACTORS: dict[str, ProseExtractor] = {
    '.md': MarkdownExtractor(),
    '.rst': MarkdownExtractor(),
    '.txt': MarkdownExtractor(),
    '.py': PythonExtractor(),
}


def extractor_for(path: str) -> ProseExtractor | None:
    """Pick an extractor by file extension, or None when the file type carries no prose we score."""
    return _EXTRACTORS.get(PurePath(path).suffix.lower())
