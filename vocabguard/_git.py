"""Thin subprocess wrappers around git. Paths are always repo-root relative so refs and cwd agree."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic_ai.exceptions import UserError

from .extract import extractor_for


@dataclass(kw_only=True)
class Commit:
    sha: str
    date: str
    subject: str


def run(*args: str) -> str:
    try:
        completed = subprocess.run(['git', *args], check=True, capture_output=True, text=True, errors='replace')
    except FileNotFoundError as error:
        raise UserError('git is not installed or not on PATH') from error
    except subprocess.CalledProcessError as error:
        raise UserError(f'git {args[0]} failed: {error.stderr.strip()}') from error
    return completed.stdout


def prose_files(ref: str, paths: Sequence[str]) -> list[str]:
    """Files at `ref` whose extension has an extractor, limited to `paths` when given."""
    listing = run('ls-tree', '-r', '--name-only', '--full-name', ref, '--', *paths)
    return [path for path in listing.splitlines() if path and extractor_for(path) is not None]


def read_file(ref: str, path: str) -> str | None:
    """File content at `ref`, or None when the path does not exist there (a new file)."""
    try:
        return run('show', f'{ref}:{path}')
    except UserError:
        return None


def commits_since(ref: str) -> list[Commit]:
    log = run('log', '--reverse', '--format=%H%x1f%cI%x1f%s', f'{ref}..HEAD')
    commits: list[Commit] = []
    for line in log.splitlines():
        sha, date, subject = line.split('\x1f', 2)
        commits.append(Commit(sha=sha, date=date, subject=subject))
    return commits


def added_lines_by_file(sha: str) -> dict[str, str]:
    """Added lines per prose file in one commit, parsed from a zero-context unified diff."""
    diff = run('show', sha, '--format=', '--unified=0', '--no-color', '--no-renames')
    return _parse_added(diff.splitlines())


def _parse_added(lines: Iterable[str]) -> dict[str, str]:
    added: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in lines:
        if line.startswith('+++ '):
            path = line[4:]
            path = path.removeprefix('b/')
            current = added.setdefault(path, []) if extractor_for(path) is not None else None
        elif line.startswith('+') and current is not None:
            current.append(line[1:])
    return {path: '\n'.join(content) for path, content in added.items() if content}
