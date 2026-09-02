from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from vocabguard.cli import main

CLEAN = (
    'The parser reads each line, splits it on tabs, and hands the fields to the renderer, which '
    'writes one row per record and skips blank lines without complaint.'
)
CONTAMINATED = (
    'We leverage a robust pipeline here, and this section will delve into why the robust design '
    'lets us leverage every stage without extra work from the caller or the reader.'
)


def git(*args: str) -> None:
    subprocess.run(['git', *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    git('init', '-q', '-b', 'main')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    (tmp_path / 'README.md').write_text(f'# Parser\n\n{CLEAN}\n')
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'mod.py').write_text(f'"""{CLEAN}"""\n\n\ndef leverage() -> None:\n    pass\n')
    (tmp_path / 'data.json').write_text('{"leverage": true}')
    git('add', '-A')
    git('commit', '-q', '-m', 'base')
    git('tag', 'base')
    return tmp_path


def test_baseline_is_frozen_once_written(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 0
    counts = json.loads((repo / 'baseline.json').read_text())
    assert counts['total'] > 0
    assert 'parser' in counts['counts']
    assert 'leverage' not in counts['counts']
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 2
    assert 'frozen' in capsys.readouterr().err
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json', '--force']) == 0


def test_baseline_path_filter(repo: Path) -> None:
    assert main(['baseline', '--ref', 'base', '--path', 'src', '-o', 'src.json']) == 0
    counts = json.loads((repo / 'src.json').read_text())
    # README.md is excluded, so its heading is gone while the module docstring's prose remains.
    assert 'parser' in counts['counts']
    assert counts['counts']['parser'] == 1


def test_rewrite_uses_the_named_model_and_resumes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['rewrite', '--ref', 'base', '--model', 'test', '-o', 'corpus']) == 0
    rewritten = repo / 'corpus' / 'README.md.md'
    assert rewritten.exists()
    assert (repo / 'corpus' / 'src' / 'mod.py.md').exists()
    assert not (repo / 'corpus' / 'data.json.md').exists()
    assert main(['rewrite', '--ref', 'base', '--model', 'test', '-o', 'corpus']) == 0
    assert 'already rewritten' in capsys.readouterr().err


def test_contrast_writes_watchlist_and_merges_curated(repo: Path) -> None:
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 0
    corpus = repo / 'corpus'
    corpus.mkdir()
    (corpus / 'README.md.md').write_text(CONTAMINATED)
    (repo / 'curated.json').write_text(json.dumps({'replacements': {'leverage': 'use'}, 'banned_patterns': ['\u2014']}))
    assert (
        main(
            [
                'contrast',
                '--baseline',
                'baseline.json',
                '--corpus',
                'corpus',
                '-o',
                'watchlist.json',
                '--alpha0',
                '10',
                '--z',
                '1.0',
                '--curated',
                'curated.json',
            ]
        )
        == 0
    )
    watchlist = json.loads((repo / 'watchlist.json').read_text())
    assert watchlist['terms']['leverage'] > 1.0
    assert 'parser' not in watchlist['terms']
    assert watchlist['replacements'] == {'leverage': 'use'}
    assert watchlist['banned_patterns'] == ['\u2014']


def test_contrast_rejects_empty_corpus(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 0
    (repo / 'empty').mkdir()
    assert main(['contrast', '--baseline', 'baseline.json', '--corpus', 'empty']) == 2
    assert 'no prose found' in capsys.readouterr().err


def test_check_reports_hits_and_exit_code(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (repo / 'bad.md').write_text(CONTAMINATED)
    assert main(['check', 'README.md', 'src']) == 0
    assert main(['check', 'bad.md', 'README.md']) == 1
    out = capsys.readouterr().out
    assert out.startswith('bad.md: vocabulary score')
    assert "'leverage'" in out
    assert main(['check', 'bad.md', '--threshold', '100']) == 0


def test_check_diff_base_scores_only_additions(repo: Path) -> None:
    bad = repo / 'bad.md'
    bad.write_text(f'{CONTAMINATED}\n')
    git('add', 'bad.md')
    git('commit', '-q', '-m', 'add contaminated file')
    git('tag', 'contaminated')
    bad.write_text(f'{CONTAMINATED}\n\n{CLEAN}\n')
    assert main(['check', 'bad.md']) == 1
    assert main(['check', 'bad.md', '--diff-base', 'contaminated']) == 0
    assert main(['check', 'bad.md', '--diff-base', 'base']) == 1


def test_report_rows_per_commit(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 0
    (repo / 'bad.md').write_text(f'{CONTAMINATED}\n')
    git('add', 'bad.md')
    git('commit', '-q', '-m', 'drifting commit')
    (repo / 'data.json').write_text('{}')
    git('add', 'data.json')
    git('commit', '-q', '-m', 'no prose here')
    (repo / 'README.md').write_text(f'# Parser\n\n{CLEAN}\n\n{CLEAN}\n')
    git('add', 'README.md')
    git('commit', '-q', '-m', 'same voice as the baseline')

    assert main(['report', '--baseline', 'baseline.json', '--since', 'base', '--format', 'json']) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row['subject'] for row in rows] == ['drifting commit', 'same voice as the baseline']
    drifting, same = rows
    assert drifting['hits_per_thousand_words'] > 0
    assert same['hits_per_thousand_words'] == 0
    assert drifting['js_divergence'] > same['js_divergence']

    assert main(['report', '--baseline', 'baseline.json', '--since', 'base']) == 0
    csv_lines = capsys.readouterr().out.splitlines()
    assert csv_lines[0] == 'sha,date,subject,words,js_divergence,hits_per_thousand_words'
    assert len(csv_lines) == 3


def test_git_failure_is_a_user_error(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['baseline', '--ref', 'no-such-ref', '-o', 'baseline.json']) == 2
    assert 'git ls-tree failed' in capsys.readouterr().err
