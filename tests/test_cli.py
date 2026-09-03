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


def test_contrast_top_keeps_only_the_highest_z(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(['baseline', '--ref', 'base', '-o', 'baseline.json']) == 0
    corpus = repo / 'corpus'
    corpus.mkdir()
    (corpus / 'README.md.md').write_text(CONTAMINATED)
    args = ['contrast', '--baseline', 'baseline.json', '--corpus', 'corpus', '--alpha0', '10', '--z', '0.5']
    assert main([*args, '-o', 'full.json']) == 0
    assert main([*args, '-o', 'capped.json', '--top', '2']) == 0
    full = json.loads((repo / 'full.json').read_text())['terms']
    capped = json.loads((repo / 'capped.json').read_text())['terms']
    assert len(full) > 2
    assert list(capped) == list(full)[:2]
    assert main([*args, '-o', 'bad.json', '--top', '0']) == 2
    assert '--top' in capsys.readouterr().err


def test_evaluate_grades_a_split(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    human = repo / 'human'
    model = repo / 'model'
    human.mkdir()
    model.mkdir()
    for index in range(5):
        (human / f'{index}.md').write_text(f'{CLEAN} Version {index} adds nothing new.\n')
        (model / f'{index}.md').write_text(f'{CONTAMINATED} Version {index} adds nothing new.\n')
    args = ['evaluate', '--baseline-dir', 'human', '--corpus-dir', 'model', '--alpha0', '10', '--holdout', '0.4']
    assert main([*args, '--z', '1.0']) == 0
    out = capsys.readouterr().out
    assert 'held out: 2 baseline / 2 model documents' in out
    assert 'AUC: 1.000' in out
    assert 'flags 100% of model documents and 0% of baseline documents' in out
    assert main([*args, '--z', '50']) == 2
    assert 'no n-grams cleared' in capsys.readouterr().err
    assert main([*args, '--holdout', '1.5']) == 2
    assert main(['evaluate', '--baseline-dir', 'missing', '--corpus-dir', 'model']) == 2


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


def test_baseline_and_rewrite_from_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / 'reference'
    (corpus / 'nested').mkdir(parents=True)
    (corpus / 'one.md').write_text(f'# One\n\n{CLEAN}\n')
    (corpus / 'nested' / 'two.md').write_text(CLEAN)
    (corpus / 'ignored.json').write_text('{"leverage": 1}')

    assert main(['baseline', '--dir', 'reference', '-o', 'baseline.json']) == 0
    counts = json.loads((tmp_path / 'baseline.json').read_text())
    assert counts['counts']['parser'] == 2
    assert 'leverage' not in counts['counts']
    assert counts['source'] == 'baseline reference'

    assert main(['baseline', '--dir', 'reference', '--path', 'nested', '-o', 'nested.json', '--force']) == 0
    assert json.loads((tmp_path / 'nested.json').read_text())['counts']['parser'] == 1

    assert main(['rewrite', '--dir', 'reference', '--model', 'test', '-o', 'corpus']) == 0
    assert (tmp_path / 'corpus' / 'one.md.md').exists()
    assert (tmp_path / 'corpus' / 'nested' / 'two.md.md').exists()


def test_source_options_are_exclusive_and_required(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(['baseline', '-o', str(tmp_path / 'b.json')])
    with pytest.raises(SystemExit):
        main(['baseline', '--ref', 'HEAD', '--dir', str(tmp_path), '-o', str(tmp_path / 'b.json')])
    assert main(['baseline', '--dir', str(tmp_path / 'missing'), '-o', str(tmp_path / 'b.json')]) == 2
    assert 'not a directory' in capsys.readouterr().err


def test_scrape_uses_the_openrouter_key_without_a_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # With a key in the environment no OAuth flow runs; the first model request is then refused by the
    # test conftest, which proves the OpenRouter model was built and asked.
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-test')
    with pytest.raises(RuntimeError, match='Model requests are not allowed'):
        main(['scrape', '-o', str(tmp_path / 'corpus')])
