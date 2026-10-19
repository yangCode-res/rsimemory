"""Supplement completed replay reports with E2E metrics without editing runs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.evaluation import aggregate


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def build_report(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    immutable = run.parent if (run.parent / '_archive-manifest.json').exists() else run
    if output.is_relative_to(immutable):
        raise ValueError('Supplement must be outside the original run archive')
    if output.exists():
        raise FileExistsError('Use a fresh supplement directory')
    results = {}
    sources = {}
    directories = [run / 'plain-bm25', run / 'plain-semantic']
    directories.extend(sorted((run / 'branches').glob('candidate-*')))
    for policy in directories:
        for fold in ('train', 'validation'):
            report_path = policy / fold / 'report.json'
            if not report_path.exists():
                continue
            path = report_path.parent
            progress = json.loads((path / 'progress.json').read_text())
            with (path / 'answers.jsonl').open() as stream:
                rows = [json.loads(line) for line in stream]
            if (progress.get('status') != 'completed'
                    or progress.get('total_questions') != len(rows)
                    or progress.get('completed') != len(rows)):
                raise ValueError('Cannot publish E2E metrics for incomplete replay')
            original = json.loads(report_path.read_text())
            result = aggregate(rows)
            if result['AVG'] != original['AVG'] or any(
                    result['counts'][key] != original['counts'][key]
                    for key in ('total', 'scored', 'abnormal')):
                raise ValueError('Strict metrics do not reproduce the original report')
            results.setdefault(policy.name, {})[fold] = result
            sources[str(report_path.relative_to(run))] = {
                'report_sha256': digest(report_path),
                'answers_sha256': digest(path / 'answers.jsonl'),
                'strict_metrics_reproduced': True,
            }
    if not results:
        raise ValueError('No completed train/validation reports found')
    try:
        commit = subprocess.check_output(
            ['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        commit = None
    summary = {'metric_version': 'strict-plus-e2e-v1',
               'created_at': datetime.now(timezone.utc).isoformat(),
               'implementation_commit': commit, 'source_run': str(run),
               'original_run_modified': False, 'model_calls': 0,
               'test_evaluated': False, 'selection_rules_changed': False,
               'reports': results, 'sources': sources}
    os.umask(0o077)
    output.mkdir(parents=True, mode=0o700)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# E2E 补充报告', '',
             '未作答 actor_failed 计 0；judge 异常、缺类及基础设施中断不补为 0。',
             '六类分别计算后等权平均。原严格 AVG、原日志和选择规则保留。', '',
             '|版本|划分|严格 AVG|E2E AVG|异常数/题数|异常率|',
             '|---|---|---:|---:|---:|---:|']
    for policy, folds in results.items():
        for fold, report in folds.items():
            counts = report['counts']
            lines.append(f'|{policy}|{fold}|{report["AVG"]}|{report["E2E_AVG"]}|'
                         f'{counts["abnormal"]}/{counts["total"]}|{report["abnormal_rate"]}|')
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = build_report(args.run, args.output)
    print(json.dumps({'output': str(args.output), 'reports': summary['reports']}))


if __name__ == '__main__':
    main()
