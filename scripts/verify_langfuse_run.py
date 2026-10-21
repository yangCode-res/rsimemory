"""Compare delivered Langfuse records with a prepared run, on the server only."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen


def verify(output):
    from scripts import export_langfuse as otlp
    output = Path(output).resolve()
    manifest = json.loads((output / 'prepare.json').read_text())
    source = output / 'traces.jsonl'
    if otlp._input_sha256(source) != manifest['prepared_sha256']:
        raise ValueError('Prepared telemetry hash mismatch')
    root = Path(__file__).resolve().parents[1]
    env = dict(line.split('=', 1) for line in
               (root / 'observability/langfuse/.env').read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    authorization = 'Basic ' + base64.b64encode(('clickhouse:' + env['CLICKHOUSE_PASSWORD']).encode()).decode()

    def query(sql):
        req = Request('http://127.0.0.1:8123/', data=(sql + ' FORMAT JSON').encode(),
                      headers={'Authorization': authorization})
        with urlopen(req, timeout=60) as response:
            return json.load(response)['data']

    identity = manifest['run_id']
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in identity):
        raise ValueError('Unsupported run identity')
    scope = "project_id='memory-policy-rsi' AND session_id='" + identity + "'"
    environment = manifest.get('environment') or next(otlp.iter_records(source)).get('environment')
    if environment:
        if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in environment):
            raise ValueError('Unsupported environment identifier')
        scope += " AND environment='" + environment + "'"
    actual = query('SELECT span_id,trace_id,name,usage_details,prompt_name,prompt_version,environment '
                   'FROM default.events_core FINAL WHERE ' + scope)
    actual_by_id = {row['span_id']: row for row in actual}
    expected = {}
    usage = {}
    mismatches = {'trace_id': 0, 'usage': 0, 'prompt': 0, 'environment': 0}
    for record in otlp.iter_records(source):
        identifier = record['span_id']
        expected[identifier] = record['trace_id']
        row = actual_by_id.get(identifier)
        if row is None:
            continue
        mismatches['trace_id'] += int(row['trace_id'] != record['trace_id'])
        mismatches['environment'] += int(row['environment'] != record['environment'])
        if record.get('kind') == 'generation':
            target = otlp._normalized_usage(record.get('usage'))
            mismatches['usage'] += int(any(row['usage_details'].get(k) != v for k, v in target.items()))
            mismatches['prompt'] += int(row['prompt_name'] != record.get('prompt_name', '') or
                                        row['prompt_version'] != record.get('prompt_version'))
            totals = usage.setdefault(record['name'], {'input': 0, 'output': 0, 'total': 0, 'calls': 0})
            totals['calls'] += 1
            for key in ['input', 'output', 'total']:
                totals[key] += row['usage_details'].get(key, 0)
    score_rows = query('SELECT id,name,value,trace_id,observation_id,environment FROM default.scores FINAL '
                       "WHERE project_id='memory-policy-rsi' AND trace_id IN "
                       '(SELECT trace_id FROM default.events_core WHERE ' + scope + ')')
    score_by_id = {row['id']: row for row in score_rows}
    expected_scores = json.loads((output / 'scores.json').read_text())['scores']
    expected_score_ids = {row['id'] for row in expected_scores}
    score_mismatches = 0
    for score in expected_scores:
        row = score_by_id.get(score['id'])
        if row:
            score_mismatches += int(abs(row['value'] - score['value']) > 1e-9 or
                                    row['trace_id'] != score['traceId'] or
                                    row['observation_id'] != score['observationId'] or
                                    row['environment'] != score.get('environment', 'default'))
    missing = set(expected) - set(actual_by_id)
    extra = set(actual_by_id) - set(expected)
    missing_scores = expected_score_ids - set(score_by_id)
    result = {'run_id': identity, 'expected_observations': len(expected),
              'actual_unique_observations': len(actual_by_id),
              'missing_observations': len(missing), 'extra_observations': len(extra),
              'duplicate_observation_rows': len(actual) - len(actual_by_id),
              'expected_scores': len(expected_scores), 'actual_unique_scores': len(score_by_id),
              'missing_scores': len(missing_scores), 'score_mismatches': score_mismatches,
              'extra_scores': len(set(score_by_id) - expected_score_ids),
              'field_mismatches': mismatches, 'usage': usage,
              'usage_matches': usage == manifest['usage']}
    result['verified'] = (not missing and not extra and not missing_scores and not score_mismatches
                          and not any(mismatches.values()) and result['usage_matches']
                          and not result['duplicate_observation_rows'] and not result['extra_scores'])
    otlp._write_receipt(output / 'verification.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--inside-namespace', action='store_true')
    args = p.parse_args()
    if args.inside_namespace:
        result = verify(args.output)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result['verified'] else 2)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'deploy/langfuse'))
    import hpc
    result = subprocess.run(hpc.namespace_prefix() + ['env', 'PYTHONPATH=' + str(root), str(root / '.venv/bin/python'),
        '-m', 'scripts.verify_langfuse_run', '--output', str(args.output.resolve()), '--inside-namespace'],
        cwd=root, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end='')
    if result.returncode not in (0, 2):
        raise RuntimeError('Server-side verification failed; no clinical payload exported')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
