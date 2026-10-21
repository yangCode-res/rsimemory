"""Group existing question traces by pseudonymous patient without rerunning them."""
import argparse
import json
import os
from pathlib import Path
from scripts import export_langfuse as otlp
from scripts.langfuse_run import Client
from rsi.tracing import patient_label


def index_patients(run):
    run = Path(run)
    config = json.loads((run / 'config.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    policies = ['plain-bm25', 'plain-semantic'] + [b['id'] for b in summary.get('branches', [])]
    lookup, groups = {}, {}
    for policy in policies:
        base = run / ('branches/' + policy if policy.startswith('candidate-') else policy)
        for fold in ('train', 'validation'):
            path = base / fold / 'answers.jsonl'
            if not path.exists():
                continue
            for line in path.open():
                row = json.loads(line)
                label = patient_label(row['persona_id'], config.get('data_revision'))
                key = (fold, str(row['query_id']))
                if key in lookup and lookup[key] != label:
                    raise ValueError('Ambiguous question-to-patient mapping')
                lookup[key] = label
                group = groups.setdefault(label, {'fold': fold, 'policies': {}})
                if group['fold'] != fold:
                    raise ValueError('A patient occurs in more than one split')
                group['policies'].setdefault(policy, []).append(
                    {key: row.get(key) for key in ('query_type', 'score', 'status')})
    return lookup, groups


def group_record(record, lookup, identity):
    metadata = record.get('metadata', {})
    label = metadata.get('patient')
    if not label and metadata.get('query_id') is not None:
        label = lookup.get((metadata.get('fold'), str(metadata['query_id'])))
    if label:
        metadata['patient'] = label
        metadata['patient_session_id'] = identity + '/' + label
        record['session_id'] = metadata['patient_session_id']
        tags = record.setdefault('tags', [])
        if label not in tags:
            tags.append(label)
    return record


def publish(run, imported, client, env):
    run, imported = Path(run), Path(imported)
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root / 'deploy/langfuse/observability.json').read_text())
    if not profile.get('patient_grouping', {}).get('enabled', False):
        return {'enabled': False}
    version = profile['patient_grouping']['version']
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in version):
        raise ValueError('Invalid patient view version')
    manifest = json.loads((imported / 'prepare.json').read_text())
    identity = manifest['run_id']
    config = json.loads((run / 'config.json').read_text())
    environment = 'fixture' if config.get('fixture') else 'research'
    lookup, groups = index_patients(run)
    target = imported / version
    target.mkdir(exist_ok=True, mode=0o700)
    # Keep experiment context when enriching a previously imported v4 run.
    source = imported / 'experiment-enrichment.jsonl'
    if not source.exists():
        source = imported / 'traces.jsonl'
    path = target / 'traces.jsonl'
    patients, questions, trace_ids = {}, 0, set()
    temporary = target / 'traces.preparing.jsonl'
    if not path.exists():
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream:
            for record in otlp.iter_records(source):
                if record.get('metadata', {}).get('query_id') is None:
                    continue
                record = group_record(record, lookup, identity)
                if not record.get('session_id'):
                    raise ValueError('A question observation could not be assigned to a patient')
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        os.replace(temporary, path)
    usage = {}
    count = 0
    for record in otlp.iter_records(path):
        label = record['metadata']['patient']
        trace_ids.add(record['trace_id'])
        count += 1
        group = patients.setdefault(label, {'session_id': record['session_id'], 'question_count': 0})
        if record['name'] == 'question':
            questions += 1
            group['question_count'] += 1
        if record.get('kind') == 'generation':
            values = otlp._normalized_usage(record.get('usage'))
            totals = usage.setdefault(record['name'], {'input': 0, 'output': 0, 'total': 0, 'calls': 0})
            totals['calls'] += 1
            for key in ('input', 'output', 'total'):
                totals[key] += values.get(key, 0)
    original_scores = json.loads((imported / 'scores.json').read_text())['scores']
    otlp._write_receipt(target / 'scores.json', {'scores': [s for s in original_scores if s['traceId'] in trace_ids]})
    otlp._write_receipt(target / 'prepare.json', {
        'run_id': identity, 'source_sha256': manifest['source_sha256'],
        'prepared_sha256': otlp._input_sha256(path), 'usage': usage, 'environment': environment})
    receipt = otlp.export(path, target / 'trace-receipt.json', env=env, max_batch_bytes=2 * 1024 * 1024)
    if receipt.get('delivery_status') != 'accepted':
        raise ValueError('Patient grouping delivery incomplete')
    result = {'run_id': identity, 'patients': patients, 'questions': questions,
              'observations': count, 'trace_count': len(trace_ids), 'environment': environment}
    otlp._write_receipt(target / 'index.json', result)
    # No new scores or generations: this projection only assigns session context.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--env-file', required=True, type=Path)
    args = parser.parse_args()
    env = dict(line.split('=', 1) for line in args.env_file.read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    env.update(LANGFUSE_BASE_URL='http://127.0.0.1:3000',
               LANGFUSE_PUBLIC_KEY=env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY'],
               LANGFUSE_SECRET_KEY=env['LANGFUSE_INIT_PROJECT_SECRET_KEY'])
    result = publish(args.run, args.output, Client(env), env)
    print(json.dumps({key: value for key, value in result.items() if key != 'patients'}), flush=True)


if __name__ == '__main__':
    main()
