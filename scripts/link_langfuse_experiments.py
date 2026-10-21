"""Enrich existing traces with native v4 experiment context; never rerun models."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from urllib.parse import quote
from scripts.langfuse_run import Client, digest
from scripts import export_langfuse as otlp


def link(run, output, client, env):
    run, output = Path(run), Path(output)
    manifest = json.loads((output / 'prepare.json').read_text())
    verification = json.loads((output / 'verification.json').read_text())
    if not verification.get('verified'):
        raise ValueError('Verify observations and scores before linking experiments')
    config = json.loads((run / 'config.json').read_text())
    identity = manifest['run_id']
    if config.get('fixture'):
        raise ValueError('Use separate synthetic API verification, not a real dataset namespace')
    questions = [r for r in otlp.iter_records(output / 'traces.jsonl') if r['name'] == 'question']
    if any(r['metadata'].get('fold') not in ('train', 'validation') for r in questions):
        raise ValueError('Only already-evaluated train and validation traces are supported')
    revision = digest([config.get('data_revision'), config.get('split_version'), config.get('language')])[:12]
    datasets = {fold: 'rsi-medmemory-' + config['language'] + '-' + revision + '-' + fold
                for fold in ('train', 'validation')}
    receipt_path = output / 'experiment-receipt.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {
        'source_sha256': manifest['source_sha256'], 'items': {}, 'links': {}, 'datasets': {}}
    if receipt['source_sha256'] != manifest['source_sha256']:
        raise ValueError('Experiment receipt source mismatch')
    for fold, name in datasets.items():
        if fold in receipt['datasets']:
            continue
        try:
            dataset = client.request('/api/public/datasets/' + quote(name, safe=''))
        except Exception as exc:
            if getattr(exc, 'code', None) != 404:
                raise
            dataset = client.request('/api/public/datasets', {
                'name': name, 'description': 'Frozen local MedMemoryBench ' + fold + '; no test data.',
                'metadata': {'language': config['language'], 'split_version': config['split_version'],
                             'data_revision': config.get('data_revision'), 'expected_output_status': 'not_imported'}})
        receipt['datasets'][fold] = {'name': name, 'id': dataset['id']}
        otlp._write_receipt(receipt_path, receipt)
    unique = {}
    for record in questions:
        metadata = record['metadata']
        fold = metadata['fold']
        identifier = digest([datasets[fold], metadata['query_id']])
        value = {'id': identifier, 'datasetName': datasets[fold], 'input': record['input'],
                 'metadata': {k: metadata.get(k) for k in ('query_id', 'query_type', 'checkpoint')}}
        if identifier in unique and unique[identifier]['input'] != value['input']:
            raise ValueError('Question identity maps to inconsistent input across policies')
        unique[identifier] = value

    def item_work(value):
        try:
            item = client.request('/api/public/dataset-items/' + quote(value['id'], safe=''))
            if item['input'] != value['input']:
                raise ValueError('Existing dataset item differs from frozen question')
        except Exception as exc:
            if getattr(exc, 'code', None) != 404:
                raise
            item = client.request('/api/public/dataset-items', value)
        return value['id'], item['id']

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(item_work, value) for key, value in unique.items() if key not in receipt['items']]
        for future in as_completed(futures):
            key, value = future.result()
            receipt['items'][key] = value
            otlp._write_receipt(receipt_path, receipt)

    contexts = {}
    expected = {}
    for root in questions:
        metadata = root['metadata']
        fold, policy = metadata['fold'], metadata['policy']
        name = identity + '/' + policy + '/' + fold
        identifier = 'rsi-' + digest(name)[:32]
        item_id = receipt['items'][digest([datasets[fold], metadata['query_id']])]
        attributes = {'langfuse.experiment.id': identifier,
                      'langfuse.experiment.name': name,
                      'langfuse.experiment.dataset.id': receipt['datasets'][fold]['id'],
                      'langfuse.experiment.item.id': item_id,
                      'langfuse.experiment.item.root_observation_id': root['span_id']}
        contexts[root['trace_id']] = attributes
        group = expected.setdefault(identifier, {'name': name, 'items': {}})
        group['items'][root['span_id']] = root['trace_id']
    enrichment = output / 'experiment-enrichment.jsonl'
    if not enrichment.exists():
        temporary = output / 'experiment-enrichment.preparing.jsonl'
        fd = os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream:
            for record in otlp.iter_records(output / 'traces.jsonl'):
                if record['trace_id'] not in contexts:
                    continue
                record['experiment_attributes'] = dict(contexts[record['trace_id']])
                if record['name'] == 'question':
                    record['experiment_attributes']['langfuse.experiment.description'] = (
                        'Frozen actor; recorded replay, no model rerun. Per-question means are not six-type macro AVG; '
                        'original AVG/E2E_AVG remain on replay observations.')
                stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
        os.replace(temporary, enrichment)
    result = otlp.export(enrichment, output / 'experiment-v4-delivery.json', env=env,
                         max_batch_bytes=2 * 1024 * 1024)
    if result.get('delivery_status') != 'accepted':
        raise ValueError('Experiment enrichment delivery incomplete')
    from datetime import datetime, timezone
    first = min(root['start_time_ns'] for root in questions)
    since = datetime.fromtimestamp(first / 1e9 - 86400, timezone.utc).isoformat()
    report = {'run_id': identity, 'dataset_items': len(unique), 'runs': {}, 'verified': False}
    for identifier, group in expected.items():
        actual = {}
        cursor = None
        while True:
            path = ('/api/public/experiment-items?experimentId=' + quote(identifier, safe='') +
                    '&fromStartTime=' + quote(since, safe='') + '&limit=100')
            if cursor:
                path += '&cursor=' + quote(cursor, safe='')
            data = client.request(path)
            for item in data['data']:
                actual[item['id']] = item['traceId']
            next_cursor = data.get('meta', {}).get('nextCursor') or data.get('meta', {}).get('cursor')
            if not next_cursor:
                break
            if next_cursor == cursor:
                raise ValueError('Experiment API pagination did not advance')
            cursor = next_cursor
        report['runs'][group['name']] = {'expected': len(group['items']), 'actual': len(actual),
                                        'matches': actual == group['items'], 'id': identifier}
    report['verified'] = all(value['matches'] for value in report['runs'].values())
    otlp._write_receipt(output / 'experiment-verification.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--env-file', required=True, type=Path)
    args = p.parse_args()
    env = dict(line.split('=', 1) for line in args.env_file.read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    env.update(LANGFUSE_BASE_URL='http://127.0.0.1:3000',
               LANGFUSE_PUBLIC_KEY=env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY'],
               LANGFUSE_SECRET_KEY=env['LANGFUSE_INIT_PROJECT_SECRET_KEY'])
    result = link(args.run, args.output, Client(env), env)
    print(json.dumps(result), flush=True)
    raise SystemExit(0 if result['verified'] else 2)


if __name__ == '__main__':
    main()
