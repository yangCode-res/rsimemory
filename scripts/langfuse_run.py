"""Publish a completed RSI run without changing its frozen artifacts.

All inputs and derived telemetry stay on the experiment server. Receipt files
make metadata/score delivery resumable; trace delivery uses the OTLP receipt.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request
from scripts import export_langfuse as otlp


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Client:
    def __init__(self, env):
        self.base = otlp._base_url(env)
        if not otlp._is_loopback(self.base):
            raise ValueError('Only the server loopback endpoint is supported')
        self.auth = base64.b64encode((env['LANGFUSE_PUBLIC_KEY'] + ':' + env['LANGFUSE_SECRET_KEY']).encode()).decode()

    def request(self, path, body=None):
        req = Request(self.base + path,
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Authorization': 'Basic ' + self.auth, 'Content-Type': 'application/json'})
        with otlp._safe_urlopen(req, timeout=60) as response:
            return json.load(response)


def score(record, name, value, comment=''):
    return {'id': digest([record['trace_id'], record['span_id'], name]),
            'traceId': record['trace_id'], 'observationId': record['span_id'],
            'name': name, 'value': value, 'dataType': 'NUMERIC', 'comment': comment,
            'environment': record.get('environment', 'default')}


def question_scores(record):
    output = record.get('output') or {}
    value = output.get('score')
    status = output.get('status')
    scores = []
    if status == 'scored' and type(value) is int and value in (0, 1):
        scores.append(score(record, 'correctness', value, 'Original custom engineering rubric; not clinically calibrated.'))
        scores.append(score(record, 'e2e_correctness', value))
    elif status == 'actor_failed' and value is None:
        scores.append(score(record, 'e2e_correctness', 0, 'Unanswered counted as zero only under strict-plus-e2e-v1.'))
    scores.append(score(record, 'grading_abnormal', int(status != 'scored')))
    return scores


def prompt_spec(record):
    messages = (record.get('input') or {}).get('messages', [])
    systems = [m for m in messages if m.get('role') == 'system']
    if not systems:
        return None
    role = record['name'].removeprefix('llm.')
    return {'name': 'rsi/' + role + '/system-' + digest(systems)[:16],
            'type': 'chat', 'prompt': systems, 'labels': [],
            'tags': ['rsi', 'frozen-system-prompt'],
            'config': {'scope': 'system_messages_only', 'source_sha256': digest(systems),
                       'full_request_location': 'generation.input.messages'}}


def prepare(run, output, client, fixture=False):
    run = Path(run).resolve()
    output = Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError('Derived telemetry must be outside the archived run')
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    source = run / 'traces.jsonl'
    config = json.loads((run / 'config.json').read_text())
    identity = run.parent.name if run.name == 'result' else run.name
    source_hash = otlp._input_sha256(source)
    manifest_path = output / 'prepare.json'
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing['source_sha256'] != source_hash:
            raise ValueError('Source changed; choose a new output directory')
        return existing
    records = list(otlp.iter_records(source))
    if fixture:
        if not config.get('fixture'):
            raise ValueError('Fixture selection requires a synthetic fixture run')
    summary = json.loads((run / 'summary.json').read_text())
    commits = {b['id']: b.get('commit') for b in summary.get('branches', [])}
    prompts = {}
    registry_path = output / 'prompts.json'
    registry = json.loads(registry_path.read_text()) if registry_path.exists() else {}
    for record in records:
        if record.get('kind') == 'generation':
            spec = prompt_spec(record)
            if spec and spec['name'] not in registry:
                # Names bind content hashes: reuse an existing immutable version on retry.
                try:
                    result = client.request('/api/public/v2/prompts/' + quote(spec['name'], safe='') + '?version=1')
                except Exception as exc:
                    if getattr(exc, 'code', None) != 404:
                        raise
                    result = client.request('/api/public/v2/prompts', spec)
                if result.get('prompt') != spec['prompt']:
                    raise ValueError('Prompt content hash/name conflict')
                registry[spec['name']] = result['version']
                otlp._write_receipt(registry_path, registry)
            if spec:
                prompts[record['span_id']] = (spec['name'], registry[spec['name']])
    scores = []
    trace_path = output / 'traces.jsonl'
    counts = {'records': 0, 'questions': 0, 'generations': 0, 'errors': 0}
    totals = {}
    temporary_trace = output / 'traces.preparing.jsonl'
    fd = os.open(temporary_trace, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as stream:
        for record in records:
            metadata = record.setdefault('metadata', {})
            metadata['original_run_id'] = metadata.get('run_id')
            metadata.update(run_id=identity, telemetry_version='rsi-langfuse-v2',
                            source_trace_sha256=source_hash, currency_cost_status='unpriced_local_compute',
                            fixture=bool(config.get('fixture')), language=config.get('language'),
                            split_version=config.get('split_version'))
            if metadata.get('policy') in commits:
                metadata['policy_commit'] = commits[metadata['policy']]
            record['environment'] = 'fixture' if config.get('fixture') else 'research'
            record['tags'] = ['rsi', identity, record['environment']]
            for key in ['policy', 'fold']:
                if metadata.get(key): record['tags'].append(str(metadata[key]))
            if record['span_id'] in prompts:
                record['prompt_name'], record['prompt_version'] = prompts[record['span_id']]
            if record['name'] == 'question':
                counts['questions'] += 1
                scores.extend(question_scores(record))
                metadata['answer_status'] = (record.get('output') or {}).get('status')
                metadata['score_scope'] = 'per_question_not_macro_AVG'
            if record['name'] == 'replay':
                report = record.get('output') or {}
                for key in ['AVG', 'E2E_AVG', 'abnormal_rate']:
                    if isinstance(report.get(key), (int, float)):
                        scores.append(score(record, key, report[key], 'Six task types equally weighted for AVG/E2E_AVG.'))
                cost = report.get('cost', {})
                rows = cost.get('policy', [])
                seconds = sum(row.get('sandbox_seconds', row.get('ingest_seconds', 0) + row.get('search_seconds', 0)) for row in rows) + cost.get('search_seconds', 0)
                storage = max([row.get('peak_storage_bytes', row.get('storage_bytes', 0)) for row in rows] or [0])
                scores.extend([score(record, 'policy_seconds', seconds, 'Includes sandbox overhead for generated policies.'),
                               score(record, 'policy_peak_bytes', storage)])
            if record.get('kind') == 'generation':
                counts['generations'] += 1
                usage = otlp._normalized_usage(record.get('usage'))
                role = record['name']
                target = totals.setdefault(role, {'input': 0, 'output': 0, 'total': 0, 'calls': 0})
                target['calls'] += 1
                for key in ['input', 'output', 'total']: target[key] += usage.get(key, 0)
            counts['records'] += 1
            counts['errors'] += int(record.get('status') == 'error')
            stream.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
    os.replace(temporary_trace, trace_path)
    otlp._write_receipt(output / 'scores.json', {'scores': scores})
    manifest = {'source_sha256': source_hash, 'run_id': identity, 'counts': counts,
                'score_count': len(scores), 'usage': totals, 'prompts': registry,
                'prepared_sha256': otlp._input_sha256(trace_path),
                'root_trace_id': next(r['trace_id'] for r in records if r['name'] == 'rsi-evolution-round')}
    otlp._write_receipt(manifest_path, manifest)
    return manifest


def deliver_scores(output, client):
    output = Path(output)
    path = output / 'score-receipt.json'
    payload = json.loads((output / 'scores.json').read_text())['scores']
    receipt = json.loads(path.read_text()) if path.exists() else {'accepted_ids': []}
    accepted = set(receipt['accepted_ids'])
    pending = [s for s in payload if s['id'] not in accepted]
    from datetime import datetime, timezone
    for i in range(0, len(pending), 100):
        batch = pending[i:i + 100]
        events = [{'id': 'event-' + s['id'], 'type': 'score-create',
                   'timestamp': datetime.now(timezone.utc).isoformat(), 'body': s} for s in batch]
        result = client.request('/api/public/ingestion', {'batch': events})
        if result.get('errors'):
            raise RuntimeError('Score ingestion reported errors; receipt preserved')
        acknowledged = {s['id'] for s in result.get('successes', [])}
        if acknowledged != {e['id'] for e in events}:
            raise RuntimeError('Score ingestion did not acknowledge every event')
        accepted.update(s['id'] for s in batch)
        receipt['accepted_ids'] = sorted(accepted)
        otlp._write_receipt(path, receipt)
    return len(accepted)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--env-file', required=True, type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--fixture', action='store_true')
    args = p.parse_args()
    env = dict(line.split('=', 1) for line in args.env_file.read_text().splitlines()
               if line and not line.startswith('#') and '=' in line)
    env['LANGFUSE_PUBLIC_KEY'] = env.get('LANGFUSE_PUBLIC_KEY') or env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']
    env['LANGFUSE_SECRET_KEY'] = env.get('LANGFUSE_SECRET_KEY') or env['LANGFUSE_INIT_PROJECT_SECRET_KEY']
    env['LANGFUSE_BASE_URL'] = 'http://127.0.0.1:3000'
    client = Client(env)
    manifest = prepare(args.run, args.output, client, args.fixture)
    print(json.dumps(manifest), flush=True)
    if args.prepare_only:
        return
    receipt = otlp.export(args.output / 'traces.jsonl', args.output / 'trace-receipt.json',
                          env=env, max_batch_bytes=2 * 1024 * 1024)
    if receipt.get('delivery_status') != 'accepted':
        raise RuntimeError('Trace delivery incomplete; inspect receipt')
    score_count = deliver_scores(args.output, client)
    print(json.dumps({'accepted_spans': len(receipt['accepted_ids']), 'accepted_scores': score_count}), flush=True)


if __name__ == '__main__':
    main()
