"""Server-only retrieval ablation; never export clinical records or policy source.

Use one fixed middle checkpoint per validation patient, all its questions,
and the exact first successful search input recorded for candidate-3.
No model calls, test split access, answer scoring or candidate selection.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.data import chunk_session, normalize_query
from rsi.policy import GeneratedPolicy, PlainPolicy


def instrument(source, character_tokens=False):
    tree = ast.parse(source)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    tokenizer = functions[0]
    if character_tokens:
        replacement = ast.parse('''
def replacement(text):
    normalized = " ".join(str(text).casefold().split()).strip()
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return [normalized[i:i+2] for i in range(len(normalized)-1)]
''').body[0]
        replacement.name = tokenizer.name
        tree.body[tree.body.index(tokenizer)] = replacement
    search = functions[13]
    sorts = [n for n in search.body if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == 'sorted']
    assert len(sorts) == 2, 'Unexpected policy structure'
    first = sorts[0]
    key = next(k.value for k in first.value.keywords if k.arg == 'key')
    score_variable = key.body.elts[0].func.value.id
    probe = ast.parse('print("DIAG_ZERO_BM25", int(not any(' + score_variable + '.values())))').body
    position = search.body.index(first)
    search.body[position:position] = probe
    # Bulk initialization invokes the exact original ingest in chronological
    # order, avoiding one sandbox launch and snapshot per source chunk.
    # It does not alter the candidate's data structures or update functions.
    bootstrap = ast.parse('''
_diag_initialize = initialize
_diag_ingest = ingest
def initialize(config):
    chunks = config.get("__diagnostic_chunks", [])
    clean = {k:v for k,v in config.items() if k != "__diagnostic_chunks"}
    state = _diag_initialize(clean)
    for chunk in chunks:
        state = _diag_ingest(state, chunk)
    return state
''')
    tree.body.extend(bootstrap.body)
    return ast.unparse(ast.fix_missing_locations(tree))


def mean(items):
    return round(statistics.mean(items), 6) if items else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:4]))
    import pandas as pd

    root = args.run
    config = json.loads((root / 'config.json').read_text())
    source = (root / 'branches/candidate-3/policy.py').read_text()
    rows = [json.loads(line) for line in
            (root / 'branches/candidate-3/validation/answers.jsonl').open()]
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row['persona_id']].append(row)
    selected = {}
    for patient, patient_rows in by_patient.items():
        checkpoints = sorted({row['session_id'] for row in patient_rows})
        checkpoint = checkpoints[(len(checkpoints)-1)//2]
        selected[patient] = [r for r in patient_rows if r['session_id'] == checkpoint]
    chosen_ids = {r['query_id'] for rs in selected.values() for r in rs}
    search_inputs = {}
    for line in (root / 'traces.jsonl').open():
        span = json.loads(line)
        meta = span.get('metadata', {})
        qid = meta.get('query_id')
        if (span.get('name') == 'memory.search' and span.get('status') == 'ok'
                and meta.get('policy') == 'candidate-3' and meta.get('fold') == 'validation'
                and qid in chosen_ids):
            previous = search_inputs.get(qid)
            if previous is None or span['start_time_ns'] < previous['start_time_ns']:
                search_inputs[qid] = {'start_time_ns': span['start_time_ns'],
                                      'input': span['input']}
    if set(search_inputs) != chosen_ids:
        raise ValueError('Cannot pair every selected query with its recorded successful search')
    data_root = args.dataset / 'data/zh'
    filters = [('persona_id', 'in', list(selected))]
    query_frame = pd.read_parquet(data_root / 'queries.parquet', filters=filters)
    queries = {q['query_id']: q for q in
               map(normalize_query, query_frame.to_dict('records')) if q['query_id'] in chosen_ids}
    dialogue = pd.read_parquet(data_root / 'dialogues.parquet', filters=filters)
    sessions = defaultdict(list)
    for (pid, sid), group in dialogue.groupby(['persona_id', 'session_id'], sort=True):
        turns = group.sort_values('turn', kind='stable')
        sessions[int(pid)].append({'persona_id': int(pid), 'session_id': int(sid),
            'text': '\n'.join(str(r.role)+': '+str(r.content) for r in turns.itertuples())})
    variants = ('candidate_original', 'candidate_rerank_disabled',
                'candidate_character_bigrams', 'plain_bm25_same_queries')
    metrics = {variant: defaultdict(list) for variant in variants}
    detail = []
    started = time.monotonic()
    for patient, patient_rows in selected.items():
        checkpoint = patient_rows[0]['session_id']
        chunks = [chunk for session in sessions[patient] if session['session_id'] <= checkpoint
                  for chunk in chunk_session(session, config['chunk_chars'], config['chunk_overlap'])]
        registry = {chunk['id']: chunk for chunk in chunks}
        for variant in variants:
            variant_config = copy.deepcopy(config)
            if variant == 'plain_bm25_same_queries':
                policy = PlainPolicy(variant_config)
                for chunk in chunks:
                    policy.ingest(chunk)
            else:
                if variant == 'candidate_rerank_disabled':
                    variant_config['rerank_weight'] = 0.0
                variant_config['__diagnostic_chunks'] = chunks
                variant_source = instrument(source, variant == 'candidate_character_bigrams')
                policy = GeneratedPolicy(variant_source, variant_config, timeout_s=180)
                policy._registry = copy.deepcopy(registry)
            zero_bm25 = []
            if isinstance(policy, GeneratedPolicy):
                original_call = policy._call

                def capture(operation, payload):
                    result = original_call(operation, payload)
                    if operation == 'search':
                        for message in result.get('stdout_excerpt', '').splitlines():
                            if message.startswith('DIAG_ZERO_BM25 '):
                                zero_bm25.append(int(message.split()[-1]))
                    return result
                policy._call = capture
            for row in patient_rows:
                q = queries[row['query_id']]
                request = search_inputs[row['query_id']]['input']
                scope = dict(request.get('filters') or {})
                scope.update(persona_id=patient, checkpoint=checkpoint)
                returned = policy.search(request['query'], scope)
                actual_ids = [entry['id'] for entry in returned]
                logged_ids = next(op['ids'] for op in row['tool_audit'] if op.get('action') == 'search')
                gold = {int(point['session_id']) for point in q['evidence']
                        if isinstance(point, dict) and 'session_id' in point
                        and 0 < int(point['session_id']) <= checkpoint}
                got = {entry['session_id'] for entry in returned}
                m = metrics[variant]
                m['exact_reproduction_of_original_ids'].append(actual_ids == logged_ids)
                m['current_session_fraction'].append(
                    sum(entry['session_id'] == checkpoint for entry in returned)/max(1, len(returned)))
                m['unique_sessions'].append(len(got))
                if gold:
                    m['gold_session_hit'].append(bool(got & gold))
                    m['gold_session_recall'].append(len(got & gold)/len(gold))
                detail.append({'variant':variant, 'query_id':row['query_id'],
                               'returned_ids':actual_ids, 'reference_sessions':sorted(gold)})
            metrics[variant]['zero_bm25'].extend(zero_bm25)
            print(json.dumps({'event':'group_complete', 'variant':variant,
                              'questions':len(patient_rows),
                              'elapsed_s':round(time.monotonic()-started, 1)}), flush=True)
    report = {'scope':'retrieval_diagnosis_only_no_answer_rescoring',
              'selection':'all queries at fifth of ten validation checkpoints per patient',
              'patients':len(selected), 'questions':len(chosen_ids),
              'variants':{v:{k:{'mean':mean(vals), 'count':len(vals), 'sum':sum(vals)}
                             for k,vals in values.items()} for v,values in metrics.items()}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'aggregate.json').write_text(json.dumps(report, indent=2)+'\n')
    # Keep all per-query identifiers on the server, outside the immutable run.
    (args.output / 'server-only-detail.json').write_text(json.dumps(detail)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
