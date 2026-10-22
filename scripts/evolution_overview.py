"""Project recorded evolution evidence into a compact Langfuse review trace."""
import json
import gzip
import math
import os
from pathlib import Path
from scripts import export_langfuse as otlp
from scripts.langfuse_run import digest, score, deliver_scores


def delta(value, parent):
    if value is None or parent is None:
        return None
    return 100 * (value - parent)


def comparison(report, baseline, include_types=True):
    result = {key: report.get(key) for key in ('AVG', 'E2E_AVG', 'abnormal_rate', 'counts')}
    result['delta_vs_parent_pp'] = {key: delta(report.get(key), baseline.get(key))
                                    for key in ('AVG', 'E2E_AVG')}
    if include_types:
        result['type_accuracy'] = report.get('type_accuracy')
        result['e2e_type_accuracy'] = report.get('e2e_type_accuracy')
        result['type_delta_vs_parent_pp'] = {
            key: delta(value, baseline.get('e2e_type_accuracy', {}).get(key))
            for key, value in report.get('e2e_type_accuracy', {}).items()}
    return result



def read_source(imported):
    """Read archived derived payloads without restoring large uncompressed copies."""
    path = Path(imported) / 'traces.jsonl'
    if path.exists():
        yield from otlp.iter_records(path)
    else:
        with gzip.open(str(path) + '.gz', 'rt') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def search_state(tree, policy):
    nodes = tree.get('nodes', {})
    node = nodes.get(policy)
    if not node:
        return {'availability': 'not_recorded', 'selected_next': None}
    visits = node.get('visits', 0)
    parent = nodes.get(node.get('parent'), {})
    c = tree.get('c', 1.414)
    result = {'availability': 'recorded_tree_snapshot', 'Q': node.get('Q'),
              'visits': visits, 'parent_visits': parent.get('visits'), 'c': c,
              'reward': tree.get('reward','strict_train_AVG_only') + '; Q includes descendant backpropagation',
              'formula': 'Q + c * sqrt(log(max(1, parent_visits)) / visits)',
              'selected_next': None, 'selection_history': 'see_recorded_search_expansions' if tree.get('validation_role')=='development' else 'not_recorded',
              'uct_kind': 'derived_snapshot_not_historical_selection',
              'status': node.get('status')}
    if node.get('status') in ('invalid', 'runtime_failed'):
        result.update(uct=None, exploration_bonus=None, priority='excluded_negative_infinity')
    elif not visits:
        result.update(uct=None, exploration_bonus=None, priority='unvisited_positive_infinity',
                      eligibility_note='Raw UCT rule; incomplete scoring is not proof of eligibility or selection.')
    elif isinstance(node.get('Q'), (int, float)) and parent.get('visits') is not None:
        bonus = c * math.sqrt(math.log(max(1, parent['visits'])) / visits)
        result.update(uct=node['Q'] + bonus, exploration_bonus=bonus, priority='finite')
    else:
        result.update(uct=None, exploration_bonus=None, priority='insufficient_recorded_state')
    return result


def diagnostic_evidence(run):
    run = Path(run)
    def load(name):
        path = run / name
        return json.loads(path.read_text()) if path.exists() else {}
    reflection = load('reflexion.json')
    summary = load('diagnostics/summary.json')
    selection = load('diagnostics/selection.json')
    # No patient identifiers or clinical case text in the aggregate summary.
    aggregate = {key: summary.get(key) for key in
                 ('total_questions', 'success_count', 'failure_count', 'abnormal_count', 'error_groups')}
    aggregate['by_type'] = summary.get('outcome_distributions', {}).get('by_type')
    return {'evidence_scope': 'parent_training_only',
            'observed_groups': aggregate,
            'group_interpretation': 'Observed proxies, not established causes; do not infer causal prevalence from these groups.',
            'sample_coverage': selection.get('actual', {'total': reflection.get('diagnostic_example_count')}),
            'clusters': reflection.get('clusters', []),
            'cluster_interpretation': 'Model-reported hypotheses on selected evidence; case counts are cited examples, not population prevalence.',
            'rationale_schema': reflection.get('rationale_schema'),
            'diagnosis_status': reflection.get('diagnosis_status', 'historical_recorded' if reflection else 'unavailable')}


def build(run, imported, profile):
    run, imported = Path(run), Path(imported)
    manifest = json.loads((imported / 'prepare.json').read_text())
    summary = json.loads((run / 'summary.json').read_text())
    tree = json.loads((run / 'tree.json').read_text())
    identity = manifest['run_id']
    version = profile.get('version', 'evolution-overview-v1')
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in version):
        raise ValueError('Overview version must be a lowercase filename-safe identifier')
    trace_id = digest([identity, version])[:32]
    root_id = digest([trace_id, 'root'])[:16]
    questions = {}
    sources = {}
    root = None
    for row in read_source(imported):
        metadata = row.get('metadata', {})
        if row['name'] == 'question' and metadata.get('fold') == 'train':
            questions[(metadata.get('policy'), metadata.get('query_id'))] = row
        if row['name'] in ('candidate.codegen', 'candidate.result'):
            sources[(metadata.get('policy'), row['name'])] = row
        if row['name'] == 'rsi-evolution-round':
            root = row
    if root is None:
        raise ValueError('Missing source evolution root')
    source_root_url = '/project/memory-policy-rsi/traces/' + root['trace_id']
    metadata = {'run_id': identity, 'view': version,
                'source_trace_sha256': manifest['source_sha256'],
                'source_execution_url': source_root_url,
                'evidence_status': 'recorded_results_and_model_reported_hypotheses',
                'timing_kind': 'source_execution_interval_not_projection_runtime',
                'scope': 'derived_view_no_new_model_calls'}
    config = json.loads((run / 'config.json').read_text())
    environment = 'fixture-overview' if config.get('fixture') else profile['environment']
    records, scores, reviews = [], [], []
    diagnostics = diagnostic_evidence(run)
    diagnosis_id = digest([trace_id, 'diagnostics'])[:16]
    for branch in summary.get('branches', []):
        policy = branch['id']
        parent = branch.get('parent') or 'plain-bm25'
        proposal = branch.get('proposal') or {}
        result = {'policy': policy, 'parent_policy': parent, 'policy_commit': branch.get('commit'),
                  'selection_status': branch.get('status'),
                  'proposal_direction': proposal.get('direction'),
                  'generation': branch.get('generation'), 'expansion': branch.get('expansion'),
                  'target_error_clusters': proposal.get('supporting_cluster_labels', []),
                  'search_state': search_state(tree, policy),
                  'diagnosis_observation_id': diagnosis_id,
                  'hypothesis_verdict': 'not_automatically_adjudicated',
                  'cost_comparison_note': 'Generated policies include worker/snapshot isolation overhead.',
                  'folds': {}}
        for fold in ('train', 'validation'):
            path = run / 'branches' / policy / fold / 'report.json'
            parent_path = run / ('branches/' + parent if parent.startswith('candidate-') else parent) / fold / 'report.json'
            if path.exists():
                report = json.loads(path.read_text())
                baseline = json.loads(parent_path.read_text()) if parent_path.exists() else {}
                result['folds'][fold] = comparison(report, baseline, profile['include_type_metrics'])
                result['folds'][fold]['status'] = 'report_available'
            else:
                progress = path.with_name('progress.json')
                result['folds'][fold] = {'status': 'report_unavailable', 'AVG': None, 'E2E_AVG': None,
                    'progress': json.loads(progress.read_text()) if progress.exists() else None}
        result['validation_cost'] = {k: summary.get('comparison', {}).get(policy, {}).get(k)
                                     for k in ('policy_seconds', 'policy_bytes', 'policy_tokens')}
        semantic = summary.get('comparison', {}).get('plain-semantic', {})
        result['validation_delta_vs_semantic_pp'] = delta(
            summary.get('comparison', {}).get(policy, {}).get('E2E_AVG'), semantic.get('E2E_AVG'))
        start = sources.get((policy, 'candidate.codegen'), root)
        end = sources.get((policy, 'candidate.result'), root)
        case_links = []
        if profile['include_training_case_links']:
            for query in proposal.get('supporting_case_ids', []):
                question = questions.get((parent, query))
                case_links.append({'query_id': query, 'available': question is not None,
                    'url': '/project/memory-policy-rsi/traces/' + question['trace_id'] if question else None})
        case_ids = proposal.get('supporting_case_ids', [])
        if not profile['include_training_case_links']:
            result['case_link_status'] = 'disabled_by_profile'
        elif any(not c['available'] for c in case_links):
            result['case_link_status'] = 'some_references_unresolved'
        else:
            result['case_link_status'] = 'resolved' if case_ids else 'no_case_references_claimed'
        result['supporting_training_cases'] = case_links
        source_links = [{'trace_id': start['trace_id'], 'span_id': start['span_id']},
                        {'trace_id': end['trace_id'], 'span_id': end['span_id']}]
        row = {'schema_version': 1, 'trace_id': trace_id, 'span_id': digest([trace_id, policy])[:16],
               'parent_span_id': root_id, 'trace_name': 'evolution.overview', 'name': policy + ' | ' + str(proposal.get('direction') or 'direction unavailable')[:120],
               'kind': 'span', 'start_time_ns': start['start_time_ns'], 'end_time_ns': end['end_time_ns'],
               'input': {'model_reported_proposal': proposal,
                         'code_url': source_root_url + '?observation=' + start['span_id']},
               'output': result, 'metadata': {**metadata, 'policy': policy, 'parent_policy': parent,
                       'policy_commit': branch.get('commit'),
                       'candidate_selection_status': branch.get('status')},
               'status': 'ok', 'provenance': 'derived_from_recorded_artifacts',
               'environment': environment, 'tags': ['evolution-overview', identity, policy],
               'links': source_links}
        records.append(row)
        reviews.append({key: result[key] for key in ('policy','parent_policy','policy_commit','selection_status','proposal_direction','target_error_clusters','search_state','validation_cost','validation_delta_vs_semantic_pp')})
        reviews[-1]['validation_macro_E2E_AVG'] = result['folds'].get('validation',{}).get('E2E_AVG')
        reviews[-1]['validation_delta_vs_parent_pp'] = result['folds'].get('validation',{}).get('delta_vs_parent_pp',{}).get('E2E_AVG')
        reviews[-1]['observation_id'] = row['span_id']
        reviews[-1]['folds'] = result['folds']
        validation = result['folds'].get('validation', {})
        values = {'validation_macro_AVG': validation.get('AVG'),
                  'validation_macro_E2E_AVG': validation.get('E2E_AVG'),
                  'validation_delta_vs_parent_pp': validation.get('delta_vs_parent_pp', {}).get('E2E_AVG'),
                  'validation_delta_vs_semantic_pp': result['validation_delta_vs_semantic_pp'],
                  'validation_policy_seconds': result['validation_cost'].get('policy_seconds'),
                  'validation_policy_peak_bytes': result['validation_cost'].get('policy_bytes')}
        for fold in ('train', 'validation'):
            fold_result = result['folds'].get(fold, {})
            for metric in ('AVG', 'E2E_AVG', 'abnormal_rate'):
                values[fold + '_' + metric] = fold_result.get(metric)
        for metric in ('Q', 'visits', 'exploration_bonus', 'uct'):
            values['search_' + metric] = result['search_state'].get(metric)
        for name, value in values.items():
            if isinstance(value, (int, float)):
                scores.append(score(row, name, value, 'Recorded aggregate; not a hypothesis/cause verdict.'))
    cluster_index = []
    for index, cluster in enumerate(diagnostics['clusters']):
        identifier = digest([trace_id, 'cluster', index])[:16]
        label = cluster.get('label', 'unnamed')
        case_ids = cluster.get('supporting_case_ids', cluster.get('case_ids', []))
        links = []
        for case in case_ids:
            question = questions.get(('plain-bm25', case))
            links.append({'query_id': case, 'available': question is not None,
                          'url': '/project/memory-policy-rsi/traces/' + question['trace_id'] if question else None})
        item = {'model_reported_cluster': cluster, 'cited_case_count': len(case_ids),
                'supporting_training_cases': links,
                'addressed_by_candidates': [r['policy'] for r in reviews if label in r['target_error_clusters']],
                'causal_verdict': 'not_automatically_adjudicated',
                'count_scope': 'cited_examples_only_not_full_training_prevalence'}
        records.append({'schema_version': 1, 'trace_id': trace_id, 'span_id': identifier,
            'parent_span_id': diagnosis_id, 'trace_name': 'evolution.overview',
            'name': 'error.cluster | ' + str(label)[:120], 'kind': 'span',
            'start_time_ns': root['start_time_ns'], 'end_time_ns': root['end_time_ns'],
            'input': {'evidence_scope': 'parent_training_only'}, 'output': item,
            'metadata': metadata, 'status': 'ok', 'provenance': 'derived_from_recorded_artifacts',
            'environment': environment, 'tags': ['evolution-overview', identity, 'error-cluster']})
        cluster_index.append({'label': label, 'cited_case_count': len(case_ids),
                              'observation_id': identifier, 'addressed_by_candidates': item['addressed_by_candidates']})
    diagnostic_output = {key: value for key, value in diagnostics.items() if key != 'clusters'}
    diagnostic_output['clusters'] = cluster_index
    records.append({'schema_version': 1, 'trace_id': trace_id, 'span_id': diagnosis_id,
        'parent_span_id': root_id, 'trace_name': 'evolution.overview',
        'name': 'diagnostics | observed errors and model hypotheses', 'kind': 'span',
        'start_time_ns': root['start_time_ns'], 'end_time_ns': root['end_time_ns'],
        'input': {'evidence_scope': 'parent_training_only'}, 'output': diagnostic_output,
        'metadata': metadata, 'status': 'ok', 'provenance': 'derived_from_recorded_artifacts',
        'environment': environment, 'tags': ['evolution-overview', identity, 'diagnostics']})
    records.append({'schema_version': 1, 'trace_id': trace_id, 'span_id': root_id,
        'trace_name': 'evolution.overview', 'name': 'evolution.overview', 'kind': 'span',
        'start_time_ns': root['start_time_ns'], 'end_time_ns': root['end_time_ns'],
        'input': {'source_execution_url': source_root_url,
                  'framework_commit': root.get('metadata', {}).get('implementation_commit'),
                  'algorithm': tree.get('algorithm')},
        'output': {'search': summary.get('search'), 'diagnostics': diagnostic_output, 'candidates': reviews, 'test_evaluated': summary.get('test_evaluated'),
                   'reading_order': 'proposal -> code -> train/validation -> type tradeoffs -> cost -> supporting cases',
                   'hypothesis_verdict': 'Requires controlled intervention; score changes alone do not prove mechanism.'},
        'metadata': metadata, 'status': 'ok', 'provenance': 'derived_from_recorded_artifacts',
        'environment': environment, 'tags': ['evolution-overview', identity],
        'links': [{'trace_id': root['trace_id'], 'span_id': root['span_id']}]})
    return records, scores, trace_id


def publish(run, imported, client, env, profile_path=None):
    root = Path(__file__).resolve().parents[1]
    profile_path = Path(profile_path) if profile_path else root / 'deploy/langfuse/observability.json'
    profile = json.loads(profile_path.read_text())
    if not profile['enabled']:
        return {'enabled': False}
    records, scores, trace_id = build(run, imported, profile)
    target = Path(imported) / profile.get('version', 'evolution-overview-v1')
    target.mkdir(exist_ok=True, mode=0o700)
    path = target / 'traces.jsonl'
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    if path.exists() and path.read_text() != payload:
        raise ValueError('Overview changed; choose a new version/directory rather than overwrite')
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as stream: stream.write(payload)
    otlp._write_receipt(target / 'scores.json', {'scores': scores})
    otlp._write_receipt(target / 'prepare.json', {
        'run_id': records[-1]['metadata']['run_id'], 'usage': {},
        'prepared_sha256': otlp._input_sha256(path),
        'source_sha256': records[-1]['metadata']['source_trace_sha256'],
        'environment': records[-1]['environment']})
    receipt = otlp.export(path, target / 'trace-receipt.json', env=env, max_batch_bytes=2 * 1024 * 1024)
    if receipt.get('delivery_status') != 'accepted':
        raise ValueError('Overview delivery incomplete')
    count = deliver_scores(target, client)
    result = {'trace_id': trace_id, 'observations': len(records), 'accepted_scores': count,
              'environment': records[-1]['environment'],
              'url': '/project/memory-policy-rsi/traces/' + trace_id}
    otlp._write_receipt(target / 'index.json', result)
    return result


def main():
    import argparse
    from scripts.langfuse_run import Client
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--env-file',required=True,type=Path)
    parser.add_argument('--profile',type=Path)
    args=parser.parse_args()
    env=dict(line.split('=',1) for line in args.env_file.read_text().splitlines()
             if line and not line.startswith('#') and '=' in line)
    env.update(LANGFUSE_BASE_URL='http://127.0.0.1:3000',
               LANGFUSE_PUBLIC_KEY=env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY'],
               LANGFUSE_SECRET_KEY=env['LANGFUSE_INIT_PROJECT_SECRET_KEY'])
    print(json.dumps(publish(args.run,args.output,Client(env),env,args.profile)),flush=True)


if __name__=='__main__':
    main()
