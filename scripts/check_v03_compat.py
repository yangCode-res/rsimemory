"""Compatibility probes for saved generated code; no medical scoring or LLM."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsi.policy import GeneratedPolicy

GRAPH_SOURCE = '''
import numpy as np
import networkx as nx
class Memory:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.vector = np.array([1., 2.])
def initialize(config): return Memory()
def ingest(state, session):
    assert isinstance(state, Memory)
    state.graph.add_node(session['id'])
    state.vector += 1.
    return state
def search(state, query, filters):
    assert isinstance(state, Memory)
    assert np.sum(state.vector) > 0
    ids = list(state.graph.nodes)
    state.graph.clear()
    return ids
'''


def probe(source):
    policy = GeneratedPolicy(source)
    for i in range(1, 4):
        policy.ingest({'id': f'synthetic-{i}', 'text': f'历史会话{i}: 血糖控制情况与用药记录。',
                       'persona_id': 1, 'session_id': i})
    before = policy.snapshot()
    first = policy.search('血糖控制', {'persona_id': 1, 'checkpoint': 3})
    second = policy.search('血糖控制', {'persona_id': 1, 'checkpoint': 3})
    assert policy.snapshot()['state'] == before['state'], 'search persisted state changes'
    policy.restore(before)
    restored = policy.search('血糖控制', {'persona_id': 1, 'checkpoint': 3})
    ids = lambda rows: [r['id'] for r in rows]
    assert ids(first) == ids(second) == ids(restored), 'query isolation or restore mismatch'
    return {'status': 'passed', 'hits': len(first), 'read_only': True, 'restored': True,
            'serializable': policy.serializable, 'cost': policy.cost}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous-run', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    previous, output = Path(args.previous_run), Path(args.output)
    if output.exists(): p.error('Use a new output directory')
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
    tree = json.loads((previous/'tree.json').read_text())
    results = {}
    for file in sorted(previous.glob('branches/*/policy.py')):
        name = file.parent.name
        source = file.read_text()
        try: result = probe(source)
        except Exception as exc:
            result = {'status': 'failed', 'error_type': type(exc).__name__,
                      'category': getattr(exc, 'category', 'runtime')}
        result['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        result['original_policy_commit'] = tree['nodes'][name].get('commit')
        results[name] = result
    try: graph = probe(GRAPH_SOURCE)
    except Exception as exc:
        graph = {'status': 'failed', 'error_type': type(exc).__name__,
                 'category': getattr(exc, 'category', 'runtime')}
    report = {'contract_version': 'v0.3', 'kind': 'synthetic-compatibility-probes',
              'implementation_commit': commit, 'previous_run': str(previous),
              'scientific_result': False, 'test_evaluated': False,
              'candidates': results, 'graph_numpy_class': graph}
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({'candidates': len(results), 'passed': sum(r['status']=='passed' for r in results.values()),
                      'graph_numpy_class': graph['status'], 'scientific_result': False}))
    if not results or graph['status'] != 'passed' or any(r['status'] != 'passed' for r in results.values()):
        raise SystemExit(1)


if __name__ == '__main__': main()
