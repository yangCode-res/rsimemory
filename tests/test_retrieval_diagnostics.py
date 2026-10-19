import sys
import unittest

from rsi.policy import GeneratedPolicy
from scripts.diagnose_candidate3_retrieval import instrument


SOURCE = '''
def _grams(text):
    return str(text).split()
class Memory:
    def __init__(self):
        self.chunks = []
    def _lexical_scores(self, query):
        if not query:
            return {c['id']: 0.0 for c in self.chunks}
        return {c['id']: sum(token in _grams(c['text']) for token in _grams(query))
                for c in self.chunks}
    def search(self, query):
        scores = self._lexical_scores(query)
        return [c['id'] for c in sorted(self.chunks,
                key=lambda c: (scores[c['id']], c['session_id']), reverse=True)[:1]]
def initialize(config):
    return Memory()
def ingest(state, session):
    state.chunks.append(session)
    return state
def search(state, query, filters):
    return state.search(query)
'''


@unittest.skipUnless(sys.platform.startswith('linux'), 'Requires isolated Linux candidate worker')
class RetrievalDiagnosticsTests(unittest.TestCase):
    def make_policy(self, fix):
        chunks = [{'id':'fixture-old', 'text':'青松片导致皮疹后停用', 'persona_id':0, 'session_id':1},
                  {'id':'fixture-new', 'text':'今天讨论天气饮食运动', 'persona_id':0, 'session_id':2}]
        policy = GeneratedPolicy(instrument(SOURCE, fix), {'__diagnostic_chunks':chunks})
        policy._registry = {c['id']:c for c in chunks}
        return policy

    def test_probe_preserves_control_and_only_tokenizer_change_recovers_older_fact(self):
        original, fixed = self.make_policy(False), self.make_policy(True)
        scope = {'persona_id':0, 'checkpoint':2}
        self.assertEqual(original.search('青松片皮疹停用', scope)[0]['id'], 'fixture-new')
        self.assertEqual(fixed.search('青松片皮疹停用', scope)[0]['id'], 'fixture-old')

    def test_numeric_probe_preserves_early_return_branch(self):
        policy = self.make_policy(False)
        result = policy._call('search', {'state':policy._state, 'query':'', 'filters':{}})
        self.assertEqual(result['ids'], ['fixture-new'])
        self.assertIn('DIAG_ZERO_BM25 1', result['stdout_excerpt'])


if __name__ == '__main__':
    unittest.main()
