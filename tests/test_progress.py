import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi import runtime


class SyntheticPolicy:
    def __init__(self):
        self._registry = {}
        self.cost = {'ingest_calls': 0, 'search_calls': 0}

    def ingest(self, chunk):
        self._registry[chunk['id']] = dict(chunk)
        self.cost['ingest_calls'] += 1

    def search(self, query, filters):
        self.cost['search_calls'] += 1
        return [row for row in self._registry.values()
                if row['persona_id'] == filters['persona_id']
                and row['session_id'] <= filters['checkpoint']]


class ProgressPersistenceTests(unittest.TestCase):
    def test_metadata_publication_failure_keeps_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'progress.json'
            runtime.dump(path, {'completed':1})
            with patch.object(runtime.os, 'replace', side_effect=OSError('synthetic publication failure')):
                with self.assertRaises(OSError):
                    runtime.dump(path, {'completed':2})
            self.assertEqual(json.loads(path.read_text()), {'completed':1})
            self.assertEqual(list(Path(temp).iterdir()), [path])

    def setUp(self):
        self.sessions = [
            {'persona_id': 'synthetic-person', 'session_id': 1, 'text': 'synthetic memory one'},
            {'persona_id': 'synthetic-person', 'session_id': 2, 'text': 'synthetic memory two'},
        ]
        self.queries = [
            {'persona_id': 'synthetic-person', 'session_id': 1, 'query_id': 'synthetic-query-1',
             'query_type': 'state_update', 'question': 'synthetic question one'},
            {'persona_id': 'synthetic-person', 'session_id': 2, 'query_id': 'synthetic-query-2',
             'query_type': 'state_update', 'question': 'synthetic question two'},
        ]
        self.config = {'chunk_chars': 1200, 'chunk_overlap': 120, 'max_steps': 4,
                       'context_chars': 16000, 'top_k': 5}

    def _run(self, output, actor_class):
        with patch.object(runtime, 'Actor', actor_class), patch.object(
                runtime, 'evaluate', return_value={'score': 1, 'status': 'scored',
                                                   'reason': 'synthetic', 'call_log': []}):
            return runtime.replay(self.sessions, self.queries, ['synthetic-person'],
                                  SyntheticPolicy, object(), object(), self.config, output)

    def test_first_answer_and_tool_operations_are_flushed_before_second_query(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'

            class CheckingActor:
                calls = 0

                def __init__(self, *args):
                    pass

                def answer(self, question, tools):
                    type(self).calls += 1
                    if type(self).calls == 2:
                        answers = (out / 'answers.jsonl').read_text().splitlines()
                        operations = [json.loads(line) for line in
                                      (out / 'operations.jsonl').read_text().splitlines()]
                        self_outer.assertEqual(len(answers), 1)
                        self_outer.assertTrue(any(row.get('query_id') == 'synthetic-query-1'
                                                  for row in operations))
                        progress = json.loads((out / 'progress.json').read_text())
                        self_outer.assertEqual(progress['completed'], 1)
                        self_outer.assertEqual(
                            {row['action'] for row in operations if row.get('query_id') == 'synthetic-query-1'},
                            {'search', 'read'})
                    evidence = tools.search('synthetic lookup')
                    if evidence:
                        tools.read([evidence[0]['id']])
                    return {'status': 'completed', 'answer': 'synthetic answer',
                            'trajectory': [], 'evidence': evidence, 'call_log': []}

            self_outer = self
            report, rows = self._run(out, CheckingActor)
            progress = json.loads((out / 'progress.json').read_text())
            self.assertEqual(len(rows), 2)
            self.assertEqual(progress, {'status': 'completed', 'total_questions': 2,
                                        'completed': 2, 'scored': 2, 'abnormal': 0})
            self.assertEqual(report['counts']['scored'], 2)
            serialized = (out / 'progress.json').read_text()
            for private_value in ('synthetic-person', 'synthetic-query',
                                  'synthetic question', 'synthetic answer'):
                self.assertNotIn(private_value, serialized)

    def test_exception_keeps_flushed_rows_and_marks_failed_without_identifiers(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'

            class FailingActor:
                calls = 0

                def __init__(self, *args):
                    pass

                def answer(self, question, tools):
                    type(self).calls += 1
                    if type(self).calls == 2:
                        # Ensure even this in-flight operation is durable before failing.
                        evidence = tools.search('synthetic lookup')
                        if evidence:
                            tools.read([evidence[0]['id']])
                        raise RuntimeError('synthetic actor failure')
                    evidence = tools.search('synthetic lookup')
                    return {'status': 'completed', 'answer': 'synthetic answer',
                            'trajectory': [], 'evidence': evidence, 'call_log': []}

            with self.assertRaisesRegex(RuntimeError, 'synthetic actor failure'):
                self._run(out, FailingActor)

            answers = [json.loads(line) for line in (out / 'answers.jsonl').read_text().splitlines()]
            operations = [json.loads(line) for line in (out / 'operations.jsonl').read_text().splitlines()]
            progress = json.loads((out / 'progress.json').read_text())
            self.assertEqual(len(answers), 1)
            self.assertTrue(any(row.get('query_id') == 'synthetic-query-2' for row in operations))
            self.assertEqual(progress, {'status': 'failed', 'total_questions': 2,
                                        'completed': 1, 'scored': 1, 'abnormal': 1})
            self.assertFalse((out / 'report.json').exists())
            serialized = (out / 'progress.json').read_text()
            for private_value in ('synthetic-person', 'synthetic-query',
                                  'synthetic question', 'synthetic answer'):
                self.assertNotIn(private_value, serialized)


if __name__ == '__main__':
    unittest.main()
