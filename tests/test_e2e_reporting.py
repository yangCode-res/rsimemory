import json
from pathlib import Path
import tempfile
import unittest

from rsi.cli import comparison_metric
from rsi.evaluation import aggregate, TYPES
from rsi.fixtures import dataset
from rsi.policy import PlainPolicy
from rsi.runtime import replay
from scripts.report_e2e import build_report


def complete_rows():
    return [{'query_type': kind, 'status': 'scored', 'score': 1} for kind in TYPES]


class E2EReportingTests(unittest.TestCase):
    def test_unanswered_keeps_denominator_and_six_type_macro_weighting(self):
        rows = complete_rows() + [{'query_type': 'state_update', 'status': 'actor_failed',
                                   'score': None}]
        report = aggregate(rows)
        self.assertIsNone(report['AVG'])
        self.assertIsNone(report['type_accuracy']['state_update'])
        self.assertEqual(report['e2e_type_accuracy']['state_update'], 0.5)
        self.assertAlmostEqual(report['E2E_AVG'], 5.5/6)
        self.assertNotAlmostEqual(report['E2E_AVG'], 6/7)
        self.assertEqual(report['abnormal_rate'], 1/7)
        self.assertEqual(report['counts']['unanswered_as_zero'], 1)
        self.assertEqual(report['counts']['e2e_unscorable'], 0)

    def test_all_unanswered_is_zero_not_null(self):
        rows = [{'query_type': kind, 'status': 'actor_failed', 'score': None} for kind in TYPES]
        report = aggregate(rows)
        self.assertEqual(report['E2E_AVG'], 0)
        self.assertEqual(report['abnormal_rate'], 1)
        self.assertIsNone(report['AVG'])

    def test_complete_reports_preserve_original_avg(self):
        report = aggregate(complete_rows())
        self.assertEqual(report['AVG'], report['E2E_AVG'])
        self.assertEqual(report['abnormal_rate'], 0)

    def test_evaluator_and_malformed_scores_are_not_treated_as_wrong_answers(self):
        for status, score in [('judge_failed', None), ('scored', True),
                              ('scored', 1.0), ('actor_failed', 1),
                              ('infrastructure_failed', None)]:
            with self.subTest(status=status, score=score):
                report = aggregate(complete_rows() + [
                    {'query_type': 'state_update', 'status': status, 'score': score}])
                self.assertIsNone(report['E2E_AVG'])
                self.assertEqual(report['counts']['unanswered_as_zero'], 0)
                self.assertEqual(report['counts']['e2e_unscorable'], 1)

    def test_missing_and_unknown_types_are_not_zero_filled(self):
        self.assertIsNone(aggregate(complete_rows()[:-1])['E2E_AVG'])
        self.assertIsNone(aggregate(complete_rows() + [
            {'query_type': 'unknown', 'status': 'actor_failed', 'score': None}])['E2E_AVG'])
        self.assertIsNone(aggregate([])['E2E_AVG'])
        self.assertIsNone(aggregate([])['abnormal_rate'])

    def test_comparison_keeps_strict_avg_and_exposes_e2e(self):
        report = aggregate(complete_rows() + [
            {'query_type': 'state_update', 'status': 'actor_failed', 'score': None}])
        report['cost'] = {'policy': []}
        result = comparison_metric(report)
        self.assertIsNone(result['AVG'])
        self.assertAlmostEqual(result['E2E_AVG'], 5.5/6)
        self.assertEqual(result['abnormal_rate'], 1/7)

    def test_actual_replay_reports_actor_protocol_nonanswers_as_zero(self):
        class NeverFinal:
            def complete(self, messages, role='actor', **kwargs):
                if role != 'actor':
                    raise AssertionError('Judge must not score missing answers')
                return {'text': '{"action":"search","query":"synthetic","filters":{}}',
                        'usage': {}, 'latency_s': 0}

        sessions, queries = dataset()
        config = {'chunk_chars':1200, 'chunk_overlap':120, 'top_k':5,
                  'max_steps':1, 'context_chars':16000}
        patient = sessions[0]['persona_id']
        with tempfile.TemporaryDirectory() as temp:
            report, rows = replay(sessions, queries, [patient], PlainPolicy,
                                  NeverFinal(), NeverFinal(), config, Path(temp)/'replay')
        self.assertTrue(rows)
        self.assertIsNone(report['AVG'])
        self.assertEqual(report['E2E_AVG'], 0)
        self.assertEqual(report['abnormal_rate'], 1)
        self.assertTrue(all(row['status'] == 'actor_failed' for row in rows))

    def make_replay(self, root, completed=True):
        path = root/'plain-bm25'/'validation'
        path.mkdir(parents=True)
        rows = complete_rows() + [{'query_type': 'state_update', 'status': 'actor_failed',
                                  'score': None, 'question': 'private-text-must-stay-in-source'}]
        (path/'answers.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        report = aggregate(rows)
        (path/'report.json').write_text(json.dumps(report))
        (path/'progress.json').write_text(json.dumps({
            'status': 'completed' if completed else 'failed',
            'total_questions': len(rows), 'completed': len(rows)}))
        return path

    def test_supplement_preserves_archive_and_never_copies_question_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp)/'run', Path(temp)/'supplement'
            path = self.make_replay(root)
            before = {p:p.read_bytes() for p in path.iterdir()}
            report = build_report(root, out)
            self.assertEqual(before, {p:p.read_bytes() for p in path.iterdir()})
            self.assertIsNone(report['reports']['plain-bm25']['validation']['AVG'])
            self.assertIsNotNone(report['reports']['plain-bm25']['validation']['E2E_AVG'])
            self.assertNotIn('private-text-must-stay-in-source', (out/'summary.json').read_text())
            self.assertNotIn('private-text-must-stay-in-source', (out/'REPORT.md').read_text())
            self.assertEqual(report['model_calls'], 0)
            self.assertFalse(report['test_evaluated'])

    def test_failed_replay_cannot_be_presented_as_completed_e2e(self):
        with tempfile.TemporaryDirectory() as temp:
            root, out = Path(temp)/'run', Path(temp)/'supplement'
            self.make_replay(root, completed=False)
            with self.assertRaisesRegex(ValueError, 'incomplete replay'):
                build_report(root, out)
            self.assertFalse(out.exists())

    def test_supplement_cannot_be_written_inside_immutable_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp)/'archive'
            root = archive/'result'
            self.make_replay(root)
            (archive/'_archive-manifest.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'outside'):
                build_report(root, archive/'supplement')


if __name__ == '__main__':
    unittest.main()
