import copy
import unittest
from rsi.tracing import patient_label
from scripts.patient_view import group_record
from scripts.export_langfuse import record_to_span

class PatientViewTests(unittest.TestCase):
    def test_alias_stable_across_policies_and_input_id_types(self):
        self.assertEqual(patient_label(42,'v1'),patient_label('42','v1'))
        self.assertNotEqual(patient_label(42,'v1'),patient_label(43,'v1'))
        self.assertNotEqual(patient_label(42,'v1'),patient_label(42,'v2'))

    def test_policy_does_not_split_patient_session(self):
        lookup={('train','q1'):'patient-abc'}
        for policy in ['plain-bm25','candidate-1']:
            row={'metadata':{'query_id':'q1','fold':'train','policy':policy}}
            result=group_record(row,lookup,'run-1')
            self.assertEqual(result['session_id'],'run-1/patient-abc')
        self.assertEqual(group_record({'metadata':{'query_id':'q1','fold':'train'}},lookup,'run-2')['session_id'],'run-2/patient-abc')

    def test_evolution_trace_does_not_enter_patient_session(self):
        row={'name':'evolution.overview','metadata':{'run_id':'r'}}
        self.assertNotIn('session_id',group_record(row,{},'r'))

    def test_explicit_session_overrides_run_id_without_duplicate_attribute(self):
        row={'trace_id':'a'*32,'span_id':'b'*16,'name':'question',
             'start_time_ns':1,'end_time_ns':2,'metadata':{'run_id':'r'},'session_id':'r/patient-x'}
        attrs=[a for a in record_to_span(row)['attributes'] if a['key']=='langfuse.session.id']
        self.assertEqual(attrs,[{'key':'langfuse.session.id','value':{'stringValue':'r/patient-x'}}])

    def test_checkpoint_and_model_input_are_unchanged(self):
        row={'metadata':{'query_id':'q1','fold':'train','checkpoint':7},
             'input':{'question':'synthetic'},'output':{'score':1},'trace_id':'a'*32}
        original=copy.deepcopy(row)
        grouped=group_record(row,{('train','q1'):'patient-x'},'r')
        for key in ['input','output','trace_id']:
            self.assertEqual(grouped[key],original[key])
        self.assertEqual(grouped['metadata']['checkpoint'],7)

    def test_patient_is_native_user_id(self):
        row={'trace_id':'a'*32,'span_id':'b'*16,'name':'question',
             'start_time_ns':1,'end_time_ns':2,'metadata':{'run_id':'r','patient':'patient-x'}}
        attrs=[a for a in record_to_span(row)['attributes'] if a['key']=='langfuse.user.id']
        self.assertEqual(attrs,[{'key':'langfuse.user.id','value':{'stringValue':'patient-x'}}])
