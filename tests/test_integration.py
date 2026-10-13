import json
from pathlib import Path
import tempfile
import unittest
from rsi.data import chunk_session, normalize_query, patient_splits
from rsi.evolution import new_node, backprop, uct, pareto, reflect, generate
from rsi.fixtures import FixtureClient, dataset
from rsi.policy import PlainPolicy
from rsi.runtime import HostTools, replay, InterventionTools

class PipelineIntegration(unittest.TestCase):
    def test_online_chunks_strip_annotations(self):
        row={'persona_id':1,'session_id':10,'text':'abc'*100,'answers':'SECRET','query_type':'SECRET'}
        for chunk in chunk_session(row,40,5):
            self.assertEqual(set(chunk),{'id','persona_id','session_id','text'})

    def test_scope_cannot_expand(self):
        p=PlainPolicy();p.ingest({'id':'a','text':'fact','persona_id':1,'session_id':10})
        t=HostTools(p,1,10)
        with self.assertRaises(PermissionError):t.search('fact',{'checkpoint':100})
        with self.assertRaises(PermissionError):t.read(['a'])

    def test_question_snapshots_independent(self):
        p=PlainPolicy();p.ingest({'id':'a','text':'fact','persona_id':1,'session_id':10})
        first,second=HostTools(p,1,10),HostTools(p,1,10)
        first.search('fact')
        with self.assertRaises(PermissionError):second.read(['a'])
        self.assertEqual(p.cost['search_calls'],0)

    def test_answer_schema_normalization(self):
        q=normalize_query({'question':'选择\nA. 正确\nB. 错误','query_type':'multiple_choice',
                           'answers':json.dumps([{'content':'正确','is_correct':True},{'content':'错误','is_correct':False}]),
                           'metadata':'{}'})
        self.assertEqual(q['correct_options'],['A'])

    def test_answer_option_prefix(self):
        q=normalize_query({'question':'选择\\nA. 正确\\nB. 错误','query_type':'multiple_choice',
                           'answers':json.dumps([{'content':'A. 正确','is_correct':True}]),'metadata':'{}'})
        self.assertEqual(q['correct_options'],['A'])

    def test_fixture_replay_has_all_types(self):
        sessions,qs=dataset();client=FixtureClient();split=patient_splits(list(range(1,7)))
        config={'chunk_chars':1200,'chunk_overlap':120,'top_k':5,'max_steps':4,'context_chars':16000}
        with tempfile.TemporaryDirectory() as d:
            report,rows=replay(sessions,qs,split['train'],PlainPolicy,client,client,config,Path(d)/'train')
            self.assertEqual(report['counts']['abnormal'],0);self.assertEqual(report['AVG'],1.)
            self.assertFalse(set(r['persona_id'] for r in rows)&set(split['test']))

    def test_tree_uses_training_only(self):
        root=new_node('root');leaf=new_node('child','root');tree={'nodes':{'root':root,'child':leaf}}
        backprop(tree,'child',.5)
        self.assertEqual(root['Q'],.5);self.assertEqual(root['visits'],1)
        self.assertEqual(uct(new_node('new'),1),float('inf'))
        backprop(tree,'child',None);self.assertEqual(leaf['visits'],1)

    def test_intervention_provenance_rejected(self):
        q={'session_id':20,'evidence':[{'content':'missing'}, {'content':'future','session_id':30},
                                      {'content':'not present','session_id':15}, {'content':'present','session_id':10}]}
        t=InterventionTools(q,{10})
        self.assertEqual(len(t.rows),1);self.assertEqual(t.rejected,3)

    def test_pareto_tradeoffs(self):
        reports={'cheap':{'AVG':.5,'policy_tokens':0,'policy_seconds':1,'policy_bytes':10},
                 'accurate':{'AVG':.8,'policy_tokens':2,'policy_seconds':2,'policy_bytes':20},
                 'dominated':{'AVG':.4,'policy_tokens':1,'policy_seconds':2,'policy_bytes':20},
                 'failed':{'AVG':None}}
        self.assertEqual(set(pareto(reports)),{'cheap','accurate'})

    def test_reflection_bounds_diagnostics_and_keeps_json(self):
        class CaptureClient(FixtureClient):
            def complete(self,messages,**kwargs):
                self.messages=messages
                return super().complete(messages,**kwargs)
        client=CaptureClient()
        rows=[{'query_id':str(i),'query_type':'state_update','question':'问'*20000,
               'answer':'答'*20000,'score':i%2,'reason':'证'*20000,
               'evidence':[{'text':'记'*20000}]*20,'trajectory':[]} for i in range(8)]
        lookup={str(i):{'positive_answers':['金'*20000],'evidence':[]} for i in range(8)}
        result=reflect(client,rows,lookup,[],'baseline'*1200)
        payload=json.loads(client.messages[1]['content'])
        self.assertLessEqual(result['input_chars'],24000)
        self.assertEqual(payload['parent_code'],'baseline'*1200)
        self.assertTrue(payload['diagnostic_excerpts'])
        self.assertTrue(payload['examples'])

    def test_rejected_source_is_preserved_without_execution(self):
        class InvalidCodeClient:
            def complete(self,*args,**kwargs):
                return {'text':json.dumps({'code':'def broken(:\n    pass'}),'usage':{'completion_tokens':3},
                        'latency_s':1.,'role':'codegen'}
        with self.assertRaises(Exception) as caught:
            generate(InvalidCodeClient(),{},'parent')
        self.assertEqual(caught.exception.candidate_source,'def broken(:\n    pass')
        self.assertEqual(caught.exception.codegen_cost['usage']['completion_tokens'],3)

if __name__=='__main__':unittest.main()
