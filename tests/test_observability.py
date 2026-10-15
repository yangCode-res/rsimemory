import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.evolution import reflect
from rsi.fixtures import FixtureClient
from rsi.model import ModelClient
from rsi.tracing import trace_run


class ObservabilityIntegration(unittest.TestCase):
    def test_fixture_cli_groups_question_traces_without_evaluating_test(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'fixture-traced'
            result=subprocess.run([sys.executable,'-m','rsi.cli','--fixture','--output',str(out)],
                                  capture_output=True,text=True,check=True)
            spans=[json.loads(x) for x in (out/'traces.jsonl').read_text().splitlines()]
            summary=json.loads((out/'summary.json').read_text())
            questions=[s for s in spans if s['name']=='question']
            self.assertTrue(questions)
            self.assertTrue(all(q['metadata']['fold'] in ('train','validation') for q in questions))
            self.assertTrue(all(q['metadata']['policy']=='plain-bm25' for q in questions))
            self.assertTrue(all(q.get('links') for q in questions))
            self.assertFalse(summary['test_evaluated'])
            self.assertEqual(summary['comparison']['plain-bm25']['AVG'],1.)
            self.assertEqual(sum(s['name']=='rsi-evolution-round' for s in spans),1)
            self.assertTrue(any(s['name']=='memory.search' for s in spans))
            self.assertTrue(any(s['name']=='grade' for s in spans))
            self.assertTrue(all(s['metadata'].get('run_id')=='fixture-traced' for s in spans))

    def test_actual_truncated_model_output_is_retained_without_request_credentials(self):
        payload={'choices':[{'finish_reason':'length','message':{'content':'partial generated source'}}],
                 'usage':{'completion_tokens':3072}}
        class Response:
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def read(self,*args): return json.dumps(payload).encode()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'traces.jsonl'
            with trace_run(path,'synthetic'), patch('urllib.request.urlopen',return_value=Response()):
                with self.assertRaisesRegex(ValueError,'truncated'):
                    ModelClient(api_key='secret-must-not-be-captured').complete(
                        [{'role':'user','content':'synthetic request'}],role='codegen')
            serialized=path.read_text()
            generation=next(json.loads(s) for s in serialized.splitlines()
                            if json.loads(s)['name']=='llm.codegen')
            self.assertEqual(generation['output'],payload)
            self.assertEqual(generation['status'],'error')
            self.assertNotIn('secret-must-not-be-captured',serialized)

    def test_proposal_cannot_claim_support_from_case_absent_from_actual_prompt(self):
        class UnsupportedEvidence(FixtureClient):
            def complete(self,messages,**kwargs):
                result=super().complete(messages,**kwargs)
                obj=json.loads(result['text'])
                obj['proposals'][0].update(basis='training_cases',
                    supporting_case_ids=['case-never-provided'],rationale='unverified claim')
                result['text']=json.dumps(obj)
                return result
        rows=[{'query_id':'train-case','query_type':'state_update','question':'synthetic',
               'answer':'synthetic','score':0,'reason':'synthetic','evidence':[],'trajectory':[]}]
        with self.assertRaisesRegex(ValueError,'absent'):
            reflect(UnsupportedEvidence(),rows,{'train-case':{'positive_answers':['synthetic']}},[],
                    'def initialize(config): return {}')

    def test_recorded_diagnostic_ids_match_exact_bounded_input(self):
        class CapturingClient(FixtureClient):
            def complete(self,messages,**kwargs):
                self.input=json.loads(messages[1]['content'])
                return super().complete(messages,**kwargs)
        rows=[{'query_id':str(i),'query_type':'state_update','question':'synthetic',
               'answer':'synthetic','score':i%2,'reason':'synthetic','evidence':[],
               'trajectory':[]} for i in range(8)]
        client=CapturingClient()
        result=reflect(client,rows,{str(i):{'positive_answers':['synthetic']} for i in range(8)},[],
                       'def initialize(config): return {}')
        self.assertEqual(result['diagnostic_case_ids'],[e['query_id'] for e in client.input['examples']])
        self.assertEqual(result['diagnostic_example_count'],5)
        self.assertEqual(result['rationale_status'],'model_reported_hypotheses_not_verified_causes')


if __name__=='__main__': unittest.main()
