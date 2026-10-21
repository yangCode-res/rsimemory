import unittest
from scripts.langfuse_run import question_scores, prompt_spec
from scripts.export_langfuse import record_to_span

class IntegrationTests(unittest.TestCase):
    def record(self, status, score):
        return {'trace_id':'a'*32,'span_id':'b'*16,
                'output':{'status':status,'score':score}}

    def test_judge_failures_are_not_zero_correctness(self):
        scores=question_scores(self.record('judge_failed',None))
        self.assertEqual([(s['name'],s['value']) for s in scores],[('grading_abnormal',1)])

    def test_actor_failure_only_gets_e2e_zero(self):
        scores=question_scores(self.record('actor_failed',None))
        self.assertEqual([(s['name'],s['value']) for s in scores],
                         [('e2e_correctness',0),('grading_abnormal',1)])

    def test_score_identity_is_stable_and_metric_specific(self):
        r=self.record('scored',1)
        a=question_scores(r);b=question_scores(r)
        self.assertEqual(a,b)
        self.assertEqual(len(set(s['id'] for s in a)),3)

    def test_prompt_identity_ignores_dynamic_patient_messages(self):
        r={'name':'llm.actor','input':{'messages':[{'role':'system','content':'fixed'},
                                                  {'role':'user','content':'case A'}]}}
        a=prompt_spec(r)
        r['input']['messages'][1]['content']='case B'
        self.assertEqual(a,prompt_spec(r))
        self.assertEqual(len(a['prompt']),1)

    def test_native_attributes_reach_otlp(self):
        r={'trace_id':'a'*32,'span_id':'b'*16,'name':'llm.actor','kind':'generation',
           'start_time_ns':1,'end_time_ns':2,'metadata':{'max_tokens':10},
           'environment':'fixture','tags':['rsi'],'prompt_name':'rsi/actor','prompt_version':1}
        attrs={a['key']:a['value'] for a in record_to_span(r)['attributes']}
        self.assertEqual(attrs['langfuse.environment'],{'stringValue':'fixture'})
        self.assertEqual(attrs['langfuse.observation.prompt.version'],{'intValue':'1'})
        self.assertIn('langfuse.observation.model.parameters',attrs)


class NativeExperimentTests(unittest.TestCase):
    def test_resume_links_existing_traces_without_creating_new_generations(self):
        import json
        import tempfile
        from pathlib import Path
        from urllib.error import HTTPError
        from scripts.link_langfuse_experiments import link

        class Client:
            def __init__(self):
                self.items={};self.links=[];self.datasets={};self.link_posts=0;self.item_reads=0
            def request(self,path,body=None):
                if body is None:
                    if path.startswith('/api/public/experiment-items?'):
                        self.item_reads+=1
                        if '&cursor=' in path:return {'data':[], 'meta':{}}
                        return {'data':[{'id':'b'*16,'traceId':'a'*32}], 'meta':{'cursor':'next-page'}}
                    if path.startswith('/api/public/dataset-items/'):
                        key=path.rsplit('/',1)[1]
                        if key in self.items:return self.items[key]
                    if path.startswith('/api/public/datasets/'):
                        name=path.rsplit('/',1)[1]
                        if name in self.datasets:return self.datasets[name]
                    raise HTTPError(path,404,'missing',{},None)
                if path=='/api/public/datasets':
                    value={'id':'dataset','name':body['name']}
                    self.datasets[body['name']]=value;return value
                if path=='/api/public/dataset-items':
                    self.items[body['id']]=body;return body
                if path=='/api/public/dataset-run-items':
                    self.link_posts+=1
                    value={**body,'id':'link','datasetRunId':'native-run'}
                    self.links.append(value);return value
                raise AssertionError('Unexpected API endpoint')

        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'original';out=root/'derived';run.mkdir();out.mkdir()
            (run/'config.json').write_text(json.dumps({'fixture':False,'language':'zh','split_version':'v2'}))
            (out/'prepare.json').write_text(json.dumps({'run_id':'run-1','source_sha256':'source'}))
            (out/'verification.json').write_text('{"verified":true}')
            question={'name':'question','trace_id':'a'*32,'span_id':'b'*16,
                      'start_time_ns':1700000000000000000,
                      'input':{'question':'synthetic'},'metadata':{'fold':'train','policy':'baseline','query_id':'q1'}}
            (out/'traces.jsonl').write_text(json.dumps(question)+'\n')
            from unittest import mock
            client=Client()
            with mock.patch('scripts.link_langfuse_experiments.otlp.export',return_value={'delivery_status':'accepted'}):
                first=link(run,out,client,{});second=link(run,out,client,{})
            self.assertTrue(first['verified']);self.assertTrue(second['verified'])
            self.assertEqual(client.link_posts,0)
            self.assertEqual(client.item_reads,4)
            self.assertEqual(len(first['runs']),1)


class BatchSizeTests(unittest.TestCase):
    def test_byte_budget_matches_utf8_payload_and_preserves_records(self):
        import json
        from scripts.export_langfuse import _batches,_payload
        records=[{'trace_id':'a'*32,'span_id':format(i,'016x'),'name':'中文',
                  'start_time_ns':1,'end_time_ns':2,'input':'数据'*20}
                 for i in range(1,8)]
        batches=list(_batches(records,1500))
        self.assertGreater(len(batches),1)
        self.assertEqual(sum(len(batch) for batch in batches),len(records))
        for batch in batches:
            self.assertLessEqual(len(json.dumps(_payload(batch),separators=(',',':')).encode()),1500)
