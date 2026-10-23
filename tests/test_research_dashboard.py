import json,tempfile,unittest,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('research_dashboard',Path(__file__).resolve().parents[1]/'deploy/langfuse/research_dashboard.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class DashboardTests(unittest.TestCase):
    def test_running_reports_missing_values_and_fixture_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def write(run,name,value):
                path=root/'runs'/run/'result'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            name='full-v03-local-123'
            write(name,'config.json',{'fixture':False})
            write(name,'summary.json',{'branches':[{'id':'candidate-1','parent':'plain-bm25','proposal':{'direction':'test'}}]})
            write(name,'run-status.json',{'status':'running','stage':'candidates.parallel'})
            write(name,'branches/candidate-1/train/progress.json',{'completed':10,'total_questions':100,'status':'running'})
            write(name,'diagnostics/summary.json',{'total_questions':100,'patients':{'private':1}})
            write(name,'reflexion.json',{'clusters':[{'label':'retrieval','supporting_case_ids':['private-case']} ]})
            write('full-v03-local-999','config.json',{'fixture':True})
            write('full-v03-local-999','summary.json',{'branches':[]})
            data=m.snapshot(root);self.assertEqual(len(data['runs']),1)
            r=data['runs'][0];c=r['policies'][-1]
            self.assertFalse(c['folds']['train']['report_available'])
            self.assertIsNone(c['folds']['train']['AVG'])
            self.assertEqual(c['folds']['train']['progress']['completed'],10)
            self.assertEqual(r['clusters'][0]['cited_cases'],1)
            self.assertNotIn('private',json.dumps(data))
            self.assertNotIn('overview',r['links'])

    def test_only_verified_overview_links_are_exposed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=root/'runs/full-v03-local-123/result';r.mkdir(parents=True)
            (r/'config.json').write_text('{"fixture":false}')
            (r/'summary.json').write_text('{"branches":[]}')
            v=root/'runs/langfuse-import-123-v2/evolution-overview-v2';v.mkdir(parents=True)
            (v/'index.json').write_text('{"url":"/project/memory-policy-rsi/traces/abc"}')
            self.assertNotIn('overview',m.snapshot(root)['runs'][0]['links'])
            (v/'verification.json').write_text('{"verified":true}')
            self.assertTrue(m.snapshot(root)['runs'][0]['links']['overview'].endswith('/abc'))


    def test_baseline_running_before_summary_exists_is_visible(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);r=root/'runs/full-v03-local-123/result';r.mkdir(parents=True)
            (r/'config.json').write_text('{"fixture":false}')
            (r/'run-status.json').write_text('{"status":"running","stage":"baseline.train"}')
            data=m.snapshot(root)
            self.assertEqual(len(data['runs']),1)
            self.assertEqual(data['runs'][0]['stage'],'baseline.train')
            self.assertEqual(len(data['runs'][0]['policies']),2)
