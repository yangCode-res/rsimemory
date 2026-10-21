import copy
import unittest
from scripts.evolution_overview import comparison,delta

class OverviewTests(unittest.TestCase):
    def test_null_strict_score_never_becomes_zero(self):
        self.assertIsNone(delta(None,.2))
        self.assertIsNone(delta(.2,None))
        result=comparison({'AVG':None,'E2E_AVG':.3},{'AVG':.2,'E2E_AVG':.2})
        self.assertIsNone(result['delta_vs_parent_pp']['AVG'])
        self.assertAlmostEqual(result['delta_vs_parent_pp']['E2E_AVG'],10.)

    def test_type_regression_remains_visible(self):
        candidate={'AVG':.3,'E2E_AVG':.3,'e2e_type_accuracy':{'MCD':.4,'MQ':.2}}
        baseline={'AVG':.3,'E2E_AVG':.3,'e2e_type_accuracy':{'MCD':.3,'MQ':.3}}
        result=comparison(candidate,baseline)
        self.assertAlmostEqual(result['type_delta_vs_parent_pp']['MCD'],10.)
        self.assertAlmostEqual(result['type_delta_vs_parent_pp']['MQ'],-10.)

    def test_view_configuration_does_not_change_original_report(self):
        candidate={'AVG':.3,'E2E_AVG':.3,'e2e_type_accuracy':{'MCD':.4}}
        original=copy.deepcopy(candidate)
        result=comparison(candidate,{'AVG':.2,'E2E_AVG':.2},False)
        self.assertNotIn('type_accuracy',result)
        self.assertEqual(candidate,original)

    def test_missing_baseline_type_is_unknown(self):
        result=comparison({'e2e_type_accuracy':{'MCD':.4}}, {})
        self.assertIsNone(result['type_delta_vs_parent_pp']['MCD'])

    def test_build_links_only_parent_training_and_isolates_fixture(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.evolution_overview import build
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run=root/'run';imported=root/'imported';run.mkdir();imported.mkdir()
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            write(run/'config.json',{'fixture':True})
            write(imported/'prepare.json',{'run_id':'fixture','source_sha256':'hash'})
            write(run/'summary.json',{'branches':[{'id':'candidate-1','parent':'plain-bm25',
                  'status':'evaluated','proposal':{'supporting_case_ids':['case1']}}]})
            write(run/'tree.json',{'algorithm':'single expansion'})
            for fold in ['train','validation']:
                write(run/'plain-bm25'/fold/'report.json',{'AVG':.2,'E2E_AVG':.2})
                write(run/'branches/candidate-1'/fold/'report.json',{'AVG':.3,'E2E_AVG':.3})
            rows=[{'name':'rsi-evolution-round','trace_id':'c'*32,'span_id':'1'*16,
                   'start_time_ns':0,'end_time_ns':10},
                  {'name':'question','trace_id':'a'*32,'span_id':'2'*16,
                   'metadata':{'fold':'train','policy':'plain-bm25','query_id':'case1'}},
                  {'name':'question','trace_id':'b'*32,'span_id':'3'*16,
                   'metadata':{'fold':'validation','policy':'plain-bm25','query_id':'case1'}}]
            (imported/'traces.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            profile={'environment':'research-overview','include_type_metrics':True,
                     'include_training_case_links':True}
            records,scores,identifier=build(run,imported,profile)
            self.assertTrue(all(r['environment']=='fixture-overview' for r in records))
            self.assertTrue(records[0]['output']['supporting_training_cases'][0]['url'].endswith('a'*32))
            self.assertEqual(records[0]['output']['hypothesis_verdict'],'not_automatically_adjudicated')
            profile['include_training_case_links']=False
            records,_,_=build(run,imported,profile)
            self.assertEqual(records[0]['output']['case_link_status'],'disabled_by_profile')
