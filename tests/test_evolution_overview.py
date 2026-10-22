import copy
import unittest
from scripts.evolution_overview import comparison,delta,search_state,diagnostic_evidence,read_source

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
            write(run/'reflexion.json',{'clusters':[{'label':'retrieval',
                'supporting_case_ids':['case1'], 'attribution_status':'unresolved'}]})
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
            cluster=next(r for r in records if r['name'].startswith('error.cluster'))
            self.assertEqual(cluster['output']['cited_case_count'],1)
            self.assertTrue(cluster['output']['supporting_training_cases'][0]['url'].endswith('a'*32))
            self.assertEqual(records[-1]['output']['diagnostics']['clusters'][0]['label'],'retrieval')
            self.assertEqual(records[0]['output']['folds']['train']['E2E_AVG'],.3)
            profile['include_training_case_links']=False
            records,_,_=build(run,imported,profile)
            self.assertEqual(records[0]['output']['case_link_status'],'disabled_by_profile')


class SearchEvidenceTests(unittest.TestCase):
    def test_uct_matches_executor_without_claiming_selection(self):
        from rsi.evolution import uct
        tree={'c':1.414,'nodes':{'root':{'visits':5},'child':{
            'parent':'root','visits':2,'Q':.3,'status':'evaluated'}}}
        state=search_state(tree,'child')
        self.assertAlmostEqual(state['uct'],uct(tree['nodes']['child'],5))
        self.assertIsNone(state['selected_next'])
        self.assertEqual(state['selection_history'],'not_recorded')

    def test_nonfinite_priorities_are_json_safe_and_not_zero(self):
        import json
        for status,priority in [('pending','unvisited_positive_infinity'),
                                ('incomplete_scoring','unvisited_positive_infinity'),
                                ('invalid','excluded_negative_infinity')]:
            state=search_state({'nodes':{'c':{'visits':0,'status':status}}},'c')
            self.assertIsNone(state['uct'])
            self.assertEqual(state['priority'],priority)
            json.dumps(state,allow_nan=False)
        self.assertEqual(search_state({},'missing')['availability'],'not_recorded')

    def test_diagnostics_keep_sampling_separate_and_omit_patient_map(self):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'diagnostics').mkdir()
            (p/'diagnostics/summary.json').write_text(json.dumps({
                'total_questions':100,'failure_count':60,'patients':{'private':100},
                'error_groups':{'tool_error_observed':5}}))
            (p/'reflexion.json').write_text(json.dumps({'clusters':[{
                'label':'retrieval','supporting_case_ids':['example'],
                'attribution_status':'unresolved'}],'diagnostic_example_count':16}))
            result=diagnostic_evidence(p)
            self.assertEqual(result['observed_groups']['failure_count'],60)
            self.assertEqual(result['sample_coverage']['total'],16)
            self.assertNotIn('patients',result['observed_groups'])
            self.assertEqual(result['clusters'][0]['attribution_status'],'unresolved')

    def test_compressed_payload_remains_readable(self):
        import tempfile,gzip,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            with gzip.open(Path(d)/'traces.jsonl.gz','wt') as f:
                f.write(json.dumps({'name':'question'})+'\n')
            self.assertEqual(list(read_source(d)),[{'name':'question'}])
