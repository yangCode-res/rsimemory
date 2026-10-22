import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rsi.continuous import run, select_parent, objective, atomic, read
from rsi.evolution import new_node
from rsi.model import ModelServiceUnavailable


class ContinuousSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.out=Path(self.temp.name)
        self.args=SimpleNamespace(max_candidates=6,children_per_expansion=2,max_depth=3,
            search_c=1.414,search_max_hours=0,fixture=False,no_diagnosis=True,reflexion_max_corrections=1)
        self.config={'language':'zh','policy_auxiliary_model':'disabled',
            **{k:1 for k in ('top_k','max_steps','context_chars','chunk_chars','chunk_overlap')}}
        self.report={'AVG':.4,'E2E_AVG':.4,'cost':{'policy':[]},'counts':{}}
        atomic(self.out/'plain-bm25/train/report.json',self.report)
        atomic(self.out/'plain-bm25/validation/report.json',self.report)
        (self.out/'plain-bm25/train/answers.jsonl').write_text('{"query_id":"root-case"}\n')
        (self.out/'parent-policy.py').write_text('# root')
        self.observed=[];self.codegen=[];self.evaluations=[]
        def reflect(client,rows,lookup,interventions,code,*args,**kwargs):
            self.observed.append((copy.deepcopy(rows),code))
            return {'clusters':[],'proposals':[{'direction':str(i)} for i in range(kwargs['proposal_count'])]}
        def generate(client,proposal,code,**kwargs):
            self.codegen.append(code)
            return code+'\n# version '+str(len(self.codegen)),{}
        def evaluate(jobs,args,config,sessions,queries,ids,actor,judge,accept):
            for job in jobs:
                self.evaluations.append(job['name'])
                for fold in ('train','validation'):
                    atomic(job['branch']/fold/'report.json',self.report)
                    (job['branch']/fold/'answers.jsonl').write_text(json.dumps({'query_id':job['name']})+'\n')
                node=copy.deepcopy(job['node']);node.update(status='evaluated',train_AVG=.4,validation_AVG=.4)
                accept((job['name'],node,{'AVG':.4,'E2E_AVG':.4}))
        self.evaluate=evaluate
        for name,value in [('rsi.continuous.reflect',reflect),('rsi.continuous.generate',generate),
                           ('rsi.cli._run_candidate_jobs',evaluate),('rsi.cli.commit_policy',lambda *a:'commit')]:
            patcher=patch(name,value);patcher.start();self.addCleanup(patcher.stop)

    def execute(self,initial=False):
        return run(self.args,self.out,self.config,{}, {'train':['a'],'validation':['b']},
            {'train':[],'validation':[]},[],None,None,None,
            initial=('# root','root-commit',self.report,self.report,{}, {},None) if initial else None)

    def test_budget_parent_evidence_and_completed_resume(self):
        result=self.execute(True)
        self.assertEqual(len(result['branches']),6)
        self.assertEqual(result['search']['stop_reason'],'candidate_budget')
        self.assertEqual(len(result['search']['expansions']),3)
        self.assertNotEqual(self.observed[1][0][0]['query_id'],'root-case')
        self.assertIn('version',self.observed[1][1])
        self.assertEqual(len(self.codegen),6)
        before=read(self.out/'search-state.json')['tree']
        self.execute()
        self.assertEqual(read(self.out/'search-state.json')['tree'],before)
        self.assertEqual(len(self.evaluations),6)
        self.assertTrue(all(n['generation']<=3 for n in result['branches']))

    def test_depth_limit(self):
        self.args.max_depth=1
        result=self.execute(True)
        self.assertEqual(len(result['branches']),2)
        self.assertEqual(result['search']['stop_reason'],'no_expandable_nodes')

    def test_recover_after_one_completed_candidate_without_double_reward(self):
        def interrupt(jobs,*args):
            self.evaluate(jobs[:1],*args)
            raise ModelServiceUnavailable('synthetic interruption')
        with patch('rsi.cli._run_candidate_jobs',interrupt):
            with self.assertRaises(ModelServiceUnavailable):self.execute(True)
        result=self.execute()
        self.assertEqual(len(result['branches']),6)
        self.assertEqual(self.evaluations.count('candidate-0001'),1)
        state=read(self.out/'search-state.json')
        self.assertEqual(state['tree']['nodes']['plain-bm25']['visits'],7)
        self.assertTrue(all(n.get('reward_applied') for n in result['branches']))

    def test_tampered_artifact_refuses_resume(self):
        self.execute(True)
        file=self.out/'branches/candidate-0001/train/report.json'
        file.write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.execute()

    def test_validation_objective_is_independent_of_strict_train_status(self):
        self.assertEqual(objective({'E2E_AVG':.2,'AVG':None}),.2)
        self.assertIsNone(objective({'E2E_AVG':None}))
        self.assertIsNone(objective({'E2E_AVG':float('nan')}))

    def test_uct_filters_expanded_failed_and_depth_limited_nodes(self):
        root=new_node('root');root.update(generation=0,expanded=True,search_eligible=True)
        child=new_node('child','root');child.update(generation=1,expanded=False,search_eligible=True,Q=.5,visits=1)
        root['visits']=3
        tree={'nodes':{'root':root,'child':child},'c':1.414}
        chosen,entries=select_parent(tree,3)
        self.assertEqual(chosen,'child');self.assertGreater(entries[0]['uct'],.5)
        child['search_eligible']=False
        self.assertIsNone(select_parent(tree,3)[0])

    def test_non_multiple_budget(self):
        self.args.max_candidates=3
        result=self.execute(True)
        self.assertEqual(len(result['branches']),3)
        self.assertEqual([e['count'] for e in result['search']['expansions']],[2,1])

    def test_invalid_codegen_consumes_budget(self):
        with patch('rsi.continuous.generate',side_effect=ValueError('invalid')):
            result=self.execute(True)
        self.assertEqual(len(result['branches']),2)
        self.assertEqual(result['search']['stop_reason'],'no_expandable_nodes')
        self.assertTrue(all(n['status']=='invalid' for n in result['branches']))
