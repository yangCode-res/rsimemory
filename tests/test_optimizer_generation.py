import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.cli import main
from rsi.evolution import reflect
from rsi.fixtures import FixtureClient, dataset
from rsi.model import ModelClient, ModelOutputTruncated


class OptimizerGeneration(unittest.TestCase):
    def test_role_budgets_sampling_and_timeout_do_not_change_actor_or_judge(self):
        settings = {
            'reflexion': {'max_tokens':8192, 'temperature':.7, 'timeout_s':900},
            'codegen': {'max_tokens':16384, 'temperature':.4, 'timeout_s':900},
        }
        client = ModelClient(model='synthetic', role_settings=settings)
        seen = []
        def respond(request, timeout):
            seen.append((json.loads(request.data), timeout))
            return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop',
                'message':{'content':'complete'}}]}).encode())
        with patch('rsi.model.urllib.request.urlopen', side_effect=respond):
            for role in ('actor','reflexion','codegen','judge','codegen','actor'):
                client.complete([], role=role)
        self.assertEqual([(b['max_tokens'],b['temperature'],t) for b,t in seen],
            [(1024,0.,180.),(8192,.7,900),(16384,.4,900),(1024,0.,180.),
             (16384,.4,900),(1024,0.,180.)])
        self.assertTrue(all(b['chat_template_kwargs']=={'enable_thinking':False} for b,_ in seen))

    def test_length_finish_is_generation_failure_and_retains_usage(self):
        client = ModelClient(role_settings={'codegen':{'max_tokens':16384}})
        payload = {'choices':[{'finish_reason':'length','message':{'content':'incomplete'}}],
                   'usage':{'completion_tokens':16384}}
        with patch('rsi.model.urllib.request.urlopen',
                   return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaises(ModelOutputTruncated) as caught:
                client.complete([], role='codegen')
        self.assertEqual(caught.exception.category,'generation_truncated')
        self.assertEqual(client.calls[-1]['usage']['completion_tokens'],16384)

    def test_cli_wires_separate_optimizer_settings_and_persists_custom_values(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            argv = ['rsi','--dataset',str(Path(temp)/'synthetic-data'),'--output',str(out),
                    '--codegen-max-tokens','12000','--reflexion-temperature','0.8']
            with patch.object(sys,'argv',argv), patch('rsi.cli.load_medmemory',return_value=dataset()), \
                    patch('rsi.cli.ModelClient',side_effect=[FixtureClient(),FixtureClient(),FixtureClient()]) as factory, \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            config = json.loads((out/'config.json').read_text())
            self.assertEqual(config['optimizer_generation_settings']['codegen']['max_tokens'],12000)
            self.assertEqual(config['optimizer_generation_settings']['reflexion']['temperature'],.8)
            self.assertEqual(config['actor_max_tokens'],1024)
            self.assertEqual(factory.call_args_list[0].kwargs,{'max_tokens':1024})
            self.assertEqual(factory.call_args_list[1].kwargs,{'max_tokens':1024})
            self.assertEqual(factory.call_args_list[2].kwargs['role_settings'],
                             config['optimizer_generation_settings'])

    def test_cli_classifies_truncated_codegen_separately_from_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with patch.object(sys,'argv',['rsi','--fixture','--evolve','--output',str(out)]), \
                    patch('rsi.cli.generate',side_effect=ModelOutputTruncated('synthetic length limit')), \
                    contextlib.redirect_stdout(io.StringIO()):
                main()
            summary = json.loads((out/'summary.json').read_text())
            self.assertEqual(len(summary['branches']),3)
            self.assertTrue(all(n['failure_category']=='generation_truncated'
                                and n['failure_stage']=='codegen' for n in summary['branches']))

    def test_no_diagnosis_does_not_receive_train_metrics_or_cases(self):
        class Capture(FixtureClient):
            def complete(self,messages,**kwargs):
                self.prompt=json.loads(messages[1]['content'])
                return super().complete(messages,**kwargs)
        client=Capture()
        context={'active_parent_config':{'semantic_weight':0},
                 'parent_train_metrics':{'AVG':.1}}
        result=reflect(client,[],{},[],'parent',False,policy_context=context)
        self.assertEqual(client.prompt['examples'],[])
        self.assertNotIn('parent_train_metrics',client.prompt['parent_context'])
        self.assertEqual(client.prompt['parent_context']['active_parent_config']['semantic_weight'],0)
        self.assertEqual(result['rationale_schema'],'evidence-v3')
        self.assertIn('mechanism',result['proposals'][0])
        self.assertIn('falsification_test',result['proposals'][0])

    def test_confidence_word_is_not_accepted_as_causal_explanation(self):
        class ConfidenceOnly(FixtureClient):
            def complete(self,messages,**kwargs):
                response=super().complete(messages,**kwargs)
                obj=json.loads(response['text']);obj['clusters'][0]['attribution']='medium'
                response['text']=json.dumps(obj)
                return response
        with self.assertRaisesRegex(ValueError,'explanation'):
            reflect(ConfidenceOnly(),[],{},[],'parent')


if __name__=='__main__':
    unittest.main()
