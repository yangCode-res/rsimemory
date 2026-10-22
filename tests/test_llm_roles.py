import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rsi.llm_roles import DEFAULT_CONFIG_PATH, install, resolve, RoutedClient


class RoleConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'roles.yaml'
        self.config={'model':'local-actor','judge_model':'local-judge','actor_max_tokens':1024,
            'optimizer_generation_settings':{'reflexion':{'max_tokens':8192,'temperature':.7,'timeout_s':900},
                                            'codegen':{'max_tokens':16384,'temperature':.4,'timeout_s':900}}}

    def write(self,text):
        self.path.write_text('version: llm-roles-v1\n'+text)
        return resolve(self.path,self.config)

    def test_defaults_and_five_distinct_nodes(self):
        result=self.write('roles:\n  judge:\n    model: independent-judge\n  diagnosis:\n    model: diagnostic-model\n  proposals:\n    model: proposal-model\n')
        r=result['resolved']
        self.assertEqual(r['actor']['model'],'local-actor')
        self.assertEqual(r['judge']['model'],'independent-judge')
        self.assertNotEqual(r['diagnosis']['model'],r['proposals']['model'])
        self.assertEqual(r['diagnosis']['max_tokens'],8192)
        self.assertEqual(r['codegen']['temperature'],.4)
        self.assertIsNone(r['actor']['base_url'])

    def test_default_yaml_uses_historical_caps_and_provider_context_for_optimizer(self):
        result=resolve(DEFAULT_CONFIG_PATH,self.config)['resolved']
        self.assertEqual(result['actor']['max_tokens'],4096)
        self.assertEqual(result['judge']['max_tokens'],1024)
        for role in ('diagnosis','proposals','codegen'):
            with self.subTest(role=role):self.assertIsNone(result[role]['max_tokens'])

    def test_install_syncs_legacy_actor_and_judge_caps(self):
        config={**self.config,'judge_max_tokens':512}
        self.write('roles:\n  actor:\n    max_tokens: 4096\n  judge:\n    max_tokens: 1024\n')
        profile=install(config,self.path)
        self.assertEqual(config['actor_max_tokens'],profile['resolved']['actor']['max_tokens'])
        self.assertEqual(config['judge_max_tokens'],profile['resolved']['judge']['max_tokens'])

    def test_invalid_fields_values_and_credentials(self):
        for text in ['roles:\n  typo: {}\n','defaults:\n  api_key: secret\n',
                     'defaults:\n  temperature: .nan\n','defaults:\n  max_tokens: false\n',
                     'defaults:\n  max_tokens: 0\n','defaults:\n  max_tokens: -1\n',
                     'defaults:\n  base_url: https://user:password@example.com/v1\n',
                     'defaults:\n  model: first\n  model: second\n']:
            with self.subTest(text=text),self.assertRaises(ValueError):self.write(text)

    def test_real_request_routing_and_generation_accounting(self):
        profile=self.write('roles:\n  actor:\n    model: actor-a\n  diagnosis:\n    model: diagnosis-b\n    base_url: https://example.com/v1\n    api_key_env: EXPERIMENT_TEST_KEY\n  proposals:\n    model: proposals-c\n    max_tokens: 4321\n')
        requests=[]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return json.dumps({'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}}).encode()
        def serve(request,**kwargs):
            requests.append((request.full_url,json.loads(request.data),request.get_header('Authorization')))
            return Response()
        with patch.dict(os.environ,{'EXPERIMENT_TEST_KEY':'test-key-must-stay-private'}),patch('urllib.request.urlopen',side_effect=serve):
            client=RoutedClient({**self.config,'llm_roles':profile},'http://127.0.0.1:7863/v1')
            client.complete([{'role':'user','content':'synthetic'}],role='actor')
            client.for_stage('diagnosis').complete([{'role':'user','content':'synthetic'}],role='reflexion')
            client.for_stage('proposals').complete([{'role':'user','content':'synthetic'}],role='reflexion')
        self.assertEqual([r[1]['model'] for r in requests],['actor-a','diagnosis-b','proposals-c'])
        self.assertTrue(requests[0][0].startswith('http://127.0.0.1:7863/'))
        self.assertTrue(requests[1][0].startswith('https://example.com/'))
        self.assertEqual(requests[2][1]['max_tokens'],4321)
        self.assertEqual(requests[1][2],'Bearer test-key-must-stay-private')
        self.assertEqual(len(client.calls),3)
        self.assertNotIn('test-key-must-stay-private',json.dumps(profile)+json.dumps(client.calls))

    def test_null_override_omits_http_field(self):
        profile=self.write('roles:\n  proposals:\n    max_tokens: null\n')
        requests=[]
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,*args):return json.dumps({'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}]}).encode()
        def serve(request,**kwargs):
            requests.append(json.loads(request.data));return Response()
        with patch('urllib.request.urlopen',side_effect=serve):
            client=RoutedClient({**self.config,'llm_roles':profile},'http://localhost/v1')
            client.complete([{'role':'user','content':'x'}],role='proposals')
        self.assertNotIn('max_tokens',requests[0])

    def test_yaml_null_is_only_allowed_for_max_tokens(self):
        resolved=self.write('defaults:\n  max_tokens: null\nroles: {}\n')
        self.assertIsNone(resolved['resolved']['actor']['max_tokens'])
        for key in ('temperature','timeout_s'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.write(f'defaults:\n  {key}: null\n')

    def test_yaml_change_changes_recorded_fingerprint(self):
        first=self.write('roles: {}\n')
        second=self.write('roles:\n  judge:\n    model: another-model\n')
        self.assertNotEqual(first['source_sha256'],second['source_sha256'])
        self.assertNotEqual(first['resolved_sha256'],second['resolved_sha256'])


    def test_cli_resume_rejects_yaml_change_before_creating_clients(self):
        import contextlib,io,sys
        from rsi.cli import main
        profile=self.write('roles: {}\n')
        out=Path(self.temp.name)/'run';out.mkdir()
        (out/'config.json').write_text(json.dumps({'continuous':True,'evolve':True,
            'output':str(out),'llm_config':str(self.path),'llm_roles':profile}))
        self.path.write_text('version: llm-roles-v1\nroles:\n  judge:\n    model: changed\n')
        with patch.object(sys,'argv',['cli','--resume-search','--output',str(out)]), \
             patch('rsi.cli.ModelClient') as client,contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main()
            self.assertEqual(error.exception.code,2)
            client.assert_not_called()
