import ast
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.cli import main
from rsi.fixtures import FixtureClient
from rsi.policy import GeneratedPolicy, PlainPolicy, _grams
from rsi.policy_repair import repair_tokenizer
from rsi.policy_source import baseline_source, source_manifest
from tests.test_retrieval_diagnostics import SOURCE


class ParentSourceTests(unittest.TestCase):
    def test_bundle_runs_without_host_globals_and_matches_baseline(self):
        source = baseline_source()
        tree = ast.parse(source)
        self.assertTrue({'_grams','PlainPolicy','initialize','ingest','search'} <=
                        {n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.level for n in tree.body))
        self.assertNotIn('class GeneratedPolicy', source)
        namespace = {'__name__':'standalone_parent_test'}
        exec(compile(source,'<standalone-parent>','exec'),namespace)
        config = {'top_k':1, 'semantic_weight':0}
        state = namespace['initialize'](config)
        actual = PlainPolicy(config)
        chunks = [{'id':'old', 'text':'青松片引起皮疹，随后停用。', 'persona_id':0, 'session_id':1},
                  {'id':'new', 'text':'今天安排饮食和运动。', 'persona_id':0, 'session_id':2},
                  {'id':'future', 'text':'青松片皮疹停用', 'persona_id':0, 'session_id':99}]
        for chunk in chunks:
            state = namespace['ingest'](state, chunk)
            actual.ingest(chunk)
        for query in ('青松片皮疹停用', '皮疹', '饮食', '   '):
            scope = {'persona_id':0, 'checkpoint':2}
            expected = [r['id'] for r in actual.search(query,scope)]
            self.assertEqual(namespace['search'](copy.deepcopy(state),query,scope),expected)

    def test_cli_sends_complete_parent_and_language_to_both_optimizer_stages(self):
        class Capture(FixtureClient):
            def __init__(self):
                super().__init__()
                self.optimizer_inputs = []
            def complete(self, messages, **kwargs):
                if kwargs.get('role') in ('reflexion','codegen'):
                    self.optimizer_inputs.append((kwargs['role'],json.loads(messages[-1]['content'])))
                return super().complete(messages,**kwargs)
        client = Capture()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'run'
            with patch.object(sys,'argv',['rsi','--fixture','--evolve','--output',str(out)]), \
                 patch('rsi.cli.FixtureClient',return_value=client), \
                 contextlib.redirect_stdout(io.StringIO()):
                main()
            source = (out/'parent-policy.py').read_text()
            manifest = json.loads((out/'parent-source.json').read_text())
            self.assertEqual(manifest,source_manifest(source,'zh'))
            config = json.loads((out/'config.json').read_text())
            self.assertEqual(config['parent_source_sha256'],manifest['sha256'])
        self.assertEqual(len(client.optimizer_inputs),4)
        for role, payload in client.optimizer_inputs:
            self.assertEqual(payload['parent_code'],source)
            self.assertIn('def _grams(',payload['parent_code'])
            self.assertIn('import re',payload['parent_code'])
            context = (payload['parent_context']['active_parent_config'] if role == 'reflexion'
                       else payload['runtime_context'])
            self.assertEqual(context['language'],'zh')
            self.assertIn('character bigrams',context['baseline_tokenization'])

    def test_repair_changes_only_one_function_and_matches_baseline_tokenization(self):
        original = ast.parse(SOURCE).body[0]
        lines = SOURCE.splitlines(keepends=True)
        prefix,suffix = ''.join(lines[:original.lineno-1]),''.join(lines[original.end_lineno:])
        fixed, metadata = repair_tokenizer(SOURCE)
        self.assertTrue(fixed.startswith(prefix))
        self.assertTrue(fixed.endswith(suffix))
        self.assertTrue(metadata['other_source_bytes_preserved'])
        namespace = {}
        exec(compile(fixed,'<synthetic-repair>','exec'),namespace)
        for value in ('青松片皮疹停用','Aspirin Rash','Straße\t疼痛\n缓解','  ', '甲', ''):
            self.assertEqual(namespace['_grams'](value),_grams(value))

    def test_repair_rejects_unexpected_module_instead_of_guessing(self):
        with self.assertRaisesRegex(ValueError,'one undecorated'):
            repair_tokenizer('def other():\n    pass\n')


@unittest.skipUnless(sys.platform.startswith('linux'),'Requires Linux isolated worker')
class IsolatedParentTests(unittest.TestCase):
    def test_standalone_parent_and_repaired_policy_run_in_real_worker(self):
        fixed,_ = repair_tokenizer(SOURCE)
        for source in (baseline_source(),fixed):
            with self.subTest(bundle=source == fixed):
                policy = GeneratedPolicy(source,{'top_k':1})
                for sid,text in ((1,'青松片引起皮疹后停用'),(2,'今天安排运动饮食')):
                    policy.ingest({'id':f'fixture-{sid}','text':text,'persona_id':0,'session_id':sid})
                self.assertEqual(policy.search('青松片皮疹停用',{'persona_id':0,'checkpoint':2})[0]['id'],
                                 'fixture-1')


if __name__ == '__main__':
    unittest.main()
