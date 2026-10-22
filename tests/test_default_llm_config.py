import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.cli import main
from rsi.llm_roles import DEFAULT_CONFIG_PATH


class DefaultLLMConfigTests(unittest.TestCase):
    def run_fixture(self, output, extra_args=(), env=None):
        argv = ['rsi', '--fixture', '--output', str(output), *extra_args]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, env or {}, clear=False), \
                contextlib.redirect_stdout(io.StringIO()):
            main()
        return json.loads((Path(output) / 'config.json').read_text())

    def test_default_yaml_is_used_and_resolved_profile_is_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            config = self.run_fixture(Path(temp) / 'run')
            self.assertEqual(Path(config['llm_config']), DEFAULT_CONFIG_PATH)
            self.assertEqual(config['llm_roles']['source_sha256'],
                             hashlib.sha256(DEFAULT_CONFIG_PATH.read_bytes()).hexdigest())
            resolved = config['llm_roles']['resolved']
            self.assertEqual(resolved['actor']['max_tokens'], 4096)
            self.assertEqual(resolved['judge']['max_tokens'], 1024)
            for node in ('diagnosis', 'proposals', 'codegen'):
                with self.subTest(node=node):
                    self.assertIsNone(resolved[node]['max_tokens'])
            self.assertEqual(config['actor_max_tokens'], 4096)

    def test_missing_selected_yaml_fails_before_clients_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            missing = Path(temp) / 'missing.yaml'
            argv = ['rsi', '--fixture', '--output', str(out), '--llm-config', str(missing)]
            with patch.object(sys, 'argv', argv), patch('rsi.cli.ModelClient') as client, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)
            client.assert_not_called()
            self.assertFalse(out.exists())

    def test_missing_default_yaml_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'run'
            missing = Path(temp) / 'missing-default.yaml'
            with patch.object(sys, 'argv', ['rsi', '--fixture', '--output', str(out)]), \
                    patch('rsi.cli.DEFAULT_CONFIG_PATH', missing), \
                    patch('rsi.cli.ModelClient') as client, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)
            client.assert_not_called()
            self.assertFalse(out.exists())

    def test_explicit_yaml_overrides_default(self):
        with tempfile.TemporaryDirectory() as temp:
            custom = Path(temp) / 'custom.yaml'
            custom.write_text('version: llm-roles-v1\nroles:\n  actor:\n    max_tokens: 2048\n')
            config = self.run_fixture(Path(temp) / 'run', ['--llm-config', str(custom)])
            self.assertEqual(Path(config['llm_config']), custom.resolve())
            self.assertEqual(config['llm_roles']['resolved']['actor']['max_tokens'], 2048)
            self.assertEqual(config['actor_max_tokens'], 2048)

    def test_environment_yaml_overrides_default(self):
        with tempfile.TemporaryDirectory() as temp:
            custom = Path(temp) / 'environment.yaml'
            custom.write_text('version: llm-roles-v1\nroles:\n  judge:\n    max_tokens: 768\n')
            config = self.run_fixture(Path(temp) / 'run', env={'RSI_LLM_CONFIG': str(custom)})
            self.assertEqual(Path(config['llm_config']), custom.resolve())
            self.assertEqual(config['llm_roles']['resolved']['judge']['max_tokens'], 768)

    def test_continuous_fixture_resumes_with_same_yaml_from_new_path(self):
        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / 'original.yaml'
            replacement = Path(temp) / 'replacement.yaml'
            original.write_text('version: llm-roles-v1\nroles: {}\n')
            replacement.write_bytes(original.read_bytes())
            out = Path(temp) / 'continuous'
            first = self.run_fixture(out, ['--evolve', '--continuous', '--max-candidates', '1',
                                           '--llm-config', str(original)])
            self.assertTrue(first['continuous'])
            original.unlink()
            resumed = self.run_fixture(out, ['--resume-search'],
                                       env={'RSI_LLM_CONFIG': str(replacement)})
            self.assertEqual(resumed['max_candidates'], 1)
            self.assertEqual(resumed['llm_roles']['source_sha256'],
                             hashlib.sha256(replacement.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
