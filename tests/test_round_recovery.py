"""Recovery and terminal failure integration using synthetic data only."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rsi.cli import main
from rsi.fixtures import FixtureClient
from rsi.policy import PlainPolicy
from rsi.reflexion_protocol import ReflexionStageFailure
from rsi.runtime import replay


class RoundRecoveryTests(unittest.TestCase):
    def run_cli(self, out, *args):
        with patch.object(sys,'argv',['rsi','--fixture','--output',str(out),*args]), \
             contextlib.redirect_stdout(io.StringIO()):
            main()

    def seal_fixture(self, source):
        # Synthetic archive: explicitly declare its trusted runtime for a local
        # mirror without .git; production sources must prove their real commit.
        config=json.loads((source/'config.json').read_text())
        config['implementation_dirty']=False
        (source/'config.json').write_text(json.dumps(config))
        (source/'config-hash.json').write_text(json.dumps({
            'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}))
        manifest={'verified':True,'files':{}}
        for path in source.rglob('*'):
            if path.is_file() and '.git' not in path.parts:
                manifest['files'][path.relative_to(source).as_posix()]={
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size}
        (source/'_archive-manifest.json').write_text(json.dumps(manifest))

    def test_resume_skips_baseline_calls_and_replays_every_candidate_from_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'original'
            self.run_cli(source)
            self.seal_fixture(source)
            before={p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            out=Path(temp)/'resumed'
            fresh=[]
            def factory(source,config,**kwargs):
                policy=PlainPolicy(config)
                fresh.append(policy)
                return policy
            with patch('rsi.cli.GeneratedPolicy',side_effect=factory), \
                 patch('rsi.cli.replay',wraps=replay) as calls:
                self.run_cli(out,'--evolve','--resume-baselines-from',str(source))
            self.assertEqual(calls.call_count,6)
            self.assertTrue(all('branches' in str(c.args[-1]) for c in calls.call_args_list))
            self.assertEqual(len({id(p) for p in fresh}),len(fresh))
            self.assertEqual(before,{p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()})
            self.assertTrue(json.loads((out/'baseline-reuse.json').read_text())['source_manifest_verified'])
            self.assertEqual(json.loads((out/'run-status.json').read_text())['status'],'completed')
            self.assertEqual(len(json.loads((out/'summary.json').read_text())['branches']),3)
            self.assertEqual((source/'plain-bm25/train/report.json').read_bytes(),
                             (out/'plain-bm25/train/report.json').read_bytes())

    def test_exhausted_protocol_is_failed_and_keeps_baseline_scores(self):
        class Invalid(FixtureClient):
            def complete(self,messages,**kwargs):
                response=super().complete(messages,**kwargs)
                if kwargs.get('role')=='reflexion':
                    response['text']='{}'
                return response
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'failed'
            with patch('rsi.cli.FixtureClient',Invalid), \
                 patch('rsi.cli.generate',side_effect=AssertionError('no code generation')):
                with self.assertRaises(ReflexionStageFailure):
                    self.run_cli(out,'--evolve')
            summary=json.loads((out/'summary.json').read_text())
            self.assertEqual(summary['status'],'failed')
            self.assertEqual(summary['comparison']['plain-bm25']['AVG'],1.)
            self.assertEqual(summary['branches'],[])
            self.assertEqual(json.loads((out/'run-status.json').read_text())['stage'],'diagnosis')
            record=json.loads((out/'reflexion/diagnosis/result.json').read_text())
            self.assertEqual(record['status'],'failed')
            self.assertEqual(len(record['attempts']),2)
            self.assertTrue(all(a['input_messages'] and a['response'] and a['validation_errors']
                                for a in record['attempts']))

    def test_process_exit_is_nonzero_on_reflexion_failure(self):
        code='''import sys
from unittest.mock import patch
from rsi.cli import main
from rsi.reflexion_protocol import ReflexionStageFailure
with patch('rsi.cli.reflect',side_effect=ReflexionStageFailure('diagnosis','validation_error',{'attempts':[]})):
    main()
'''
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'failed'
            result=subprocess.run([sys.executable,'-c',code,'--fixture','--evolve','--output',str(out)],
                                  capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(json.loads((out/'run-status.json').read_text())['status'],'failed')

    def test_resume_mismatch_fails_before_any_new_model_or_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'rejected'
            with patch('rsi.cli.replay',side_effect=AssertionError('must verify before replay')):
                with self.assertRaises(ValueError):
                    self.run_cli(out,'--evolve','--resume-baselines-from',str(Path(temp)/'missing'))
            self.assertEqual(json.loads((out/'run-status.json').read_text())['stage'],'baseline.verification')


if __name__=='__main__':
    unittest.main()
