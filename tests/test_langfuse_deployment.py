import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('langfuse_hpc',
    Path(__file__).resolve().parents[1] / 'deploy/langfuse/hpc.py')
hpc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hpc)


class LangfuseDeployment(unittest.TestCase):
    def test_cached_rootfs_starts_without_old_image_store(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'state').mkdir();(root/'config/images').mkdir(parents=True)
            (root/'rootfs/postgres').mkdir(parents=True)
            (root/'state/rootfs-postgres.ready').write_text(hpc.IMAGES['postgres'])
            (root/'config/images/postgres.json').write_text(json.dumps({
                'image_id':hpc.IMAGES['postgres'],
                'config':{'Env':['PATH=/usr/bin:/bin'],'Entrypoint':['/entrypoint'],
                          'Cmd':['postgres'],'WorkingDir':''}}))
            calls=[]
            def engine(*args,**kwargs):
                calls.append(args)
                if args[:2]==('container','inspect'):
                    return subprocess.CompletedProcess(args,1,stdout='',stderr='')
                if args[0]=='image':
                    self.fail('Migrated service must not require the old image store')
                return subprocess.CompletedProcess(args,0,stdout='synthetic',stderr='')
            with patch.object(hpc,'DATA',root), patch.object(hpc,'engine',side_effect=engine), \
                    patch.object(hpc,'environment',return_value={'POSTGRES_PASSWORD':'synthetic'}), \
                    contextlib.redirect_stdout(io.StringIO()):
                hpc.launch('postgres')
            run=next(c for c in calls if c[0]=='run')
            position=run.index(str(root/'rootfs/postgres'))
            self.assertIn('--rootfs',run[:position])
            self.assertIn('--cgroups=disabled',run[:position])
            self.assertEqual(run[position+1],'/entrypoint')
            self.assertNotIn('synthetic',run)
            self.assertEqual((root/'config/postgres.env').stat().st_mode & 0o777,0o600)

    def test_namespace_pid_comes_from_runtime_not_persistent_data(self):
        with tempfile.TemporaryDirectory() as temp:
            runtime=Path(temp);(runtime/'state').mkdir()
            (runtime/'state/podman-ns.pid').write_text('123')
            with patch.object(hpc,'RUNTIME',runtime), \
                    patch.object(hpc.os,'sched_getaffinity',return_value={1,2,3,4,5},create=True):
                command=hpc.namespace_prefix()
            self.assertIn('123',command)
            self.assertEqual(command[:3],['taskset','-c','1,2,3,4'])

    def test_unfinished_cache_cannot_skip_archive_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'persistent';runtime=Path(temp)/'runtime'
            (root/'images').mkdir(parents=True);(root/'config/images').mkdir(parents=True)
            (runtime/'rootfs/postgres').mkdir(parents=True);(runtime/'state').mkdir()
            (runtime/'rootfs/postgres/.rsi-copy-complete').touch()
            (root/'images/postgres.rootfs.tar').write_bytes(b'incomplete synthetic archive')
            (root/'config/images/postgres.json').write_text(json.dumps({
                'image_id':hpc.IMAGES['postgres'],'rootfs_sha256':'0'*64,'config':{}}))
            def engine(*args,**kwargs):
                if args[0] in ('image','run'):
                    self.fail('Incomplete cache must not run or require an image store')
                return subprocess.CompletedProcess(args,1 if args[0]=='container' else 0,
                                                   stdout='',stderr='')
            with patch.object(hpc,'DATA',root),patch.object(hpc,'RUNTIME',runtime), \
                    patch.object(hpc,'engine',side_effect=engine):
                with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                    hpc.launch('postgres')


if __name__=='__main__':
    unittest.main()
