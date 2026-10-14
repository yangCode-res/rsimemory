import errno
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import round_storage


class RoundStorageTests(unittest.TestCase):
    def test_create_rejects_network_filesystem_and_probes_local_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            network = Path(temp) / "network"
            local = Path(temp) / "local"
            network.mkdir()
            local.mkdir()
            free = 20 * 1024**3
            with mock.patch.object(round_storage, "filesystem_type",
                                   side_effect=lambda p: "fuse.sshfs" if Path(p) == network else "xfs"), \
                 mock.patch.object(round_storage, "_free_bytes", return_value=free):
                scratch, reported = round_storage.create_scratch(
                    "123", candidates=[str(network), str(local)])
            self.assertEqual(scratch.parent, local)
            self.assertTrue(scratch.name.startswith("rsi-123-"))
            self.assertEqual(reported, free)
            self.assertEqual(scratch.stat().st_mode & 0o777, 0o700)
            self.assertFalse((scratch / ".storage-probe").exists())

    def test_create_requires_ten_gibibytes_by_default(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(round_storage, "filesystem_type", return_value="xfs"), \
                 mock.patch.object(round_storage, "_free_bytes", return_value=9 * 1024**3):
                with self.assertRaisesRegex(OSError, "no suitable local scratch"):
                    round_storage.create_scratch("123", candidates=[temp])

    def test_sync_retries_transient_eio_and_excludes_cache_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            (source / "result" / "policy-git" / ".git").mkdir(parents=True)
            (source / "result" / "policy-git" / ".git" / "HEAD").write_text("ref: main\n")
            (source / "result" / "__pycache__").mkdir()
            (source / "result" / "__pycache__" / "junk.pyc").write_bytes(b"cache")
            (source / "result" / ".rsi-atomic-progress.json.tmp").write_text("partial")
            (source / "result" / "answers.jsonl").write_text('{"answer":1}\n')
            original = round_storage._copy_hash
            calls = 0

            def flaky(src, tmp):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError(errno.EIO, "injected transient I/O error")
                return original(src, tmp)

            with mock.patch.object(round_storage, "_copy_hash", side_effect=flaky), \
                 mock.patch.object(round_storage.time, "sleep"):
                report = round_storage.archive(source, destination, final=False, retries=3)
            self.assertFalse(report["consistent"])
            self.assertEqual(report["copied_files"], 2)
            self.assertTrue((destination / "result" / "policy-git" / ".git" / "HEAD").exists())
            self.assertFalse((destination / "result" / "__pycache__").exists())
            self.assertFalse((destination / "result" / ".rsi-atomic-progress.json.tmp").exists())
            self.assertFalse((destination / round_storage.MANIFEST).exists())

    def test_finalize_writes_manifest_with_exact_verified_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            source.mkdir()
            (source / "answers.jsonl").write_bytes(b"answer\n")
            report = round_storage.archive(source, destination, final=True, retry_delay=0)
            manifest = json.loads((destination / round_storage.MANIFEST).read_text())
            self.assertTrue(report["consistent"])
            self.assertTrue(manifest["verified"])
            self.assertEqual(manifest["files"]["answers.jsonl"]["size"], 7)
            self.assertEqual(round_storage._hash(destination / "answers.jsonl")[0],
                             manifest["files"]["answers.jsonl"]["sha256"])

    def test_finalize_fails_when_destination_changes_before_verification(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            source.mkdir()
            (source / "answers.jsonl").write_bytes(b"answer\n")
            original = round_storage._atomic_copy

            def corrupt_after_copy(src, target, *args, **kwargs):
                item = original(src, target, *args, **kwargs)
                target.write_bytes(b"corrupt")
                return item

            with mock.patch.object(round_storage, "_atomic_copy", side_effect=corrupt_after_copy):
                with self.assertRaisesRegex(OSError, "checksum/size mismatch"):
                    round_storage.archive(source, destination, final=True, retry_delay=0)
            self.assertFalse((destination / round_storage.MANIFEST).exists())
            self.assertEqual((source / "answers.jsonl").read_bytes(), b"answer\n")

    def test_atomic_copy_retries_transient_directory_creation_eio(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            source.write_bytes(b"data")
            target = Path(temp) / "destination" / "nested" / "file"
            original = Path.mkdir
            failures = 0

            def flaky_mkdir(path, *args, **kwargs):
                nonlocal failures
                if path == target.parent and failures == 0:
                    failures += 1
                    raise OSError(errno.EIO, "injected mkdir error")
                return original(path, *args, **kwargs)

            with mock.patch.object(Path, "mkdir", flaky_mkdir), \
                 mock.patch.object(round_storage.time, "sleep"):
                item = round_storage._atomic_copy(source, target, retries=3,
                                                  retry_delay=0, require_stable=True)
            self.assertEqual(failures, 1)
            self.assertEqual(target.read_bytes(), b"data")
            self.assertEqual(item["size"], 4)

    def test_manifest_publication_retries_transient_eio(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            source.mkdir()
            (source / "answer").write_bytes(b"data")
            original = round_storage._write_json_atomic
            attempts = 0

            def flaky_manifest(path, value):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError(errno.EIO, "injected transient manifest error")
                return original(path, value)

            with mock.patch.object(round_storage, "_write_json_atomic", side_effect=flaky_manifest), \
                 mock.patch.object(round_storage.time, "sleep"):
                result = round_storage.archive(source, destination, final=True, retries=3)
            self.assertTrue(result["consistent"])
            self.assertEqual(attempts, 2)
            self.assertTrue(json.loads((destination / round_storage.MANIFEST).read_text())["verified"])

    def test_sync_invalidates_old_verified_manifest_when_archive_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            source.mkdir()
            (source / "answer").write_bytes(b"data")
            round_storage.archive(source, destination, final=True, retry_delay=0)
            original = round_storage._copy_hash

            def fail_copy(src, target):
                raise OSError(errno.EIO, "persistent injected error")

            with mock.patch.object(round_storage, "_copy_hash", side_effect=fail_copy), \
                 mock.patch.object(round_storage.time, "sleep"):
                with self.assertRaises(OSError):
                    round_storage.archive(source, destination, final=False, retries=2)
            self.assertFalse((destination / round_storage.MANIFEST).exists())
            self.assertEqual((source / "answer").read_bytes(), b"data")

    def test_sync_reports_changed_source_as_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "src"
            destination = Path(temp) / "archive"
            source.mkdir()
            (source / "answer").write_bytes(b"data")
            original = round_storage._copy_hash

            def changing_copy(src, target):
                item = original(src, target)
                with src.open("ab") as stream:
                    stream.write(b"+")
                return item

            with mock.patch.object(round_storage, "_copy_hash", side_effect=changing_copy), \
                 mock.patch.object(round_storage.time, "sleep"):
                result = round_storage.archive(source, destination, final=False,
                                               retries=2, retry_delay=0)
            self.assertEqual(result["skipped_changed"], ["answer"])
            self.assertEqual(result["copied_files"], 0)


if __name__ == "__main__":
    unittest.main()
