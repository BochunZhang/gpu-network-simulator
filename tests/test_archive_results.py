import io
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import archive_results as importer
import run_experiments as runner


class ArchiveResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.incoming = self.root / 'results-remote'
        self.results = self.root / 'results'
        self.status = self.root / 'status'
        self.incoming.mkdir()

    def package(self, filename='download.tar.gz', metadata=None, summary=b'completed data', extra=None):
        row = {'id': '000', 'slug': 'system-info', 'name': 'GPU inventory', 'version': 1,
               'kind': 'gpu_inventory', 'claim': 'attempt-one', 'host_id': 'remote-host', 'worker': 'node-a',
               'source_commit': 'a' * 40, 'task_sha256': 'b' * 64, 'state': 'succeeded',
               'command': 'bash tools/runner.sh --task 000 --once',
               'archive_files': ['result.json', 'summary.txt']}
        if metadata:
            row.update(metadata)
        path = self.incoming / filename
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in [('result.json', json.dumps(row).encode()), ('summary.txt', summary)]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            if extra:
                for info, data in extra:
                    archive.addfile(info, io.BytesIO(data) if data is not None else None)
        return path, row

    def cli(self):
        return [sys.executable, '-B', str(REPO / 'tools/archive_results.py'),
                '--incoming-dir', str(self.incoming), '--results-dir', str(self.results),
                '--status-dir', str(self.status), '--lock-dir', str(self.root / '.lock')]

    def test_import_exact_package_and_idempotent_receipt(self):
        package, row = self.package(metadata={'state': 'failed'})
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        self.assertFalse(package.exists())
        self.assertEqual(receipt_path, self.status / '000-system-info/v1.archived.json')
        archived = self.results / '000-system-info/v1.tar.gz'
        self.assertEqual(archived.read_bytes(), exact)
        self.assertEqual(json.loads((self.results / '000-system-info/v1/result.json').read_text()), row)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['state'], 'failed')
        self.assertEqual(receipt['command_source'], 'recorded')
        self.assertEqual(receipt['archive_sha256'], runner.sha256_file(archived))
        duplicate = self.incoming / 'again.tar.gz'
        duplicate.write_bytes(exact)
        with patch.object(runner, 'timestamp', return_value='different later timestamp'):
            importer.archive_package(duplicate, self.results, self.status)
        self.assertFalse(duplicate.exists())
        self.assertEqual(json.loads(receipt_path.read_text()), receipt)

    def test_verified_case_manifest_is_exposed_in_receipt(self):
        manifest = b'{"schema_version": 1, "cases": {}}'
        checksum = hashlib.sha256(manifest).hexdigest()
        member = tarfile.TarInfo('artifacts-manifest.json')
        member.size = len(manifest)
        counts = {'executed': 1, 'reused': 2, 'failed': 0, 'skipped': 0}
        package, _ = self.package(metadata={
            'artifacts_manifest_sha256': checksum, 'case_counts': counts,
            'archive_files': ['result.json', 'summary.txt', 'artifacts-manifest.json'],
        }, extra=[(member, manifest)])
        receipt = json.loads(importer.archive_package(package, self.results, self.status).read_text())
        self.assertEqual(receipt['artifacts_manifest_sha256'], checksum)
        self.assertEqual(receipt['case_counts'], counts)
        bad, _ = self.package('bad-manifest.tar.gz', metadata={
            'artifacts_manifest_sha256': '0' * 64,
            'archive_files': ['result.json', 'summary.txt', 'artifacts-manifest.json'],
        }, extra=[(member, manifest)])
        with self.assertRaisesRegex(importer.ArchiveError, 'manifest checksum'):
            importer.archive_package(bad, self.results, self.status)
        self.assertTrue(bad.exists())

    def test_legacy_package_infers_slug_without_changing_tar_or_record(self):
        package, row = self.package()
        with tarfile.open(package) as archive:
            summary = archive.extractfile('summary.txt').read()
        row.pop('slug')
        row.pop('command')
        with tarfile.open(package, 'w:gz') as archive:
            for name, data in [('result.json', json.dumps(row).encode()), ('summary.txt', summary)]:
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['slug'], 'system-info')
        self.assertEqual(receipt['command_source'], 'replay')
        self.assertEqual(receipt['command'], 'bash tools/runner.sh --task 000 --once')
        self.assertEqual((self.results / '000-system-info/v1.tar.gz').read_bytes(), exact)
        self.assertNotIn('slug', json.loads((self.results / '000-system-info/v1/result.json').read_text()))

    def test_conflicting_claim_preserves_incoming_and_archived_data(self):
        first, _ = self.package()
        receipt = importer.archive_package(first, self.results, self.status)
        receipt_bytes = receipt.read_bytes()
        original = (self.results / '000-system-info/v1.tar.gz').read_bytes()
        conflict, _ = self.package('conflict.tar.gz', metadata={'claim': 'attempt-two'})
        with self.assertRaisesRegex(importer.ArchiveError, 'incompatible archival receipt'):
            importer.archive_package(conflict, self.results, self.status)
        self.assertTrue(conflict.exists())
        self.assertEqual(receipt.read_bytes(), receipt_bytes)
        self.assertEqual((self.results / '000-system-info/v1.tar.gz').read_bytes(), original)

    def test_partial_local_import_can_resume_before_receipt(self):
        package, row = self.package()
        target = self.results / '000-system-info/v1'
        target.mkdir(parents=True)
        with tarfile.open(package) as archive:
            (target / 'result.json').write_bytes(archive.extractfile('result.json').read())
        exact = package.read_bytes()
        target.with_suffix('.tar.gz').write_bytes(exact)
        receipt_path = importer.archive_package(package, self.results, self.status)
        self.assertTrue(receipt_path.is_file())
        self.assertEqual((target / 'summary.txt').read_text(), 'completed data')
        self.assertEqual(target.with_suffix('.tar.gz').read_bytes(), exact)

    def test_unsafe_tar_and_incomplete_manifest_are_preserved(self):
        traversal = tarfile.TarInfo('../escaped.txt')
        traversal.size = 1
        symlink = tarfile.TarInfo('link')
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = '/tmp/escaped'
        duplicate = tarfile.TarInfo('summary.txt')
        duplicate.size = 1
        undeclared = tarfile.TarInfo('undeclared.txt')
        undeclared.size = 1
        for index, member in enumerate([(traversal, b'x'), (symlink, None),
                                        (duplicate, b'x'), (undeclared, b'x')]):
            with self.subTest(member=member[0].name):
                package, _ = self.package(f'unsafe-{index}.tar.gz', extra=[member])
                with self.assertRaises(importer.ArchiveError):
                    importer.archive_package(package, self.results, self.status)
                self.assertTrue(package.exists())
                self.assertFalse((self.status / '000-system-info/v1.archived.json').exists())
        self.assertFalse((self.root / 'escaped.txt').exists())

    def test_corrupt_and_nonterminal_packages_are_not_acknowledged(self):
        corrupt = self.incoming / 'broken.tar.gz'
        corrupt.write_bytes(b'not a tar package')
        result = subprocess.run(self.cli(), text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(corrupt.exists())
        self.assertFalse(self.status.exists())
        package, _ = self.package(metadata={'state': 'running'})
        with self.assertRaisesRegex(importer.ArchiveError, 'not terminal'):
            importer.archive_package(package, self.results, self.status)
        self.assertTrue(package.exists())

    def test_gzip_trailer_and_crc_are_verified_before_acknowledgment(self):
        package, _ = self.package()
        original = package.read_bytes()
        for damaged in (original[:-4], original[:-8] + bytes([original[-8] ^ 1]) + original[-7:]):
            package.write_bytes(damaged)
            result = subprocess.run(self.cli(), text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertTrue(package.exists())
            self.assertFalse((self.status / '000-system-info/v1.archived.json').exists())

    def test_two_import_processes_use_shared_mutex(self):
        package, _ = self.package()
        script = ('import sys\nfrom pathlib import Path\n'
                  f'sys.path.insert(0, {str(REPO / "tools")!r})\nimport archive_results as a\n'
                  f'with a.import_lock(Path({str(self.root / ".lock")!r})):\n'
                  '    print("locked", flush=True)\n    sys.stdin.read()\n')
        holder = subprocess.Popen([sys.executable, '-B', '-c', script], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(holder.stdout.readline().strip(), 'locked')
            blocked = subprocess.run(self.cli(), text=True, capture_output=True, timeout=15)
            self.assertEqual(blocked.returncode, 1)
            self.assertIn('already locked', blocked.stderr)
            self.assertTrue(package.exists())
        finally:
            holder.communicate(timeout=10)
        completed = subprocess.run(self.cli(), text=True, capture_output=True, timeout=15)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertFalse(package.exists())

    def test_loose_result_directories_are_reported_without_deleting(self):
        loose = self.incoming / '000-system-info/v1'
        loose.mkdir(parents=True)
        (loose / 'result.json').write_text('{}')
        result = subprocess.run(self.cli(), text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 1)
        self.assertIn('original .tar.gz', result.stderr)
        self.assertTrue((loose / 'result.json').exists())
