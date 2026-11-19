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
        archived = self.results / '000-system-info/000-system-info.v1.tar.gz'
        self.assertEqual(archived.read_bytes(), exact)
        self.assertEqual(json.loads((self.results / '000-system-info/v1/result.json').read_text()), row)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt['state'], 'failed')
        self.assertEqual(receipt['command_source'], 'recorded')
        self.assertEqual(receipt['archive_name'], '000-system-info.v1.tar.gz')
        self.assertEqual(receipt['archive_sha256'], runner.sha256_file(archived))
        duplicate = self.incoming / 'again.tar.gz'
        duplicate.write_bytes(exact)
        with patch.object(runner, 'timestamp', return_value='different later timestamp'):
            importer.archive_package(duplicate, self.results, self.status)
        self.assertFalse(duplicate.exists())
        self.assertEqual(json.loads(receipt_path.read_text()), receipt)

    def test_failed_profile_round_trips_native_report_and_result_receipt(self):
        task_id = 'gedes-000-fat-tree48-27648-01'
        label = 'v1-261119-044456'
        row = {
            'id': task_id, 'parent_id': 'gedes-000-fat-tree48-27648', 'subtask_id': '01',
            'slug': 'fat-tree48-27648', 'name': 'GeDES FatTree48 profile',
            'task_status': '2026-11-18T13:29:35Z', 'subtask_status': '2026-11-19T04:44:56Z',
            'version': 1795063496, 'version_label': label, 'kind': 'gedes_fat_tree48',
            'claim': 'profile-attempt', 'host_id': 'remote-host', 'worker': 'node-a',
            'source_commit': 'a' * 40, 'task_sha256': 'b' * 64, 'state': 'failed', 'exit_code': 1,
            'command': f'bash tools/runner.sh --task {task_id} --once',
        }
        remote_results = self.root / 'cpfs' / 'results'
        output = runner.run_path(remote_results, row)
        output.mkdir(parents=True)
        trace = hashlib.shake_128(b'nsys-trace-fixture').digest(2_340_000)
        trace_sha = hashlib.sha256(trace).hexdigest()
        summary = {'passed': False, 'error': 'flow completion times differ'}
        (output / 'profile.nsys-rep').write_bytes(trace)
        (output / 'profile.sqlite').write_bytes(b'sqlite export')
        runner.write_json(output / 'summary.json', summary)

        runner.prepare_archive(output, row)
        self.assertEqual(set(row['archive_files']), {'result.json', 'summary.json', 'profile.nsys-rep'})
        remote_package = runner.archive_path(remote_results, row)
        original = remote_package.read_bytes()
        original_sha = hashlib.sha256(original).hexdigest()
        package = self.incoming / remote_package.name
        package.write_bytes(original)
        ordinary_limit = 1024**2
        tar_limit = ordinary_limit + importer.MAX_MEMBERS * 1024 + 1024**2
        self.assertGreater(len(original), ordinary_limit)
        with patch.multiple(importer, MAX_BYTES=ordinary_limit, MAX_TAR_BYTES=tar_limit):
            receipt_path = importer.archive_package(package, self.results, self.status)

        archived_output = self.results / task_id / label
        self.assertFalse(package.exists())
        self.assertEqual((archived_output / 'profile.nsys-rep').read_bytes(), trace)
        self.assertEqual(runner.sha256_file(archived_output / 'profile.nsys-rep'), trace_sha)
        self.assertEqual(runner.read_json(archived_output / 'summary.json'), summary)
        self.assertEqual(runner.read_json(archived_output / 'result.json')['state'], 'failed')
        self.assertFalse((archived_output / 'profile.sqlite').exists())
        self.assertEqual((output / 'profile.sqlite').read_bytes(), b'sqlite export')
        self.assertEqual((archived_output.parent / remote_package.name).read_bytes(), original)
        receipt = runner.read_json(receipt_path)
        self.assertEqual(receipt['archive_sha256'], original_sha)
        self.assertEqual(receipt['state'], 'failed')

    def test_full_warp_capture_round_trip_with_bulk_members(self):
        row = {
            'id': 'gedes-000-fat-tree48-27648-02', 'parent_id': 'gedes-000-fat-tree48-27648',
            'subtask_id': '02', 'task_status': '2026-11-18T13:29:35Z',
            'subtask_status': '2026-11-19T04:44:56Z', 'version': 1795063496,
            'version_label': 'v1-261119-044456', 'slug': 'fat-tree48-27648',
            'kind': 'gedes_fat_tree48', 'task': {'params': {'warp_trace': True}},
            'source_commit': 'a'*40, 'task_sha256': 'b'*64, 'worker': 'node-a',
            'host_id': 'host', 'claim': 'test', 'state': 'succeeded',
        }
        remote = self.root / 'cpfs/results'
        output = runner.run_path(remote, row)
        (output / 'warp').mkdir(parents=True)
        contents = {name: bytes([i + 1])*2345 for i, name in enumerate(runner.WARP_TRACE_FILES)}
        for name, data in contents.items():
            (output / name).write_bytes(data)
        with patch.object(runner, 'FILE_LIMIT', 1000):
            runner.prepare_archive(output, row)
        archive = runner.archive_path(remote, row)
        with patch.object(importer, 'MAX_BYTES', 4096):
            receipt = importer.archive_package(archive, self.results, self.status)
        self.assertEqual(runner.read_json(receipt)['state'], 'succeeded')
        local = runner.run_path(self.results, row)
        for name, data in contents.items():
            self.assertEqual((local / name).read_bytes(), data)

    def test_classified_receipt_remains_idempotent_after_redownload(self):
        child = 'gedes-000-smoke-02'
        fields = {'parent_id': 'gedes-000-smoke', 'subtask_id': '02',
                  'task_status': '2026-11-17T00:00:00Z', 'subtask_status': '2026-11-18T00:00:00Z',
                  'id': child, 'version': 1794960000, 'slug': 'smoke'}
        package, _ = self.package(metadata=fields)
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        self.assertEqual(receipt_path, self.status / child / 'v1794960000.archived.json')
        receipt = runner.read_json(receipt_path)
        self.assertEqual(receipt['parent_id'], fields['parent_id'])
        self.assertEqual((self.results / child / (child + '.v1794960000.tar.gz')).read_bytes(), exact)
        package.write_bytes(exact)
        with patch.object(runner, 'timestamp', return_value='later'):
            importer.archive_package(package, self.results, self.status)
        self.assertEqual(runner.read_json(receipt_path), receipt)

    def test_labelled_archive_keeps_original_bytes_and_uses_release_paths(self):
        child = 'gedes-000-smoke-02'
        label = 'v1-261118-000000'
        archive_name = f'{child}.{label}.tar.gz'
        fields = {'parent_id': 'gedes-000-smoke', 'subtask_id': '02', 'id': child, 'slug': 'smoke',
                  'task_status': '2026-11-17T00:00:00Z', 'subtask_status': '2026-11-18T00:00:00Z',
                  'version': 1794960000, 'version_label': label, 'archive_name': archive_name}
        package, row = self.package(metadata=fields)
        exact = package.read_bytes()
        status_dir = self.status / child
        status_dir.mkdir(parents=True)
        (status_dir / 'unrelated.archived.json').write_text('not a receipt')

        receipt_path = importer.archive_package(package, self.results, self.status)

        self.assertEqual(receipt_path, status_dir / f'{label}.archived.json')
        receipt = runner.read_json(receipt_path)
        self.assertEqual(receipt['version'], 1794960000)
        self.assertEqual(receipt['version_label'], label)
        self.assertEqual(receipt['archive_name'], archive_name)
        self.assertEqual((self.results / child / archive_name).read_bytes(), exact)
        self.assertEqual(runner.read_json(self.results / child / label / 'result.json'), row)
        self.assertFalse((status_dir / 'v1794960000.archived.json').exists())
        package.write_bytes(exact)
        with patch.object(runner, 'timestamp', return_value='later'):
            self.assertEqual(importer.archive_package(package, self.results, self.status), receipt_path)
        self.assertEqual(runner.read_json(receipt_path), receipt)

    def test_labelled_and_unlabelled_receipts_cannot_claim_same_publication(self):
        child = 'gedes-000-smoke-02'
        fields = {'parent_id': 'gedes-000-smoke', 'subtask_id': '02', 'id': child, 'slug': 'smoke',
                  'task_status': '2026-11-17T00:00:00Z', 'subtask_status': '2026-11-18T00:00:00Z',
                  'version': 1794960000}
        package, _ = self.package(metadata=fields)
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        receipt_bytes = receipt_path.read_bytes()
        labelled, _ = self.package('labelled.tar.gz', {**fields, 'version_label': 'v1-261118-000000'})

        with self.assertRaisesRegex(importer.ArchiveError, 'incompatible archival receipt'):
            importer.archive_package(labelled, self.results, self.status)

        self.assertTrue(labelled.is_file())
        self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
        self.assertEqual((self.results / child / f'{child}.v1794960000.tar.gz').read_bytes(), exact)
        self.assertFalse((self.results / child / 'v1-261118-000000').exists())

    def test_labelled_import_rejects_a_receipt_alias_for_the_same_publication(self):
        child = 'gedes-000-smoke-02'
        label = 'v1-261118-000000'
        fields = {'parent_id': 'gedes-000-smoke', 'subtask_id': '02', 'id': child, 'slug': 'smoke',
                  'task_status': '2026-11-17T00:00:00Z', 'subtask_status': '2026-11-18T00:00:00Z',
                  'version': 1794960000, 'version_label': label}
        package, _ = self.package(metadata=fields)
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        receipt = runner.read_json(receipt_path)
        package.write_bytes(exact)
        for alias, archived in [('renamed', receipt), ('v2-261118-000000', {
                **receipt, 'version_label': 'v2-261118-000000',
                'archive_name': f'{child}.v2-261118-000000.tar.gz'})]:
            with self.subTest(alias=alias):
                alias_path = receipt_path.with_name(f'{alias}.archived.json')
                runner.write_json(alias_path, archived)
                with self.assertRaisesRegex(importer.ArchiveError, 'incompatible archival receipt'):
                    importer.archive_package(package, self.results, self.status)
                self.assertTrue(package.is_file())
                self.assertEqual(runner.read_json(receipt_path), receipt)
                alias_path.unlink()

    def test_classified_record_and_receipt_must_identify_the_same_publication(self):
        fields = {'parent_id': 'gedes-000-smoke', 'subtask_id': '02', 'id': 'gedes-000-smoke-02',
                  'task_status': '2026-11-17T00:00:00Z', 'subtask_status': '2026-11-18T00:00:00Z',
                  'version': 1794960000}
        package, row = self.package(metadata=fields)
        receipt = importer.receipt_for(row, package)
        for invalid in ({'version': 1794960001}, {'subtask_status': None}, {'task_status': 'ignore'},
                        {'task_status': '2026-11-19T00:00:00Z'}, {'subtask_id': '03'}):
            with self.subTest(invalid=invalid):
                with self.assertRaises(runner.RunnerError):
                    runner.validate_record({**row, **invalid}, package)
                with self.assertRaises(runner.RunnerError):
                    runner.validate_receipt({**receipt, **invalid})
        stripped = {k: v for k, v in row.items() if k not in runner.RELEASE_FIELDS}
        with self.assertRaises(runner.RunnerError):
            runner.validate_record(stripped, package)

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
        self.assertEqual((self.results / '000-system-info/000-system-info.v1.tar.gz').read_bytes(), exact)
        self.assertNotIn('slug', json.loads((self.results / '000-system-info/v1/result.json').read_text()))

    def test_conflicting_claim_preserves_incoming_and_archived_data(self):
        first, _ = self.package()
        receipt = importer.archive_package(first, self.results, self.status)
        receipt_bytes = receipt.read_bytes()
        original = (self.results / '000-system-info/000-system-info.v1.tar.gz').read_bytes()
        conflict, _ = self.package('conflict.tar.gz', metadata={'claim': 'attempt-two'})
        with self.assertRaisesRegex(importer.ArchiveError, 'incompatible archival receipt'):
            importer.archive_package(conflict, self.results, self.status)
        self.assertTrue(conflict.exists())
        self.assertEqual(receipt.read_bytes(), receipt_bytes)
        self.assertEqual((self.results / '000-system-info/000-system-info.v1.tar.gz').read_bytes(), original)

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
        self.assertEqual((target.parent / '000-system-info.v1.tar.gz').read_bytes(), exact)
        self.assertFalse(target.with_suffix('.tar.gz').exists())

    def test_legacy_receipt_and_package_are_renamed_without_recompression(self):
        package, _ = self.package()
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        canonical = self.results / '000-system-info/000-system-info.v1.tar.gz'
        legacy = canonical.parent / 'v1.tar.gz'
        canonical.rename(legacy)
        receipt = json.loads(receipt_path.read_text())
        receipt.pop('archive_name')
        receipt_path.write_text(json.dumps(receipt))
        duplicate = self.incoming / 'v1.tar.gz'
        duplicate.write_bytes(exact)

        with patch.object(runner, 'timestamp', return_value='later import'):
            importer.archive_package(duplicate, self.results, self.status)

        expected = {**receipt, 'archive_name': '000-system-info.v1.tar.gz'}
        self.assertEqual(json.loads(receipt_path.read_text()), expected)
        self.assertEqual(canonical.read_bytes(), exact)
        self.assertEqual(runner.sha256_file(canonical), receipt['archive_sha256'])
        self.assertFalse(legacy.exists())
        self.assertFalse(duplicate.exists())

    def test_numeric_directory_legacy_package_is_normalized(self):
        package, _ = self.package()
        exact = package.read_bytes()
        legacy = self.results / '000/v1.tar.gz'
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(exact)

        importer.archive_package(package, self.results, self.status)

        canonical = self.results / '000-system-info/000-system-info.v1.tar.gz'
        self.assertEqual(canonical.read_bytes(), exact)
        self.assertFalse(legacy.exists())

    def test_matching_legacy_duplicates_are_removed_after_import(self):
        package, _ = self.package()
        exact = package.read_bytes()
        importer.archive_package(package, self.results, self.status)
        legacy_paths = [self.results / '000-system-info/v1.tar.gz', self.results / '000/v1.tar.gz']
        for legacy in legacy_paths:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_bytes(exact)
        duplicate = self.incoming / 'downloaded-again.tar.gz'
        duplicate.write_bytes(exact)

        importer.archive_package(duplicate, self.results, self.status)

        self.assertEqual((self.results / '000-system-info/000-system-info.v1.tar.gz').read_bytes(), exact)
        self.assertTrue(all(not path.exists() for path in legacy_paths))
        self.assertFalse(duplicate.exists())

    def test_conflicting_legacy_and_canonical_packages_are_preserved(self):
        package, _ = self.package()
        exact = package.read_bytes()
        receipt_path = importer.archive_package(package, self.results, self.status)
        receipt_bytes = receipt_path.read_bytes()
        legacy = self.results / '000-system-info/v1.tar.gz'
        legacy.write_bytes(b'other result data')
        duplicate = self.incoming / 'again.tar.gz'
        duplicate.write_bytes(exact)

        with self.assertRaisesRegex(importer.ArchiveError, 'archived package differs'):
            importer.archive_package(duplicate, self.results, self.status)

        self.assertEqual(legacy.read_bytes(), b'other result data')
        self.assertEqual((self.results / '000-system-info/000-system-info.v1.tar.gz').read_bytes(), exact)
        self.assertEqual(receipt_path.read_bytes(), receipt_bytes)
        self.assertTrue(duplicate.exists())

    def test_declared_package_name_must_match_identity(self):
        package, _ = self.package(metadata={'archive_name': '001-other-task.v9.tar.gz'})
        with self.assertRaisesRegex((importer.ArchiveError, runner.RunnerError), 'archive'):
            importer.archive_package(package, self.results, self.status)
        self.assertTrue(package.exists())
        self.assertFalse(self.status.exists())

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
