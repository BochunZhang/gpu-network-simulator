from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import run_experiments as runner
from retired_experiments import reconcile_ignored


class RetirementTest(unittest.TestCase):
    parent = 'nsx-000-test'

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.store = runner.ClaimStore(root / '.lock', root / 'status')
        self.results = root / 'results'
        self.archive = root / 'results-archive'

    def definition(self, child='00', **kwargs):
        return {'parent_id': self.parent, 'subtask_id': child, 'task_status': '2026-11-01T00:00:00Z',
                'subtask_status': 'ignore', **kwargs}

    def run_row(self, task_id, version, state, child='00', finish=True):
        task = {'id': task_id, 'version': version, 'name': 'test', 'slug': 'test', 'kind': 'fixture',
                'command': 'test'}
        if task_id.startswith(self.parent):
            published = datetime.fromtimestamp(version, timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            task.update(self.definition(child, subtask_status=published))
        row = self.store.claim(task, 'c' * 64, 'a' * 40, self.results, 'worker')
        if not finish:
            return row
        return self.complete(row, state)

    def complete(self, row, state):
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        (output / 'worker.log').write_text('keep raw')
        row = {**row, 'state': state, 'exit_code': 0 if state == 'succeeded' else 1}
        runner.prepare_archive(output, row)
        runner.write_json(output / 'result.json', row)
        self.store.finish(row)
        return row

    def retire(self, *tasks, **kwargs):
        return reconcile_ignored([(task, None) for task in tasks], self.store, self.results,
                                 self.archive, **kwargs)

    def test_child_retirement_archives_all_releases_and_marks_pending_siblings_only_on_parent_ignore(self):
        old = self.run_row('001', 1, 'failed')
        new = self.run_row(self.parent + '-00', 1794960000, 'succeeded')
        sibling = self.run_row(self.parent + '-01', 1794960001, 'succeeded', child='01')
        before = [runner.archive_path(self.results, row).read_bytes() for row in (old, new)]
        child = self.definition(legacy_id='001')
        self.assertEqual(self.retire(child, self.definition('01', subtask_status='later')), [])
        for row, original in zip([old, new], before):
            status = self.store.read()[row['id'], row['version']]
            self.assertEqual((status['state'], status['delivery_state'], status['disposition']),
                             (row['state'], 'archived', 'ignored'))
            self.assertEqual(runner.archive_path(self.archive, row).read_bytes(), original)
            self.assertTrue((runner.run_path(self.archive, row) / 'worker.log').exists())
            self.assertTrue(self.store.reservation(row['id'], row['version']).exists())
        self.assertTrue(runner.run_path(self.results, sibling).exists())
        self.assertFalse((self.store.status_dir / self.parent / 'ignored.json').exists())
        self.assertEqual(self.retire(child), [])
        parent_ignore = self.definition('01', task_status='ignore', subtask_status='later')
        self.assertEqual(self.retire(parent_ignore, self.definition('02', task_status='ignore')), [])
        self.assertTrue((self.store.status_dir / self.parent / 'ignored.json').exists())
        self.assertTrue((self.store.status_dir / (self.parent + '-02') / 'ignored.json').exists())
        self.assertTrue(runner.run_path(self.archive, sibling).exists())

    def test_running_claim_finishes_before_retirement(self):
        row = self.run_row('001', 1, 'failed', finish=False)
        child = self.definition(legacy_id='001')
        self.assertEqual(self.retire(child), [])
        self.assertTrue(self.store.guard('001').exists())
        self.assertEqual(self.store.read()['001', 1]['state'], 'running')
        finished = self.complete(row, 'failed')
        self.assertEqual(self.retire(child), [])
        self.assertTrue(runner.run_path(self.archive, finished).exists())

    def test_ignored_definition_cannot_create_a_claim(self):
        task = {'id': self.parent + '-00', 'version': 1794960000, 'slug': 'old', 'kind': 'fixture',
                'name': 'old', 'command': 'test', **self.definition()}
        for fields in ({}, {'task_status': 'ignore', 'subtask_status': '2026-11-18T00:00:00Z'}):
            self.assertIsNone(self.store.claim({**task, **fields}, 'b' * 64, 'c' * 40, self.results, 'worker'))
        self.assertFalse(self.store.lock_dir.exists())

    def test_unstarted_subtask_and_legacy_alias_block_stale_claims(self):
        child = self.definition(legacy_id='001')
        self.assertEqual(self.retire(child), [])
        stale = {'id': '001', 'version': 1, 'slug': 'old', 'kind': 'fixture', 'name': 'old',
                 'command': 'test'}
        for task in (stale, {**stale, **self.definition(subtask_status='2026-11-11T08:00:00Z'),
                             'legacy_id': '001', 'id': self.parent + '-00', 'version': 1794384000}):
            self.assertIsNone(self.store.claim(task, 'b' * 64, 'c' * 40, self.results, 'old-runner'))
        self.assertEqual(self.store.read(), {})
        self.assertFalse(any(self.store.lock_dir.glob('*.v*.lock')))

    def test_abandoned_unsealed_work_stays_in_place(self):
        row = self.run_row('001', 1, 'failed', finish=False)
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        (output / 'raw.log').write_text('partial')
        self.store.abandon(self.results, '001', 1)
        self.assertEqual(len(self.retire(self.definition(legacy_id='001'))), 1)
        after = self.store.read()['001', 1]
        self.assertEqual(after['state'], 'abandoned')
        self.assertEqual(after['delivery_state'], 'awaiting_download')
        self.assertEqual((output / 'raw.log').read_text(), 'partial')
        self.assertFalse(runner.run_path(self.archive, row).exists())

    def test_abandoned_raw_identity_and_interrupted_checkpoint(self):
        row = self.run_row('001', 1, 'failed', finish=False)
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        child = self.definition(legacy_id='001')
        (output / 'worker.log').write_text('partial execution evidence')
        runner.write_json(output / 'run.json', {**row, 'claim': 'wrong'})
        self.store.abandon(self.results, '001', 1)
        self.assertEqual(len(self.retire(child)), 1)
        self.assertTrue(output.is_dir())
        runner.write_json(output / 'run.json', row)
        with patch.object(self.store, 'save', side_effect=OSError('interrupted after rename')):
            self.assertEqual(len(self.retire(child)), 1)
        self.assertFalse(output.exists())
        self.assertEqual(self.retire(child), [])
        destination = runner.run_path(self.archive, row)
        self.assertEqual((destination / 'worker.log').read_text(), 'partial execution evidence')
        status = self.store.read()['001', 1]
        self.assertEqual((status['state'], status['delivery_state'], status['disposition']),
                         ('abandoned', 'archived', 'ignored'))
        self.assertEqual(status['archive_contents'], 'partial_raw_only')
        self.assertNotIn('archive_sha256', status)
        self.assertFalse(runner.archive_path(self.archive, row).exists())
        self.assertTrue(self.store.reservation('001', 1).is_dir())

    def test_abandoned_raw_with_existing_package_or_terminal_record_requires_sealed_archive(self):
        row = self.run_row('001', 1, 'failed', finish=False)
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / 'run.json', row)
        self.store.abandon(self.results, '001', 1)
        child = self.definition(legacy_id='001')
        package = runner.archive_path(self.results, row)
        package.write_bytes(b'truncated package')
        self.assertEqual(len(self.retire(child)), 1)
        self.assertTrue(output.exists())
        self.assertEqual(package.read_bytes(), b'truncated package')
        package.unlink()
        runner.write_json(output / 'result.json', row)
        self.assertEqual(len(self.retire(child)), 1)
        self.assertTrue(output.exists())

    def test_corruption_and_partial_move_retain_evidence_and_retry(self):
        row = self.run_row('001', 1, 'failed')
        child = self.definition(legacy_id='001')
        package = runner.archive_path(self.results, row)
        raw = runner.run_path(self.results, row)
        original = package.read_bytes()
        package.write_bytes(b'corrupt')
        self.assertEqual(len(self.retire(child)), 1)
        self.assertEqual(package.read_bytes(), b'corrupt')
        self.assertTrue(raw.is_dir())
        package.write_bytes(original)
        rename = Path.rename
        def interrupt(path, destination):
            if path == package:
                raise OSError('interrupted')
            return rename(path, destination)
        with patch.object(Path, 'rename', interrupt):
            self.assertEqual(len(self.retire(child)), 1)
        self.assertTrue(runner.run_path(self.archive, row).is_dir())
        self.assertTrue(package.exists())
        self.assertEqual(self.retire(child), [])
        self.assertEqual(runner.archive_path(self.archive, row).read_bytes(), original)
        self.assertFalse(self.store.path('001', 1).with_suffix('.archive-error.json').exists())

    def test_preexisting_receipt_preserves_prior_delivery_and_rejects_mismatch(self):
        row = self.run_row('001', 1, 'succeeded')
        child = self.definition(legacy_id='001')
        receipt = {**row, 'schema_version': 1, 'archived_at': 'earlier',
                   'archive_sha256': runner.sha256_file(runner.archive_path(self.results, row))}
        self.assertEqual(len(self.retire(child, receipts=[{**receipt, 'archive_sha256': 'f' * 64}])), 1)
        self.assertTrue(runner.run_path(self.results, row).is_dir())
        self.assertEqual(self.store.archive_receipts([receipt], self.results, self.archive), [])
        self.assertEqual(self.retire(child, receipts=[receipt]), [])
        status = self.store.read()['001', 1]
        self.assertEqual(status['archived_at'], 'earlier')
        self.assertEqual(status['disposition'], 'ignored')
        self.assertNotIn('archive_reason', status)
