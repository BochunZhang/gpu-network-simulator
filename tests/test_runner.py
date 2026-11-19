import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import run_experiments as runner


class RunnerTest(unittest.TestCase):
    def git(self, cwd, *args):
        return subprocess.run([self.real_git, *args], cwd=cwd, text=True,
                              capture_output=True, check=True).stdout.strip()

    def setUp(self):
        self.real_git = shutil.which('git')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote, self.author, self.checkout = (self.root / name for name in ('origin.git', 'author', 'shared'))
        self.git(self.root, 'init', '--bare', '--initial-branch=main', str(self.remote))
        hook = self.remote / 'hooks' / 'pre-receive'
        hook.write_text('#!/bin/sh\necho uploads-forbidden >&2\nexit 1\n')
        hook.chmod(0o755)
        self.git(self.root, 'init', '--initial-branch=main', str(self.author))
        self.git(self.author, 'config', 'user.name', 'Test')
        self.git(self.author, 'config', 'user.email', 'test@localhost')
        self.git(self.author, 'config', 'commit.gpgsign', 'false')
        (self.author / 'tasks').mkdir()
        (self.author / 'tools').mkdir()
        for module in ('run_experiments.py', 'incremental.py', 'task_catalog.py', 'retired_experiments.py', 'archive_results.py'):
            shutil.copyfile(REPO / 'tools' / module, self.author / 'tools' / module)
        self.commit_task()
        self.git(self.root, 'clone', str(self.remote), str(self.checkout))
        self.lock_dir, self.status_dir, self.results = (self.checkout / name for name in ('.lock', 'status', 'results'))
        self.queue = runner.Queue(self.checkout)
        self.store = runner.ClaimStore(self.lock_dir, self.status_dir)

    def commit_task(self, task_id='000', version=1, marker='original', **fields):
        task = {'id': task_id, 'name': 'Smoke task', 'version': version,
                'kind': 'new_kind', 'params': {}, 'slug': 'new-kind',
                'command': f'bash tools/runner.sh --task {task_id} --once', **fields}
        runner.write_json(self.author / f'tasks/{task_id}.json', task)
        (self.author / 'tools/experiment_handlers.py').write_text(
            'def prepare(kind, params):\n'
            '    def run(source, output):\n'
            f"        (output / 'summary.txt').write_text({marker!r})\n"
            "        (output / 'raw.log').write_text('excluded')\n"
            '    return run\n', encoding='utf-8')
        self.git(self.author, 'add', '.')
        self.git(self.author, 'commit', '-m', f'{task_id} v{version} {marker}')
        # Local fixture authoring bypasses the receiver. Workers remain pull-only.
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        return task

    def source_task(self, task_id='000'):
        source = self.queue.source()
        task, digest = next(pair for pair in self.queue.tasks(source) if pair[0]['id'] == task_id)
        return source, task, digest

    def commit_case_task(self, version, execution, fail=False, value='unchanged', comment=''):
        task = {'id': '000', 'slug': 'new-kind', 'name': 'Incremental fixture', 'version': version,
                'kind': 'case_fixture', 'command': 'bash tools/runner.sh --task 000 --once',
                'params': {'fail': fail, 'value': value}, 'execution': execution}
        runner.write_json(self.author / 'tasks/000.json', task)
        (self.author / 'tools/experiment_handlers.py').write_text(
            f'# {comment}\n'
            'def case_plan(task, source):\n'
            '    def stable(output, dependencies):\n'
            "        (output / 'value.txt').write_text('stable result')\n"
            '    def mutable(output, dependencies):\n'
            "        (output / 'value.txt').write_text(task['params']['value'])\n"
            "        if task['params']['fail']: raise RuntimeError('fixture failure')\n"
            "    return [{'id':'stable','implementation':'stable-v1','inputs':{},'reuse':True,'run':stable},\n"
            "            {'id':'mutable','implementation':'mutable-v1','inputs':task['params'],'reuse':True,'run':mutable}]\n",
            encoding='utf-8')
        self.git(self.author, 'add', '.')
        self.git(self.author, 'commit', '-m', f'incremental fixture v{version} {comment}')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        return task

    def case_cli(self, expected=0):
        process = subprocess.run([sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'),
                                  '--task', '000', '--once'], text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, expected, process.stdout + process.stderr)
        return process

    def publish_catalog(self, manifest):
        (self.author / 'tasks/000.json').unlink(missing_ok=True)
        runner.write_json(self.author / 'tasks' / f"{manifest['id']}.json", manifest)
        self.git(self.author, 'add', 'tasks')
        self.git(self.author, 'commit', '-m', 'Publish catalog update')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')

    def test_cli_publishes_extends_reissues_and_retires(self):
        parent = 'nsx-000-topologies'
        initial = '2026-11-18T00:00:00Z'
        batch = runner.task_catalog.release_version(initial)
        first = {'status': initial, 'kind': 'new_kind', 'params': {'topos': ['A', 'B', 'C']}}
        manifest = {'id': parent, 'name': 'Topology checks', 'status': initial, 'subtasks': {'00': first}}
        self.publish_catalog(manifest)

        def once(selector):
            proc = subprocess.run([sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'),
                                   '--task', selector, '--once'], text=True, capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            return proc

        child0, child1 = parent + '-00', parent + '-01'
        first_execution = once(parent)
        self.assertIn(f'complete {child0} v1-261118-000000 state=succeeded '
                      f'archive_name={child0}.v1-261118-000000.tar.gz', first_execution.stdout)
        first_run = self.store.read()[child0, batch]
        package = runner.archive_path(self.results, first_run)
        checksum = runner.sha256_file(package)
        manifest['subtasks']['01'] = {**first, 'params': {'topos': ['D']}}
        self.publish_catalog(manifest)
        once(parent)
        self.assertEqual(len(self.store.read()), 2)
        first['status'] = '2026-11-18T00:00:01Z'
        self.publish_catalog(manifest)
        once(child1)
        self.assertEqual(len(self.store.read()), 2)
        once(child0)
        self.assertEqual(set(self.store.read()), {(child0, batch), (child1, batch), (child0, batch + 1)})
        self.assertEqual(runner.sha256_file(package), checksum)
        self.assertEqual(self.store.read()[child0, batch]['claim'], first_run['claim'])
        self.assertEqual(package.name, f'{child0}.v1-261118-000000.tar.gz')
        self.assertTrue((self.results / child0 / 'v2-261118-000001' / 'result.json').is_file())
        self.assertTrue((self.status_dir / child1 / 'v1-261118-000000.json').is_file())

        first['status'] = 'ignore'
        self.publish_catalog(manifest)
        once(parent)
        rows = self.store.read()
        for release in (batch, batch + 1):
            self.assertEqual(rows[child0, release]['disposition'], 'ignored')
            self.assertEqual(rows[child0, release]['state'], 'succeeded')
        self.assertFalse(package.exists())
        self.assertEqual(runner.sha256_file(runner.archive_path(self.checkout / 'results-archive', first_run)),
                         checksum)
        self.assertEqual(rows[child1, batch]['delivery_state'], 'awaiting_download')
        manifest['status'] = 'ignore'
        self.publish_catalog(manifest)
        once(parent)
        self.assertEqual(self.store.read()[child1, batch]['disposition'], 'ignored')

    def test_compound_category_has_its_own_task_indices(self):
        stamp = '2026-11-18T00:00:00Z'
        child = {'status': stamp, 'kind': 'new_kind', 'params': {}}
        manifest = {'name': 'Fixture', 'status': stamp, 'subtasks': {'00': child}}
        parent = 'days-ago-000-fat-tree48-18432'
        self.publish_catalog({**manifest, 'id': 'days-000-topology'})
        self.publish_catalog({**manifest, 'id': parent})
        _, task, _ = self.source_task(parent + '-00')
        self.assertEqual(task['slug'], 'fat-tree48-18432')

        process = subprocess.run([sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'),
                                  '--task', parent, '--once'], text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        record = self.store.read()[task['id'], task['version']]
        self.assertTrue(runner.archive_path(self.results, record).is_file())

        self.publish_catalog({**manifest, 'id': 'days-ago-002-other'})
        self.publish_catalog({**manifest, 'id': 'days-ago-001-backdated'})
        with self.assertRaisesRegex(runner.RunnerError, 'must follow the published days-ago indices'):
            self.queue.tasks(self.queue.source())

    def test_fresh_controller_validates_parent_history_and_accepts_a_correction(self):
        manifest = {'id': 'nsx-000-topology', 'name': 'Topology', 'status': '2026-11-18T00:00:00Z',
                    'subtasks': {'00': {'kind': 'new_kind', 'params': {}, 'status': '2026-11-18T00:00:10Z'}}}
        self.publish_catalog(manifest)
        manifest['status'] = '2026-11-18T00:00:01Z'
        self.publish_catalog(manifest)
        with self.assertRaisesRegex(runner.RunnerError, 'immutable'):
            self.queue.tasks(self.queue.source())
        manifest['status'] = '2026-11-18T00:00:00Z'
        self.publish_catalog(manifest)
        source = self.queue.source()
        fresh = runner.Queue(self.checkout)
        self.assertEqual([t['id'] for t, _ in fresh.tasks(source)], ['nsx-000-topology-00'])

    def test_publication_labels_include_unclaimed_releases_and_survive_restart(self):
        first = '2026-11-18T00:00:10Z'
        second = '2026-11-18T00:00:20Z'
        third = '2026-11-18T00:00:30Z'
        parent = 'days-ago-000-topology'
        child = {'kind': 'new_kind', 'params': {}, 'status': first}
        manifest = {'id': parent, 'name': 'Topology', 'status': first, 'subtasks': {'00': child}}
        self.publish_catalog(manifest)
        _, task, digest = self.source_task(parent + '-00')
        self.assertEqual(task['version_label'], 'v1-261118-000010')
        self.assertEqual(runner.task_catalog.execution_digest(task), digest)

        child.update(enabled=False, description='Paused')
        self.publish_catalog(manifest)
        _, paused, paused_digest = self.source_task(parent + '-00')
        self.assertEqual((paused['version_label'], paused_digest), (task['version_label'], digest))

        child['status'] = second
        self.publish_catalog(manifest)
        self.publish_catalog({**manifest, 'id': 'gedes-000-other'})
        child.update(status=third, enabled=True)
        manifest['subtasks']['01'] = {**child}
        self.publish_catalog(manifest)
        source = self.queue.source()
        current = self.queue.tasks(source)
        self.assertEqual(current, runner.Queue(self.checkout).tasks(source))
        self.assertEqual({task['subtask_id']: task['version_label'] for task, _ in current
                          if task['parent_id'] == parent},
                         {'00': 'v3-261118-000030', '01': 'v1-261118-000030'})

        child['status'] = 'ignore'
        self.publish_catalog(manifest)
        _, retired, _ = self.source_task(parent + '-00')
        self.assertEqual(retired['subtask_status'], 'ignore')
        _, _, _, history = self.queue._catalog[parent]
        self.assertEqual(list(history[retired['id']].values()),
                         ['v1-261118-000010', 'v2-261118-000020', 'v3-261118-000030'])

    def test_publication_label_cannot_be_guessed_from_a_shallow_clone(self):
        stamp = '2026-11-18T00:00:10Z'
        child = {'status': stamp, 'kind': 'new_kind', 'params': {}}
        self.publish_catalog({'id': 'gedes-000-topology', 'name': 'Topology',
                              'status': stamp, 'subtasks': {'00': child}})
        shallow = self.root / 'shallow'
        self.git(self.root, 'clone', '--depth=1', self.remote.as_uri(), str(shallow))
        with self.assertRaisesRegex(runner.RunnerError, 'publication labels require the complete task history'):
            runner.Queue(shallow).tasks(self.queue.source())

    def test_archive_boundary_frees_slots_and_preserves_future_publication_history(self):
        parent = 'nsx-000-topology'
        stamp = '2026-11-18T00:00:00Z'
        manifest = {'id': parent, 'name': 'Old topology', 'status': stamp,
                    'subtasks': {'00': {'kind': 'new_kind', 'params': {}, 'status': stamp}}}
        legacy = {'id': '000', 'slug': 'new-kind', 'version': 1, 'published_at': stamp}
        self.publish_catalog({**manifest, 'legacy': legacy})
        old_commit = self.queue.source()
        active_file = self.author / 'tasks' / f'{parent}.json'
        backup_file = self.author / 'archive' / active_file.name
        backup_file.parent.mkdir()
        active_file.rename(backup_file)
        (self.author / 'tasks/README.md').write_text('Reviewed tasks only.\n')

        def publish_change(message):
            self.git(self.author, 'add', '.')
            self.git(self.author, 'commit', '-m', message)
            self.git(self.remote, 'fetch', str(self.author), 'main:main')
            return self.queue.source()

        archived = publish_change('Archive old experiments')
        with patch.object(runner, 'ARCHIVED_CATALOG_COMMIT', old_commit):
            self.assertEqual(self.queue.tasks(archived), [])
            self.commit_task()
            shallow = self.root / 'shallow'
            self.git(self.root, 'clone', '--depth=1', self.remote.as_uri(), str(shallow))
            source = self.queue.source()
            for controller in (self.queue, runner.Queue(shallow)):
                with self.assertRaisesRegex(runner.RunnerError, 'invalid classified task path'):
                    controller.tasks(source)
            # Restoring the backed-up alias also cannot re-activate a numeric execution.
            self.publish_catalog({**manifest, 'legacy': legacy})
            with self.assertRaisesRegex(runner.RunnerError, 'historical execution aliases'):
                self.queue.tasks(self.queue.source())
            reviewed = '2026-11-19T00:00:00Z'
            manifest.update(name='Reviewed topology', status=reviewed)
            manifest['subtasks']['00']['status'] = reviewed
            self.publish_catalog(manifest)
            published = self.queue.source()
            [(task, _)] = self.queue.tasks(published)
            self.assertEqual(task['id'], parent + '-00')
            self.assertNotIn('legacy_id', task)
            self.assertNotIn('legacy_base', task)
            self.assertTrue(backup_file.exists())

            active_file.unlink()
            deleted = publish_change('Invalid removal of reviewed task')
            for controller in (self.queue, runner.Queue(self.checkout)):
                with self.assertRaisesRegex(runner.RunnerError, 'must remain in the catalog'):
                    controller.tasks(deleted)

    def test_incremental_cli_reuses_passed_case_from_failed_archived_v2(self):
        self.commit_case_task(2, {'mode': 'full'}, fail=True, value='v2')
        failed = self.case_cli(expected=1)
        self.assertIn('complete 000 v2 state=failed archive_name=000-new-kind.v2.tar.gz', failed.stdout)
        output2 = self.results / '000-new-kind/v2'
        result2 = runner.read_json(output2 / 'result.json')
        self.assertEqual(result2['case_counts'], {'executed': 2, 'reused': 0, 'failed': 1, 'skipped': 0})
        receipt = {key: result2[key] for key in ('id', 'slug', 'name', 'version', 'claim', 'source_commit',
                                                'task_sha256', 'state', 'command')}
        receipt.update(schema_version=1, archived_at='2026-11-18T00:00:00+00:00',
                       archive_sha256=runner.sha256_file(runner.archive_path(self.results, result2)))
        self.store.archive_receipts([receipt], self.results, self.checkout / 'results-archive')
        self.assertFalse(output2.exists())
        execution = {'mode': 'incremental', 'base': {'id': '000', 'version': 2,
                     'artifacts_manifest_sha256': result2['artifacts_manifest_sha256']}}
        self.commit_case_task(3, execution, fail=False, value='v3', comment='unrelated source text changed')
        self.case_cli()
        output3 = self.results / '000-new-kind/v3'
        result3 = runner.read_json(output3 / 'result.json')
        self.assertEqual(result3['case_counts'], {'executed': 1, 'reused': 1, 'failed': 0, 'skipped': 0})
        self.assertEqual(result3['base']['version'], 2)
        manifest = runner.read_json(output3 / 'artifacts-manifest.json')
        self.assertEqual(manifest['cases']['stable']['origin'], 'reused')
        self.assertEqual(manifest['cases']['mutable']['origin'], 'executed')
        self.assertEqual((output3 / 'cases/mutable/value.txt').read_text(), 'v3')
        self.assertEqual((self.checkout / 'results-archive/000-new-kind/v2/cases/mutable/value.txt').read_text(), 'v2')
        with tarfile.open(runner.archive_path(self.results, result3)) as archive:
            self.assertIn('artifacts-manifest.json', archive.getnames())
            self.assertIn('cases/stable/value.txt', archive.getnames())
        self.case_cli()
        self.assertEqual(self.store.read()['000', 3]['state'], 'succeeded')
        self.commit_case_task(3, execution, value='changed without increasing version')
        rejected = self.case_cli(expected=1)
        self.assertIn('changed without increasing version', rejected.stderr)
        self.assertEqual((output3 / 'cases/mutable/value.txt').read_text(), 'v3')

    def test_missing_incremental_base_waits_without_claim_and_other_task_runs(self):
        self.commit_case_task(3, {'mode': 'incremental', 'base': {'id': '000', 'version': 2,
                              'artifacts_manifest_sha256': '0' * 64}})
        self.case_cli()
        self.assertFalse(self.store.guard('000').exists())
        self.assertFalse(self.store.reservation('000', 3).exists())
        waiting = runner.read_json(self.status_dir / '000-new-kind/v3.waiting.json')
        self.assertEqual(waiting['state'], 'waiting_for_base')
        self.assertEqual(self.store.read(), {})
        self.commit_task(task_id='001', marker='unblocked independent task')
        process = subprocess.run([sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'), '--once'],
                                 text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(self.store.read()['001', 1]['state'], 'succeeded')
        self.assertFalse(self.store.reservation('000', 3).exists())

    def test_case_finalizer_cannot_mutate_committed_artifacts(self):
        self.commit_case_task(2, {'mode': 'full'})
        handler = self.author / 'tools/experiment_handlers.py'
        handler.write_text(handler.read_text() +
                           '\ndef finalize_cases(task, output, manifest):\n'
                           "    (output / 'cases/stable/value.txt').write_text('invalid mutation')\n"
                           "    manifest.clear()\n")
        self.git(self.author, 'add', '.')
        self.git(self.author, 'commit', '-m', 'unsafe fixture finalizer')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        self.case_cli(expected=1)
        output = self.results / '000-new-kind/v2'
        result = runner.read_json(output / 'result.json')
        manifest = runner.read_json(output / 'artifacts-manifest.json')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(manifest['state'], 'failed')
        self.assertEqual(manifest['cases']['stable']['state'], 'failed')
        self.assertEqual(result['case_counts']['failed'], 1)

    def claim(self, task_id='000'):
        source, task, digest = self.source_task(task_id)
        row = self.store.claim(task, digest, source, self.results, 'test-host')
        return task, row

    def completed_fixture(self, legacy=False, task_id='000'):
        task, row = self.claim(task_id)
        if legacy:
            row = {key: value for key, value in row.items() if key not in {'slug', 'command'}}
            self.store.path(task_id, 1).unlink()
            legacy_status = self.status_dir / task_id / 'v1.json'
            legacy_status.parent.mkdir(parents=True)
            runner.write_json(legacy_status, row)
            output = self.results / task_id / 'v1'
        else:
            output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / 'run.json', {**row, 'task': task})
        self.assertEqual(runner.execute(self.queue, output), 0)
        self.store.reconcile(self.results)
        result = runner.read_json(output / 'result.json')
        receipt = {key: result[key] for key in
                   ('id', 'slug', 'name', 'version', 'claim', 'source_commit', 'task_sha256', 'state', 'command')}
        receipt.update({key: result[key] for key in runner.RELEASE_FIELDS if key in result})
        receipt.update(schema_version=1, archive_sha256=runner.sha256_file(runner.archive_path(self.results, result)),
                       archived_at='2026-11-18T00:00:00+00:00')
        return output, receipt

    def test_slug_metadata_legacy_digest_and_status_filter(self):
        source, task, digest = self.source_task()
        legacy = {key: value for key, value in task.items() if key not in {'slug', 'command'}}
        definition = {key: value for key, value in legacy.items() if key not in {'description', 'enabled'}}
        import hashlib
        self.assertEqual(digest, hashlib.sha256(runner.encode(definition).encode()).hexdigest())
        _, row = self.claim()
        self.assertEqual(runner.task_label(row), '000-new-kind')
        self.assertEqual(row['command'], 'bash tools/runner.sh --task 000 --once')
        receipt_path = self.status_dir / '000-new-kind' / 'v1.archived.json'
        runner.write_json(receipt_path, {'not': 'a runtime status'})
        self.assertEqual(set(self.store.read()), {('000', 1)})

    def test_package_name_and_metadata_identify_task_and_version(self):
        output, receipt = self.completed_fixture()
        package = runner.archive_path(self.results, receipt)
        self.assertEqual(package, self.results / '000-new-kind/000-new-kind.v1.tar.gz')
        self.assertTrue(package.is_file())
        self.assertFalse(output.with_suffix('.tar.gz').exists())
        with tarfile.open(package) as archive:
            record = json.loads(archive.extractfile('result.json').read())
        self.assertEqual(record['archive_name'], '000-new-kind.v1.tar.gz')
        self.assertEqual(self.store.read()['000', 1]['archive_name'], record['archive_name'])
        with self.assertRaisesRegex(runner.RunnerError, 'archive name'):
            runner.validate_receipt({**receipt, 'archive_name': 'different.v1.tar.gz'})

    def test_legacy_package_is_renamed_without_recompressing(self):
        output, receipt = self.completed_fixture(legacy=True)
        canonical = runner.archive_path(self.results, receipt)
        self.assertTrue(canonical.is_file())  # Even a legacy raw directory gets a canonical package.
        legacy_package = output.with_suffix('.tar.gz')
        with tarfile.open(canonical) as archive:
            contents = [(member, archive.extractfile(member).read()) for member in archive.getmembers()]
        import io
        with tarfile.open(legacy_package, 'w:gz') as archive:
            for member, data in contents:
                if member.name == 'result.json':
                    record = json.loads(data)
                    record.pop('archive_name', None)
                    data = json.dumps(record).encode()
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        canonical.unlink()
        original = legacy_package.read_bytes()
        receipt['archive_sha256'] = runner.sha256_file(legacy_package)
        archive_dir = self.checkout / 'results-archive'
        self.store.archive_receipts([receipt], self.results, archive_dir)
        destination = runner.archive_path(archive_dir, receipt)
        self.assertEqual(destination.read_bytes(), original)
        with tarfile.open(destination) as archive:
            self.assertNotIn('archive_name', json.loads(archive.extractfile('result.json').read()))
        self.assertFalse(legacy_package.exists())

    def test_identical_package_aliases_merge_into_canonical_archive(self):
        output, receipt = self.completed_fixture()
        canonical = runner.archive_path(self.results, receipt)
        original = canonical.read_bytes()
        legacy = output.with_suffix('.tar.gz')
        shutil.copyfile(canonical, legacy)
        archive_dir = self.checkout / 'results-archive'
        archived_legacy = runner.run_path(archive_dir, receipt).with_suffix('.tar.gz')
        archived_legacy.parent.mkdir(parents=True)
        shutil.copyfile(canonical, archived_legacy)
        self.store.archive_receipts([receipt], self.results, archive_dir)
        destination = runner.archive_path(archive_dir, receipt)
        self.assertEqual(destination.read_bytes(), original)
        self.assertFalse(canonical.exists())
        self.assertFalse(legacy.exists())
        self.assertFalse(archived_legacy.exists())
        self.assertEqual(self.store.read()['000', 1]['archive_name'], destination.name)
        self.store.archive_receipts([receipt], self.results, archive_dir)
        self.assertEqual(destination.read_bytes(), original)
        destination.rename(archived_legacy)
        self.store.archive_receipts([receipt], self.results, archive_dir)
        self.assertEqual(destination.read_bytes(), original)
        self.assertFalse(archived_legacy.exists())

    def test_conflicting_package_alias_preserves_pending_files(self):
        output, receipt = self.completed_fixture()
        canonical = runner.archive_path(self.results, receipt)
        original = canonical.read_bytes()
        legacy = output.with_suffix('.tar.gz')
        legacy.write_bytes(b'other experiment')
        errors = self.store.archive_receipts([receipt], self.results, self.checkout / 'results-archive')
        self.assertEqual(len(errors), 1)
        self.assertIn('checksum', errors[0]['reason'])
        self.assertEqual(canonical.read_bytes(), original)
        self.assertEqual(legacy.read_bytes(), b'other experiment')
        self.assertTrue(output.is_dir())

    def test_nonregular_package_destination_preserves_pending_files(self):
        output, receipt = self.completed_fixture()
        package = runner.archive_path(self.results, receipt)
        original = package.read_bytes()
        archive_dir = self.checkout / 'results-archive'
        destination = runner.archive_path(archive_dir, receipt)
        destination.mkdir(parents=True)
        for symlink in (False, True):
            if symlink:
                destination.rmdir()
                destination.symlink_to(package)
            errors = self.store.archive_receipts([receipt], self.results, archive_dir)
            self.assertEqual(len(errors), 1)
            self.assertIn('not a regular file', errors[0]['reason'])
            self.assertTrue(output.is_dir())
            self.assertEqual(package.read_bytes(), original)

    def test_missing_package_records_error_and_flat_rename_recovers(self):
        output, receipt = self.completed_fixture()
        package = runner.archive_path(self.results, receipt)
        original = package.read_bytes()
        package.unlink()
        archive_dir = self.checkout / 'results-archive'
        errors = self.store.archive_receipts([receipt], self.results, archive_dir)
        self.assertEqual(len(errors), 1)
        self.assertIn('archival package is missing', errors[0]['reason'])
        error_path = self.status_dir / '000-new-kind/v1.archive-error.json'
        saved = runner.read_json(error_path)
        self.assertEqual(saved['execution_state'], 'succeeded')
        self.assertEqual(saved['archive_sha256'], receipt['archive_sha256'])
        self.assertEqual(saved['controller_sha256'], runner.CONTROLLER_SHA256)
        self.assertIn(str(self.results / runner.archive_name(receipt)), saved['expected_package_paths'])
        self.assertTrue(output.is_dir())
        self.assertTrue(self.store.reservation('000', 1).is_dir())
        self.assertEqual(self.store.read()['000', 1]['delivery_state'], 'awaiting_download')
        flat = self.results / runner.archive_name(receipt)
        flat.write_bytes(original)
        self.assertEqual(self.store.archive_receipts([receipt], self.results, archive_dir), [])
        self.assertEqual(runner.archive_path(archive_dir, receipt).read_bytes(), original)
        self.assertFalse(flat.exists())
        self.assertFalse(error_path.exists())
        self.assertEqual(self.store.read()['000', 1]['delivery_state'], 'archived')

    def test_interim_hyphen_names_archive_without_recompression(self):
        output, receipt = self.completed_fixture()
        package = runner.archive_path(self.results, receipt)
        original = package.read_bytes()
        name = '000-new-kind-v1.tar.gz'
        nested = output.parent / name
        package.rename(nested)
        archive_dir = self.checkout / 'results-archive'
        self.assertEqual(self.store.archive_receipts([receipt], self.results, archive_dir), [])
        destination = runner.archive_path(archive_dir, receipt)
        self.assertEqual(destination.read_bytes(), original)
        for alias in (self.results / name, self.results / '000' / name):
            alias.parent.mkdir(parents=True, exist_ok=True)
            alias.write_bytes(original)
            self.assertEqual(self.store.archive_receipts([receipt], self.results, archive_dir), [])
            self.assertFalse(alias.exists())
            self.assertEqual(destination.read_bytes(), original)

    def test_bad_receipt_does_not_block_following_receipt(self):
        first, first_receipt = self.completed_fixture()
        runner.archive_path(self.results, first_receipt).unlink()
        self.commit_task(task_id='001', marker='second complete task')
        second, second_receipt = self.completed_fixture(task_id='001')
        archive_dir = self.checkout / 'results-archive'
        errors = self.store.archive_receipts([first_receipt, second_receipt], self.results, archive_dir)
        self.assertEqual([row['id'] for row in errors], ['000'])
        self.assertTrue(first.is_dir())
        self.assertFalse(second.exists())
        self.assertTrue(runner.archive_path(archive_dir, second_receipt).is_file())
        self.assertEqual(self.store.read()['001', 1]['delivery_state'], 'archived')

    def test_missing_archival_package_does_not_block_new_task_cli(self):
        output, receipt = self.completed_fixture()
        runner.archive_path(self.results, receipt).unlink()
        receipt_path = self.author / 'status/000-new-kind/v1.archived.json'
        receipt_path.parent.mkdir(parents=True)
        runner.write_json(receipt_path, receipt)
        self.commit_task(task_id='001', marker='continue after archive error')
        result = subprocess.run([sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'),
                                 '--once'], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('archive pending for 000-new-kind v1', result.stderr)
        self.assertNotIn('poll failed:', result.stderr)
        self.assertIn('controller_sha256=', result.stdout)
        self.assertIn('running task 001 v1', result.stdout)
        self.assertEqual(self.store.read()['001', 1]['state'], 'succeeded')
        self.assertTrue(output.is_dir())
        self.assertTrue((self.status_dir / '000-new-kind/v1.archive-error.json').is_file())

    def test_legacy_running_status_finishes_without_rerun(self):
        output, receipt = self.completed_fixture(legacy=True)
        self.assertEqual(output.parent.name, '000')
        self.store.reconcile(self.results)
        self.assertEqual(self.store.read()['000', 1]['state'], 'succeeded')
        source, task, digest = self.source_task()
        self.assertIsNone(self.store.claim(task, digest, source, self.results, 'again'))
        archive = self.checkout / 'results-archive'
        self.store.archive_receipts([receipt], self.results, archive)
        self.assertTrue((archive / '000-new-kind' / '000-new-kind.v1.tar.gz').is_file())
        self.assertFalse((self.results / '000').exists())

    def test_receipt_moves_and_preserves_execution_and_newer_task(self):
        output, receipt = self.completed_fixture()
        self.commit_task(version=2)
        _, second = self.claim()
        old_guard = runner.read_json(self.store.guard('000') / 'owner.json')
        archive = self.checkout / 'results-archive'
        self.store.archive_receipts([receipt], self.results, archive)
        self.assertTrue((archive / '000-new-kind/v1/raw.log').is_file())
        self.assertTrue((archive / '000-new-kind/000-new-kind.v1.tar.gz').is_file())
        self.assertFalse(output.exists())
        self.assertTrue(self.store.reservation('000', 1).is_dir())
        self.assertEqual(runner.read_json(self.store.guard('000') / 'owner.json')['claim'], old_guard['claim'])
        status = self.store.read()['000', 1]
        self.assertEqual(status['state'], 'succeeded')
        self.assertEqual(status['delivery_state'], 'archived')
        # Reconciliation of an old terminal record cannot undo delivery state.
        raw_record = runner.read_json(archive / '000-new-kind/v1/result.json')
        self.store.finish(raw_record)
        self.assertEqual(self.store.read()['000', 1]['delivery_state'], 'archived')
        self.store.archive_receipts([receipt], self.results, archive)
        self.assertEqual(self.store.read()['000', 2]['claim'], second['claim'])

    def test_receipt_checksum_and_destination_collision_preserve_data(self):
        output, receipt = self.completed_fixture()
        archive = self.checkout / 'results-archive'
        errors = self.store.archive_receipts([{**receipt, 'archive_sha256': '0' * 64}], self.results, archive)
        self.assertEqual(len(errors), 1)
        self.assertIn('checksum', errors[0]['reason'])
        self.assertTrue(output.exists())
        target = runner.run_path(archive, receipt)
        target.mkdir(parents=True)
        (target / 'unrelated.log').write_text('retain')
        errors = self.store.archive_receipts([receipt], self.results, archive)
        self.assertEqual(len(errors), 1)
        self.assertIn('both pending and archived', errors[0]['reason'])
        self.assertTrue((output / 'raw.log').is_file())
        self.assertEqual((target / 'unrelated.log').read_text(), 'retain')
        self.assertFalse(list(self.lock_dir.glob('archive-*.lock')))

    def test_partial_archive_move_replays_safely(self):
        output, receipt = self.completed_fixture()
        archive = self.checkout / 'results-archive'
        package = runner.archive_path(self.results, receipt)
        real_rename = Path.rename
        def fail_package(path, target):
            if path == package:
                raise OSError('simulated move interruption')
            return real_rename(path, target)
        with patch.object(Path, 'rename', fail_package):
            errors = self.store.archive_receipts([receipt], self.results, archive)
            self.assertEqual(len(errors), 1)
            self.assertIn('move interruption', errors[0]['reason'])
        self.assertFalse(output.exists())
        self.assertTrue(package.is_file())
        self.assertEqual(self.store.read()['000', 1]['delivery_state'], 'awaiting_download')
        self.store.archive_receipts([receipt], self.results, archive)
        self.assertFalse(package.exists())
        self.assertFalse(output.parent.exists())
        self.assertEqual(self.store.read()['000', 1]['delivery_state'], 'archived')

    def test_published_receipt_is_read_from_fetched_source(self):
        output, receipt = self.completed_fixture()
        target = self.author / 'status' / runner.task_label(receipt)
        target.mkdir(parents=True)
        runner.write_json(target / 'v1.archived.json', receipt)
        self.git(self.author, 'add', '.')
        self.git(self.author, 'commit', '-m', 'confirm local archival')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        source = self.queue.source()
        self.assertEqual(self.queue.archived_receipts(source), [receipt])
        self.store.archive_receipts(self.queue.archived_receipts(source), self.results,
                                    self.checkout / 'results-archive')
        self.assertFalse(output.exists())

    def test_labeled_receipt_archives_the_matching_claim(self):
        stamp = '2026-11-18T13:29:35Z'
        parent = 'days-ago-000-topology'
        child = {'status': stamp, 'kind': 'new_kind', 'params': {}}
        self.publish_catalog({'id': parent, 'name': 'Fixture', 'status': stamp, 'subtasks': {'00': child}})
        task_id = parent + '-00'
        output, receipt = self.completed_fixture(task_id=task_id)
        package = runner.archive_path(self.results, receipt)
        self.assertEqual(output.name, 'v1-261118-132935')
        self.assertEqual(receipt['version'], 1795008575)
        target = self.author / 'status' / task_id
        target.mkdir(parents=True)
        runner.write_json(target / 'v1-261118-132935.archived.json', receipt)
        self.git(self.author, 'add', 'status')
        self.git(self.author, 'commit', '-m', 'Confirm labeled result')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        receipts = self.queue.archived_receipts(self.queue.source())
        self.assertEqual(receipts, [receipt])

        archive = self.checkout / 'results-archive'
        wrong = {**receipt, 'version_label': 'v2-261118-132935'}
        errors = self.store.archive_receipts([wrong], self.results, archive)
        self.assertIn('does not match execution', errors[0]['reason'])
        self.assertTrue(package.is_file())
        self.assertEqual(self.store.archive_receipts(receipts, self.results, archive), [])
        self.assertFalse(output.exists())
        self.assertEqual(runner.sha256_file(runner.archive_path(archive, receipt)), receipt['archive_sha256'])

    def test_two_processes_claim_same_version_once(self):
        source, task, digest = self.source_task()
        barrier = self.root / 'start'
        script = (
            'import json,sys,time\nfrom pathlib import Path\n'
            f'sys.path.insert(0,{str(REPO / "tools")!r})\n'
            'import run_experiments as r\n'
            f'while not Path({str(barrier)!r}).exists(): time.sleep(.01)\n'
            f's=r.ClaimStore(Path({str(self.lock_dir)!r}),Path({str(self.status_dir)!r}))\n'
            f'row=s.claim({task!r},{digest!r},{source!r},Path({str(self.results)!r}),sys.argv[1])\n'
            'print(json.dumps(row))\n'
        )
        processes = [subprocess.Popen([sys.executable, '-B', '-c', script, worker],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                     for worker in ('host-a', 'host-b')]
        barrier.touch()
        rows = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, stderr)
            rows.append(json.loads(stdout))
        self.assertEqual(sum(row is not None for row in rows), 1)
        winner = next(row for row in rows if row)
        self.assertEqual(self.store.read()['000', 1]['claim'], winner['claim'])
        self.assertTrue(self.store.guard('000').is_dir())

    def test_cross_version_guard_and_other_task_progress(self):
        _, first = self.claim()
        self.commit_task(version=2)
        source, task, digest = self.source_task()
        self.assertIsNone(self.store.claim(task, digest, source, self.results, 'other-host'))
        self.commit_task(task_id='001', marker='independent')
        _, independent = self.claim('001')
        self.assertIsNotNone(independent)
        self.store.finish({**first, 'state': 'succeeded', 'finished_at': 'now', 'exit_code': 0})
        self.assertTrue(self.store.reservation('000', 1).is_dir())
        self.assertFalse(self.store.guard('000').exists())
        _, second = self.claim()
        self.assertEqual(second['version'], 2)
        # Replaying old recovery must not remove the newer guard.
        self.store.finish({**first, 'state': 'succeeded', 'finished_at': 'now', 'exit_code': 0})
        self.assertEqual(runner.read_json(self.store.guard('000') / 'owner.json')['claim'], second['claim'])

    def test_migrated_child_keeps_old_guard_and_sibling_can_run_independently(self):
        source, original, digest = self.source_task()
        row = self.store.claim(original, digest, source, self.results, 'old-runner')
        parent = 'system-000-init-environment'
        newer = {**original, 'id': parent + '-00', 'legacy_id': '000', 'version': 1794960000,
                 'parent_id': parent, 'subtask_id': '00', 'task_status': '2026-11-17T00:00:00Z',
                 'subtask_status': '2026-11-18T00:00:00Z'}
        sibling = {**original, 'id': parent + '-01', 'version': newer['version'],
                   'parent_id': parent, 'subtask_id': '01', 'task_status': newer['task_status'],
                   'subtask_status': newer['subtask_status']}
        def claim(task):
            return self.store.claim(task, 'f' * 64, source, self.results, 'worker')
        self.assertIsNone(claim(newer))
        other = claim(sibling)
        self.assertIsNotNone(other)
        self.store.finish({**row, 'state': 'failed'})
        reissue = claim(newer)
        self.assertTrue(self.store.guard('000').is_dir())
        self.assertEqual(self.store.read(newer['id']), {(newer['id'], newer['version']): reissue})
        self.assertGreaterEqual(len(self.store.read()), 3)
        self.store.finish({**reissue, 'state': 'succeeded'})
        self.assertFalse(self.store.guard('000').exists())
        self.assertTrue(self.store.guard(sibling['id']).exists())
        self.assertIsNone(claim(newer))
        self.store.finish({**other, 'state': 'succeeded'})
        (self.status_dir / parent).mkdir()
        (self.status_dir / parent / 'ignored.json').touch()
        self.assertIsNone(claim({**newer, 'version': newer['version'] + 1,
                                 'subtask_status': '2026-11-18T00:00:01Z'}))

    def test_terminal_reservation_blocks_stale_status_listing(self):
        source, task, digest = self.source_task()
        row = self.store.claim(task, digest, source, self.results, 'host')
        self.store.finish({**row, 'state': 'failed', 'finished_at': 'now', 'exit_code': 1})
        self.store.path('000', 1).unlink()
        self.assertIsNone(self.store.claim(task, digest, source, self.results, 'other-host'))
        self.assertFalse(self.store.guard('000').exists())

    def test_pinned_archive_and_original_host_recovery(self):
        self.commit_task(marker='pinned')
        task, row = self.claim()
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / 'run.json', {**row, 'task': task})
        self.commit_task(version=2, marker='later')
        self.assertEqual(runner.execute(self.queue, output), 0)
        self.assertEqual((output / 'summary.txt').read_text(), 'pinned')
        self.assertTrue(self.store.guard('000').is_dir())
        with patch.object(runner, 'host_identity', return_value='another-host'):
            self.store.reconcile(self.results)
        self.assertEqual(self.store.read()['000', 1]['state'], 'running')
        self.assertTrue(self.store.guard('000').exists())
        self.store.reconcile(self.results)
        self.assertEqual(self.store.read()['000', 1]['state'], 'succeeded')
        self.assertFalse(self.store.guard('000').exists())
        with tarfile.open(runner.archive_path(self.results, row)) as archive:
            self.assertEqual(archive.getnames(), ['result.json', 'summary.txt'])
            record = json.loads(archive.extractfile('result.json').read())
            self.assertEqual(record['source_commit'], row['source_commit'])
            self.assertIn('pinned_handler_sha256', record)

    def test_claim_and_git_recovery_require_original_host(self):
        _, row = self.claim()
        with patch.object(runner, 'host_identity', return_value='foreign-host'):
            with self.assertRaisesRegex(runner.RunnerError, 'original execution host'):
                self.store.abandon(self.results, '000', 1)
        self.store.abandon(self.results, '000', 1)
        self.assertFalse(self.store.guard('000').exists())
        self.assertEqual(self.store.read()['000', 1]['state'], 'abandoned')
        self.assertTrue(self.store.reservation('000', 1).is_dir())
        git_lock = self.lock_dir / 'git.lock'
        git_lock.mkdir()
        owner = {'claim': 'git-owner', 'host_id': runner.host_identity()}
        runner.write_json(git_lock / 'owner.json', owner)
        with patch.object(runner, 'host_identity', return_value='foreign-host'):
            with self.assertRaisesRegex(runner.RunnerError, 'original host'):
                self.queue.recover_git_lock()
        self.queue.recover_git_lock()
        self.assertFalse(git_lock.exists())

    def test_partial_claim_and_ownerless_lock_recovery(self):
        _, row = self.claim()
        # Crash after task guard ownership, before version reservation metadata.
        (self.store.reservation('000', 1) / 'owner.json').unlink()
        self.store.reservation('000', 1).rmdir()
        self.store.path('000', 1).unlink()
        self.store.abandon(self.results, '000', 1)
        self.assertEqual(self.store.read()['000', 1]['state'], 'abandoned')
        self.store.guard('001').mkdir()
        with self.assertRaisesRegex(runner.RunnerError, 'ownerless'):
            self.store.abandon(self.results, '001', 1)
        self.assertTrue(self.store.guard('001').exists())

    def test_manifest_schema_disabled_resources_and_digest(self):
        self.commit_task(enabled=False)
        self.commit_task(task_id='001', resources={'gpus': 4}, marker='four GPUs')
        self.commit_task(task_id='002', resources={'gpus': 1}, marker='one GPU')
        source = self.queue.source()
        def fake_worker(args, **kwargs):
            return object() if '--_execute' in args else REAL_POPEN(args, **kwargs)
        with patch.object(runner.subprocess, 'Popen', side_effect=fake_worker) as popen:
            active = runner.start_task(self.queue, source, self.store, self.results, 'host', 0, 1)
        self.assertEqual(active[1], self.results / '002-new-kind/v1')
        self.assertEqual(sum('--_execute' in call.args[0] for call in popen.call_args_list), 1)
        self.commit_task(task_id='002', resources={'gpus': 1}, name='changed without version', marker='changed')
        with self.assertRaisesRegex(runner.RunnerError, 'changed without increasing version'):
            # Guard is still held, so test the same-version digest after completion.
            row = self.store.read()['002', 1]
            self.store.finish({**row, 'state': 'succeeded'})
            self.claim('002')
        invalid = self.author / 'tasks/003.json'
        invalid.write_text(json.dumps({'id': '3', 'name': 'invalid', 'version': 1, 'kind': 'x', 'params': {}}))
        self.git(self.author, 'add', '.')
        self.git(self.author, 'commit', '-m', 'invalid id')
        self.git(self.remote, 'fetch', str(self.author), 'main:main')
        with self.assertRaisesRegex(runner.RunnerError, 'invalid task manifest'):
            self.queue.tasks(self.queue.source())

    def test_source_fetch_and_export_share_git_mutex(self):
        source, _, _ = self.source_task()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            fetch = pool.submit(self.queue.source)
            exported = self.root / 'exported'
            export = pool.submit(self.queue.export, source, exported)
            self.assertEqual(fetch.result(), source)
            export.result()
        self.assertTrue((exported / 'tools/experiment_handlers.py').is_file())
        self.assertFalse((self.lock_dir / 'git.lock').exists())
        self.assertFalse((self.checkout / '.git/index.lock').exists())
        self.assertFalse((self.checkout / '.git/worktrees').exists())

    def test_git_timeout_stops_descendants_before_releasing_mutex(self):
        bindir = self.root / 'timed-git'
        bindir.mkdir()
        child_started = self.root / 'child-started'
        orphan_write = self.root / 'orphan-wrote-after-unlock'
        driver = bindir / 'git'
        child_code = f"import time;from pathlib import Path;time.sleep(2);Path({str(orphan_write)!r}).touch()"
        driver.write_text(f'#!{sys.executable}\nimport subprocess,sys,time\nfrom pathlib import Path\n'
                          f'child=subprocess.Popen([sys.executable,"-c",{child_code!r}])\n'
                          f'Path({str(child_started)!r}).touch()\ntime.sleep(30)\n')
        driver.chmod(0o755)
        def ready_process(*args, **kwargs):
            process = REAL_POPEN(*args, **kwargs)
            deadline = time.monotonic() + 5
            while not child_started.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                time.sleep(0.01)
            return process
        with patch.dict(os.environ, {'PATH': str(bindir) + os.pathsep + os.environ['PATH']}):
            with patch.object(runner.subprocess, 'Popen', side_effect=ready_process):
                with self.assertRaisesRegex(runner.RunnerError, 'timed out'):
                    with self.queue.operation():
                        self.queue.git('fetch', timeout=0.1)
        self.assertTrue(child_started.exists())
        self.assertFalse((self.lock_dir / 'git.lock').exists())
        time.sleep(2.1)
        self.assertFalse(orphan_write.exists())

    def test_cli_read_only_once_resume_and_new_version(self):
        guard_dir = self.root / 'bin'
        guard_dir.mkdir()
        log = self.root / 'git-calls.jsonl'
        guard = guard_dir / 'git'
        guard.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                         f'with open({str(log)!r}, "a") as stream: stream.write(json.dumps(sys.argv[1:])+"\\n")\n'
                         'if any(word in {"push","send-pack","credential"} for word in sys.argv[1:]): sys.exit(98)\n'
                         f'os.execv({self.real_git!r}, [{self.real_git!r}, *sys.argv[1:]])\n')
        guard.chmod(0o755)
        env = {**os.environ, 'PATH': str(guard_dir) + os.pathsep + os.environ['PATH'],
               'PYTHONDONTWRITEBYTECODE': '1', 'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_CONFIG_SYSTEM': '/dev/null'}
        args = [sys.executable, '-B', str(self.checkout / 'tools/run_experiments.py'), '--once']
        for version, executes in ((1, True), (1, False), (2, True)):
            if version == 2:
                self.commit_task(version=2, marker='v2')
            process = subprocess.run(args, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertEqual(self.store.read()['000', version]['state'], 'succeeded')
            lines = [line for line in process.stdout.splitlines() if line.startswith('complete ')]
            expected = [f'complete 000 v{version} state=succeeded archive_name=000-new-kind.v{version}.tar.gz']
            self.assertEqual(lines, expected if executes else [])
        self.assertEqual((self.results / '000-new-kind/v2/summary.txt').read_text(), 'v2')
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(any('fetch' in call for call in calls))
        self.assertFalse(any('push' in call or 'credential' in call for call in calls))
        self.assertEqual(self.git(self.remote, 'for-each-ref', '--format=%(refname)'), 'refs/heads/main')

    def test_launcher_swaps_controller_after_active_task_finishes(self):
        checkout_tools = self.checkout / 'tools'
        author_tools = self.author / 'tools'
        for name in ('runner.sh', 'runner_loop.py'):
            shutil.copyfile(REPO / 'tools' / name, author_tools / name)
        (author_tools / 'setup_python.py').write_text('import sys\nprint(sys.executable)\n')
        controller = author_tools / 'run_experiments.py'
        initial_code = controller.read_text().replace('POLL_SECONDS = 10\n', 'POLL_SECONDS = 0.05\n')
        controller.write_text(initial_code)
        release = self.root / 'finish-v1'
        handler = author_tools / 'experiment_handlers.py'
        handler.write_text(
            'from pathlib import Path\n'
            'import time\n'
            'def prepare(kind, params):\n'
            '    def run(source, output):\n'
            '        if params["marker"] == "v1":\n'
            '            (output / "ready.txt").touch()\n'
            '            deadline = time.monotonic() + 10\n'
            f'            while not Path({str(release)!r}).is_file():\n'
            '                if time.monotonic() >= deadline: raise RuntimeError("fixture not released")\n'
            '                time.sleep(0.01)\n'
            '        (output / "summary.txt").write_text(params["marker"])\n'
            '        (output / "controller-proof.fixture").write_text("new archive format")\n'
            '        if params["marker"] == "failed": raise RuntimeError("fixture workload failed")\n'
            '    return run\n',
        )
        task_file = self.author / 'tasks' / '000.json'
        task = runner.read_json(task_file)

        def publish(version, marker):
            runner.write_json(task_file, {**task, 'version': version, 'params': {'marker': marker}})
            self.git(self.author, 'add', '.')
            self.git(self.author, 'commit', '-m', f'Publish controller fixture v{version}')
            self.git(self.remote, 'fetch', str(self.author), 'main:main')
            return self.git(self.author, 'rev-parse', 'HEAD')

        first_commit = publish(1, 'v1')
        self.git(self.checkout, 'pull', '--ff-only')
        initial_sha = runner.sha256_file(controller)
        args = ['bash', str(checkout_tools / 'runner.sh'), '--task', '000', '--worker-id', 'fixture node']
        env = {**os.environ, 'GPU_NET_SIM_PYTHON': sys.executable}
        log = self.root / 'launcher.log'
        with log.open('w') as stream:
            launcher = subprocess.Popen(args, cwd=self.root, env=env, stdout=stream, stderr=subprocess.STDOUT)

        def wait_for(predicate):
            deadline = time.monotonic() + 10
            while not predicate():
                if launcher.poll() is not None or time.monotonic() >= deadline:
                    self.fail(log.read_text())
                time.sleep(0.02)

        task_results = self.results / '000-new-kind'
        first_output = task_results / 'v1'
        try:
            wait_for(lambda: (first_output / 'ready.txt').exists())
            first_run = runner.read_json(first_output / 'run.json')
            self.assertEqual(first_run['runner_sha256'], initial_sha)
            self.assertEqual(first_run['source_commit'], first_commit)
            self.assertEqual(first_run['runner_controller_commit'], first_commit)
            with self.assertRaisesRegex(runner.RunnerError, 'already locked'):
                with runner.device_lock(runner.LOCK_PATH):
                    self.fail('launcher did not keep the host lock while the worker was active')

            controller.write_text(initial_code.replace('RESULT_SUFFIXES = {', "RESULT_SUFFIXES = {'.fixture', ", 1))
            updated_sha = runner.sha256_file(controller)
            self.assertNotEqual(initial_sha, updated_sha)
            handler.write_text(handler.read_text().replace('new archive format', 'updated worker source'))
            second_commit = publish(2, 'v2')
            wait_for(lambda: self.git(self.checkout, 'rev-parse', 'refs/gpu-network-simulator/source') == second_commit)
            self.assertFalse((first_output / 'result.json').exists())
            self.assertFalse((task_results / 'v2').exists())
            release.touch()
            wait_for(lambda: 'complete 000 v2 state=succeeded' in log.read_text())

            first = runner.read_json(first_output / 'result.json')
            second = runner.read_json(task_results / 'v2' / 'result.json')
            self.assertEqual(first['runner_sha256'], initial_sha)
            self.assertNotIn('controller-proof.fixture', first['archive_files'])
            self.assertEqual(second['runner_sha256'], updated_sha)
            self.assertEqual(second['source_commit'], second_commit)
            self.assertEqual(second['runner_controller_commit'], second_commit)
            self.assertEqual(second['runner_controller_tree'], self.git(self.author, 'rev-parse', f'{second_commit}:tools'))
            with tarfile.open(runner.archive_path(self.results, second)) as archive:
                self.assertEqual(archive.extractfile('controller-proof.fixture').read(), b'updated worker source')

            third_commit = publish(3, 'task-only publication')
            wait_for(lambda: 'complete 000 v3 state=succeeded' in log.read_text())
            third = runner.read_json(task_results / 'v3' / 'result.json')
            self.assertEqual(third['source_commit'], third_commit)
            self.assertEqual(third['runner_sha256'], updated_sha)
            self.assertEqual(third['runner_controller_commit'], second_commit)
            transcript = log.read_text()
            updated_startups = [line for line in transcript.splitlines()
                                if line.startswith('CPFS worker ') and updated_sha in line]
            self.assertEqual(len(updated_startups), 1, transcript)
            self.assertIn('fixture node', updated_startups[0])
            self.assertLess(transcript.index('complete 000 v1'), transcript.index(updated_startups[0]))
            self.assertEqual(self.git(self.checkout, 'rev-parse', 'HEAD'), first_commit)
            self.assertEqual(runner.sha256_file(checkout_tools / 'run_experiments.py'), initial_sha)
            self.assertEqual(self.git(self.checkout, 'status', '--porcelain', '--untracked-files=no'), '')
        finally:
            release.touch()
            if launcher.poll() is None:
                launcher.terminate()
                try:
                    launcher.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    launcher.kill()
                    launcher.wait()
                    raise
        self.assertEqual(launcher.returncode, 143, log.read_text())

        publish(4, 'failed')
        once = subprocess.run([*args, '--once'], cwd=self.root, env=env, text=True,
                              capture_output=True, timeout=15)
        self.assertEqual(once.returncode, 1, once.stdout + once.stderr)
        self.assertEqual(once.stdout.count('complete 000 v4 state=failed'), 1, once.stdout)
        self.assertEqual(runner.read_json(task_results / 'v4' / 'result.json')['runner_sha256'], updated_sha)

    def test_completion_precedes_network_and_child_crash_retains_claim(self):
        task, row = self.claim()
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / 'result.json', {**row, 'state': 'succeeded', 'exit_code': 0})
        args = SimpleNamespace(once=True, results_dir=self.results, worker_id='host', gpu_count=0)
        with patch.object(self.queue, 'source', side_effect=runner.RunnerError('offline')):
            self.assertEqual(runner.poll(self.queue, self.store, args, 0), 1)
        self.assertEqual(self.store.read()['000', 1]['state'], 'succeeded')
        self.commit_task(version=2)
        _, second = self.claim()
        dead = SimpleNamespace(returncode=9, poll=lambda: 9, wait=lambda timeout: 9)
        args.once = False
        with patch.object(runner, 'start_task', return_value=(dead, self.results / '000-new-kind/v2', second['claim'])) as start:
            with patch.object(self.store, 'read', return_value={}), patch('builtins.print') as printed:
                self.assertEqual(runner.poll(self.queue, self.store, args, 0), 1)
        self.assertFalse(any(call.args[0].startswith('complete ') for call in printed.call_args_list))
        self.assertEqual(start.call_count, 1)
        self.assertTrue(self.store.guard('000').exists())

    def test_complete_reports_missing_package_before_next_fetch(self):
        source, task, digest = self.source_task()
        args = SimpleNamespace(once=True, results_dir=self.results, worker_id='host', gpu_count=0)

        def finish_without_archive(*_):
            row = self.store.claim(task, digest, source, self.results, 'host')
            output = runner.run_path(self.results, row)
            output.mkdir(parents=True)
            runner.write_json(output / 'result.json', {**row, 'state': 'failed', 'exit_code': 1,
                                                        'archive_error': 'disk full'})
            child = SimpleNamespace(returncode=1, poll=lambda: 1, wait=lambda timeout: 1)
            return child, output, row['claim']

        with patch.object(self.queue, 'source', side_effect=[source, runner.RunnerError('offline')]):
            with patch.object(runner, 'start_task', side_effect=finish_without_archive), patch('builtins.print') as printed:
                self.assertEqual(runner.poll(self.queue, self.store, args, 0), 1)
        self.assertEqual([call.args[0] for call in printed.call_args_list],
                         ['complete 000 v1 state=failed archive_name=-', 'poll failed: offline'])
        self.assertEqual(self.store.read()['000', 1]['state'], 'failed')

    def test_gpu_visibility_and_inherited_local_lock(self):
        response = SimpleNamespace(returncode=0, stdout='0, GPU-a\n1, GPU-b\n2, GPU-c\n3, GPU-d\n')
        with patch.object(runner.subprocess, 'run', return_value=response):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1,3'}):
                self.assertEqual(runner.gpu_count(), 2)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
                self.assertEqual(runner.gpu_count(), 0)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-c'}):
                self.assertEqual(runner.gpu_count(), 2)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'MIG-unsupported'}):
                with self.assertRaisesRegex(runner.RunnerError, 'unsupported'):
                    runner.gpu_count()
        lock = self.root / 'host.lock'
        with runner.device_lock(lock) as fd:
            child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'],
                                     stdin=subprocess.PIPE, pass_fds=(fd,))
        try:
            with self.assertRaisesRegex(runner.RunnerError, 'already locked'):
                with runner.device_lock(lock):
                    self.fail('host lock lost while child was still running')
        finally:
            child.communicate(timeout=10)
        with runner.device_lock(lock):
            pass


REAL_POPEN = subprocess.Popen
