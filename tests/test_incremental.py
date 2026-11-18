import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import incremental as engine


class IncrementalTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.results = self.root / 'results'
        self.archive = self.root / 'results-archive'
        self.calls = []
        self.task = {'id': '000', 'slug': 'system-info', 'name': 'test',
                     'kind': 'fixture', 'version': 2, 'params': {}}

    def record(self, task):
        return {'id': task['id'], 'version': task['version'], 'kind': task['kind'],
                'claim': f"claim-{task['version']}", 'source_commit': str(task['version']) * 40,
                'task_sha256': str(task['version']) * 64}

    def case(self, name, value='stable', dependencies=None, implementation='v1', reuse=True,
             fail=False, mutate_dependency=False):
        def execute(output, parents):
            self.calls.append(name)
            if mutate_dependency:
                (next(iter(parents.values())) / 'value.txt').write_text('mutated')
            if fail:
                (output / 'failure.log').write_text('diagnostic evidence')
                raise RuntimeError('injected case failure')
            (output / 'value.txt').write_text(value)
        return {'id': name, 'depends_on': dependencies or [], 'implementation': implementation,
                'inputs': {'value': value}, 'reuse': reuse, 'run': execute}

    def full(self, cases):
        output = self.results / '000-system-info/v2'
        output.mkdir(parents=True)
        record = self.record(self.task)
        manifest = engine.run_cases(self.task, cases, output, record)
        engine.write_json(output / 'result.json', {**record, 'state': manifest['state']})
        return output, manifest

    def incremental_task(self, output):
        return {**self.task, 'version': 3, 'execution': {
            'mode': 'incremental', 'base': {'id': '000', 'version': 2,
                'artifacts_manifest_sha256': engine.file_digest(output / 'artifacts-manifest.json')},
            'reuse': 'verified-compatible'}}

    def rerun(self, task, cases, snapshot_name='snapshot'):
        base = engine.prepare_base(task, self.results, self.archive, self.root / snapshot_name)
        output = self.results / '000-system-info/v3'
        output.mkdir(parents=True)
        result = engine.run_cases(task, cases, output, self.record(task), base)
        return output, result

    def test_reuses_passed_cases_and_reruns_failed_and_new_cases(self):
        output, old = self.full([self.case('passed'), self.case('failed', fail=True),
                                 self.case('dependent', dependencies=['failed'])])
        self.assertEqual(old['state'], 'failed')
        task = self.incremental_task(output)
        self.calls.clear()
        newer, result = self.rerun(task, [self.case('passed'), self.case('failed'),
                                        self.case('dependent', dependencies=['failed']), self.case('new')])
        self.assertEqual(self.calls, ['failed', 'dependent', 'new'])
        self.assertEqual(result['state'], 'succeeded')
        self.assertEqual(result['cases']['passed']['origin'], 'reused')
        self.assertEqual(result['cases']['passed']['reused_from']['version'], 2)
        self.assertEqual(result['summary']['reused'], 1)
        self.assertEqual(result['summary']['executed'], 3)
        self.assertEqual((output / 'cases/failed/failure.log').read_text(), 'diagnostic evidence')
        self.assertEqual((newer / 'cases/passed/value.txt').read_text(), 'stable')

    def test_changed_implementation_invalidates_dependency_chain(self):
        output, _ = self.full([self.case('producer'), self.case('consumer', dependencies=['producer']),
                               self.case('independent')])
        self.calls.clear()
        _, manifest = self.rerun(self.incremental_task(output), [self.case('producer', implementation='v2'),
                     self.case('consumer', dependencies=['producer']), self.case('independent')])
        self.assertEqual(self.calls, ['producer', 'consumer'])
        self.assertEqual(manifest['cases']['independent']['origin'], 'reused')

    def test_fresh_probe_can_leave_unchanged_derived_case_reusable(self):
        output, _ = self.full([self.case('fresh', reuse=False), self.case('derived', dependencies=['fresh'])])
        self.calls.clear()
        _, manifest = self.rerun(self.incremental_task(output), [self.case('fresh', reuse=False),
                                                     self.case('derived', dependencies=['fresh'])])
        self.assertEqual(self.calls, ['fresh'])
        self.assertEqual(manifest['cases']['derived']['origin'], 'reused')

    def test_archived_base_and_corrupt_artifact_recomputation(self):
        output, _ = self.full([self.case('first'), self.case('second')])
        task = self.incremental_task(output)
        target = self.archive / '000-system-info/v2'
        target.parent.mkdir(parents=True)
        output.rename(target)
        (target / 'cases/first/value.txt').write_text('corrupt')
        self.calls.clear()
        _, manifest = self.rerun(task, [self.case('first'), self.case('second')])
        self.assertEqual(self.calls, ['first'])
        self.assertEqual(manifest['cases']['second']['origin'], 'reused')
        self.assertIn('artifact unavailable', manifest['cases']['first']['reason'])
        self.assertEqual((target / 'cases/first/value.txt').read_text(), 'corrupt')

    def test_missing_or_mismatched_manifest_blocks_preparation(self):
        output, _ = self.full([self.case('first')])
        task = self.incremental_task(output)
        (output / 'artifacts-manifest.json').write_text('{}')
        with self.assertRaisesRegex(engine.BaseUnavailable, 'SHA-256 mismatch'):
            engine.prepare_base(task, self.results, self.archive, self.root / 'snapshot')
        self.assertFalse((self.root / 'snapshot').exists())
        shutil.rmtree(output)
        with self.assertRaisesRegex(engine.BaseUnavailable, 'missing baseline'):
            engine.prepare_base(task, self.results, self.archive, self.root / 'snapshot')

    def test_targets_force_and_checkpoint_failures(self):
        output, _ = self.full([self.case('first'), self.case('second')])
        task = self.incremental_task(output)
        task['execution'].update(targets=['first'], force_rerun=['first'])
        self.calls.clear()
        target, manifest = self.rerun(task, [self.case('first', fail=True), self.case('second')])
        self.assertEqual(self.calls, ['first'])
        self.assertEqual(manifest['state'], 'failed')
        self.assertEqual(set(manifest['cases']), {'first'})
        self.assertEqual(json.loads((target / 'artifacts-manifest.json').read_text())['cases']['first']['state'], 'failed')
        self.assertEqual((target / 'cases/first/failure.log').read_text(), 'diagnostic evidence')

    def test_dependency_mutation_cannot_finish_successfully(self):
        output, manifest = self.full([self.case('producer'),
                                     self.case('consumer', dependencies=['producer'], mutate_dependency=True)])
        self.assertEqual(manifest['state'], 'failed')
        self.assertEqual(manifest['cases']['producer']['state'], 'failed')
        self.assertEqual(manifest['cases']['consumer']['state'], 'failed')
        self.calls.clear()
        _, retried = self.rerun(self.incremental_task(output), [self.case('producer'),
                                        self.case('consumer', dependencies=['producer'])])
        self.assertEqual(self.calls, ['producer', 'consumer'])
        self.assertEqual(retried['state'], 'succeeded')

    def test_cycles_invalid_configs_and_unsafe_artifact_manifest(self):
        invalid = self.root / 'invalid'
        invalid.mkdir()
        with self.assertRaisesRegex(engine.IncrementalError, 'cycle'):
            engine.run_cases(self.task, [self.case('a', dependencies=['b']), self.case('b', dependencies=['a'])],
                             invalid, self.record(self.task))
        output, manifest = self.full([self.case('passed')])
        manifest['cases']['passed']['artifacts'][0]['path'] = 'cases/passed/../../escape'
        engine.write_json(output / 'artifacts-manifest.json', manifest)
        task = self.incremental_task(output)
        with self.assertRaises(engine.BaseUnavailable):
            engine.prepare_base(task, self.results, self.archive, self.root / 'snapshot')
        task['execution']['base']['version'] = 3
        with self.assertRaisesRegex(engine.IncrementalError, 'earlier version'):
            engine.validate_execution(task)

    def test_snapshot_corruption_is_not_reused(self):
        output, _ = self.full([self.case('first')])
        task = self.incremental_task(output)
        base = engine.prepare_base(task, self.results, self.archive, self.root / 'snapshot')
        (Path(base['root']) / 'cases/first/value.txt').write_text('changed after preflight')
        target = self.results / '000-system-info/v3'
        target.mkdir(parents=True)
        self.calls.clear()
        manifest = engine.run_cases(task, [self.case('first')], target, self.record(task), base)
        self.assertEqual(self.calls, ['first'])
        self.assertEqual(manifest['cases']['first']['origin'], 'executed')

    def test_modified_snapshot_manifest_is_rejected(self):
        output, _ = self.full([self.case('first')])
        task = self.incremental_task(output)
        base = engine.prepare_base(task, self.results, self.archive, self.root / 'snapshot')
        base['manifest']['cases']['first']['key'] = '0' * 64
        target = self.results / '000-system-info/v3'
        target.mkdir(parents=True)
        with self.assertRaisesRegex(engine.BaseUnavailable, 'differs from its pinned manifest'):
            engine.run_cases(task, [self.case('first')], target, self.record(task), base)

    def test_finalizer_mutation_invalidates_committed_case_and_descendants(self):
        output, manifest = self.full([self.case('producer'), self.case('consumer', dependencies=['producer'])])
        (output / 'cases/producer/value.txt').write_text('modified by finalizer')
        # Verify a serialized/reloaded map, whose ordering is alphabetic.
        manifest = json.loads(json.dumps(manifest, sort_keys=True))
        checked = engine.verify_outputs(output, manifest)
        self.assertEqual(checked['state'], 'failed')
        self.assertEqual(checked['summary']['failed'], 2)
        self.assertEqual(checked['cases']['consumer']['state'], 'failed')
        self.assertEqual(json.loads((output / 'artifacts-manifest.json').read_text())['state'], 'failed')

    def test_reused_executable_preserves_permissions(self):
        def build(output, dependencies):
            script = output / 'probe.sh'
            script.write_text('#!/bin/sh\nprintf \"reused executable\\n\"\n')
            script.chmod(0o755)
        case = {'id': 'build', 'implementation': 'compiler-v1', 'inputs': {}, 'run': build}
        output, _ = self.full([case])
        newer, manifest = self.rerun(self.incremental_task(output), [case])
        self.assertEqual(manifest['cases']['build']['origin'], 'reused')
        executable = newer / 'cases/build/probe.sh'
        self.assertTrue(executable.stat().st_mode & 0o111)
        completed = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.strip(), 'reused executable')
