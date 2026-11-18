import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import experiment_handlers as handlers
import incremental


class InventoryCasesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = [[str(index), f'GPU-fixture{index}', 'GB200', '10.0', '190000', '580', f'PCI{index}']
                     for index in range(4)]
        self.calls = []
        self.failed_probe = None

    def task(self, version):
        return {'id': '000', 'version': version, 'slug': 'system-info', 'kind': 'gpu_inventory',
                'name': 'GB200 inventory', 'params': {}}

    def record(self, version):
        return {'id': '000', 'version': version, 'slug': 'system-info', 'kind': 'gpu_inventory',
                'name': 'GB200 inventory', 'claim': f'claim-{version}', 'source_commit': 'a' * 40,
                'task_sha256': 'b' * 64, 'host_id': 'fixture-host', 'worker': 'fixture'}

    def fake_run(self, args, stdout, **kwargs):
        self.calls.append(tuple(args))
        if args[0] == 'nvidia-smi' and args[1].startswith('--query-gpu='):
            text = '\n'.join(', '.join(row) for row in self.rows) + '\n'
        elif args[:2] == ['nvidia-smi', 'topo']:
            text = 'GPU0 GPU1 GPU2 GPU3\nNV18 NV18 NV18 NV18\n'
        elif args[:2] == ['nvidia-smi', 'nvlink']:
            text = 'All fixture NVLink links active\n'
        else:
            text = f'{args[0]} fixture version 1\n'
        stdout.write(text.encode())
        return subprocess.CompletedProcess(args, 3 if args[1] == self.failed_probe else 0)

    @contextlib.contextmanager
    def probes(self, missing=None):
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=True))
            stack.enter_context(patch.object(handlers.platform, 'platform', return_value='fixture-linux-aarch64'))
            stack.enter_context(patch.object(handlers.subprocess, 'run', side_effect=self.fake_run))
            stack.enter_context(patch.object(handlers.shutil, 'which', side_effect=lambda name: None if name == missing else f'/fixture/{name}'))
            yield

    def run_without_engine(self, version=2):
        output = self.root / f'v{version}'
        output.mkdir()
        paths = {}
        for case in handlers.case_plan(self.task(version), self.root):
            path = output / 'cases' / case['id']
            path.mkdir(parents=True)
            case['run'](path, {name: paths[name] for name in case['depends_on']})
            paths[case['id']] = path
        return output, paths

    def test_case_probe_artifacts_are_stable_and_journal_remains_at_task_root(self):
        with self.probes(missing='nvcc'):
            first, _ = self.run_without_engine(2)
            second, _ = self.run_without_engine(3)
        first_files = {path.relative_to(first).as_posix(): path.read_bytes()
                       for path in (first / 'cases').rglob('*') if path.is_file()}
        second_files = {path.relative_to(second).as_posix(): path.read_bytes()
                        for path in (second / 'cases').rglob('*') if path.is_file()}
        self.assertEqual(first_files, second_files)
        self.assertFalse(list((first / 'cases').rglob('commands.json')))
        journal = json.loads((first / 'commands.json').read_text())
        self.assertTrue(all(item['log'].startswith('cases/') for item in journal))
        summary = json.loads((first / 'cases/inventory-report/summary.json').read_text())
        self.assertEqual(summary['gpu_count'], 4)
        self.assertEqual(summary['probes']['nvcc'], 'not installed')
        self.assertEqual(summary['probes']['topology'], 'ok')

    def test_failed_runtime_probe_preserves_feedback_for_next_version(self):
        output = self.root / 'failed' / 'cases' / 'topology'
        output.mkdir(parents=True)
        self.failed_probe = 'topo'
        with self.probes():
            with self.assertRaisesRegex(RuntimeError, 'returned 3'):
                handlers._inventory_topology(output, {})
        feedback = json.loads((output / 'probe.json').read_text())
        self.assertEqual(feedback['status'], 'failed')
        journal = json.loads((output.parent.parent / 'commands.json').read_text())
        self.assertEqual(journal[0]['state'], 'failed')
        self.assertEqual(journal[0]['returncode'], 3)

    def test_other_handlers_keep_legacy_full_execution_path(self):
        self.assertIsNone(handlers.case_plan({'kind': 'gedes_smoke'}, self.root))
        self.assertIsNone(handlers.case_plan({'kind': 'days_cuda_smoke'}, self.root))
        self.assertEqual(handlers.prepare('gedes_smoke', {}), handlers.gedes_smoke)
        with self.assertRaises(ValueError):
            handlers.case_plan({**self.task(2), 'params': {'unexpected': True}}, self.root)

    def baseline(self):
        output = self.root / 'results' / '000-system-info' / 'v2'
        output.mkdir(parents=True)
        task, record = self.task(2), self.record(2)
        manifest = incremental.run_cases(task, handlers.case_plan(task, self.root), output, record)
        handlers.finalize_cases(task, output, manifest)
        handlers.write_json(output / 'result.json', {**record, 'state': manifest['state']})
        return output, manifest

    def run_incremental(self, baseline):
        task = self.task(3)
        task['execution'] = {
            'mode': 'incremental',
            'base': {'id': '000', 'version': 2,
                     'artifacts_manifest_sha256': incremental.file_digest(baseline / 'artifacts-manifest.json')},
        }
        snapshot = incremental.prepare_base(task, self.root / 'results', self.root / 'results-archive',
                                            self.root / 'baseline-snapshot')
        output = self.root / 'results' / '000-system-info' / 'v3'
        output.mkdir()
        manifest = incremental.run_cases(task, handlers.case_plan(task, self.root), output, self.record(3), snapshot)
        handlers.finalize_cases(task, output, manifest)
        return output, manifest

    def test_incremental_inventory_refreshes_probes_and_reuses_identical_report(self):
        with self.probes():
            baseline, first = self.baseline()
            initial_call_count = len(self.calls)
            output, manifest = self.run_incremental(baseline)
        self.assertEqual(first['state'], 'succeeded')
        self.assertEqual(manifest['state'], 'succeeded')
        self.assertEqual(manifest['summary']['executed'], 4)
        self.assertEqual(manifest['summary']['reused'], 1)
        self.assertEqual(len(self.calls), initial_call_count * 2)
        report = manifest['cases']['inventory-report']
        self.assertEqual(report['origin'], 'reused')
        self.assertEqual(report['reused_from']['version'], 2)
        self.assertEqual((output / 'cases/inventory-report/summary.json').read_bytes(),
                         (baseline / 'cases/inventory-report/summary.json').read_bytes())
        self.assertEqual(json.loads((output / 'summary.json').read_text())['gpu_count'], 4)
        self.assertTrue((output / 'gpu-list.csv').is_file())

    def test_changed_driver_invalidates_derived_report(self):
        with self.probes():
            baseline, _ = self.baseline()
            self.rows[0][5] = '581'
            _, manifest = self.run_incremental(baseline)
        self.assertEqual(manifest['state'], 'succeeded')
        report = manifest['cases']['inventory-report']
        self.assertEqual(report['origin'], 'executed')
        self.assertEqual(report['reason'], 'implementation, inputs or dependency changed')

    def test_runtime_probe_failure_is_a_failed_case_and_partial_summary(self):
        with self.probes():
            self.failed_probe = 'nvlink'
            output, manifest = self.baseline()
        self.assertEqual(manifest['state'], 'failed')
        self.assertEqual(manifest['cases']['nvlink-check']['state'], 'failed')
        self.assertNotEqual(manifest['cases']['inventory-report']['state'], 'succeeded')
        self.assertEqual(manifest['cases']['toolchain']['state'], 'succeeded')
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['case_states']['nvlink-check'], 'failed')
        self.assertEqual(summary['probes']['nvlink'], 'failed')
