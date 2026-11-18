import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import days_ago_fat_tree48 as days
import experiment_handlers as handlers


class DaysFatTree48Test(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.worker_bin = self.output / 'worker-bin'
        self.cargo_bin = self.output / 'cargo home' / 'bin'
        self.worker_env = {'PATH': str(self.worker_bin), 'CARGO_HOME': str(self.cargo_bin.parent),
                           'RUSTUP_HOME': str(self.output / 'rustup home'), 'CUDA_VISIBLE_DEVICES': 'GPU-test'}
        self.write_rust_tools(self.worker_bin)
        self.install_calls = 0
        self.fail_install = False
        self.rust_commands = {}
        self.smoke_hash = '0123456789abcdef'
        self.wide_sourced = days.EXPECTED_PACKETS
        self.fail_build = False
        self.fixture_bytes = b'mock fixture'
        digest = hashlib.sha256(self.fixture_bytes).hexdigest()
        for patcher in (
            patch.object(handlers, 'selected_gpu', return_value=('10.0', self.worker_env)),
            patch.object(handlers, 'checkout', side_effect=self.checkout),
            patch.object(handlers, 'command', side_effect=self.command),
            patch.object(days, 'SMOKE_FIXTURE_SHA256', digest),
            patch.object(days, 'WIDE_FIXTURE_SHA256', digest),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_rust_tools(self, directory, *, broken=None):
        directory.mkdir(parents=True, exist_ok=True)
        for tool in ('rustc', 'cargo'):
            executable = directory / tool
            executable.write_text(f'#!/bin/sh\nexit {int(broken == tool)}\n')
            executable.chmod(0o755)

    def checkout(self, upstream, scratch, output, env):
        source = scratch / 'source'
        for fixture in (handlers.DAYS_FIXTURE, days.WIDE_FIXTURE):
            path = source / fixture
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.fixture_bytes)
        return source

    def command(self, args, log, cwd=None, env=None):
        if args == ['sh', REPO / 'tools/install_rust.sh']:
            self.install_calls += 1
            if self.fail_install:
                log.write_text('rustup download failed\n')
                raise RuntimeError(f'sh returned 1; see {log.name}')
            self.write_rust_tools(self.cargo_bin)
        elif args[0] in ('rustc', 'cargo'):
            resolved = shutil.which(args[0], path=env['PATH'])
            self.assertIsNotNone(resolved)
            self.assertEqual(env['RUSTUP_HOME'], self.worker_env['RUSTUP_HOME'])
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], self.worker_env['CUDA_VISIBLE_DEVICES'])
            self.rust_commands[tuple(args[:2])] = (Path(resolved).parent, env.get('RUSTUP_TOOLCHAIN'))
        if args[:2] == ['cargo', 'build']:
            if self.fail_build:
                log.write_bytes(b'x' * (20 * 1024) + b'compiler failure\n')
                raise RuntimeError('cargo returned 101')
            binary = Path(env['CARGO_TARGET_DIR']) / 'release/days'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'same binary across engines')
        elif '--engine' in args:
            engine = args[args.index('--engine') + 1]
            sourced = self.wide_sourced if log.name.startswith('fat-tree48') else 100
            digest = self.smoke_hash if log.name == 'smoke-cuda.txt' else '0123456789abcdef'
            lines = []
            if engine == 'cuda':
                lines.append('record=days_device engine=cuda wall_ns=8 device_ns=7 rounds=6 '
                             'transitions=1000 continuation_relaunches=0 retry_count=1 '
                             'grown_stream_count=1 channel_capacity_distribution=2:1 '
                             'retry_trace=[CapacityRetry { old: 1, new: 2 }] '
                             'same_time_continuations=0 round_kernel=plain')
            lines.append(f'record=days_result engine={engine} lowering_ns=2 run_ns=10 '
                         f'result_bytes=200 result_fnv1a64={digest} pending_events=50 resident_packets=10 '
                         f'sourced_packets={sourced} departed_packets=400 received_packets=80 dropped_packets=4')
            log.write_text('\n'.join(lines))
        return 0.25

    def summary(self):
        return json.loads((self.output / 'summary.json').read_text())

    def test_matching_smoke_then_fixed_window_with_timing(self):
        days.run(REPO, self.output, handlers)
        result = self.summary()
        self.assertEqual(result['state'], 'succeeded')
        self.assertTrue(result['smoke']['fingerprints_match'])
        self.assertEqual(result['wide']['expected_sourced_packets'], 269_653_860)
        self.assertEqual(result['wide']['run']['device']['transitions'], 1000)
        self.assertEqual(result['wide']['run']['process_seconds'], 0.25)
        self.assertEqual(result['wide']['run']['run_ns'], 10)
        self.assertEqual(self.install_calls, 0)
        self.assertEqual(self.rust_commands['cargo', 'build'], (self.worker_bin, None))

    def test_existing_cargo_bin_is_used_without_install(self):
        (self.worker_bin / 'cargo').unlink()
        self.write_rust_tools(self.cargo_bin)
        days.run(REPO, self.output, handlers)
        self.assertEqual(self.install_calls, 0)
        self.assertEqual(set(self.rust_commands.values()), {(self.cargo_bin, None)})

    def test_missing_cargo_installs_once_and_uses_the_restored_tools(self):
        (self.worker_bin / 'cargo').unlink()
        days.run(REPO, self.output, handlers)
        self.assertEqual(self.install_calls, 1)
        self.assertEqual(set(self.rust_commands),
                         {('rustc', '--version'), ('cargo', '--version'), ('cargo', 'build')})
        self.assertEqual(set(self.rust_commands.values()), {(self.cargo_bin, 'stable')})
        self.assertEqual(self.worker_env['PATH'], str(self.worker_bin))
        self.assertNotIn('RUSTUP_TOOLCHAIN', self.worker_env)

    def test_unusable_rust_installer_failure_leaves_a_diagnostic(self):
        self.write_rust_tools(self.worker_bin, broken='rustc')
        self.fail_install = True
        with self.assertRaisesRegex(RuntimeError, 'sh returned 1'):
            days.run(REPO, self.output, handlers)
        self.assertEqual(self.install_calls, 1)
        self.assertEqual(self.summary()['stage'], 'rust-toolchain')
        self.assertEqual(self.summary()['state'], 'failed')
        self.assertEqual((self.output / 'rust-install.txt').read_text(), 'rustup download failed\n')
        self.assertEqual(self.rust_commands, {})

    def test_smoke_mismatch_prevents_wide_run(self):
        self.smoke_hash = '0123456789abcdee'
        with self.assertRaisesRegex(RuntimeError, 'smoke fingerprints differ'):
            days.run(REPO, self.output, handlers)
        self.assertEqual(self.summary()['stage'], 'smoke')
        self.assertEqual(self.summary()['state'], 'failed')
        self.assertFalse((self.output / 'fat-tree48-cuda.txt').exists())

    def test_wrong_work_is_retained_as_failure(self):
        self.wide_sourced -= 1
        with self.assertRaisesRegex(RuntimeError, 'fixed-window packet gate failed'):
            days.run(REPO, self.output, handlers)
        result = self.summary()
        self.assertEqual((result['state'], result['stage']), ('failed', 'fat-tree48'))
        self.assertEqual(result['wide']['run']['sourced_packets'], self.wide_sourced)

    def test_build_failure_includes_bounded_archive_diagnostic(self):
        self.fail_build = True
        with self.assertRaisesRegex(RuntimeError, 'cargo returned 101'):
            days.run(REPO, self.output, handlers)
        self.assertEqual(self.summary()['stage'], 'build')
        tail = (self.output / 'build-error.txt').read_bytes()
        self.assertEqual(len(tail), 16 * 1024)
        self.assertTrue(tail.endswith(b'compiler failure\n'))


if __name__ == '__main__':
    unittest.main()
