import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

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
        self.profile_hash = '0123456789abcdef'
        self.profile_transitions = 1000
        self.retry_count = 0
        self.capture_seconds = None
        self.wide_commands = {}
        self.wide_sourced = days.EXPECTED_PACKETS
        self.fail_build = False
        for patcher in (
            patch.object(handlers, 'selected_gpu', return_value=('10.0', self.worker_env)),
            patch.object(handlers, 'checkout', side_effect=AssertionError('expected maintained source')),
            patch.object(handlers, 'command', side_effect=self.command),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_rust_tools(self, directory, *, broken=None):
        directory.mkdir(parents=True, exist_ok=True)
        for tool in ('rustc', 'cargo'):
            executable = directory / tool
            executable.write_text(f'#!/bin/sh\nexit {int(broken == tool)}\n')
            executable.chmod(0o755)

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
            self.assertEqual(cwd, REPO / days.SIMULATOR_DIR)
            if self.fail_build:
                log.write_bytes(b'x' * (20 * 1024) + b'compiler failure\n')
                raise RuntimeError('cargo returned 101')
            binary = Path(env['CARGO_TARGET_DIR']) / 'release/days'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'same binary across engines')
        elif '--engine' in args:
            engine = args[args.index('--engine') + 1]
            self.assertEqual(bool(env.get('FT48_WARP_OUTPUT')), log.name == 'fat-tree48-cuda-warp.txt')
            if 'FT48_WARP_OUTPUT' in env:
                directory = Path(env['FT48_WARP_OUTPUT'])
                directory.mkdir(exist_ok=True)
                (directory / 'graph.dot').write_text('digraph captured { a -> b }\n')
            wide = log.name.startswith('fat-tree48')
            profiled = log.name in ('fat-tree48-cuda-nsys.txt', 'fat-tree48-cuda-warp.txt')
            if wide:
                self.wide_commands[log.name] = args
            sourced = self.wide_sourced if wide else 100
            digest = self.profile_hash if profiled else '0123456789abcdef'
            if log.name == 'smoke-cuda.txt':
                digest = self.smoke_hash
            lines = []
            expected_capture = self.capture_seconds if log.name == 'fat-tree48-cuda-nsys.txt' else None
            self.assertEqual(env.get(days.CAPTURE_ENV), expected_capture)
            if expected_capture:
                lines.append('record=days_capture first_wave=0 end_wave=2 first_round=0 end_round=5 '
                             f'wall_ns=10100000000 requested_seconds={expected_capture}')
            if engine == 'cuda':
                device_ns = 9 if profiled else 7
                transitions = self.profile_transitions if profiled else 1000
                lines.append(f'record=days_device engine=cuda wall_ns=10 device_ns={device_ns} rounds=6 '
                             f'transitions={transitions} continuation_relaunches=0 retry_count={self.retry_count} '
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
        self.assertNotIn('profile', result['wide'])
        self.assertEqual(self.install_calls, 0)
        self.assertEqual(self.rust_commands['cargo', 'build'], (self.worker_bin, None))

    def test_profiling_reuses_baseline_binary_and_accepts_different_timing(self):
        profiler = Mock()
        profiler.preflight.return_value = Path('/mock/nsys')
        profile_prefix = [profiler.preflight.return_value, 'profile']
        profiler.prefix.return_value = profile_prefix
        profiler.collect.return_value = {'kernel_count': 72}
        with patch.object(handlers, '_support_module', return_value=profiler) as support:
            days.run(REPO, self.output, handlers, profile=True)
        support.assert_called_once_with(REPO, 'nsys_ft48')
        result = self.summary()
        self.assertEqual(result['state'], 'succeeded')
        profile = result['wide']['profile']
        self.assertEqual(profile['run']['device']['device_ns'], 9)
        self.assertEqual(result['wide']['run']['device']['device_ns'], 7)
        self.assertEqual(profile['work_differences'], [])
        self.assertEqual(profile['trace']['kernel_count'], 72)
        self.assertEqual(self.wide_commands['fat-tree48-cuda-nsys.txt'],
                         [*profile_prefix, *self.wide_commands['fat-tree48-cuda.txt']])
        capture_env = profiler.collect.call_args.args[2]
        self.assertEqual(capture_env['PATH'], self.worker_env['PATH'])
        profiler.collect.assert_called_once_with(profiler.preflight.return_value, self.output,
                                                 capture_env, handlers, simulator='days-ago', rounds=6)

    def test_profile_work_mismatch_keeps_both_runs_and_fails(self):
        profiler = Mock()
        profiler.prefix.return_value = ['nsys', 'profile']
        self.profile_hash = '0123456789abcdee'
        self.profile_transitions = 999
        with patch.object(handlers, '_support_module', return_value=profiler):
            with self.assertRaisesRegex(RuntimeError, 'profiled work differs from baseline'):
                days.run(REPO, self.output, handlers, profile=True)
        result = self.summary()
        self.assertEqual((result['state'], result['stage']), ('failed', 'fat-tree48-nsys'))
        self.assertEqual(result['wide']['profile']['work_differences'],
                         ['result_fnv1a64', 'device.transitions'])
        self.assertEqual(result['wide']['run']['device']['transitions'], 1000)
        self.assertEqual(result['wide']['profile']['run']['device']['transitions'], 999)
        self.assertTrue((self.output / 'fat-tree48-cuda.txt').is_file())
        self.assertTrue((self.output / 'fat-tree48-cuda-nsys.txt').is_file())
        profiler.collect.assert_not_called()

    def test_compute_triage_collects_scoped_waves_before_paired_mismatch(self):
        self.capture_seconds = '10'
        self.worker_env[days.CAPTURE_ENV] = '999'
        self.profile_hash = '0123456789abcdee'
        profiler = Mock()
        profiler.preflight.return_value = 'nsys'
        profiler.prefix.return_value = ['nsys', 'profile']
        profiler.collect.return_value = {'passed': True}
        with patch.object(handlers, '_support_module', return_value=profiler):
            with self.assertRaisesRegex(RuntimeError, 'profiled work differs from baseline'):
                days.run(REPO, self.output, handlers, profile=True, capture_seconds=10, metric_set='gb10x-ct')
        profile = self.summary()['wide']['profile']
        self.assertEqual(profile['work_differences'], ['result_fnv1a64'])
        self.assertEqual(profile['capture_attempt'], 0)
        self.assertTrue(profile['trace']['passed'])
        self.assertEqual(self.worker_env[days.CAPTURE_ENV], '999')
        self.assertEqual(profile['capture_window'], {'first_wave': 0, 'end_wave': 2, 'first_round': 0,
                                                    'end_round': 5, 'wall_ns': 10_100_000_000,
                                                    'requested_seconds': 10})
        self.assertEqual(profiler.preflight.call_args.kwargs, {'metric_set': 'gb10x-ct'})
        profiler.prefix.assert_called_once_with('nsys', self.output, capture_range=True, metric_set='gb10x-ct')
        self.assertEqual(profiler.collect.call_args.kwargs,
                         {'simulator': 'days-ago', 'rounds': 6, 'capture_window': profile['capture_window'],
                          'metric_set': 'gb10x-ct'})
        self.assertNotIn(days.CAPTURE_ENV, profiler.collect.call_args.args[2])

    def test_capture_of_retried_run_keeps_trace_without_accepting_final_counters(self):
        self.capture_seconds = '10'
        self.retry_count = 1
        profiler = Mock()
        profiler.prefix.return_value = ['nsys', 'profile']
        profiler.collect.return_value = {'passed': True}
        with patch.object(handlers, '_support_module', return_value=profiler):
            with self.assertRaisesRegex(RuntimeError, 'retried attempt'):
                days.run(REPO, self.output, handlers, profile=True, capture_seconds=10, metric_set='gb10x-ct')
        self.assertEqual(self.summary()['state'], 'failed')
        self.assertTrue(self.summary()['wide']['profile']['trace']['passed'])
        profiler.collect.assert_called_once()

    def test_capture_marker_must_be_unique_and_match_request(self):
        log = self.output / 'fat-tree48-cuda-nsys.txt'
        record = ('record=days_capture first_wave=0 end_wave=3 first_round=0 end_round=130 '
                  'wall_ns=10000000001 requested_seconds=10\n')
        log.write_text(record * 2)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            days._read_capture_window(log, 10)
        log.write_text(record)
        with self.assertRaisesRegex(ValueError, 'requested duration'):
            days._read_capture_window(log, 8)

    def test_warp_profile_uses_maintained_build_and_compares_work(self):
        warp = Mock()
        warp.prepare.side_effect = lambda source, output, scratch, env, handlers: dict(env)
        warp.capture_environment.side_effect = lambda source, output, env, kind: {
            **env, 'FT48_WARP_OUTPUT': str(output / 'warp')
        }
        warp.collect.return_value = {'warp_instances': 123}
        with patch.object(handlers, '_support_module', return_value=warp) as support:
            days.run(REPO, self.output, handlers, warp_trace=True)
        support.assert_called_once_with(REPO, 'warp_ft48')
        profiled = self.summary()['wide']['profile']
        self.assertEqual(profiled['trace']['warp_instances'], 123)
        self.assertEqual(profiled['work_differences'], [])
        warp.collect.assert_called_once_with(REPO, self.output, 'days', 6)

    def test_profile_preflight_failure_stops_before_the_build(self):
        profiler = Mock()
        profiler.preflight.side_effect = RuntimeError('nsys unavailable')
        with patch.object(handlers, '_support_module', return_value=profiler):
            with self.assertRaisesRegex(RuntimeError, 'nsys unavailable'):
                days.run(REPO, self.output, handlers, profile=True)
        result = self.summary()
        self.assertEqual((result['state'], result['stage']), ('failed', 'nsys-preflight'))
        self.assertNotIn(('cargo', 'build'), self.rust_commands)
        self.assertNotIn('run', result['wide'])

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
