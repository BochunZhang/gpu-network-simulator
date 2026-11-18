import hashlib
import json
from pathlib import Path
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
        self.smoke_hash = '0123456789abcdef'
        self.wide_sourced = days.EXPECTED_PACKETS
        self.fail_build = False
        self.fixture_bytes = b'mock fixture'
        digest = hashlib.sha256(self.fixture_bytes).hexdigest()
        for patcher in (
            patch.object(handlers, 'selected_gpu', return_value=('10.0', {})),
            patch.object(handlers, 'checkout', side_effect=self.checkout),
            patch.object(handlers, 'command', side_effect=self.command),
            patch.object(days, 'SMOKE_FIXTURE_SHA256', digest),
            patch.object(days, 'WIDE_FIXTURE_SHA256', digest),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def checkout(self, upstream, scratch, output, env):
        source = scratch / 'source'
        for fixture in (handlers.DAYS_FIXTURE, days.WIDE_FIXTURE):
            path = source / fixture
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.fixture_bytes)
        return source

    def command(self, args, log, cwd=None, env=None):
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
