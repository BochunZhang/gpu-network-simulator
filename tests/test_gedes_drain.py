import csv
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from test_gedes_experiments import run_line

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gedes_drain as drain
import gedes_experiments as v3


class SnapshotFixture:
    def __init__(self, blocked_queue=False):
        self.blocked_queue = blocked_queue

    def command(self, args, log, **kwargs):
        path = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--output=')))
        k = int(next(arg.split('=', 1)[1] for arg in args if arg.startswith('--ft_k=')))
        hosts = k ** 3 // 4
        early = '--audit_early_stop=1' in args
        log.write_text(run_line(finished=not early, rounds=1 if early else 3,
                                hosts=hosts, channel_blocks=6 * hosts // 32))
        if early:
            log.write_text(log.read_text().replace(f'completed_nodes={hosts}', 'completed_nodes=0'))
        with path.open('w', newline='') as flows, Path(str(path) + '.connections.csv').open('w', newline='') as connections:
            fct = csv.writer(flows)
            tcp = csv.writer(connections)
            fct.writerow(['source_ip', 'dst_ip', 'flow_size', 'start_timestamp', 'completed_timestamp',
                          'flow_completion_time'])
            tcp.writerow(['node', 'source_ip', 'dst_ip', 'generated_bytes', 'generated_start_ns',
                          'device_bytes', 'device_start_ns', 'device_end_ns', 'device_byte_offset',
                          'pending_flows', 'acked_flows', 'planned_bytes', 'acked_bytes', 'snd', 'una',
                          'expected_seq', 'record_count', 'has_records', 'unacked_packets'])
            for node in range(hosts):
                peer = (node + hosts // 2) % hosts
                size = 1460 * (node + 1)
                peer_size = 1460 * (peer + 1)
                end = -1 if early else 11000
                fct.writerow([node, peer, size, 5000, end, end - 5000])
                sent = 0 if early else size
                tcp.writerow([node, node, peer, size, 5000, size, 5000, end, -1 if early else size,
                              int(early), int(not early), sent, sent, sent, sent,
                              0 if early else peer_size, 0, 0, 0])
        with Path(str(path) + '.queues.csv').open('w', newline='') as queues:
            writer = csv.writer(queues)
            writer.writerow(['group', 'index', 'entries'])
            for group, multiple in drain.QUEUE_COUNTS.items():
                for index in range(hosts * multiple):
                    writer.writerow([group, index, int(self.blocked_queue and group == 'tcp_egress' and index == 0)])
        return 0.2


class GeDESDrainTest(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)

    def run_fixture(self, case, k, early=False, blocked=False):
        fixture = SnapshotFixture(blocked)
        reference = self.output / 'reference.csv'
        fixture.command([f'--output={reference}', f'--ft_k={k}'], self.output / 'reference.txt')
        expected = v3.flow_statistics(reference, k ** 3 // 4)
        with patch.dict(drain.REFERENCE_CSV, {k: expected['csv_sha256']}), patch.dict(
                drain.REFERENCE_INPUTS, {k: expected['reported_input_sha256']}):
            return drain._case(Path('GeDES'), case, k, early, self.output, {}, v3, fixture)

    def test_complete_fixture_covers_bidirectional_flows_and_queues(self):
        result = self.run_fixture('k4', 4)
        self.assertTrue(result['passed'], result['failures'])
        self.assertEqual(result['snapshot']['acked_flows'], 16)
        self.assertEqual(result['snapshot']['queues']['channel_egress']['queues'], 96)

    def test_nonempty_queue_fails_even_with_a_true_upstream_finish_flag(self):
        result = self.run_fixture('k4-pending', 4, blocked=True)
        self.assertEqual(result['measurements']['upstream_finished'], 1)
        self.assertEqual(result['upstream_gate_failures'], [])
        self.assertFalse(result['passed'])
        self.assertEqual(result['failures'], ['tcp_egress contains 1 entries'])

    def test_intentional_early_stop_requires_pending_flows(self):
        result = self.run_fixture('k8-early-stop', 8, early=True)
        self.assertTrue(result['passed'], result['failures'])
        self.assertEqual(result['snapshot']['pending_flows'], 128)
        self.assertTrue(result['drain_gate_failures'])


if __name__ == '__main__':
    unittest.main()
