import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from test_gedes_experiments import run_line

import gedes_drain as drain
import gedes_experiments as audit
import gedes_fat_tree48 as study


class FatTree48Test(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.output = Path(scratch.name)

    def test_fct_inventory_requires_the_pinned_half_tree_pairs(self):
        flows = self.output / 'flows.csv'
        with flows.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['source_ip', 'dst_ip', 'flow_size'])
            for node in range(study.HOSTS):
                peer = (node + study.HOSTS // 2) % study.HOSTS
                writer.writerow([study._host_ip(node), study._host_ip(peer), 1460])
        study._validate_flows(flows)

        with flows.open('a', newline='') as stream:
            csv.writer(stream).writerow([study._host_ip(0), study._host_ip(study.HOSTS // 2), 1460])
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            study._validate_flows(flows)

        flows.write_text(f'source_ip,dst_ip,flow_size\n{study._host_ip(0)},{study._host_ip(1)},1460\n')
        with self.assertRaisesRegex(ValueError, 'unexpected or duplicate'):
            study._validate_flows(flows)

    def test_audit_phases_and_packet_pool_errors_are_recorded(self):
        log = self.output / 'simulation.txt'
        log.write_text(run_line(rounds=40_000, hosts=study.HOSTS, channel_blocks=study.CHANNEL_BLOCKS)
                       + '\nPacket pool is not enough!\nrecord=gedes_drain seconds=1.5\n')
        phases, seconds, errors = study._run_records(log, audit)
        self.assertEqual(phases['total_nodes'], 27_648)
        self.assertEqual(phases['total_channel_blocks'], 5_184)
        self.assertEqual(seconds, 1.5)
        self.assertEqual(errors, ['Packet pool is not enough!'])

    def test_streaming_queue_gate_rejects_duplicate_readbacks(self):
        queues = self.output / 'queues.csv'
        queues.write_text('group,index,entries\ntcp_ingress,0,0\ntcp_ingress,0,0\n')
        with self.assertRaisesRegex(ValueError, 'queue inventory'):
            drain.inspect_queues(queues, study.HOSTS)


if __name__ == '__main__':
    unittest.main()
