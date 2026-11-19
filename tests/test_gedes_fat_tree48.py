import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from test_gedes_experiments import MAIN_FIXTURE, run_line

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

    def test_capture_stops_at_round_boundary_and_both_paths_finish(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('A host C++ compiler is required for the instrumentation fixture')
        header = self.output / 'cuda_profiler_api.h'
        header.write_text('''#include <iostream>
inline int cudaProfilerStart() { std::cout << "capture-start" << std::endl; return 0; }
inline int cudaProfilerStop() { std::cout << "capture-stop" << std::endl; return 0; }
''')
        fixture = audit.instrument_main(MAIN_FIXTURE).replace(
            'namespace VDES {', '''struct FixtureClock : std::chrono::steady_clock {
    inline static time_point current{};
    static time_point now() { return current; }
};
namespace VDES {''', 1
        ).replace(
            'using AuditClock = std::chrono::steady_clock;', 'using AuditClock = FixtureClock;',
        ).replace(
            'void RunGraph() { ++rounds; }',
            '''void RunGraph() {
    std::cout << "round=" << ++rounds << std::endl;
    FixtureClock::current += std::chrono::milliseconds(5100);
}''',
        )
        source = self.output / 'fixture.cc'
        binary = self.output / 'fixture'
        source.write_text(study._instrument_capture(fixture, 10))
        subprocess.run([compiler, '-std=c++17', '-I', str(self.output), str(source), '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        flow_output = self.output / 'flows.csv'
        command = [str(binary), f'--output={flow_output}']
        env = {key: value for key, value in os.environ.items() if key != study.CAPTURE_ENV}
        baseline = subprocess.run(command, env=env, check=True, capture_output=True, text=True).stdout
        baseline_flows = flow_output.read_bytes()
        profiled = subprocess.run(command, env={**env, study.CAPTURE_ENV: '1'}, check=True,
                                  capture_output=True, text=True).stdout
        self.assertNotIn('capture-', baseline)
        self.assertEqual(audit.parse_audit(baseline)['rounds'], 3)
        self.assertEqual(audit.parse_audit(profiled)['rounds'], 3)
        self.assertEqual(flow_output.read_bytes(), baseline_flows)
        self.assertLess(profiled.index('capture-start'), profiled.index('round=1'))
        self.assertLess(profiled.index('round=2'), profiled.index('capture-stop'))
        self.assertLess(profiled.index('capture-stop'), profiled.index('round=3'))
        log = self.output / 'capture.txt'
        log.write_text(profiled)
        self.assertEqual(study._read_capture_window(log, 3, 10), {
            'first_round': 0, 'end_round': 2, 'wall_ns': 10_200_000_000, 'requested_seconds': 10,
        })
        log.write_text(profiled.replace('end_round=2 rounds=2', 'end_round=2 rounds=3'))
        with self.assertRaisesRegex(ValueError, 'capture window boundaries'):
            study._read_capture_window(log, 3, 10)

    def test_short_profile_collects_trace_before_reporting_fct_mismatch(self):
        binary = self.output / 'GeDES'
        nsys = self.output / 'nsys'
        profiler = Mock()
        profiler.preflight.return_value = nsys
        profiler.prefix.return_value = ['nsys', 'profile']
        profiler.collect.return_value = {'passed': True}
        drain_fixture = Mock()
        drain_fixture.build_audit.return_value = binary
        helpers = Mock(GEDES=(None, audit.UPSTREAM_REVISION))
        modules = {'gedes_experiments': audit, 'gedes_drain': drain_fixture, 'nsys_ft48': profiler}
        helpers._support_module.side_effect = lambda source, name: modules[name]
        helpers.selected_gpu.return_value = ('10.0', {study.CAPTURE_ENV: 'inherited'})
        helpers.write_json.side_effect = lambda path, record: path.write_text(json.dumps(record))
        (self.output / 'gpu.json').write_text(json.dumps({'uuid': 'GPU-fixture', 'name': 'NVIDIA GB200'}))
        (self.output / 'gpu-free-mib.txt').write_text('188000')

        def run_case(executable, output, env, audit, drain, helpers, record, **kwargs):
            self.assertEqual(executable, binary)
            profiled = output.name == 'profile'
            self.assertEqual(env.get(study.CAPTURE_ENV), '1' if profiled else None)
            if profiled:
                (output / 'simulation.txt').write_text(
                    'record=gedes_capture first_round=0 end_round=2 rounds=2 '
                    'wall_ns=10000000000 requested_seconds=10\n')
            record.update(measurements={'rounds': 3, 'completed_nodes': 1, 'completed_channel_blocks': 1},
                          flows={'csv_sha256': str(profiled), 'reported_input_sha256': 'same'},
                          snapshot={'acked_bytes': 1460})

        with patch.object(study, '_run_case', side_effect=run_case) as run_cases:
            with self.assertRaisesRegex(RuntimeError, 'flows.csv_sha256'):
                study.run(self.output, self.output, helpers, profile=True, capture_seconds=10)
        self.assertEqual(run_cases.call_count, 2)
        self.assertIn('patch_main', drain_fixture.build_audit.call_args.kwargs)
        profiler.prefix.assert_called_once_with(nsys, self.output, capture_range=True)
        window = {'first_round': 0, 'end_round': 2, 'wall_ns': 10_000_000_000, 'requested_seconds': 10}
        profiler.collect.assert_called_once_with(nsys, self.output, {}, helpers, simulator='gedes',
                                                rounds=3, capture_window=window)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertFalse(summary['passed'])
        self.assertFalse(summary['profile']['passed'])
        self.assertEqual(summary['profile']['work_differences'], ['flows.csv_sha256'])
        self.assertEqual(summary['profile']['capture_window'], window)
        self.assertTrue(summary['profile']['trace']['passed'])


if __name__ == '__main__':
    unittest.main()
