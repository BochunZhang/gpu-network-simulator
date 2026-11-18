import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import gedes_experiments as audit


# An authored host-only stand-in tests the patched main without a CUDA toolkit.
MAIN_FIXTURE = r'''#include <iostream>
#include <fstream>
#define ENABLE_GPU_MEM_POOL 1
#define ENABLE_HUGE_GRAPH 1
using cudaError_t = int;
const int cudaSuccess = 0;
int cudaDeviceSynchronize() { return 0; }
int cudaGetLastError() { return 0; }
const char *cudaGetErrorString(int) { return "fixture"; }
namespace VDES {
int THREAD_POOL_SIZE = 1;
void SetFlowTimeRange(int) {}
void create_thread_pool(int) {}
void SetInitialMemoryPoolSize(int) {}
void InitializeMemoryPools() {}
struct Topology {
 int rounds = 0;
 void SetFattreeProperties(int,int,int,int) {}
 void SetChannelParams(int,int) {}
 void SetTimeslotInfo(int,int) {}
 void SetLookAheadTimeslot(int) {}
 void SetTransmissionPlanParams(int,double,unsigned) {}
 void BuildTopology() {}
 void BuildGraphOnlyGPU() {}
 void BuildHugeGraph() {}
 bool IsFinished() {
#ifdef FIXTURE_NEVER_FINISH
 return false;
#else
 return rounds >= 3;
#endif
 }
 void RunGraph() { ++rounds; }
 void RecordFlowResults(std::string path) {
   std::ofstream out(path);
   out << "source_ip,dst_ip,flow_size,start_timestamp,completed_timestamp,flow_completion_time\n";
   for (int i=0;i<16;++i) out << i << ",99,1460,1,11,10\n";
 }
};
}
int main(int argc, char *argv[]) {
    VDES::Topology topology;
    int flow_time_range=1000000,packet_pool_size=1000000,ft_k=4,ip_group_size=48;
    int ip_base_addr=0,ip_mask=0,tx_rate=100,popogation_delay=1000;
    int timeslot_start=0,timeslot_end=1000,look_ahead_timeslot=6,expected_packet_num=10000;
    double deviate=0.5;
    std::string output_file_name;
    for (int i=1;i<argc;++i) {
        std::string arg=argv[i];
        if (arg.find("--output=")==0) output_file_name=arg.substr(9);
            else
            {
                std::cout << "Unknown argument: " << arg << std::endl;
                return 1;
            }
    }
    std::cout << "Building Network"<< std::endl;
    return 0;
}
'''


def run_line(detail=False, finished=True, rounds=3):
    return ('record=gedes_run rounds=' + str(rounds) + ' max_rounds=1000000 '
            f'upstream_finished={int(finished)} detail={int(detail)} '
            'init_seconds=0.1 loop_seconds=0.2 tail_sync_seconds=0.01 output_seconds=0.03 '
            f'finish_check_seconds={0.001 if detail else -1} run_graph_seconds={0.15 if detail else -1}\n')


def write_flows(path, count=16, malformed=False):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['source_ip', 'dst_ip', 'flow_size', 'start_timestamp',
                         'completed_timestamp', 'flow_completion_time'])
        for index in range(count):
            writer.writerow([str(index), '99', 1460, 1, 11, 9 if malformed and index == 0 else 10])


class Helpers:
    def __init__(self):
        self.calls = []
        self.finished = True

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value))

    def command(self, args, log, **kwargs):
        self.calls.append(args)
        output = next(item.split('=', 1)[1] for item in args if item.startswith('--output='))
        k = int(next(item.split('=', 1)[1] for item in args if item.startswith('--ft_k=')))
        write_flows(Path(output), k ** 3 // 4)
        log.write_text(run_line(detail='--audit_detail=1' in args, finished=self.finished,
                                rounds=3 if self.finished else audit.MAX_ROUNDS))
        return 0.4


class GeDESAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.helpers = Helpers()
        self.build = self.root / 'cases/build'
        self.build.mkdir(parents=True)
        (self.build / 'GeDES').write_text('binary fixture')
        (self.build / 'gpu.json').write_text(json.dumps({'uuid': 'GPU-fixture'}))

    def test_host_main_compiles_and_reports_normal_detailed_and_cap_exit(self):
        compiler = shutil.which('c++')
        if not compiler:
            self.skipTest('A host C++ compiler is required for the instrumentation fixture')
        patched = audit.instrument_main(MAIN_FIXTURE)
        source = self.root / 'main.cc'
        source.write_text(patched)
        executable = self.root / 'audit-main'
        subprocess.run([compiler, '-std=c++17', '-O2', str(source), '-o', str(executable)], check=True,
                       capture_output=True)
        for detail in (False, True):
            args = [str(executable), '--output=' + str(self.root / 'flows.csv')]
            if detail:
                args.append('--audit_detail=1')
            result = subprocess.run(args, check=True, capture_output=True, text=True)
            measured = audit.parse_audit(result.stdout)
            flows = audit.flow_statistics(self.root / 'flows.csv', 16)
            self.assertEqual(measured['rounds'], 3)
            self.assertEqual(audit.validate_run(measured, flows, detail), [])
        subprocess.run([compiler, '-std=c++17', '-O2', '-DFIXTURE_NEVER_FINISH', str(source),
                        '-o', str(executable)], check=True, capture_output=True)
        result = subprocess.run([str(executable), '--output=' + str(self.root / 'flows.csv')],
                                check=True, capture_output=True, text=True)
        measured = audit.parse_audit(result.stdout)
        self.assertEqual(measured['rounds'], audit.MAX_ROUNDS)
        self.assertEqual(measured['upstream_finished'], 0)
        self.assertIn('completion predicate', audit.validate_run(
            measured, audit.flow_statistics(self.root / 'flows.csv', 16), False)[0])

    def test_seed_patch_is_anchored_to_active_create_flows_only(self):
        original = ('std::mt19937 unrelated(rd());\n'
                    '  void Topology::CreateFlows(int64_t start_timestamp)\n  {\n'
                    '    auto poison_inter = [](int64_t avg) -> int64_t\n    {\n'
                    '      static std::random_device rd;\n      static std::mt19937 gen(rd());\n'
                    '    };\n  }\n')
        seeded = audit.seed_workload(original)
        self.assertIn('static std::mt19937 gen(41U);', seeded)
        self.assertIn('std::mt19937 unrelated(rd());', seeded)
        with self.assertRaisesRegex(ValueError, 'seed layout'):
            audit.seed_workload(original.replace('gen(rd())', 'gen(99)'))
        with self.assertRaisesRegex(ValueError, 'main layout'):
            audit.instrument_main(MAIN_FIXTURE.replace('Building Network', 'other'))

    def test_cmake_patch_uses_source_directory_and_selected_compiler(self):
        original = ('set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")\n'
                    '"/path/to/code/dir/GeDES_code/code/c++/include/"\n'
                    'set(SOURCE_DIR /path/to/code/dir/GeDES_code/code/c++/src/)\n')
        result = audit.patch_cmake(original)
        self.assertNotIn('/path/to/code', result)
        self.assertNotIn('g++-12', result)
        self.assertEqual(result.count('${CMAKE_CURRENT_SOURCE_DIR}'), 2)

    def test_measurement_and_csv_gates_reject_false_completion_and_invalid_records(self):
        for text in ('', run_line() + run_line(), run_line().replace('loop_seconds=0.2', 'loop_seconds=nan'),
                     run_line().replace('finish_check_seconds=-1', 'finish_check_seconds=0')):
            with self.assertRaises(ValueError):
                audit.parse_audit(text)
        path = self.root / 'flows.csv'
        write_flows(path, malformed=True)
        flows = audit.flow_statistics(path, 16)
        measured = audit.parse_audit(run_line(finished=False, rounds=audit.MAX_ROUNDS))
        failures = audit.validate_run(measured, flows, False)
        self.assertEqual(len(failures), 2)
        self.assertEqual(flows['valid_completed_rows'], 15)
        # Completion timestamps do not enter the reported input digest.
        input_digest = flows['reported_input_sha256']
        write_flows(path)
        self.assertEqual(audit.flow_statistics(path, 16)['reported_input_sha256'], input_digest)

    def test_fct_median_uses_both_middle_values_for_even_count(self):
        path = self.root / 'median.csv'
        path.write_text('source_ip,dst_ip,flow_size,start_timestamp,completed_timestamp,flow_completion_time\n'
                        '0,99,1460,1,11,10\n1,99,1460,1,21,20\n')
        self.assertEqual(audit.flow_statistics(path, 2)['fct_median_ns'], 15)

    def test_cap_exit_preserves_audit_summary_before_case_failure(self):
        self.helpers.finished = False
        output = self.root / 'cases/normal-k8-a'
        output.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'completion predicate'):
            audit._run(output, {'build': self.build}, self.helpers, 8)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertFalse(summary['passed'])
        self.assertEqual(summary['flows']['valid_completed_rows'], 128)
        self.assertEqual(summary['measurements']['rounds'], audit.MAX_ROUNDS)
        self.assertIn('transport drain unverified', summary['scope'])

    def test_build_failure_keeps_log_tail_in_downloadable_feedback(self):
        (self.build / 'configure.log').write_text('configure diagnostic')
        (self.build / 'build.log').write_text('compile diagnostic')
        with patch.object(audit, '_build_impl', side_effect=RuntimeError('compiler failed')):
            with self.assertRaisesRegex(RuntimeError, 'compiler failed'):
                audit._build(self.build, {}, self.helpers)
        feedback = (self.build / 'failure.txt').read_text()
        self.assertIn('configure diagnostic', feedback)
        self.assertIn('compile diagnostic', feedback)

    def test_case_plan_keeps_all_measurements_independent_and_fresh(self):
        task = {'kind': 'gedes_audit', 'params': {}}
        cases = audit.case_plan(task, self.root, self.helpers)
        self.assertEqual([case['id'] for case in cases],
                         ['build', 'normal-k4', 'normal-k8-a', 'normal-k8-b', 'detailed-k8'])
        self.assertTrue(all(case['depends_on'] == ['build'] for case in cases[1:]))
        self.assertTrue(all(case['reuse'] is False for case in cases))
        self.assertEqual(cases[1]['inputs']['average_flow_size'], 10000)
        with self.assertRaises(ValueError):
            audit.case_plan({'kind': 'gedes_audit', 'params': {'ft_k': 32}}, self.root, self.helpers)

    def test_repeat_input_gate_records_mismatch_without_changing_case_outputs(self):
        cases = {}
        for index, name in enumerate(('normal-k8-a', 'normal-k8-b', 'detailed-k8')):
            path = self.root / 'cases' / name
            path.mkdir()
            self.helpers.write_json(path / 'summary.json', {
                'flows': {'reported_input_sha256': 'changed' if index == 2 else 'same'},
            })
            cases[name] = {'state': 'succeeded'}
        with self.assertRaisesRegex(RuntimeError, 'inputs differ'):
            audit.finalize_cases({'kind': 'gedes_audit'}, self.root, {'cases': cases, 'summary': {}}, self.helpers)
        result = json.loads((self.root / 'summary.json').read_text())
        self.assertFalse(result['k8_reported_inputs_match'])
        self.assertEqual(len(result['runs']), 3)
