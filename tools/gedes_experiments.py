"""Fixed GeDES completion and host-phase audit, built from a pinned upstream tree."""

import csv
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import statistics
import threading
from tempfile import TemporaryDirectory
import time


UPSTREAM_REVISION = '1376c638a0ae66116430f953da137d0c536675ad'
MAX_ROUNDS = 1_000_000
PARAMETERS = {'packet_pool_size': 1_000_000, 'average_flow_size': 10_000,
              'flow_time_range': 1_000_000}

# The upstream loop launches and stream-synchronizes a CUDA graph each round.
# Host main is instrumented and workload RNG is seeded; protocol and graph stay intact.
AUDIT_BODY = r'''    std::cout << "Building Network" << std::endl;
    using AuditClock = std::chrono::steady_clock;
    const auto audit_init_start = AuditClock::now();
    VDES::SetFlowTimeRange(flow_time_range);
    VDES::create_thread_pool(VDES::THREAD_POOL_SIZE);
    VDES::SetInitialMemoryPoolSize(packet_pool_size);
    VDES::InitializeMemoryPools();
    topology.SetFattreeProperties(ft_k, ip_group_size, ip_base_addr, ip_mask);
    topology.SetChannelParams(tx_rate, popogation_delay);
    topology.SetTimeslotInfo(timeslot_start, timeslot_end);
    topology.SetLookAheadTimeslot(look_ahead_timeslot);
    topology.SetTransmissionPlanParams(expected_packet_num, deviate, 41U);
    topology.BuildTopology();
#if ENABLE_GPU_MEM_POOL
    topology.BuildGraphOnlyGPU();
#elif ENABLE_HUGE_GRAPH
    topology.BuildHugeGraph();
#endif
    cudaError_t audit_init_error = cudaDeviceSynchronize();
    const auto audit_init_end = AuditClock::now();
    if (audit_init_error != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(audit_init_error) << std::endl;
        return 1;
    }
    int audit_rounds = 0;
    const int audit_max_rounds = 1000000;
    double audit_finish_check_seconds = 0;
    double audit_run_graph_seconds = 0;
    auto audit_finished = [&]() {
        if (!audit_detail) return topology.IsFinished();
        const auto begin = AuditClock::now();
        const bool result = topology.IsFinished();
        audit_finish_check_seconds += std::chrono::duration<double>(AuditClock::now() - begin).count();
        return result;
    };
    std::cout << "Simulation Start" << std::endl;
    const auto audit_loop_start = AuditClock::now();
    for (int i = 0; i < audit_max_rounds && !audit_finished(); ++i) {
#if ENABLE_HUGE_GRAPH
        if (audit_detail) {
            const auto begin = AuditClock::now();
            topology.RunGraph();
            audit_run_graph_seconds += std::chrono::duration<double>(AuditClock::now() - begin).count();
        } else {
            topology.RunGraph();
        }
#endif
        ++audit_rounds;
        if (i % 20000 == 0) {
            std::cout << "Simulation timeline " << i / 1000000.0 << "s" << std::endl;
        }
    }
    const auto audit_loop_end = AuditClock::now();
    const auto audit_sync_start = AuditClock::now();
    cudaError_t audit_sync_error = cudaDeviceSynchronize();
    const auto audit_sync_end = AuditClock::now();
    const bool audit_upstream_finished = topology.IsFinished();
    const double audit_loop_seconds = std::chrono::duration<double>(audit_loop_end - audit_loop_start).count();
    const double audit_sync_seconds = std::chrono::duration<double>(audit_sync_end - audit_sync_start).count();
    std::cout << "Simulation Duration: " << audit_loop_seconds + audit_sync_seconds << "s" << std::endl;
    cudaError_t audit_last_error = cudaGetLastError();
    if (audit_sync_error != cudaSuccess || audit_last_error != cudaSuccess) {
        std::cerr << "Cuda error: " << cudaGetErrorString(audit_sync_error != cudaSuccess ? audit_sync_error : audit_last_error) << std::endl;
        return 1;
    }
    const auto audit_output_start = AuditClock::now();
    topology.RecordFlowResults(output_file_name);
    const auto audit_output_end = AuditClock::now();
    std::cout << std::setprecision(17)
              << "record=gedes_run rounds=" << audit_rounds
              << " max_rounds=" << audit_max_rounds
              << " upstream_finished=" << (audit_upstream_finished ? 1 : 0)
              << " detail=" << (audit_detail ? 1 : 0)
              << " init_seconds=" << std::chrono::duration<double>(audit_init_end - audit_init_start).count()
              << " loop_seconds=" << audit_loop_seconds
              << " tail_sync_seconds=" << audit_sync_seconds
              << " output_seconds=" << std::chrono::duration<double>(audit_output_end - audit_output_start).count()
              << " finish_check_seconds=" << (audit_detail ? audit_finish_check_seconds : -1.0)
              << " run_graph_seconds=" << (audit_detail ? audit_run_graph_seconds : -1.0)
              << std::endl;
    return 0;
}'''


def instrument_main(original):
    start = '    std::cout << "Building Network"<< std::endl;'
    finish = '    return 0;\n}'
    if original.count(start) != 1 or original.count('    std::string output_file_name;') != 1:
        raise ValueError('unexpected GeDES main layout at the pinned revision')
    begin = original.index(start)
    end = original.index(finish, begin) + len(finish)
    patched = original[:begin] + AUDIT_BODY + original[end:]
    patched = patched.replace('#include <iostream>', '#include <iostream>\n#include <chrono>\n#include <iomanip>', 1)
    patched = patched.replace('    std::string output_file_name;',
                              '    bool audit_detail = false;\n    std::string output_file_name;', 1)
    unknown = '            else\n            {\n                std::cout << "Unknown argument: "'
    if patched.count(unknown) != 1:
        raise ValueError('unexpected GeDES argument parser at the pinned revision')
    patched = patched.replace(unknown, '            else if (arg == "--audit_detail=1")\n'
                              '            {\n                audit_detail = true;\n            }\n' + unknown, 1)
    return patched


def patch_cmake(original):
    placeholder = '/path/to/code/dir/GeDES_code'
    if original.count(placeholder) != 2:
        raise ValueError('unexpected GeDES CMake source paths')
    return original.replace(placeholder, '${CMAKE_CURRENT_SOURCE_DIR}').replace(
        'set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")\n', '')


def seed_workload(original):
    anchor = ("  void Topology::CreateFlows(int64_t start_timestamp)\n  {\n"
              "    auto poison_inter = [](int64_t avg) -> int64_t\n    {\n"
              "      static std::random_device rd;\n      static std::mt19937 gen(rd());")
    if original.count(anchor) != 1:
        raise ValueError('unexpected active GeDES flow-generator seed layout')
    return original.replace(anchor, anchor.replace(
        '      static std::random_device rd;\n      static std::mt19937 gen(rd());',
        '      static std::mt19937 gen(41U);'), 1)


def parse_audit(log):
    lines = [line for line in log.splitlines() if line.startswith('record=gedes_run ')]
    if len(lines) != 1:
        raise ValueError('expected exactly one GeDES completion record')
    fields = dict(item.split('=', 1) for item in lines[0].split()[1:])
    integers = ('rounds', 'max_rounds', 'upstream_finished', 'detail')
    seconds = ('init_seconds', 'loop_seconds', 'tail_sync_seconds', 'output_seconds',
               'finish_check_seconds', 'run_graph_seconds')
    try:
        values = {key: int(fields[key]) for key in integers}
        values.update({key: float(fields[key]) for key in seconds})
    except (KeyError, ValueError) as error:
        raise ValueError('invalid GeDES completion record') from error
    if (values['max_rounds'] != MAX_ROUNDS or not 0 <= values['rounds'] <= MAX_ROUNDS
            or values['upstream_finished'] not in (0, 1) or values['detail'] not in (0, 1)
            or any(not math.isfinite(values[key]) for key in seconds)
            or any(values[key] < 0 for key in seconds[:4])
            or values['loop_seconds'] <= 0):
        raise ValueError('invalid GeDES phase measurements')
    expected_unmeasured = not values['detail']
    if any((values[key] != -1 if expected_unmeasured else values[key] < 0)
           for key in seconds[4:]):
        raise ValueError('GeDES detailed measurements do not match the timing mode')
    return values


def flow_statistics(path, expected_rows):
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'source_ip', 'dst_ip', 'flow_size', 'start_timestamp', 'completed_timestamp',
                    'flow_completion_time'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('missing GeDES flow CSV fields')
        completed = []
        reported_inputs = []
        count = 0
        for row in reader:
            count += 1
            size = int(row['flow_size'])
            start, end = int(row['start_timestamp']), int(row['completed_timestamp'])
            duration = int(row['flow_completion_time'])
            reported_inputs.append((row['source_ip'], row['dst_ip'], size, start))
            if size > 0 and end > start >= 0 and duration == end - start:
                completed.append(duration)
    result = {'csv_rows': count, 'valid_completed_rows': len(completed),
              'expected_rows': expected_rows, 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'reported_input_sha256': hashlib.sha256(json.dumps(sorted(reported_inputs), separators=(',', ':')).encode()).hexdigest(),
              'input_digest_scope': 'source/destination/size/start fields of reported first flows'}
    if completed:
        ordered = sorted(completed)
        result.update(fct_min_ns=ordered[0], fct_max_ns=ordered[-1],
                      fct_median_ns=statistics.median(ordered))
    return result


def validate_run(measurements, flows, detail):
    reasons = []
    if measurements['detail'] != int(detail):
        reasons.append('timing mode does not match the requested case')
    if not measurements['upstream_finished']:
        reasons.append('upstream completion predicate was false at loop exit')
    if measurements['rounds'] == 0:
        reasons.append('no graph rounds executed')
    if flows['csv_rows'] != flows['expected_rows'] or flows['valid_completed_rows'] != flows['csv_rows']:
        reasons.append('flow CSV does not contain one valid completion per host connection')
    return reasons


def _source_audit(src):
    paths = ['main.cc', 'code/c++/src/components/topology.cc',
             'code/c++/src/components/vdes_timer.cc', 'code/c++/src/components/vdes_timer_kernel.cu',
             'code/c++/src/transport/tcp_controller.cc', 'code/c++/src/transport/tcp_controller.cu',
             'code/c++/include/conf.h', 'code/c++/src/config/config.cc']
    return {
        'revision': UPSTREAM_REVISION,
        'files': {name: hashlib.sha256((src / name).read_bytes()).hexdigest() for name in paths},
        'facts': {
            'run_graph': 'cudaGraphLaunch followed by cudaStreamSynchronize on every round',
            'is_finished': 'cached host boolean copied from device by the timer CUDA graph',
            'completion': 'upstream node-payload and channel-completion predicate',
            'csv': 'RecordFlowResults writes only the first flow per connection',
            'average_flow_size_flag': 'passed as expected packets per flow, not bytes',
            'kernel_instrumentation': False,
            'protocol_modifications': False,
            'workload_rng': 'active CreateFlows exponential generator seeded to 41U',
            'v1_workload_equivalence': 'not asserted; upstream used random_device',
        },
    }


def _build_impl(output, dependencies, helpers):
    root = output.parent.parent
    if helpers.GEDES[1] != UPSTREAM_REVISION:
        raise ValueError('GeDES audit requires its fixed upstream revision')
    capability, env = helpers.selected_gpu(root)
    helpers.command(['nvcc', '--version'], output / 'nvcc.txt', env=env, journal_dir=root)
    helpers.command(['cmake', '--version'], output / 'cmake.txt', env=env, journal_dir=root)
    helpers.command(['c++', '--version'], output / 'cxx.txt', env=env, journal_dir=root)
    with TemporaryDirectory(prefix='.gedes-audit-', dir=output) as temporary:
        src = helpers.checkout(helpers.GEDES, Path(temporary), output, env) / 'GeDES_code'
        audit = _source_audit(src)
        original_main = (src / 'main.cc').read_text()
        original_cmake = (src / 'CMakeLists.txt').read_text()
        patched_main = instrument_main(original_main)
        patched_cmake = patch_cmake(original_cmake)
        topology_file = src / 'code/c++/src/components/topology.cc'
        seeded_topology = seed_workload(topology_file.read_text())
        topology_file.write_text(seeded_topology)
        (output / 'topology-seeded.txt').write_text(seeded_topology)
        (src / 'main.cc').write_text(patched_main)
        (src / 'CMakeLists.txt').write_text(patched_cmake)
        (output / 'main-audited.txt').write_text(patched_main)
        (output / 'CMakeLists-audited.txt').write_text(patched_cmake)
        build = Path(temporary) / 'build'
        helpers.command(['cmake', '-S', str(src), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                         '-DCMAKE_CUDA_ARCHITECTURES=' + capability.replace('.', '')],
                        output / 'configure.log', env=env, journal_dir=root)
        helpers.command(['cmake', '--build', str(build), '--parallel', '4'],
                        output / 'build.log', env=env, journal_dir=root)
        shutil.copy2(build / 'GeDES', output / 'GeDES')
        shutil.copyfile(build / 'CMakeCache.txt', output / 'CMakeCache.txt')
        audit.update(main_patched_sha256=hashlib.sha256(patched_main.encode()).hexdigest(),
                     cmake_patched_sha256=hashlib.sha256(patched_cmake.encode()).hexdigest(),
                     topology_seeded_sha256=hashlib.sha256(seeded_topology.encode()).hexdigest(),
                     binary_sha256=hashlib.sha256((output / 'GeDES').read_bytes()).hexdigest(),
                     compute_capability=capability, build_mode='Release')
        helpers.write_json(output / 'build.json', audit)
    helpers.write_json(output / 'gpu.json', json.loads((root / 'gpu.json').read_text()))


def _build(output, dependencies, helpers):
    try:
        _build_impl(output, dependencies, helpers)
    except Exception as error:
        feedback = [f'{type(error).__name__}: {error}']
        for name in ('configure.log', 'build.log'):
            path = output / name
            if path.is_file():
                feedback.append(f'\n{name} tail:\n' + path.read_text(errors='replace')[-65536:])
        (output / 'failure.txt').write_text('\n'.join(feedback) + '\n')
        raise


class GPUSampler:
    """Optional one-second management samples for the invasive profile case."""
    def __init__(self, uuid, helpers):
        self.uuid = uuid
        self.helpers = helpers
        self.samples = []
        self.errors = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            try:
                sample = subprocess.run([
                    'nvidia-smi', '--id=' + self.uuid,
                    '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5,
                    pass_fds=self.helpers.pass_lock())
                if sample.returncode:
                    raise RuntimeError(sample.stderr.strip() or f'nvidia-smi rc={sample.returncode}')
                fields = [field.strip() for field in sample.stdout.strip().split(',')]
                if len(fields) != 4:
                    raise ValueError('unexpected GPU management sample')
                self.samples.append({'elapsed_seconds': time.monotonic() - self.started,
                                     **dict(zip(('gpu_util_percent', 'memory_util_percent',
                                                'memory_used_mib', 'power_watts'), fields))})
            except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
                if len(self.errors) < 5:
                    self.errors.append(str(error))
            self.stop.wait(1)

    def __enter__(self):
        self.started = time.monotonic()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(timeout=6)

    def save(self, output):
        fields = ('elapsed_seconds', 'gpu_util_percent', 'memory_util_percent', 'memory_used_mib', 'power_watts')
        with (output / 'gpu-samples.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.samples)
        self.helpers.write_json(output / 'sampling.json', {
            'scope': 'invasive one-second nvidia-smi management samples; not kernel occupancy',
            'samples': len(self.samples), 'errors': self.errors,
        })


def _run(output, dependencies, helpers, k, detail=False):
    root = output.parent.parent
    gpu = json.loads((dependencies['build'] / 'gpu.json').read_text())
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu['uuid']}
    flow_path = output / 'flows.csv'
    args = [str(dependencies['build'] / 'GeDES'), f'--ft_k={k}',
            *[f'--{key}={value}' for key, value in PARAMETERS.items()], f'--output={flow_path}']
    if detail:
        args.append('--audit_detail=1')
    if detail:
        sampler = GPUSampler(gpu['uuid'], helpers)
        try:
            with sampler:
                wall = helpers.command(args, output / 'simulation.txt', env=env, cwd=output, journal_dir=root)
        finally:
            sampler.save(output)
    else:
        wall = helpers.command(args, output / 'simulation.txt', env=env, cwd=output, journal_dir=root)
    text = (output / 'simulation.txt').read_text(errors='replace')
    measurements = parse_audit(text)
    flows = flow_statistics(flow_path, k ** 3 // 4)
    reasons = validate_run(measurements, flows, detail)
    summary = {
        'scope': 'upstream completion predicate and first-flow-per-connection CSV audit; transport drain unverified',
        'timing_mode': 'invasive-host-profile' if detail else 'normal-phase-timing',
        'ft_k': k, 'parameters': PARAMETERS, 'process_seconds': wall,
        'measurements': measurements, 'flows': flows, 'passed': not reasons, 'gate_failures': reasons,
        'instrumentation': 'host main phase timing plus CreateFlows RNG seed41; protocol and CUDA graph unchanged',
    }
    helpers.write_json(output / 'summary.json', summary)
    if reasons:
        raise RuntimeError('; '.join(reasons))


def case_plan(task, source, helpers):
    if task['kind'] != 'gedes_audit':
        return None
    if task['params'] != {}:
        raise ValueError('gedes_audit uses fixed parameters in this version')
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = [{'id': 'build', 'depends_on': [], 'run': functools.partial(_build, helpers=helpers),
               'implementation': implementation, 'inputs': {'revision': UPSTREAM_REVISION}, 'reuse': False}]
    for case_id, k, detail in [('normal-k4', 4, False), ('normal-k8-a', 8, False),
                               ('normal-k8-b', 8, False), ('detailed-k8', 8, True)]:
        dependencies = ['build']
        result.append({'id': case_id, 'depends_on': dependencies,
                       'run': functools.partial(_run, helpers=helpers, k=k, detail=detail),
                       'implementation': implementation,
                       'inputs': {'ft_k': k, **PARAMETERS, 'detail': detail}, 'reuse': False})
    return result


def finalize_cases(task, output, manifest, helpers):
    if task['kind'] != 'gedes_audit':
        return
    measurements = {}
    for case_id in manifest['cases']:
        path = output / 'cases' / case_id / 'summary.json'
        if path.is_file():
            measurements[case_id] = json.loads(path.read_text())
    k8_inputs = {key: value['flows']['reported_input_sha256'] for key, value in measurements.items()
                 if key in {'normal-k8-a', 'normal-k8-b', 'detailed-k8'}}
    reported_inputs_match = len(k8_inputs) == 3 and len(set(k8_inputs.values())) == 1
    normal_csv_hashes = {key: value['flows'].get('csv_sha256') for key, value in measurements.items()
                         if key in {'normal-k8-a', 'normal-k8-b'}}
    normal_csv_match = (len(normal_csv_hashes) == 2 and None not in normal_csv_hashes.values()
                        and len(set(normal_csv_hashes.values())) == 1)
    helpers.write_json(output / 'summary.json', {
        'scope': 'GeDES completion/host-phase audit; reported FCT records and upstream finish predicate',
        'revision': UPSTREAM_REVISION, 'case_states': {key: row['state'] for key, row in manifest['cases'].items()},
        'case_counts': manifest['summary'], 'runs': measurements,
        'k8_reported_inputs_match': reported_inputs_match, 'k8_reported_input_hashes': k8_inputs,
        'normal_k8_csv_sha256_equal': normal_csv_match, 'normal_k8_csv_sha256': normal_csv_hashes,
        'notes': ['normal measurements and invasive host profile are separate cases',
                  'RunGraph includes stream synchronization; its time is not isolated kernel execution',
                  'reported CSV contains only the first flow of each connection',
                  'CreateFlows generation uses fixed seed 41; v1 random-device input is not a matched baseline',
                  'no comparison against a CPU reference or native transport-drain assertion'],
    })
    if len(k8_inputs) == 3 and not reported_inputs_match:
        raise RuntimeError('seeded k8 reported first-flow inputs differ across normal/profile runs')
