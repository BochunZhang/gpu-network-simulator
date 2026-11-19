"""Run the pinned GeDES public artifact's FatTree48 workload on one GB200."""

import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
import time


K = 48
HOSTS = K ** 3 // 4
CHANNEL_BLOCKS = 6 * HOSTS // 32
TIMESLOT_NS = 1000
CAPTURE_ENV = 'GEDES_FT48_CAPTURE'
PARAMETERS = {'ft_k': K, 'packet_pool_size': 20_000_000,
              'average_flow_size': 10_000, 'flow_time_range': 10_000_000}
ERROR_MARKERS = ('memory exhausted!', 'packet pool is not enough!', 'error malloc:', 'cudamalloc failed',
                 'cuda error:', 'gedes completion readback failed:', 'gedes terminal snapshot failed:')


def _instrument_capture(source, seconds):
    declaration = '    int audit_rounds = 0;'
    first_graph = '#if ENABLE_HUGE_GRAPH\n        if (audit_detail)'
    round_complete = '        ++audit_rounds;'
    loop_complete = '    const auto audit_loop_end = AuditClock::now();'
    if any(source.count(anchor) != 1 for anchor in (declaration, first_graph, round_complete, loop_complete)):
        raise ValueError('unexpected audited GeDES main layout for capture')
    source = source.replace(declaration, declaration + f'''
    const bool capture_enabled = std::getenv("{CAPTURE_ENV}") != nullptr;
    const auto capture_target = std::chrono::seconds({seconds});
    bool capture_active = false;
    AuditClock::time_point capture_start;
    auto stop_capture = [&](AuditClock::duration elapsed) {{
        capture_active = false;
        const auto status = cudaProfilerStop();
        if (status != cudaSuccess) {{
            std::cerr << "Cuda error: cudaProfilerStop: " << cudaGetErrorString(status) << std::endl;
        }}
        std::cout << "record=gedes_capture first_round=0 end_round=" << audit_rounds
                  << " rounds=" << audit_rounds
                  << " wall_ns=" << std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count()
                  << " requested_seconds=" << capture_target.count() << std::endl;
    }};
''')
    source = source.replace(first_graph, '''#if ENABLE_HUGE_GRAPH
        if (capture_enabled && i == 0) {
            const auto status = cudaProfilerStart();
            if (status == cudaSuccess) {
                capture_start = AuditClock::now();
                capture_active = true;
            } else {
                std::cerr << "Cuda error: cudaProfilerStart: " << cudaGetErrorString(status) << std::endl;
            }
        }
        if (audit_detail)''')
    source = source.replace(round_complete, round_complete + '''
        if (capture_active) {
            const auto elapsed = AuditClock::now() - capture_start;
            if (elapsed >= capture_target) stop_capture(elapsed);
        }''')
    source = source.replace(loop_complete, '''    if (capture_active) {
        stop_capture(AuditClock::now() - capture_start);
    }
''' + loop_complete)
    return '#include <cuda_profiler_api.h>\n#include <cstdlib>\n' + source


def _read_capture_window(log, full_rounds, seconds):
    with log.open(encoding='utf-8', errors='replace') as stream:
        records = [line.strip() for line in stream if line.startswith('record=gedes_capture ')]
    if len(records) != 1:
        raise ValueError('expected exactly one GeDES capture window record')
    match = re.fullmatch(r'record=gedes_capture first_round=(\d+) end_round=(\d+) rounds=(\d+) '
                         r'wall_ns=(\d+) requested_seconds=(\d+)', records[0])
    if not match:
        raise ValueError('malformed GeDES capture window record')
    first, end, count, wall_ns, requested = map(int, match.groups())
    if not (first == 0 and 0 < end <= full_rounds and count == end - first and requested == seconds):
        raise ValueError('invalid GeDES capture window boundaries')
    return {'first_round': first, 'end_round': end, 'wall_ns': wall_ns, 'requested_seconds': requested}


def _host_ip(node):
    group, offset = divmod(node, K // 2)
    return 0xc0a80000 + group * 48 + offset


def _validate_flows(path):
    expected = {_host_ip(node): _host_ip((node + HOSTS // 2) % HOSTS) for node in range(HOSTS)}
    max_flow_bytes = 1460 * 2 * PARAMETERS['average_flow_size']
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            src, dst = int(row['source_ip']), int(row['dst_ip'])
            if expected.pop(src, None) != dst:
                raise ValueError(f'unexpected or duplicate FatTree48 flow pair {src}->{dst}')
            size = int(row['flow_size'])
            if not 0 < size <= max_flow_bytes or size % 1460:
                raise ValueError(f'flow {src}->{dst} violates the configured segment-size workload')
    if expected:
        raise ValueError(f'FatTree48 FCT CSV is missing {len(expected)} directed pairs')


def _run_records(path, audit):
    phase_records, drain_records, errors = [], [], []
    with path.open(encoding='utf-8', errors='replace') as stream:
        for line in stream:
            if line.startswith('record=gedes_run '):
                phase_records.append(line)
            if line.startswith('record=gedes_drain seconds='):
                drain_records.append(line.split('=', 2)[-1])
            if len(errors) < len(ERROR_MARKERS) and any(marker in line.lower() for marker in ERROR_MARKERS):
                errors.append(line.strip()[:512])
    phases = audit.parse_audit(''.join(phase_records))
    if len(drain_records) != 1:
        raise ValueError('expected exactly one GeDES terminal audit timing record')
    seconds = float(drain_records[0])
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('invalid GeDES terminal audit duration')
    return phases, seconds, errors


def _file_record(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(chunk)
    return {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def _raw_files(output):
    names = ('flows.csv', 'flows.csv.connections.csv', 'flows.csv.queues.csv', 'simulation.txt')
    paths = (output / name for name in names)
    return {path.name: _file_record(path) for path in paths if path.is_file()}


def _run_case(binary, output, env, audit, drain, helpers, record, *, prefix=(), journal_dir=None):
    flow_path = output / 'flows.csv'
    log = output / 'simulation.txt'
    args = [*prefix, str(binary), *[f'--{key}={value}' for key, value in PARAMETERS.items()],
            f'--output={flow_path}']
    record['process_seconds'] = helpers.command(args, log, env=env, cwd=output, journal_dir=journal_dir)
    measurements, drain_seconds, cuda_errors = _run_records(log, audit)
    record.update(measurements=measurements, terminal_audit_seconds=drain_seconds,
                  virtual_round_span_ns=measurements['rounds'] * TIMESLOT_NS,
                  cuda_diagnostics=cuda_errors)
    flows = record['flows'] = audit.flow_statistics(flow_path, HOSTS)
    failures = record['gate_failures'] = audit.validate_run(measurements, flows, detail=False)
    if measurements['total_nodes'] != HOSTS or measurements['total_channel_blocks'] != CHANNEL_BLOCKS:
        failures.append('upstream node or channel block count differs from FatTree48')
    if measurements['rounds'] >= measurements['max_rounds']:
        failures.append('GeDES reached the virtual round cap')
    if cuda_errors:
        failures.append('CUDA or packet-pool errors were reported')
    _validate_flows(flow_path)
    snapshot = record['snapshot'] = drain.inspect_snapshot(flow_path, K)
    failures.extend(drain.drain_failures(snapshot))
    if failures:
        raise RuntimeError('; '.join(failures))


def _work_differences(baseline, profiled):
    differences = []
    for field in ('rounds', 'completed_nodes', 'completed_channel_blocks'):
        if baseline['measurements'][field] != profiled['measurements'][field]:
            differences.append(f'measurements.{field}')
    for field in ('csv_sha256', 'reported_input_sha256'):
        if baseline['flows'][field] != profiled['flows'][field]:
            differences.append(f'flows.{field}')
    if baseline['snapshot'] != profiled['snapshot']:
        differences.append('terminal byte or queue state')
    return differences


def _build_gedes(source, output, scratch, capability, env, helpers, patch_main=None):
    repo_src = source / 'simulators/gedes/GeDES_code'
    src = repo_src
    if patch_main is not None:
        src = scratch / 'profile-source'
        shutil.copytree(repo_src, src)
        main = src / 'main.cc'
        main.write_text(patch_main(main.read_text()))
        (output / 'main.cc-profile.txt').write_bytes(main.read_bytes())
    build = scratch / 'build'
    options = ['-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=' + capability.replace('.', '')]
    if 'FT48_WARP_LIBDIR' in env:
        options.append('-DFT48_WARP_LIBDIR=' + env['FT48_WARP_LIBDIR'])
    helpers.command(['cmake', '-S', str(src), '-B', str(build), *options], output / 'configure.log', env=env)
    helpers.command(['cmake', '--build', str(build), '--parallel', '4'], output / 'build.log', env=env)
    binary = build / 'GeDES'
    helpers.write_json(output / 'build.json', {
        'source': 'simulators/gedes/GeDES_code',
        'revision': helpers.GEDES[1],
        'compiled_source_sha256': {p.relative_to(src).as_posix(): _file_record(p)['sha256']
                                   for p in sorted(src.rglob('*')) if p.is_file()},
        'binary_sha256': _file_record(binary)['sha256'],
        'compute_capability': capability, 'build_mode': 'Release',
    })
    return binary


def _paired_fct(baseline_csv, trace_csv):
    def read_rows(path):
        with path.open(encoding='utf-8', newline='') as stream:
            return {(row['source_ip'], row['dst_ip']): int(row['flow_completion_time'])
                    for row in csv.DictReader(stream)}

    baseline = read_rows(baseline_csv)
    profiled = read_rows(trace_csv)
    if baseline.keys() != profiled.keys():
        raise ValueError('paired GeDES runs have different flow endpoints')
    deltas = [abs(value - profiled[pair]) for pair, value in baseline.items()]
    return {'flow_count': len(baseline), 'different_flow_count': sum(delta != 0 for delta in deltas),
            'max_absolute_difference_ns': max(deltas), 'all_equal': not any(deltas)}


def run(source, output, helpers, profile=False, capture_seconds=None, warp_trace=False):
    audit = helpers._support_module(source, 'gedes_experiments')
    drain = helpers._support_module(source, 'gedes_drain')
    summary = {'scope': 'Single-GB200 GeDES public artifact FatTree48 TCP/DCTCP workload with seeded inputs',
               'revision': audit.UPSTREAM_REVISION, 'parameters': PARAMETERS, 'seed': 41,
               'expected': {'hosts': HOSTS, 'switches': 5 * K ** 2 // 4,
                            'directed_channels': 6 * HOSTS, 'channel_blocks': CHANNEL_BLOCKS,
                            'directed_flows': HOSTS, 'timeslot_ns': TIMESLOT_NS},
               'process_seconds_includes_terminal_audit': True,
               'build': 'pending', 'passed': False, 'raw_files': {}}
    profile_output = output / 'profile'
    try:
        if capture_seconds is not None and (not profile or type(capture_seconds) is not int or capture_seconds <= 0):
            raise ValueError('a positive integer GeDES capture duration requires profiling')
        if warp_trace and profile:
            raise ValueError('GeDES warp and Nsight captures use separate runs')
        if helpers.GEDES[1] != audit.UPSTREAM_REVISION:
            raise ValueError('unexpected GeDES source pin')
        capability, env = helpers.selected_gpu(output)
        env = {key: value for key, value in env.items()
               if key not in ('FT48_WARP_LIBDIR', 'FT48_WARP_OUTPUT', 'GEDES_FT48_GRAPH_DOT', CAPTURE_ENV)}
        build_options = {}
        if capture_seconds is not None:
            build_options['patch_main'] = lambda text: _instrument_capture(text, capture_seconds)
        gpu = json.loads((output / 'gpu.json').read_text())
        if 'GB200' not in gpu['name'].upper():
            raise ValueError(f"FatTree48 requires a GB200, selected {gpu['name']}")
        if profile:
            capture = summary['profile'] = {
                'passed': False, 'run': {},
                'process_seconds_scope': 'Nsight Systems wrapper including simulator execution and trace generation',
            }
            profiler = helpers._support_module(source, 'nsys_ft48')
            nsys = profiler.preflight(output, env, helpers)
        helpers.command(['nvidia-smi', '--id=' + gpu['uuid'], '--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'], output / 'gpu-free-mib.txt', env=env)
        summary['gpu'] = {**gpu, 'free_mib_before_build': int((output / 'gpu-free-mib.txt').read_text())}
        for tool in ('nvcc', 'cmake', 'c++'):
            helpers.command([tool, '--version'], output / f'{tool}.txt', env=env)
        summary['build'] = 'failed'
        with TemporaryDirectory(prefix='.gedes-fat-tree48-', dir=output) as temporary:
            build_start = time.monotonic()
            scratch = Path(temporary)
            if warp_trace:
                warp = helpers._support_module(source, 'warp_ft48')
                env = warp.prepare(source, output, scratch, env, helpers)
            binary = _build_gedes(source, output, scratch, capability, env, helpers, **build_options)
            summary['build_setup_seconds'] = time.monotonic() - build_start
            summary['build'] = 'succeeded'
            _run_case(binary, output, env, audit, drain, helpers, summary)
            if warp_trace:
                profile_output.mkdir()
                run = {}
                capture = summary['warp_profile'] = {'run': run, 'passed': False}
                capture_env = warp.capture_environment(source, output, env, 'gedes')
                dot = output / 'warp' / 'graph.dot'
                dot.parent.mkdir(exist_ok=True)
                capture_env['GEDES_FT48_GRAPH_DOT'] = str(dot)
                _run_case(binary, profile_output, capture_env, audit, drain, helpers, run, journal_dir=output)
                rounds = run['measurements']['rounds']
                capture['trace'] = warp.collect(source, output, 'gedes', rounds)
                if not dot.is_file() or not dot.stat().st_size:
                    raise ValueError('GeDES graph dependency DOT is missing or empty')
                capture['graph_dot'] = {'path': 'warp/graph.dot', **_file_record(dot)}
                capture['paired_fct'] = _paired_fct(output / 'flows.csv', profile_output / 'flows.csv')
                capture['round_counts_equal'] = rounds == summary['measurements']['rounds']
                differences = capture['work_differences'] = _work_differences(summary, run)
                failures = [field for field in differences if field not in ('flows.csv_sha256', 'measurements.rounds')]
                if failures:
                    raise RuntimeError(f'GeDES warp run has different inputs or terminal work: {", ".join(failures)}')
                capture['passed'] = True
            if profile:
                profile_output.mkdir()
                prefix_options = {'capture_range': True} if capture_seconds is not None else {}
                profile_env = {**env, CAPTURE_ENV: '1'} if capture_seconds is not None else env
                _run_case(binary, profile_output, profile_env, audit, drain, helpers, capture['run'],
                          prefix=profiler.prefix(nsys, output, **prefix_options), journal_dir=output)
                differences = capture['work_differences'] = _work_differences(summary, capture['run'])
                full_rounds = capture['run']['measurements']['rounds']
                collect_options = {}
                if capture_seconds is not None:
                    window = _read_capture_window(profile_output / 'simulation.txt', full_rounds, capture_seconds)
                    capture['capture_window'] = collect_options['capture_window'] = window
                if not differences or capture_seconds is not None:
                    capture['trace'] = profiler.collect(nsys, output, env, helpers, simulator='gedes',
                                                        rounds=full_rounds, **collect_options)
                if differences:
                    raise RuntimeError(f'GeDES profiled work differs from baseline: {", ".join(differences)}')
                capture['passed'] = True
        summary['passed'] = True
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        tails = [summary['error']]
        for name in ('git-fetch.log', 'configure.log', 'build.log', 'simulation.txt', 'profile/simulation.txt'):
            path = output / name
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    tails.append(f'{name} tail:\n' + stream.read().decode(errors='replace'))
        (output / 'failure.txt').write_text('\n'.join(tails) + '\n')
        raise
    finally:
        summary['raw_files'] = _raw_files(output)
        if 'warp_profile' in summary:
            summary['warp_profile']['raw_files'] = _raw_files(profile_output)
        if 'profile' in summary:
            summary['profile']['raw_files'] = _raw_files(profile_output)
        helpers.write_json(output / 'summary.json', summary)
