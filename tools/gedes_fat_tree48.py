"""Run the pinned GeDES public artifact's FatTree48 workload on one GB200."""

import csv
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time


K = 48
HOSTS = K ** 3 // 4
CHANNEL_BLOCKS = 6 * HOSTS // 32
TIMESLOT_NS = 1000
PARAMETERS = {'ft_k': K, 'packet_pool_size': 20_000_000,
              'average_flow_size': 10_000, 'flow_time_range': 10_000_000}
ERROR_MARKERS = ('memory exhausted!', 'packet pool is not enough!', 'error malloc:', 'cudamalloc failed',
                 'cuda error:', 'gedes completion readback failed:', 'gedes terminal snapshot failed:')


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


def run(source, output, helpers):
    audit = helpers._support_module(source, 'gedes_experiments')
    drain = helpers._support_module(source, 'gedes_drain')
    summary = {'scope': 'Single-GB200 GeDES public artifact FatTree48 TCP/DCTCP workload with seeded inputs',
               'revision': audit.UPSTREAM_REVISION, 'parameters': PARAMETERS, 'seed': 41,
               'expected': {'hosts': HOSTS, 'switches': 5 * K ** 2 // 4,
                            'directed_channels': 6 * HOSTS, 'channel_blocks': CHANNEL_BLOCKS,
                            'directed_flows': HOSTS, 'timeslot_ns': TIMESLOT_NS},
               'process_seconds_includes_terminal_audit': True,
               'build': 'pending', 'passed': False, 'raw_files': {}}
    flow_path = output / 'flows.csv'
    log = output / 'simulation.txt'
    raw_paths = [flow_path, output / 'flows.csv.connections.csv', output / 'flows.csv.queues.csv', log]
    try:
        if helpers.GEDES[1] != audit.UPSTREAM_REVISION:
            raise ValueError('unexpected GeDES source pin')
        capability, env = helpers.selected_gpu(output)
        gpu = json.loads((output / 'gpu.json').read_text())
        if 'GB200' not in gpu['name'].upper():
            raise ValueError(f"FatTree48 requires a GB200, selected {gpu['name']}")
        helpers.command(['nvidia-smi', '--id=' + gpu['uuid'], '--query-gpu=memory.free',
                         '--format=csv,noheader,nounits'], output / 'gpu-free-mib.txt', env=env)
        summary['gpu'] = {**gpu, 'free_mib_before_build': int((output / 'gpu-free-mib.txt').read_text())}
        for tool in ('nvcc', 'cmake', 'c++'):
            helpers.command([tool, '--version'], output / f'{tool}.txt', env=env)
        summary['build'] = 'failed'
        with TemporaryDirectory(prefix='.gedes-fat-tree48-', dir=output) as temporary:
            build_start = time.monotonic()
            binary = drain.build_audit(source, output, helpers, audit, Path(temporary), capability, env)
            summary['build_setup_seconds'] = time.monotonic() - build_start
            summary['build'] = 'succeeded'
            args = [str(binary), *[f'--{key}={value}' for key, value in PARAMETERS.items()], f'--output={flow_path}']
            summary['process_seconds'] = helpers.command(args, log, env=env, cwd=output)
        measurements, drain_seconds, cuda_errors = _run_records(log, audit)
        summary.update(measurements=measurements, terminal_audit_seconds=drain_seconds,
                       virtual_round_span_ns=measurements['rounds'] * TIMESLOT_NS,
                       cuda_diagnostics=cuda_errors)
        flows = audit.flow_statistics(flow_path, HOSTS)
        summary['flows'] = flows
        failures = audit.validate_run(measurements, flows, detail=False)
        if measurements['total_nodes'] != HOSTS or measurements['total_channel_blocks'] != CHANNEL_BLOCKS:
            failures.append('upstream node or channel block count differs from FatTree48')
        if measurements['rounds'] >= measurements['max_rounds']:
            failures.append('GeDES reached the virtual round cap')
        if cuda_errors:
            failures.append('CUDA or packet-pool errors were reported')
        summary['gate_failures'] = failures
        _validate_flows(flow_path)
        snapshot = drain.inspect_snapshot(flow_path, K)
        summary['snapshot'] = snapshot
        failures.extend(drain.drain_failures(snapshot))
        if failures:
            raise RuntimeError('; '.join(failures))
        summary['passed'] = True
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        tails = [summary['error']]
        for name in ('git-fetch.log', 'configure.log', 'build.log', log.name):
            path = output / name
            if path.is_file():
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    tails.append(f'{name} tail:\n' + stream.read().decode(errors='replace'))
        (output / 'failure.txt').write_text('\n'.join(tails) + '\n')
        raise
    finally:
        summary['raw_files'] = {path.name: _file_record(path) for path in raw_paths if path.is_file()}
        helpers.write_json(output / 'summary.json', summary)
