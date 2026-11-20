"""Compare native IKET session startup before and after CUDA Graph instantiation."""

import hashlib
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory


MODES = ('direct', 'graph-late', 'graph-early')
GRAPH_ERROR = 'IKET could not update CUDA Graph profiling state.'
REFERENCE_DRIVER = '580.105.08'
REFERENCE_NVCC = 'V13.0.48'


def validate_trace(path, mode):
    trace = json.loads(path.read_text())
    flat = trace.get('launches', [])
    graphs = trace.get('graphLaunches', {})
    if not isinstance(flat, list) or not isinstance(graphs, dict):
        raise ValueError('invalid launch tables')
    launches = [('', launch) for launch in flat]
    for key, group in graphs.items():
        if not key or not isinstance(group, list):
            raise ValueError('invalid graph launch group')
        launches.extend((key, launch) for launch in group)
    if len(launches) != 1:
        raise ValueError(f'expected exactly one launch, got {len(launches)}')
    graph_key, launch = launches[0]
    dimensions = ('gridDimX', 'gridDimY', 'gridDimZ', 'blockDimX', 'blockDimY', 'blockDimZ')
    if (mode == 'direct' and graph_key or launch['kernelName'] != 'iket_probe_kernel'
            or tuple(launch[field] for field in dimensions) != (1, 1, 1, 32, 1, 1)):
        raise ValueError('unexpected probe launch identity or geometry')
    intervals = launch['warpLifetimes']
    if not isinstance(intervals, list) or len(intervals) != 1:
        raise ValueError('expected one warp lifetime')
    interval = intervals[0]
    locations = trace['locationTable']
    index = interval['locIdx']
    if type(index) is not int or not 0 <= index < len(locations):
        raise ValueError('invalid warp location reference')
    location = locations[index]
    sm, start, end = location['smId'], interval['startTs'], interval['endTs']
    if (location['ctaId'] != [0, 0, 0] or location['warpId'] != 0
            or any(type(value) is not int for value in (sm, start, end))
            or sm < 0 or not 0 <= start <= end):
        raise ValueError('invalid CTA, warp, SM or timestamp interval')
    return {'warp_lifetimes': 1, 'graph_key': graph_key, 'sm_id': sm,
            'start_ns': start, 'end_ns': end}


def outcome(cases):
    if cases['direct']['state'] == cases['graph-early']['state'] == 'valid':
        if cases['graph-late']['state'] == 'graph-state-abort':
            return 'timing-sensitive'
        if cases['graph-late']['state'] == 'valid':
            return 'not-reproduced'
    return 'inconclusive'


def _run_case(mode, binary, output, env, handlers):
    log = output / f'{mode}.txt'
    raw = output / f'{mode}.json'
    row = {'log': log.name}
    try:
        handlers.command([binary, mode, raw], log, cwd=binary.parent, env=env)
    except RuntimeError as error:
        command = json.loads((output / 'commands.json').read_text())[-1]
        row.update(returncode=command['returncode'], error=str(error))
        graph_abort = (command['returncode'] == -signal.SIGABRT and not raw.exists()
                       and GRAPH_ERROR in log.read_text(errors='replace'))
        row['state'] = 'graph-state-abort' if graph_abort else 'failed'
    else:
        row['returncode'] = 0
        try:
            row.update(state='valid', lifetime=validate_trace(raw, mode))
        except (AttributeError, KeyError, OSError, TypeError, ValueError) as error:
            row.update(state='failed', error=f'invalid IKET trace: {error}')
    if raw.is_file():
        row['trace'] = raw.name
    return row


def run(source, output, handlers):
    summary = {
        'scope': 'Native IKET C ABI, one SM100 GPU, one kernel with one 32-thread CTA',
        'comparison': 'Both graph arms add one kernel node before IKET startup. Only graph '
                      'instantiation moves across IKET load and session Start.',
        'reference_environment': {'driver': REFERENCE_DRIVER, 'nvcc': REFERENCE_NVCC},
        'cases': {},
        'state': 'failed',
    }
    try:
        capability, env = handlers.selected_gpu(output)
        if capability != '10.0':
            raise RuntimeError(f'IKET initialization probe requires SM100, observed {capability}')
        for name in ('CUDA_INJECTION64_PATH', 'NVTX_INJECTION64_PATH', 'IKET_INJECTION_CONFIG'):
            env.pop(name, None)
        env['CUDA_LOG_FILE'] = 'stderr'
        nvcc = env.get('NVCC', 'nvcc')
        version = output / 'nvcc.txt'
        handlers.command([nvcc, '--version'], version, env=env)
        driver = json.loads((output / 'gpu.json').read_text())['driver']
        summary['reference_environment_matches'] = (
            driver == REFERENCE_DRIVER and REFERENCE_NVCC in version.read_text().split())
        with TemporaryDirectory(prefix='.iket-probe-', dir=output) as temp:
            scratch = Path(temp)
            warp = handlers._support_module(source, 'warp_ft48')
            root, iket, env = warp.prepare_library(scratch, env)
            handlers.write_json(output / 'warp-dependencies.json', {'iket': iket})
            binary = scratch / 'iket-probe'
            handlers.command([nvcc, '-std=c++17', '-arch=sm_100', '-I', root / 'include',
                              source / 'profiling/warp_trace/iket_probe.cu', '-o', binary, '-ldl'],
                             output / 'build.txt', env=env)
            summary['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
            for mode in MODES:
                summary['cases'][mode] = _run_case(mode, binary, output, env, handlers)
        summary['outcome'] = outcome(summary['cases'])
        if summary['outcome'] == 'inconclusive':
            raise RuntimeError('IKET initialization comparison inconclusive; see summary.json and case logs')
        summary['state'] = 'succeeded'
    except Exception as error:
        summary['error'] = str(error)
        raise
    finally:
        handlers.write_json(output / 'summary.json', summary)
