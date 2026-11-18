"""Four-device CUDA memory-copy and directed peer-copy baseline."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SIZES = (1024**2, 64 * 1024**2)
DEVICES = 4
ITERATIONS = 10
WARMUP = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_results(output):
    context = json.loads((output / 'cuda-context.json').read_text(encoding='utf-8'))
    devices = context.get('devices')
    if (context.get('selected_device_count') != DEVICES
            or type(context.get('visible_device_count')) is not int or context['visible_device_count'] < DEVICES
            or not isinstance(devices, list) or len(devices) != DEVICES):
        raise RuntimeError('CUDA context does not describe four selected visible devices')
    uuids = {}
    for item in devices:
        if not isinstance(item, dict):
            raise RuntimeError('CUDA device identity must be an object')
        index = item.get('logical_index')
        if (type(index) is not int or index not in range(DEVICES) or index in uuids
                or not isinstance(item.get('uuid'), str) or not item['uuid'].startswith('GPU-')
                or type(item.get('total_memory_bytes')) is not int or item['total_memory_bytes'] <= 0):
            raise RuntimeError('CUDA context has invalid logical-device identity')
        uuids[index] = item['uuid']
    if len(set(uuids.values())) != DEVICES:
        raise RuntimeError('CUDA context contains duplicate GPU identities')
    with (output / 'transfer-matrix.csv').open(newline='', encoding='utf-8') as stream:
        records = list(csv.DictReader(stream))
    expected = {(source, destination, size) for size in SIZES
                for source in range(DEVICES) for destination in range(DEVICES)}
    found = set()
    rows = []
    unsupported = []
    for record in records:
        try:
            source, destination, size = (int(record[key]) for key in ('src', 'dst', 'bytes'))
            key = source, destination, size
            if key not in expected or key in found:
                raise ValueError('unexpected or duplicate transfer row')
            found.add(key)
            kind = 'd2d' if source == destination else 'p2p'
            if (record['kind'] != kind or int(record['iterations']) != ITERATIONS
                    or int(record['warmup']) != WARMUP
                    or record['src_uuid'] != uuids[source] or record['dst_uuid'] != uuids[destination]):
                raise ValueError('transfer row metadata disagrees with its device context')
            row = {'kind': kind, 'src': source, 'dst': destination, 'bytes': size,
                   'iterations': ITERATIONS, 'warmup': WARMUP, 'peer_access': int(record['peer_access']),
                   'cuda_ms': float(record['cuda_ms']), 'wall_ms': float(record['wall_ms']),
                   'gb_per_s': float(record['gb_per_s']), 'mismatched_bytes': int(record['mismatched_bytes']),
                   'src_uuid': uuids[source], 'dst_uuid': uuids[destination]}
            if row['peer_access'] == 0 and kind == 'p2p':
                unsupported.append({'src': source, 'dst': destination, 'bytes': size})
            elif row['peer_access'] != 1:
                raise ValueError('invalid device-copy capability')
            else:
                if (any(not math.isfinite(row[key]) or row[key] <= 0 for key in ('cuda_ms', 'wall_ms', 'gb_per_s'))
                        or row['mismatched_bytes'] != 0
                        or not math.isclose(row['gb_per_s'], size * ITERATIONS / (row['cuda_ms'] * 1e6), rel_tol=1e-6)):
                    raise ValueError('copy timing, payload rate or full-byte validation failed')
            rows.append(row)
        except (KeyError, ValueError, TypeError) as error:
            raise RuntimeError(f'invalid GPU transfer output: {error}') from error
    if found != expected:
        raise RuntimeError(f'transfer matrix is incomplete: expected {len(expected)} rows, got {len(found)}')
    return {'scope': 'CUDA memory-copy and peer-copy payload throughput on four visible GPUs',
            'timing': 'destination-stream CUDA events after warmup; all timed copies completed before verification',
            'bandwidth_unit': 'decimal GB/s of payload; device-to-device copies also read and write device memory',
            'interconnect_attribution': 'CUDA peer capability and topology are recorded; physical-link attribution requires traffic counters',
            'sizes_bytes': list(SIZES), 'iterations': ITERATIONS, 'warmup': WARMUP,
            'selected_device_count': DEVICES, 'context': context, 'rows': rows,
            'unsupported_pairs': unsupported, 'all_copies_verified': not unsupported}


def save_failure(output, log_name, error):
    path = output / log_name
    tail = path.read_text(encoding='utf-8', errors='replace')[-65536:] if path.is_file() else ''
    (output / 'failure.txt').write_text(f'{error}\n\n{tail}', encoding='utf-8')


def run(source: Path, output: Path, helpers):
    """Build the pinned task source and measure the first four CUDA-visible GPUs."""
    physical = helpers.gpu_rows(output / 'gpu-list.csv')
    visible = helpers.visible_gpus(physical)
    if len(visible) < DEVICES:
        raise ValueError('GPU transfer task requires at least four visible GPUs')
    env = dict(os.environ)
    helpers.command(['nvcc', '--version'], output / 'nvcc.txt', env=env)
    helpers.command(['nvidia-smi', 'topo', '-m'], output / 'topology.txt', env=env)
    program = source / 'benchmarks' / 'gpu_transfer.cu'
    program_sha = digest(program)
    architectures = sorted({row[3].replace('.', '') for row in visible[:DEVICES]})
    helpers.write_json(output / 'benchmark.json', {
        'program_sha256': program_sha, 'sizes_bytes': list(SIZES),
        'iterations': ITERATIONS, 'warmup': WARMUP,
        'build_architectures': architectures,
        'selected_physical_uuids': [row[1] for row in visible[:DEVICES]],
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'device_buffer_bytes_per_gpu': 2 * max(SIZES),
    })
    with tempfile.TemporaryDirectory(prefix='.gpu-transfer-', dir=output) as temporary:
        binary = Path(temporary) / 'gpu-transfer'
        args = ['nvcc', '-O3', '-std=c++17', '-lineinfo']
        for architecture in architectures:
            args.extend(['-gencode', f'arch=compute_{architecture},code=sm_{architecture}'])
        try:
            build_seconds = helpers.command([*args, str(program), '-o', str(binary)], output / 'build.log', env=env)
        except RuntimeError as error:
            save_failure(output, 'build.log', error)
            raise
        binary_sha = digest(binary)
        run_error = None
        try:
            run_seconds = helpers.command([str(binary), f'--output-dir={output}'], output / 'transfer.log', env=env)
        except RuntimeError as error:
            run_error = error
            run_seconds = None
            save_failure(output, 'transfer.log', error)
        try:
            summary = parse_results(output)
        except (OSError, ValueError, RuntimeError) as error:
            save_failure(output, 'transfer.log', error)
            raise
        expected_uuids = [row[1] for row in visible[:DEVICES]]
        actual_uuids = [item['uuid'] for item in sorted(summary['context']['devices'], key=lambda item: item['logical_index'])]
        if actual_uuids != expected_uuids:
            raise RuntimeError('CUDA logical GPUs disagree with the first four visible physical GPU identities')
        summary.update(program_sha256=program_sha, binary_sha256=binary_sha,
                       build_seconds=build_seconds, process_seconds=run_seconds,
                       build_architectures=architectures,
                       cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),
                       selected_physical_uuids=expected_uuids)
        helpers.write_json(output / 'summary.json', summary)
        if run_error is not None:
            raise run_error
        if not summary['all_copies_verified']:
            raise RuntimeError('CUDA peer access is unavailable for one or more directed GPU pairs')
