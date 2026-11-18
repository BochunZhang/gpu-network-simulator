"""Verify an NSX-inspired event exchange fixture on four CUDA devices."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


DEVICES = 4
PACKETS = 192
HOPS = 4
WINDOW_NS = 10
POSITIVE_RUNS = ('single-gpu', 'four-gpu-staging', 'four-gpu-p2p')
NEGATIVE_RUN = 'negative-drop'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_reference(output):
    """Check unique, complete packet paths and stable per-owner ordering."""
    rows = set()
    last = [(-1, -1, -1)] * DEVICES
    equal_time_pairs = 0
    cross_equal_time_pairs = 0
    cross_window_boundaries = 0
    with (output / 'serial.csv').open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            owner, arrival, packet, hop, size, departure = (
                int(row[key]) for key in ('owner', 'arrival_ns', 'packet', 'hop', 'bytes', 'departure_ns'))
            key = (arrival, packet, hop)
            if (owner not in range(DEVICES) or packet not in range(PACKETS) or hop not in range(HOPS)
                    or arrival < 0 or size <= 0 or departure <= arrival
                    or (packet, hop) in rows or key <= last[owner]):
                raise RuntimeError('serial event ledger has invalid ownership, causality or duplicate delivery')
            # Fixture transitions 0→1 and 2→3 cross owners, while 1→2 is local.
            if arrival == last[owner][0]:
                equal_time_pairs += 1
                cross_equal_time_pairs += hop in (1, 3) or last[owner][2] in (1, 3)
            cross_window_boundaries += hop in (1, 3) and arrival % WINDOW_NS == 0
            last[owner] = key
            rows.add((packet, hop))
    if len(rows) != PACKETS * HOPS or not cross_equal_time_pairs or not cross_window_boundaries:
        raise RuntimeError('serial reference must cover every packet hop, cross-owner ties and boundary arrivals')
    return {'equal_time_pairs': equal_time_pairs,
            'cross_equal_time_pairs': cross_equal_time_pairs,
            'cross_window_boundary_arrivals': cross_window_boundaries}


def validate_result(output, uuids, report):
    coverage = inspect_reference(output)
    if (report['gpu_uuids'] != uuids or len(set(uuids)) != DEVICES
            or report['window_ns'] != WINDOW_NS or report['packet_count'] != PACKETS
            or report['events_per_packet'] != HOPS or report['reference_committed_events'] != PACKETS * HOPS
            or report['reference_equal_time_pairs'] != coverage['equal_time_pairs']):
        raise RuntimeError('unexpected device mapping or reference workload')
    event_bytes = report['event_record_bytes']
    if type(event_bytes) is not int or event_bytes <= 0:
        raise RuntimeError('invalid CUDA event record size')
    if set(report['runs']) != {*POSITIVE_RUNS, NEGATIVE_RUN}:
        raise RuntimeError('event exchange report does not contain every required variant')
    reference_digest = sha256(output / 'serial.csv')
    digests = {name: sha256(output / f'{name}.csv') for name in (*POSITIVE_RUNS, NEGATIVE_RUN)}
    baseline = report['runs']['single-gpu']
    for name in POSITIVE_RUNS:
        run = report['runs'][name]
        matrix = run['events_by_source_and_destination']
        if (len(matrix) != DEVICES or any(len(row) != DEVICES for row in matrix)
                or any(type(count) is not int or count <= 0 for row in matrix for count in row)
                or sum(map(sum, matrix)) != PACKETS * (HOPS - 1)
                or matrix != baseline['events_by_source_and_destination']):
            raise RuntimeError(f'{name}: invalid or incomplete directed event matrix')
        cross_events = sum(matrix[src][dst] for src in range(DEVICES) for dst in range(DEVICES) if src != dst)
        if (run['drained'] is not True or run['remaining_events'] != 0
                or run['ledger_match'] is not True or digests[name] != reference_digest
                or run['exchange_counts_match'] is not True or run['injected_drops'] != 0
                or run['rounds'] != baseline['rounds'] or not 0 < run['rounds'] < 512
                or run['committed_events'] != PACKETS * HOPS):
            raise RuntimeError(f'{name}: event ledger, drain or exactly-once gate failed')
        if name == 'single-gpu':
            if run['cross_batches'] != 0 or run['cross_bytes'] != 0:
                raise RuntimeError('single GPU control reported cross-device exchanges')
        elif run['cross_bytes'] != cross_events * event_bytes or not 12 <= run['cross_batches'] <= cross_events:
            raise RuntimeError(f'{name}: incomplete actual cross-device transfer accounting')
        timings = [run[key] for key in ('setup_seconds', 'loop_seconds', 'processing_and_control_seconds', 'exchange_seconds')]
        if (any(not math.isfinite(value) or value <= 0 for value in timings)
                or sum(timings[2:]) > timings[1] + 1e-8):
            raise RuntimeError(f'{name}: invalid diagnostic wall timing')
    negative = report['runs'][NEGATIVE_RUN]
    if (negative['injected_drops'] != 1 or negative['drained'] is not True
            or negative['remaining_events'] != 0 or negative['ledger_match'] is not False
            or negative['exchange_counts_match'] is not False or digests[NEGATIVE_RUN] == reference_digest
            or not 0 < negative['committed_events'] < PACKETS * HOPS):
        raise RuntimeError('dropped-event control did not demonstrate that drain alone is insufficient')
    return {'serial': reference_digest, **digests}, coverage


def run(source: Path, output: Path, helpers):
    visible = helpers.visible_gpus(helpers.gpu_rows(output / 'gpu-list.csv'))[:DEVICES]
    if len(visible) != DEVICES:
        raise RuntimeError('four GPU event exchange requires four visible GPUs')
    uuids = [row[1] for row in visible]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(uuids)}
    architectures = sorted({row[3].replace('.', '') for row in visible})
    program = source / 'benchmarks' / 'nsx_four_gpu.cu'
    helpers.command(['nvcc', '--version'], output / 'nvcc.txt', env=env)
    helpers.command(['nvidia-smi', 'topo', '-m'], output / 'topology.txt', env=env)
    summary = {
        'scope': 'independent NSX-inspired conservative event exchange on one and four CUDA devices',
        'protocol': 'fixed-window packet forwarding with deterministic egress service',
        'transports': ['single-device copy', 'pinned host staging', 'CUDA peer copy'],
        'transfer_scheduling': 'host staging reuses one pinned buffer serially; peer batches synchronize at each window boundary',
        'timing': 'diagnostic host wall time, including CUDA window dispatch/control and transfer synchronization',
        'gpu_uuids': uuids,
        'build_architectures': architectures,
        'program_sha256': sha256(program),
        'passed': False,
    }
    with tempfile.TemporaryDirectory(prefix='.nsx-four-gpu-', dir=output) as temporary:
        binary = Path(temporary) / 'nsx-four-gpu'
        args = ['nvcc', '-O3', '-std=c++17']
        for arch in architectures:
            args.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])
        try:
            summary['build_seconds'] = helpers.command([*args, str(program), '-o', str(binary)],
                                                       output / 'build.txt', env=env)
            summary['binary_sha256'] = sha256(binary)
            summary['process_seconds'] = helpers.command([str(binary), str(output.resolve())],
                                                         output / 'event-exchange.txt', env=env)
            report = json.loads((output / 'event-exchange.json').read_text(encoding='utf-8'))
            summary['event_exchange'] = report
            summary['ledger_sha256'], summary['reference_coverage'] = validate_result(output, uuids, report)
            summary['passed'] = True
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
            summary['error'] = str(error)
            helpers.write_json(output / 'summary.json', summary)
            raise
    helpers.write_json(output / 'summary.json', summary)
