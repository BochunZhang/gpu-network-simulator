"""Validate a clean-room NSX-inspired minimum-enqueue-time fixture."""

import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory


VARIANTS = ('global', 'local', 'inclusive-probe')
LEDGER_COLUMNS = ('owner', 'index', 'time_ns', 'source', 'sequence', 'bytes')
PROGRESS_COLUMNS = ('wave', 'forward0', 'forward1', 'sink0', 'sink1',
                    'frontier0_ns', 'source0_before', 'source1_after')
FINAL = {'sources_closed': 4, 'links_closed': 2, 'pending_inputs': 0,
         'pending_links': 0, 'overflows': 0}
INF = 1 << 30
EXPECTED_PROGRESS = {
    'global': [(0, 0, 2, 0, 2, 3, 0, 2),
               (1, 1, 3, 1, 3, 5, 1, 2),
               (2, 4, 4, 4, 4, INF, 1, 0)],
    'local': [(0, 0, 4, 0, 4, 3, 0, 2),
              (1, 1, 4, 1, 4, 5, 1, 2),
              (2, 4, 4, 4, 4, INF, 1, 0)],
    'inclusive-probe': [(0, 0, 4, 0, 4, 3, 0, 2),
                        (1, 2, 4, 1, 4, 5, 1, 1),
                        (2, 4, 4, 4, 4, INF, 1, 0)],
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_int_rows(path, columns):
    with path.open(newline='', encoding='utf-8') as stream:
        table = csv.reader(stream)
        if next(table, []) != list(columns):
            raise RuntimeError(f'{path.name}: invalid columns')
        rows = [tuple(int(value) for value in row) for row in table]
    if any(len(row) != len(columns) for row in rows):
        raise RuntimeError(f'{path.name}: invalid row width')
    return rows


def read_terminal(log):
    records = {}
    for line in log.read_text(encoding='utf-8').splitlines():
        if not line.startswith('record=nsx_met '):
            continue
        fields = dict(part.split('=', 1) for part in line.split()[1:])
        variant = fields.pop('variant')
        if variant not in VARIANTS or variant in records:
            raise RuntimeError(f'duplicate or unexpected terminal record for {variant}')
        records[variant] = {key: int(value) for key, value in fields.items()}
    if set(records) != set(VARIANTS):
        raise RuntimeError('missing CUDA terminal record')
    return records


def expected_ledger(inclusive=False):
    first = [(3, 0, 0, 200), (5, 0, 1, 201), (5, 1, 0, 210), (6, 1, 1, 211)]
    second = [(2, 2, 0, 220), (2, 3, 0, 230), (4, 2, 1, 221), (5, 3, 1, 231)]
    if inclusive:
        first[1], first[2] = first[2], first[1]
    arrival0 = [(t + 2, source, sequence, size) for t, source, sequence, size in first]
    arrival1 = [(t + 7, source, sequence, size) for t, source, sequence, size in second]
    return [(owner, index, *packet)
            for owner, packets in enumerate((first, second, arrival0, arrival1))
            for index, packet in enumerate(packets)]


def inspect_results(output):
    cpu_path = output / 'cpu.csv'
    cpu = read_int_rows(cpu_path, LEDGER_COLUMNS)
    gates = []
    if cpu != expected_ledger():
        gates.append('CPU serial reference does not match the fixture ledger')
    terminal = read_terminal(output / 'simulation.txt')
    reports = {}
    for variant in VARIANTS:
        ledger_path = output / f'{variant}.csv'
        progress_path = output / f'{variant}-progress.csv'
        ledger = read_int_rows(ledger_path, LEDGER_COLUMNS)
        progress = read_int_rows(progress_path, PROGRESS_COLUMNS)
        same_events = ledger == cpu
        rejected_at_equal_bound = ledger == expected_ledger(inclusive=True) and not same_events
        drained = terminal[variant] == FINAL
        if not drained:
            gates.append(f'{variant} did not drain all input and link queues')
        if progress != EXPECTED_PROGRESS[variant]:
            gates.append(f'{variant} had unexpected early progress or boundary behavior')
        if variant == 'inclusive-probe' and not rejected_at_equal_bound:
            gates.append('inclusive bound negative control did not produce the expected tie-order violation')
        if variant != 'inclusive-probe' and not same_events:
            gates.append(f'{variant} event ledger differs from the CPU reference')
        reports[variant] = {
            'ledger_sha256': sha256(ledger_path),
            'committed_events': len(ledger),
            'matches_cpu_ledger': same_events,
            'intentional_bound_violation_detected': rejected_at_equal_bound,
            'terminal': terminal[variant],
            'drained': drained,
            'wave_progress': [dict(zip(PROGRESS_COLUMNS, row)) for row in progress],
        }
    return {'cpu_ledger_sha256': sha256(cpu_path), 'variants': reports,
            'gate_failures': gates, 'passed': not gates}


def run(source: Path, output: Path, helpers):
    capability, env = helpers.selected_gpu(output)
    architecture = capability.replace('.', '')
    program = source / 'benchmarks' / 'nsx_single_gpu.cu'
    helpers.command(['nvcc', '--version'], output / 'nvcc.txt', env=env)
    with TemporaryDirectory(prefix='.nsx-single-gpu-', dir=output) as temp:
        binary = Path(temp) / 'nsx-single-gpu'
        build_seconds = helpers.command(
            ['nvcc', '-O3', '-std=c++17', '-lineinfo',
             '-gencode', f'arch=compute_{architecture},code=sm_{architecture}',
             str(program), '-o', str(binary)],
            output / 'build.txt', env=env)
        binary_sha256 = sha256(binary)
        process_seconds = helpers.command([binary, output], output / 'simulation.txt', env=env)
    try:
        summary = inspect_results(output)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        summary = {'passed': False, 'gate_failures': [f'invalid fixture output: {error}']}
    summary.update(
        scope='NSX-inspired local minimum-enqueue-time correctness on a fixed acyclic packet fixture',
        limits='one GPU; two forwarding owners; two sink owners; fixed link delays; no protocol or throughput claims',
        order='integer time_ns, source, sequence; strict future timestamp bound',
        waves=3, closed_source_frontier_ns=INF,
        build_architecture=f'sm_{architecture}', program_sha256=sha256(program),
        binary_sha256=binary_sha256, build_seconds=build_seconds, process_seconds=process_seconds,
    )
    helpers.write_json(output / 'summary.json', summary)
    if not summary['passed']:
        raise RuntimeError('NSX-inspired single-GPU fixture failed; see summary.json')
