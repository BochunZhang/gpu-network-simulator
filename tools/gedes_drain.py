"""Audit the terminal state of the fixed single-GPU GeDES FatTree fixture."""

import csv
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory


REFERENCE_CSV = {
    4: '192d99d8197735f95efbdac8001b7bfcdc348e35c0d12b1960f0e5d912f53555',
    8: 'a799805f97243cadd88153efecff0152bc9bfaa6ef567868f54e0cd2e550b708',
}
REFERENCE_INPUTS = {
    4: 'a1f8426191feb8fc0806ef90938b827e7b910787920c9cdce71cc042f5812db3',
    8: 'd4ff4f1f66f1c05a28bf107fea7a2c11641381a6ba56cad111a13e69912b2571',
}
QUEUE_COUNTS = {
    'tcp_ingress': 1, 'tcp_egress': 1,
    'ipv4_ingress': 1, 'ipv4_egress': 1, 'ipv4_local': 1, 'ipv4_error': 1,
    'host_nic_ingress': 1, 'host_nic_egress': 1,
    'switch_nic_ingress': 5, 'switch_nic_egress': 5,
    'channel_ingress': 6, 'channel_egress': 6,
}
SCOPE = {
    'fixture': 'Pinned single-GPU FatTree, one generated flow per directed TCP connection',
    'boundary': 'Final CUDA graph after cudaDeviceSynchronize. Both the FCT CSV and terminal snapshot use read-only device copies.',
    'channel_queues': 'Directional channel endpoints alias host or switch NIC queues. Frames in flight reside in timestamped endpoint queues.',
    'exclusions': ['Free and recycled object pools, routing tables, transmitted-flow history, and idle timer state',
                   'Disabled frame cache and unused TCP cache pointers',
                   'Historical drops and retransmissions, payload contents, CPU equivalence, and other protocol modes'],
}


def _replace(text, anchor, replacement):
    if text.count(anchor) != 1:
        raise ValueError(f'unexpected pinned GeDES source anchor: {anchor.strip()}')
    return text.replace(anchor, replacement, 1)


def instrument_sources(src, output, v3, drain_source):
    include = src / 'code/c++/include'
    topology = src / 'code/c++/src/components/topology.cc'
    main = src / 'main.cc'
    with TemporaryDirectory(prefix='.completion-headers-', dir=output) as temporary:
        v3.patch_completion_headers(src, Path(temporary))
    topology_header = include / 'topology.h'
    declaration = '    bool IsFinished();\n'
    topology_header.write_text(_replace(topology_header.read_text(), declaration,
                                        declaration + '    void WriteDrainAudit(const std::string &flow_output) const;\n'))
    patched = v3.fix_flow_start_queue(v3.seed_workload(topology.read_text()))
    patched = _replace(patched, '#include <ctime>', '#include <ctime>\n#include <stdexcept>')
    anchor = '  bool Topology::IsFinished()\n'
    topology.write_text(_replace(patched, anchor, drain_source + '\n' + anchor))

    patched = v3.instrument_main(main.read_text())
    patched = _replace(patched, '#include <iomanip>', '#include <iomanip>\n#include <exception>')
    patched = _replace(patched, '    bool audit_detail = false;\n',
                       '    bool audit_detail = false;\n    bool audit_early_stop = false;\n')
    anchor = '            else if (arg == "--audit_detail=1")\n'
    patched = _replace(patched, anchor, '''            else if (arg == "--audit_early_stop=1")
            {
                audit_early_stop = true;
            }
''' + anchor)
    patched = _replace(patched, 'i < audit_max_rounds && !audit_finished()',
                       'i < audit_max_rounds && (!audit_early_stop || i < 1) && !audit_finished()')
    anchor = '    return 0;\n}'
    patched = _replace(patched, anchor, '''    const auto drain_start = AuditClock::now();
    try {
        topology.WriteDrainAudit(output_file_name);
    } catch (const std::exception &error) {
        std::cerr << "GeDES terminal snapshot failed: " << error.what() << std::endl;
        return 1;
    }
    std::cout << "record=gedes_drain seconds="
              << std::chrono::duration<double>(AuditClock::now() - drain_start).count()
              << std::endl;
''' + anchor)
    main.write_text(patched)
    cmake = src / 'CMakeLists.txt'
    cmake.write_text(v3.patch_cmake(cmake.read_text()))

    files = (main, topology, topology_header, include / 'vdes_timer.h', cmake)
    for path in files:
        (output / f'{path.name}-drain.txt').write_bytes(path.read_bytes())
    return {path.relative_to(src).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def csv_dict(path, id_fields):
    with path.open(newline='', encoding='utf-8') as stream:
        rows = {}
        for row in csv.DictReader(stream):
            key = tuple(row[field] for field in id_fields)
            if key in rows or None in row or None in row.values():
                raise ValueError(f'malformed or duplicate row in {path.name}: {key}')
            rows[key] = row
    return rows


def inspect_queues(path, node_count):
    counts = {group: {'queues': 0, 'entries': 0, 'nonempty': 0} for group in QUEUE_COUNTS}
    seen = {group: bytearray(node_count * multiple) for group, multiple in QUEUE_COUNTS.items()}
    with path.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if None in row or None in row.values():
                raise ValueError('malformed GeDES queue occupancy row')
            group = row['group']
            index = int(row['index'])
            if (group not in seen or not 0 <= index < len(seen[group]) or str(index) != row['index']
                    or seen[group][index]):
                raise ValueError('queue inventory does not match the fixed FatTree topology')
            entries = int(row['entries'])
            if entries < 0:
                raise ValueError('negative terminal queue occupancy')
            seen[group][index] = 1
            counts[group]['queues'] += 1
            counts[group]['entries'] += entries
            counts[group]['nonempty'] += entries > 0
    if any(counts[group]['queues'] != len(indices) for group, indices in seen.items()):
        raise ValueError('queue inventory does not match the fixed FatTree topology')
    return counts


def inspect_snapshot(flow_path, k):
    node_count = k ** 3 // 4
    connections = csv_dict(Path(str(flow_path) + '.connections.csv'), ('node',))
    flows = csv_dict(flow_path, ('source_ip', 'dst_ip'))
    if set(connections) != {(str(index),) for index in range(node_count)}:
        raise ValueError('connection inventory does not cover every simulated host')
    queue_counts = inspect_queues(Path(str(flow_path) + '.queues.csv'), node_count)

    pairs = {(row['source_ip'], row['dst_ip']): row for row in connections.values()}
    if (set(pairs) != set(flows) or len(pairs) != node_count
            or len({src for src, _ in pairs}) != node_count
            or any(src == dst or (dst, src) not in pairs for src, dst in pairs)):
        raise ValueError('connections and reported flows do not cover the same directed host pairs')

    flow_failures, transport_failures = [], []
    totals = dict(generated_bytes=0, planned_bytes=0, acked_bytes=0,
                  pending_flows=0, acked_flows=0)
    for pair, row in pairs.items():
        state = {key: int(value) for key, value in row.items()}
        reported = {key: int(value) for key, value in flows[pair].items()}
        size, start = state['generated_bytes'], state['generated_start_ns']
        end = state['device_end_ns']
        if (size <= 0 or start < 0 or (size, start) != (state['device_bytes'], state['device_start_ns'])
                or (size, start, end) != (reported['flow_size'], reported['start_timestamp'],
                                           reported['completed_timestamp'])
                or reported['flow_completion_time'] != end - start):
            raise ValueError(f'generated, device, and reported flow records disagree: {pair}')
        if (end <= start or state['device_byte_offset'] != size
                or state['pending_flows'] != 0 or state['acked_flows'] != 1):
            flow_failures.append(state['node'])
        peer_bytes = int(pairs[pair[::-1]]['generated_bytes'])
        if (not size == state['planned_bytes'] == state['acked_bytes'] == state['snd'] == state['una']
                or state['expected_seq'] != peer_bytes
                or any(state[key] != 0 for key in ('record_count', 'has_records', 'unacked_packets'))):
            transport_failures.append(state['node'])
        for key in totals:
            totals[key] += state[key]

    return {'generated_flows': len(pairs), **totals,
            'flow_failures': flow_failures, 'transport_failures': transport_failures,
            'queues': queue_counts}


def drain_failures(snapshot):
    failures = []
    if snapshot['flow_failures']:
        failures.append('generated flows remain incomplete')
    if snapshot['transport_failures']:
        failures.append('TCP byte or receive/ACK state remains incomplete')
    for group, counts in snapshot['queues'].items():
        if counts['entries']:
            failures.append(f'{group} contains {counts["entries"]} entries')
    return failures


def _case(binary, case, k, early_stop, output, env, v3, helpers, prefix=()):
    folder = output / case
    folder.mkdir()
    flow_path = folder / 'flows.csv'
    args = [*prefix, str(binary), f'--ft_k={k}', *[f'--{key}={value}' for key, value in v3.PARAMETERS.items()],
            f'--output={flow_path}']
    if early_stop:
        args.append('--audit_early_stop=1')
    process_seconds = helpers.command(args, folder / 'simulation.txt', env=env, cwd=folder, journal_dir=output)
    measurements = v3.parse_audit((folder / 'simulation.txt').read_text())
    flows = v3.flow_statistics(flow_path, k ** 3 // 4)
    snapshot = inspect_snapshot(flow_path, k)
    independent = drain_failures(snapshot)
    upstream = v3.validate_run(measurements, flows, detail=False)
    expected_nodes = snapshot['generated_flows']
    expected_blocks = 6 * expected_nodes // 32
    if early_stop:
        failures = []
        if (measurements['rounds'] != 1 or measurements['upstream_finished'] != 0
                or not 0 <= measurements['completed_nodes'] < expected_nodes):
            failures.append('negative fixture did not stop with incomplete upstream state after one round')
        if snapshot['pending_flows'] <= 0 or not independent:
            failures.append('negative fixture did not expose pending flow state to the independent gate')
    else:
        failures = upstream + independent
        if flows['csv_sha256'] != REFERENCE_CSV[k]:
            failures.append('flow CSV SHA-256 differs from task 001 v3')
    if (measurements['total_nodes'] != expected_nodes or measurements['total_channel_blocks'] != expected_blocks
            or not 0 <= measurements['completed_channel_blocks'] <= expected_blocks):
        failures.append('upstream completion counts do not match the fixed topology')
    if flows['reported_input_sha256'] != REFERENCE_INPUTS[k]:
        failures.append('generated flow input SHA-256 differs from task 001 v3')
    return {'ft_k': k, 'early_stop': early_stop, 'process_seconds': process_seconds,
            'measurements': measurements, 'flows': flows, 'snapshot': snapshot,
            'upstream_gate_failures': upstream, 'drain_gate_failures': independent,
            'passed': not failures, 'failures': failures}


def build_audit(source, output, helpers, v3, scratch, capability, env, patch_main=None):
    src = helpers.checkout(helpers.GEDES, scratch, output, env) / 'GeDES_code'
    audit = v3._source_audit(src)
    drain_source = (source / 'benchmarks/gedes_drain.inc').read_text()
    hashes = instrument_sources(src, output, v3, drain_source)
    if patch_main is not None:
        main = src / 'main.cc'
        main.write_text(patch_main(main.read_text()))
        hashes['main.cc'] = hashlib.sha256(main.read_bytes()).hexdigest()
        (output / 'main.cc-profile.txt').write_bytes(main.read_bytes())
    build = scratch / 'build'
    helpers.command(['cmake', '-S', str(src), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                     '-DCMAKE_CUDA_ARCHITECTURES=' + capability.replace('.', '')],
                    output / 'configure.log', env=env)
    helpers.command(['cmake', '--build', str(build), '--parallel', '4'], output / 'build.log', env=env)
    binary = build / 'GeDES'
    helpers.write_json(output / 'build.json', {
        'revision': audit['revision'], 'upstream_source_sha256': audit['files'],
        'patched_source_sha256': hashes, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
        'compute_capability': capability, 'build_mode': 'Release',
    })
    return binary


def run(source, output, helpers):
    """Build the audit once and run positive controls and a real early-stop case."""
    v3 = helpers._support_module(source, 'gedes_experiments')
    if helpers.GEDES[1] != v3.UPSTREAM_REVISION:
        raise ValueError('GeDES drain audit requires its fixed upstream revision')
    summary = {'scope': SCOPE, 'revision': v3.UPSTREAM_REVISION, 'parameters': v3.PARAMETERS,
               'reference_csv_sha256': REFERENCE_CSV, 'reference_input_sha256': REFERENCE_INPUTS,
               'build': 'failed', 'passed': False, 'cases': {}}
    try:
        capability, env = helpers.selected_gpu(output)
        for tool in ('nvcc', 'cmake', 'c++'):
            helpers.command([tool, '--version'], output / f'{tool}.txt', env=env)
        with TemporaryDirectory(prefix='.gedes-drain-', dir=output) as temporary:
            scratch = Path(temporary)
            binary = build_audit(source, output, helpers, v3, scratch, capability, env)
            summary['build'] = 'succeeded'
            for case, k, early_stop in [('k4', 4, False), ('k8', 8, False), ('k8-early-stop', 8, True)]:
                try:
                    summary['cases'][case] = _case(binary, case, k, early_stop, output, env, v3, helpers)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    summary['cases'][case] = {'ft_k': k, 'early_stop': early_stop, 'passed': False,
                                              'failures': [f'{type(error).__name__}: {error}']}
            for case, result in summary['cases'].items():
                if not result['passed']:
                    folder = output / case
                    log = folder / 'simulation.txt'
                    tail = log.read_text(errors='replace')[-16384:] if log.is_file() else ''
                    (folder / 'failure.txt').write_text('\n'.join(result['failures']) + '\n\n' + tail)
            summary['passed'] = all(case['passed'] for case in summary['cases'].values())
    except Exception as error:
        summary['error'] = f'{type(error).__name__}: {error}'
        feedback = [summary['error']]
        for name in ('git-fetch.log', 'configure.log', 'build.log'):
            log = output / name
            if log.is_file():
                feedback.append(f'\n{name} tail:\n{log.read_text(errors="replace")[-16384:]}')
        (output / 'failure.txt').write_text('\n'.join(feedback) + '\n')
        raise
    finally:
        helpers.write_json(output / 'summary.json', summary)
    failed = [name for name, case in summary['cases'].items() if not case['passed']]
    if failed:
        raise RuntimeError('GeDES drain audit failed gates in ' + ', '.join(failed))
