"""Time and optionally trace the pinned Days AGO k48/h16 30% fixture on one SM100 GPU."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory


WIDE_FIXTURE = Path(
    'configs/benchmarks/width_via_load_k48_h16/fattree_k48_h16_load_30_sustained.toml'
)
SIMULATOR_DIR = Path('simulators/days-ago')
SMOKE_FIXTURE_SHA256 = 'a715aa8e4492583c1727224f161b61324d653382d331d270fb1850863f2a5440'
WIDE_FIXTURE_SHA256 = 'ca005a37bbab9192f85cac61ea2f3fe63c73db8aeba1d72df46096a92950ba7c'
HOST_COUNT = 18_432
FLOW_COUNT = 5_530
STOP_NS = 1_024_000
PACKET_INTERVAL_NS = 21
PACKET_BYTES = 256
EXPECTED_PACKETS = FLOW_COUNT * (STOP_NS // PACKET_INTERVAL_NS + 1)
RESULT_WORK_COUNTERS = ('result_bytes', 'pending_events', 'resident_packets', 'sourced_packets',
                        'departed_packets', 'received_packets', 'dropped_packets')
DEVICE_WORK_COUNTERS = ('rounds', 'transitions', 'continuation_relaunches', 'retry_count',
                        'grown_stream_count', 'same_time_continuations')
RESULT_COUNTERS = ('lowering_ns', 'run_ns', *RESULT_WORK_COUNTERS)
DEVICE_COUNTERS = ('wall_ns', 'device_ns', *DEVICE_WORK_COUNTERS)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _device_record(log):
    rows = [line for line in log.read_text(encoding='utf-8', errors='replace').splitlines()
            if line.startswith('record=days_device ')]
    if len(rows) != 1:
        raise RuntimeError(f'expected one CUDA timing record in {log.name}')
    fields = dict(word.split('=', 1) for word in rows[0].split() if '=' in word)
    metrics = {key: int(fields[key]) for key in DEVICE_COUNTERS}
    if (fields.get('engine') != 'cuda'
            or min(metrics[key] for key in ('wall_ns', 'device_ns', 'rounds', 'transitions')) <= 0
            or min(metrics.values()) < 0):
        raise RuntimeError(f'Days CUDA timing or work gate failed in {log.name}')
    return {**metrics, 'round_kernel': fields['round_kernel'],
            'channel_capacity_distribution': fields['channel_capacity_distribution']}


def _measure(binary, fixture, engine, log, directory, env, handlers, record, *, prefix=()):
    args = [*prefix, binary, fixture, '--engine', engine]
    if engine == 'cpu':
        args.extend(['--workers', '2', '--repetitions', '1'])
    if engine == 'cuda':
        args.extend(['--cuda-device', '0'])
    record['process_seconds'] = handlers.command(args, log, directory, env)
    fields = handlers.days_record(log, engine)
    record.update({key: int(fields[key]) for key in RESULT_COUNTERS})
    record['result_fnv1a64'] = fields['result_fnv1a64']
    if min(record['lowering_ns'], record['run_ns']) <= 0 or min(record[key] for key in RESULT_COUNTERS) < 0:
        raise RuntimeError(f'Days result timing or counters invalid in {log.name}')
    if engine == 'cuda':
        record['device'] = _device_record(log)


def _work_differences(baseline, profiled):
    differences = [key for key in (*RESULT_WORK_COUNTERS, 'result_fnv1a64')
                   if baseline[key] != profiled[key]]
    differences.extend(f'device.{key}'
                       for key in (*DEVICE_WORK_COUNTERS, 'round_kernel', 'channel_capacity_distribution')
                       if baseline['device'][key] != profiled['device'][key])
    return differences


def run(source, output, handlers, profile=False, warp_trace=False):
    wide = {'fixture': WIDE_FIXTURE.as_posix(), 'k': 48, 'hosts_per_edge': 16, 'hosts': HOST_COUNT,
            'flows': FLOW_COUNT, 'inclusive_stop_ns': STOP_NS, 'seed': 17_048, 'packet_bytes': PACKET_BYTES,
            'packet_interval_ns': PACKET_INTERVAL_NS, 'link_rate_bps': 100_000_000_000, 'link_delay_ns': 1_000,
            'expected_sourced_packets': EXPECTED_PACKETS,
            'expected_offered_bytes': EXPECTED_PACKETS * PACKET_BYTES}
    smoke = {'fixture': handlers.DAYS_FIXTURE, 'runs': {}}
    summary = {'upstream': handlers.DAYS[0], 'revision': handlers.DAYS[1],
               'source_path': SIMULATOR_DIR.as_posix(), 'cuda_target': 'sm_100', 'smoke': smoke, 'wide': wide,
               'timing_scope': {
                   'process_seconds': 'Full simulator process including scenario lowering and result fingerprinting',
                   'lowering_ns': 'Scenario lowering',
                   'run_ns': 'Executor run including capacity retries and result readback',
                   'device': 'Final successful attempt only. device_ns sums CUDA event times for graph waves.',
               },
               'stage': 'preflight'}
    build_log = output / 'build.log'
    try:
        capability, env = handlers.selected_gpu(output)
        if profile and warp_trace:
            raise ValueError('Nsight Systems and IKET require separate profiler runs')
        env = dict(env)
        env.pop('FT48_WARP_OUTPUT', None)
        env.pop('FT48_WARP_LIBDIR', None)
        if capability != '10.0':
            raise RuntimeError(f'Days FatTree48 experiment requires SM100, observed {capability}')
        handlers.command([env.get('NVCC', 'nvcc'), '--version'], output / 'nvcc.txt', env=env)
        summary['stage'] = 'rust-toolchain'
        env = handlers.days_rust_environment(source, output, env)
        if profile:
            summary['stage'] = 'nsys-preflight'
            profiler = handlers._support_module(source, 'nsys_ft48')
            nsys = profiler.preflight(output, env, handlers)
        with TemporaryDirectory(prefix='.days-fat-tree48-', dir=output) as temp:
            scratch = Path(temp)
            src = source / SIMULATOR_DIR
            if warp_trace:
                summary['stage'] = 'warp-preflight'
                warp = handlers._support_module(source, 'warp_ft48')
                env = warp.prepare(source, output, scratch, env, handlers)
            summary['stage'] = 'fixtures'
            for case, expected in ((smoke, SMOKE_FIXTURE_SHA256), (wide, WIDE_FIXTURE_SHA256)):
                case['fixture_sha256'] = _sha256(src / case['fixture'])
                if case['fixture_sha256'] != expected:
                    raise RuntimeError(f'Days pinned fixture hash mismatch: {case["fixture"]}')
            summary['stage'] = 'build'
            env['CARGO_TARGET_DIR'] = str(scratch / 'target')
            summary['build_seconds'] = handlers.command(
                ['cargo', 'build', '--locked', '--release', '--features', 'cuda', '--bin', 'days'],
                build_log, src, env,
            )
            binary = scratch / 'target/release/days'
            summary['binary_sha256'] = _sha256(binary)
            summary['stage'] = 'smoke'
            for engine in ('scalar', 'cpu', 'cuda'):
                record = smoke['runs'][engine] = {}
                _measure(binary, smoke['fixture'], engine, output / f'smoke-{engine}.txt', src, env,
                         handlers, record)
            fingerprints = {(record['result_bytes'], record['result_fnv1a64'])
                            for record in smoke['runs'].values()}
            smoke['fingerprints_match'] = len(fingerprints) == 1
            if not smoke['fingerprints_match']:
                raise RuntimeError('Days Scalar/CPU/CUDA smoke fingerprints differ')
            summary['stage'] = 'fat-tree48'
            record = wide['run'] = {}
            _measure(binary, wide['fixture'], 'cuda', output / 'fat-tree48-cuda.txt', src, env,
                     handlers, record)
            sourced, received, dropped = (record[key] for key in
                                          ('sourced_packets', 'received_packets', 'dropped_packets'))
            if sourced != EXPECTED_PACKETS or received + dropped > sourced:
                raise RuntimeError(f'Days fixed-window packet gate failed: expected {EXPECTED_PACKETS} sourced, '
                                   f'observed {sourced} sourced, {received} received and {dropped} dropped')
            if profile or warp_trace:
                label = 'warp' if warp_trace else 'nsys'
                summary['stage'] = f'fat-tree48-{label}'
                profiled = wide['profile'] = {'run': {}}
                summary['timing_scope']['profiled_process_seconds'] = (
                    'IKET lifetime-only capture including trace collection.' if warp_trace else
                    'Nsight Systems wrapper including simulator execution and trace generation.'
                )
                capture_env = warp.capture_environment(source, output, env, 'days') if warp_trace else env
                prefix = () if warp_trace else profiler.prefix(nsys, output)
                _measure(binary, wide['fixture'], 'cuda', output / f'fat-tree48-cuda-{label}.txt', src,
                         capture_env, handlers, profiled['run'], prefix=prefix)
                differences = profiled['work_differences'] = _work_differences(record, profiled['run'])
                rounds = profiled['run']['device']['rounds']
                if warp_trace:
                    summary['stage'] = 'warp-collect'
                    dot = output / 'warp/graph.dot'
                    if not dot.is_file() or dot.stat().st_size == 0:
                        raise RuntimeError('missing Days CUDA Graph dependency export')
                    profiled['graph_dependencies'] = 'warp/graph.dot'
                    profiled['trace'] = warp.collect(source, output, 'days', rounds)
                if differences:
                    summary['stage'] = f'fat-tree48-{label}'
                    raise RuntimeError(f'Days profiled work differs from baseline: {", ".join(differences)}')
                if profile:
                    summary['stage'] = 'nsys-collect'
                    profiled['trace'] = profiler.collect(nsys, output, env, handlers, simulator='days-ago',
                                                         rounds=rounds)
        summary.update(state='succeeded', stage='complete')
    except Exception as error:
        summary.update(state='failed', error=str(error))
        if summary['stage'] == 'build' and build_log.is_file():
            with build_log.open('rb') as stream:
                stream.seek(max(0, build_log.stat().st_size - 16 * 1024))
                (output / 'build-error.txt').write_bytes(stream.read())
        raise
    finally:
        handlers.write_json(output / 'summary.json', summary)
