"""Prepare the native IKET collector and decoder in a task-private build directory."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from urllib.request import urlopen
from zipfile import ZipFile


IKET_VERSION = '4.9.0.dev1'
IJSON_VERSION = '3.4.0.post0'
IKET_DIGESTS = {
    'aarch64': '9e114c569e6b8ce8464e4b811b1e3abe6d7dcec698a8c0d2ae3f045189640d33',
    'x86_64': '5aa28fe991fff888352b7756a84517ce4d2b1d82d172a4e6ac692b03e29f31fd',
}
IKET_ROOT = 'nvidia_cutlass_dsl_tools/tool_packages/iket/'


def _wheel(scratch, project, version, tag, machine, pin=None):
    with urlopen(f'https://pypi.org/pypi/{project}/{version}/json', timeout=60) as response:
        release = json.load(response)
    matches = [row for row in release['urls'] if tag in row['filename']
               and 'manylinux' in row['filename'] and row['filename'].endswith(machine + '.whl')]
    if len(matches) != 1:
        raise RuntimeError(f'expected one {project} {version} wheel for {machine} {tag}')
    row, = matches
    digest = row['digests']['sha256']
    if pin and digest != pin:
        raise RuntimeError(f'{project} wheel differs from pinned SHA-256')
    target = scratch / row['filename']
    with urlopen(row['url'], timeout=90) as response, target.open('xb') as dest:
        shutil.copyfileobj(response, dest)
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f'incomplete or corrupt wheel: {target.name}')
    return target, {'name': target.name, 'sha256': digest, 'url': row['url']}


def prepare_library(scratch, env):
    machine = platform.machine()
    if sys.platform != 'linux' or machine not in IKET_DIGESTS:
        raise RuntimeError(f'IKET capture requires supported Linux hardware, got {sys.platform}/{machine}')
    root = (scratch / 'warp-tools').resolve()
    root.mkdir()
    # Only the C ABI is used from the NVIDIA wheel. It has no Python ABI dependency.
    iket, native_record = _wheel(root, 'nvidia-cutlass-dsl-tools', IKET_VERSION,
                                 '-cp311-cp311-', machine, IKET_DIGESTS[machine])
    with ZipFile(iket) as archive:
        for name in ('include/iket/iket_profiling_session.h', 'profiler/libiket_injection.so',
                     'libiket_cubin_info.so'):
            target = root / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(archive.read(IKET_ROOT + name))
    paths = [str(root), str(root / 'profiler')]
    if env.get('LD_LIBRARY_PATH'):
        paths.append(env['LD_LIBRARY_PATH'])
    return root, native_record, {**env, 'FT48_IKET_LIBRARY': str(root / 'profiler/libiket_injection.so'),
                                 'LD_LIBRARY_PATH': os.pathsep.join(paths)}


def prepare(source, output, scratch, env, helpers):
    root, native_record, build = prepare_library(
        scratch, {k: v for k, v in env.items() if not k.startswith('FT48_WARP_')})
    abi = f'cp{sys.version_info.major}{sys.version_info.minor}'
    ijson, python_record = _wheel(root, 'ijson', IJSON_VERSION, f'-{abi}-{abi}-', platform.machine())
    with ZipFile(ijson) as archive:
        archive.extractall(root / 'python')
    tools = source / 'profiling/warp_trace'
    build.update(FT48_WARP_LIBDIR=str(root),
                 FT48_WARP_DECODER=str(tools / 'trace.py'), FT48_WARP_PYTHON=sys.executable,
                 PYTHONPATH=os.pathsep.join(filter(None, [str(root / 'python'), env.get('PYTHONPATH')])))
    helpers.write_json(output / 'warp-dependencies.json',
                       {'iket': native_record, 'ijson': python_record, 'python': sys.version})
    helpers.command(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-I', root / 'include',
                     tools / 'collector.cpp', '-o', root / 'libft48warp.so', '-ldl'],
                    output / 'warp-build.txt', env=build)
    helpers.command([sys.executable, '-c', 'import ctypes,ijson,os; '
                     'ctypes.CDLL(os.environ["FT48_IKET_LIBRARY"]); print(ijson.backend)'],
                    output / 'warp-preflight.txt', env=build)
    return build


def capture_environment(source, output, build_env, simulator, graph_replays=None):
    trace_dir = output / 'warp'
    trace_dir.mkdir(exist_ok=True)
    env = {**build_env, 'FT48_WARP_OUTPUT': str(trace_dir.resolve()),
           'FT48_WARP_SIMULATOR': simulator, 'FT48_WARP_BATCH': '1' if simulator == 'days' else '32'}
    if graph_replays is not None:
        env['FT48_WARP_GRAPH_REPLAYS'] = str(graph_replays)
    return env


def collect(source, output, simulator, rounds, graph_replays=None):
    path = source / 'profiling/warp_trace/trace.py'
    spec = importlib.util.spec_from_file_location('ft48_warp_trace', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify(output / 'warp', simulator, rounds, graph_replays)
