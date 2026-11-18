"""Fixed scientific experiment handlers. Task manifests cannot supply commands or URLs."""

import csv
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import uuid
from datetime import datetime, timezone

GEDES = ("https://github.com/mobinets/GeDES.git", "1376c638a0ae66116430f953da137d0c536675ad")
DAYS = ("https://github.com/iQua/days.git", "048d48743119257fd8d4b2e5f0fb2228153fd955")
DAYS_FIXTURE = "configs/benchmarks/tcp/fattree_k4_tcp_cubic_f16_smoke.toml"
_SUPPORT_MODULES = {}
_CASE_SUPPORT = {}


def _support_module(source, name):
    path = source / 'tools' / f'{name}.py'
    key = str(path.resolve())
    if key not in _SUPPORT_MODULES:
        module_name = 'pinned_' + name + '_' + hashlib.sha256(key.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f'cannot load pinned experiment support: {name}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _SUPPORT_MODULES[key] = module
    return _SUPPORT_MODULES[key]


def _run_pinned(kind, source, output):
    _support_module(source, kind).run(source, output, sys.modules[__name__])


def write_json(path, result):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(result, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def pass_lock():
    lock_fd = os.environ.get("GPU_NET_SIM_LOCK_FD")
    return (int(lock_fd),) if lock_fd else ()


def command(args, log, cwd=None, env=None, journal_dir=None):
    journal_root = Path(journal_dir) if journal_dir is not None else log.parent
    journal = journal_root / 'commands.json'
    records = json.loads(journal.read_text()) if journal.exists() else []
    argv = [str(argument) for argument in args]
    record = {'argv': argv, 'command': shlex.join(argv),
              'cwd': str(Path(cwd or Path.cwd()).resolve()),
              'log': log.relative_to(journal_root).as_posix(),
              'started_at': datetime.now(timezone.utc).isoformat(), 'state': 'running'}
    records.append(record)
    write_json(journal, records)
    start = time.monotonic()
    try:
        with log.open("wb") as stream:
            result = subprocess.run(
                argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                pass_fds=pass_lock(), check=False,
            )
        record.update(returncode=result.returncode,
                      state='succeeded' if result.returncode == 0 else 'failed')
        if result.returncode:
            raise RuntimeError(f"{args[0]} returned {result.returncode}; see {log.name}")
    except Exception as error:
        record.update(state='failed', error=str(error))
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - start)
        write_json(journal, records)
    return time.monotonic() - start


def gpu_rows(log_path, journal_dir=None):
    command([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version,pci.bus_id",
        "--format=csv,noheader,nounits",
    ], log_path, journal_dir=journal_dir)
    rows = [list(map(str.strip, row)) for row in csv.reader(log_path.read_text().splitlines())]
    if not rows or any(len(row) != 7 or not row[1].startswith("GPU-")
                       or not re.fullmatch(r"[0-9]{1,2}\.[0-9]", row[3]) for row in rows):
        raise RuntimeError("nvidia-smi did not report a supported physical NVIDIA GPU list")
    return rows


def visible_gpus(rows):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return rows
    if visible.strip() in {"", "-1"}:
        return []
    selected = []
    for value in visible.split(","):
        selector = value.strip()
        matches = [row for row in rows if row[0] == selector or
                   (selector.startswith("GPU-") and row[1].startswith(selector))]
        if len(matches) != 1 or matches[0] in selected:
            raise ValueError("CUDA_VISIBLE_DEVICES must contain unique physical indices or GPU UUIDs")
        selected.append(matches[0])
    return selected


def selected_gpu(output):
    rows = visible_gpus(gpu_rows(output / "gpu-list.csv"))
    if not rows:
        raise ValueError("CUDA_VISIBLE_DEVICES exposes no GPUs")
    selected = os.environ.get("GPU_NET_SIM_GPU_UUID")
    row = next((r for r in rows if r[1] == selected), None) if selected is not None else rows[0]
    if row is None:
        raise ValueError("GPU_NET_SIM_GPU_UUID must match a GPU inside CUDA_VISIBLE_DEVICES")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": row[1]}
    write_json(output / "gpu.json", {"uuid": row[1], "name": row[2], "compute_cap": row[3],
                                     "memory_mib": row[4], "driver": row[5], "pci_bus": row[6]})
    return row[3], env


def inventory(source: Path, output: Path):
    del source
    rows = gpu_rows(output / "gpu-list.csv")
    probes = {
        "topology": ["nvidia-smi", "topo", "-m"],
        "nvlink": ["nvidia-smi", "nvlink", "-s"],
        "nvcc": ["nvcc", "--version"],
        "cmake": ["cmake", "--version"],
        "rustc": ["rustc", "--version"],
        "cargo": ["cargo", "--version"],
    }
    outcomes = {}
    for label, args in probes.items():
        if not shutil.which(args[0]):
            outcomes[label] = "not installed"
            continue
        try:
            command(args, output / f"{label}.txt")
            outcomes[label] = "ok"
        except RuntimeError as exc:
            outcomes[label] = str(exc)
    write_json(output / "summary.json", {
        "scope": "GPU and toolchain inventory; no simulation run",
        "platform": platform.platform(), "python": platform.python_version(),
        "cpu_count": os.cpu_count(), "gpu_count": len(rows),
        "visible_gpu_count": len(visible_gpus(rows)),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "probes": outcomes,
    })


TOOLCHAIN_PROBES = {
    'nvcc': ['nvcc', '--version'],
    'cmake': ['cmake', '--version'],
    'rustc': ['rustc', '--version'],
    'cargo': ['cargo', '--version'],
}


def _inventory_hardware(output, dependencies):
    rows = gpu_rows(output / 'gpu-list.csv', journal_dir=output.parent.parent)
    write_json(output / 'hardware.json', {
        'platform': platform.platform(), 'python': platform.python_version(),
        'cpu_count': os.cpu_count(), 'gpu_count': len(rows),
        'visible_gpu_count': len(visible_gpus(rows)),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpus': rows,
    })


def _inventory_probe(output, label, args):
    record = {'command': args, 'status': 'not installed'}
    if shutil.which(args[0]):
        try:
            command(args, output / f'{label}.txt', journal_dir=output.parent.parent)
        except Exception as error:
            record.update(status='failed', error=str(error))
            write_json(output / 'probe.json', record)
            raise
        record['status'] = 'ok'
    write_json(output / 'probe.json', record)


def _inventory_topology(output, dependencies):
    _inventory_probe(output, 'topology', ['nvidia-smi', 'topo', '-m'])


def _inventory_nvlink(output, dependencies):
    _inventory_probe(output, 'nvlink', ['nvidia-smi', 'nvlink', '-s'])


def _inventory_toolchain(output, dependencies):
    outcomes = {}
    failures = []
    for label, args in TOOLCHAIN_PROBES.items():
        if not shutil.which(args[0]):
            outcomes[label] = 'not installed'
            continue
        try:
            command(args, output / f'{label}.txt', journal_dir=output.parent.parent)
            outcomes[label] = 'ok'
        except Exception as error:
            outcomes[label] = f'failed: {error}'
            failures.append(label)
    write_json(output / 'tools.json', outcomes)
    if failures:
        raise RuntimeError('toolchain probes failed: ' + ', '.join(failures))


def _inventory_summary(case_outputs):
    def read(case_id, filename):
        directory = case_outputs.get(case_id)
        path = directory / filename if directory is not None else None
        return json.loads(path.read_text()) if path is not None and path.is_file() else {}

    hardware = read('hardware-info', 'hardware.json')
    probes = read('toolchain', 'tools.json')
    probes['topology'] = read('topology', 'probe.json').get('status', 'not executed')
    probes['nvlink'] = read('nvlink-check', 'probe.json').get('status', 'not executed')
    return {
        'scope': 'GPU and toolchain inventory; no simulation run',
        **{key: value for key, value in hardware.items() if key != 'gpus'}, 'probes': probes,
    }


def _inventory_report(output, dependencies):
    write_json(output / 'summary.json', _inventory_summary(dependencies))


def _case_implementation(*functions):
    source = '\n'.join(inspect.getsource(function) for function in functions)
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def case_plan(task, source):
    """Describe inventory stages for the runner's case engine."""
    if task['kind'] == 'gedes_audit':
        module = _support_module(source, 'gedes_experiments')
        _CASE_SUPPORT[task['id'], task['version']] = module
        return module.case_plan(task, source, sys.modules[__name__])
    if task['kind'] != 'gpu_inventory':
        return None
    if task['params'] != {}:
        raise ValueError('gpu_inventory does not accept handler parameters')
    cases = [
        {'id': 'hardware-info', 'depends_on': [], 'run': _inventory_hardware,
         'implementation': _case_implementation(_inventory_hardware, gpu_rows, visible_gpus),
         'inputs': {'schema': 1}, 'reuse': False},
        {'id': 'topology', 'depends_on': ['hardware-info'], 'run': _inventory_topology,
         'implementation': _case_implementation(_inventory_topology, _inventory_probe),
         'inputs': {'schema': 1}, 'reuse': False},
        {'id': 'nvlink-check', 'depends_on': ['hardware-info'], 'run': _inventory_nvlink,
         'implementation': _case_implementation(_inventory_nvlink, _inventory_probe),
         'inputs': {'schema': 1}, 'reuse': False},
        {'id': 'toolchain', 'depends_on': [], 'run': _inventory_toolchain,
         'implementation': _case_implementation(_inventory_toolchain),
         'inputs': {'schema': 1, 'commands': TOOLCHAIN_PROBES}, 'reuse': False},
        {'id': 'inventory-report',
         'depends_on': ['hardware-info', 'topology', 'nvlink-check', 'toolchain'],
         'run': _inventory_report,
         'implementation': _case_implementation(_inventory_report, _inventory_summary, write_json),
         'inputs': {'schema': 1}, 'reuse': True},
    ]
    return cases


def finalize_cases(task, output, manifest):
    """Keep the compact legacy inventory outputs available at the task root."""
    if task['kind'] == 'gedes_audit':
        module = _CASE_SUPPORT.get((task['id'], task['version']))
        if module is None:
            raise RuntimeError('GeDES audit support was not loaded for this task')
        module.finalize_cases(task, output, manifest, sys.modules[__name__])
        return
    if task['kind'] != 'gpu_inventory':
        return
    case_outputs = {case_id: output / 'cases' / case_id for case_id in manifest['cases']}
    summary = _inventory_summary(case_outputs)
    summary['case_states'] = {case_id: case['state'] for case_id, case in manifest['cases'].items()}
    summary['case_counts'] = manifest.get('summary', {})
    write_json(output / 'summary.json', summary)
    exports = {'hardware-info': ['gpu-list.csv'], 'topology': ['topology.txt'],
               'nvlink-check': ['nvlink.txt'],
               'toolchain': [f'{name}.txt' for name in TOOLCHAIN_PROBES]}
    for case_id, filenames in exports.items():
        for filename in filenames:
            source = output / 'cases' / case_id / filename
            if source.is_file():
                shutil.copyfile(source, output / filename)


def checkout(upstream, scratch, output, env):
    url, revision = upstream
    path = scratch / "src"
    command(["git", "init", "-q", str(path)], output / "git-init.log", env=env)
    command(["git", "fetch", "--no-tags", "--depth=1", url, revision],
            output / "git-fetch.log", path, env)
    command(["git", "checkout", "--detach", revision], output / "git-checkout.log", path, env)
    return path


def completed_gedes_rows(csv_path):
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"flow_size", "start_timestamp", "completed_timestamp", "flow_completion_time"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("GeDES output is missing expected flow CSV fields")
        total = completed = 0
        for row in reader:
            total += 1
            size = int(row["flow_size"])
            start, end = int(row["start_timestamp"]), int(row["completed_timestamp"])
            fct = int(row["flow_completion_time"])
            completed += size > 0 and end > start >= 0 and fct == end - start
    return total, completed


def gedes_smoke(source: Path, output: Path):
    del source
    capability, env = selected_gpu(output)
    command(["nvcc", "--version"], output / "nvcc.txt", env=env)
    command(["cmake", "--version"], output / "cmake.txt", env=env)
    with TemporaryDirectory(prefix=".gedes-", dir=output) as scratch_dir:
        scratch = Path(scratch_dir)
        src = checkout(GEDES, scratch, output, env) / "GeDES_code"
        build = scratch / "build"
        cmake = src / "CMakeLists.txt"
        original = cmake.read_text(encoding="utf-8")
        placeholder = "/path/to/code/dir/GeDES_code"
        if original.count(placeholder) != 2:
            raise RuntimeError("unexpected GeDES CMake layout")
        patched = original.replace(placeholder, "${CMAKE_CURRENT_SOURCE_DIR}")
        # Let CMake select the installed compiler (or respect the operator's CXX).
        patched = patched.replace('set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")\n', "")
        cmake.write_text(patched, encoding="utf-8")
        (output / "GeDES-CMakeLists.txt").write_text(patched, encoding="utf-8")
        command(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_CUDA_ARCHITECTURES=" + capability.replace(".", "")],
                output / "configure.log", env=env)
        shutil.copyfile(build / "CMakeCache.txt", output / "GeDES-CMakeCache.txt")
        build_seconds = command(["cmake", "--build", str(build), "--parallel", "4"],
                                output / "build.log", env=env)
        csv_path = output / "gedes-flows.csv"
        args = ["--ft_k=8", "--packet_pool_size=1000000", "--average_flow_size=10000",
                "--flow_time_range=1000000", f"--output={csv_path}"]
        run_seconds = command([str(build / "GeDES"), *args], output / "gedes.log", output, env)
    log = (output / "gedes.log").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Simulation Duration:\s+([0-9.eE+-]+)s\s*$", log, re.MULTILINE)
    duration = float(match[1]) if match else float("nan")
    total, completed = completed_gedes_rows(csv_path)
    if "Cuda error:" in log or not math.isfinite(duration) or duration <= 0 or completed <= 0:
        raise RuntimeError("GeDES GPU smoke did not pass the timed-flow gate")
    write_json(output / "summary.json", {
        "scope": "GeDES author k8 GPU smoke; at least one reported completion, drain unverified",
        "upstream": GEDES[0], "revision": GEDES[1], "args": args[:-1],
        "build_seconds": build_seconds, "process_seconds": run_seconds,
        "simulator_wall_seconds": duration, "csv_rows": total, "completed_rows": completed,
    })


def days_record(log, engine):
    lines = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
             if line.startswith("record=days_result ")]
    if len(lines) != 1:
        raise RuntimeError(f"expected a single Days result in {log.name}")
    fields = dict(word.split("=", 1) for word in lines[0].split() if "=" in word)
    if (fields.get("engine") != engine or int(fields["result_bytes"]) <= 0
            or int(fields["received_packets"]) <= 0
            or not re.fullmatch(r"[0-9a-f]{16}", fields["result_fnv1a64"])):
        raise RuntimeError(f"Days positive-work gate failed for {engine}")
    return fields


def days_cuda_smoke(source: Path, output: Path):
    del source
    capability, env = selected_gpu(output)
    if capability not in {"8.6", "8.9", "12.1"}:
        raise RuntimeError(f"Days pin supports SM 8.6, 8.9 or 12.1; observed {capability}; port separately")
    command(["nvcc", "--version"], output / "nvcc.txt", env=env)
    command(["rustc", "--version"], output / "rustc.txt", env=env)
    command(["cargo", "--version"], output / "cargo.txt", env=env)
    with TemporaryDirectory(prefix=".days-", dir=output) as scratch_dir:
        scratch = Path(scratch_dir)
        src = checkout(DAYS, scratch, output, env)
        fixture = src / DAYS_FIXTURE
        fixture_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        env["CARGO_TARGET_DIR"] = str(scratch / "target")
        build_seconds = command(["cargo", "build", "--locked", "--release", "--features", "cuda",
                                 "--bin", "days"], output / "build.log", src, env)
        binary = str(scratch / "target" / "release" / "days")
        records = {}
        for engine in ("scalar", "cpu", "cuda"):
            log = output / f"days-{engine}.txt"
            args = ["--workers", "2", "--repetitions", "1"] if engine == "cpu" else []
            wall = command([binary, str(fixture), "--engine", engine, *args], log, output, env)
            records[engine] = {**days_record(log, engine), "process_seconds": wall}
    fingerprints = {(r["result_bytes"], r["result_fnv1a64"]) for r in records.values()}
    write_json(output / "summary.json", {
        "scope": "Days AGO native TCP fixture; complete-state length/fingerprint comparison",
        "upstream": DAYS[0], "revision": DAYS[1], "fixture": DAYS_FIXTURE,
        "fixture_sha256": fixture_sha256,
        "build_seconds": build_seconds, "fingerprints_match": len(fingerprints) == 1,
        "records": records,
    })
    if len(fingerprints) != 1:
        raise RuntimeError("Days Scalar/CPU/CUDA results disagree; see summary.json")


def prepare(kind, params):
    handlers = {"gpu_inventory": inventory, "gedes_smoke": gedes_smoke,
                "days_cuda_smoke": days_cuda_smoke}
    if params == {} and kind in {"gpu_transfer", "gedes_drain", "nsx_single_gpu", "nsx_four_gpu"}:
        return partial(_run_pinned, kind)
    if kind not in handlers or params != {}:
        raise ValueError(f"unsupported task kind or parameters: {kind!r}")
    return handlers[kind]
