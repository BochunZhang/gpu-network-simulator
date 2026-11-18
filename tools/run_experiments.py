#!/usr/bin/env python3
"""Poll public Git code and claim experiments through shared CPFS directories."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone

LOCK_PATH = Path('/tmp/gpu-network-simulator.lock')
POLL_SECONDS = 10
TERMINAL = {'succeeded', 'failed', 'abandoned'}
RESULT_SUFFIXES = {'.json', '.csv', '.txt'}
FILE_LIMIT = 2 * 1024**2
ARCHIVE_LIMIT = 16 * 1024**2
ID_PATTERN = r'[0-9]{3}'


class RunnerError(Exception):
    """Failure that leaves execution records available for recovery."""


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def host_identity():
    try:
        machine = Path('/etc/machine-id').read_text(encoding='ascii').strip()
    except (OSError, ValueError):
        machine = ''
    return hashlib.sha256(f'gpu-network-simulator:{machine}:{socket.gethostname()}'.encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'


def write_json(path, value):
    # Different owners never share an append stream or a temporary filename.
    tmp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with tmp.open('x', encoding='utf-8') as out:
            out.write(encode(value))
            out.flush()
            os.fsync(out.fileno())
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate_record(row, path):
    if (not isinstance(row, dict)
            or not isinstance(row.get('id'), str) or not re.fullmatch(ID_PATTERN, row['id'])
            or type(row.get('version')) is not int or row['version'] < 1
            or row.get('state') not in TERMINAL | {'running'}
            or any(not isinstance(row.get(key), str) for key in
                   ('claim', 'host_id', 'worker', 'task_sha256', 'source_commit'))):
        raise RunnerError(f'invalid execution record: {path}')
    return row


def inherited_fds():
    fd = os.environ.get('GPU_NET_SIM_LOCK_FD')
    return (int(fd),) if fd else ()


@contextlib.contextmanager
def device_lock(path):
    # This flock is local /tmp only. Shared CPFS coordination uses mkdir.
    fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f'device already locked: {path}') from exc
        yield fd
    finally:
        os.close(fd)


def owner_matches(path, row):
    try:
        owner = read_json(path / 'owner.json')
    except FileNotFoundError:
        return False
    return owner.get('claim') == row['claim'] and owner.get('host_id') == row['host_id']


def remove_owned_directory(path, row):
    # Only the original host reconciles its claims, under its local host lock.
    # No two finishers can unlink and recreate the same task guard concurrently.
    if owner_matches(path, row):
        (path / 'owner.json').unlink()
        path.rmdir()


class Queue:
    def __init__(self, root, source_branch='main', lock_dir=None):
        self.root = root
        self.source_ref = f'refs/heads/{source_branch}'
        self.lock_dir = lock_dir or root / '.lock'
        self.git('check-ref-format', self.source_ref)

    def git(self, *args, timeout=30):
        command = ['git', '-c', 'gc.auto=0', '-c', 'maintenance.auto=false', *args]
        process = subprocess.Popen(
            command, cwd=self.root, env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'},
            pass_fds=inherited_fds(), start_new_session=True,
            text=True, encoding='utf-8', errors='replace',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as exc:
            # Git may have remote-helper/index-pack descendants modifying the
            # shared object database. Stop the whole group before releasing it.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            if isinstance(exc, subprocess.TimeoutExpired):
                raise RunnerError(f'git {args[0]} timed out after {timeout} seconds') from exc
            raise
        if process.returncode:
            raise RunnerError(f'git {args[0]}: {stderr.strip() or stdout.strip()}')
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)

    @contextlib.contextmanager
    def operation(self, wait_seconds=0):
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        path = self.lock_dir / 'git.lock'
        owner = {'host_id': host_identity(), 'worker': socket.gethostname(),
                 'claim': uuid.uuid4().hex, 'created_at': timestamp(), 'pid': os.getpid()}
        deadline = time.monotonic() + wait_seconds
        while True:
            try:
                path.mkdir()
                break
            except FileExistsError as exc:
                if time.monotonic() >= deadline:
                    raise RunnerError('shared Git operation busy; stale owner can use --recover-git-lock') from exc
                time.sleep(0.1)
        try:
            write_json(path / 'owner.json', owner)
            yield
        finally:
            remove_owned_directory(path, owner)

    def recover_git_lock(self):
        path = self.lock_dir / 'git.lock'
        if not path.exists():
            return
        try:
            owner = read_json(path / 'owner.json')
        except FileNotFoundError as exc:
            raise RunnerError('ownerless shared Git lock; confirm all workers stopped before manual recovery') from exc
        if owner.get('host_id') != host_identity():
            raise RunnerError('Git lock recovery requires its original host')
        remove_owned_directory(path, owner)
        print('recovered original host shared Git lock', flush=True)

    def source(self):
        with self.operation(wait_seconds=35):
            local_ref = 'refs/gpu-network-simulator/source'
            self.git('fetch', '--quiet', '--no-tags', '--no-write-fetch-head', 'origin',
                     f'+{self.source_ref}:{local_ref}')
            return self.git('rev-parse', local_ref).stdout.strip()

    def export(self, commit, destination):
        destination.mkdir()
        with tempfile.TemporaryDirectory(prefix='gpu-network-simulator-archive-') as temp:
            archive_path = Path(temp) / 'source.tar'
            with self.operation(wait_seconds=240):
                self.git('archive', '--format=tar', f'--output={archive_path}', commit, timeout=180)
            with tarfile.open(archive_path) as archive:
                for item in archive.getmembers():
                    name = PurePosixPath(item.name)
                    if name.is_absolute() or '..' in name.parts or not (item.isfile() or item.isdir()):
                        raise RunnerError(f'unsupported path in pinned source archive: {item.name}')
                archive.extractall(destination)

    def tasks(self, source):
        paths = self.git('ls-tree', '-rz', '--name-only', source, '--', 'tasks/').stdout.split('\0')
        tasks = []
        for path in paths:
            if not re.fullmatch(r'tasks/[0-9]{3}\.json', path):
                continue  # Old six-digit manifests remain historical artifacts.
            task = json.loads(self.git('show', f'{source}:{path}').stdout)
            if not isinstance(task, dict):
                raise RunnerError(f'expected task object: {path}')
            resources = task.get('resources', {})
            if (not isinstance(task.get('id'), str) or not re.fullmatch(ID_PATTERN, task['id'])
                    or path != f"tasks/{task['id']}.json"
                    or not isinstance(task.get('name'), str) or not task['name'].strip()
                    or type(task.get('version')) is not int or task['version'] < 1
                    or not isinstance(task.get('kind'), str) or not task['kind']
                    or not isinstance(task.get('params'), dict)
                    or type(task.get('enabled', True)) is not bool
                    or not isinstance(task.get('description', ''), str)
                    or not isinstance(resources, dict) or set(resources) - {'gpus'}
                    or type(resources.get('gpus', 0)) is not int or resources.get('gpus', 0) < 0
                    or set(task) - {'id', 'name', 'version', 'kind', 'params', 'resources', 'enabled', 'description'}):
                raise RunnerError(f'invalid task manifest: {path}')
            definition = {key: value for key, value in task.items() if key not in {'description', 'enabled'}}
            digest = hashlib.sha256(encode(definition).encode()).hexdigest()
            tasks.append((task, digest))
        return sorted(tasks, key=lambda item: item[0]['id'])


class ClaimStore:
    def __init__(self, lock_dir, status_dir):
        self.lock_dir = lock_dir
        self.status_dir = status_dir

    def guard(self, task_id):
        return self.lock_dir / f'{task_id}.task.lock'

    def reservation(self, task_id, version):
        return self.lock_dir / f'{task_id}.v{version}.lock'

    def path(self, task_id, version):
        return self.status_dir / task_id / f'v{version}.json'

    def read(self, task_id=None):
        paths = self.status_dir.glob(f'{task_id}/v*.json' if task_id else '[0-9][0-9][0-9]/v*.json')
        rows = {}
        for path in paths:
            row = validate_record(read_json(path), path)
            if path != self.path(row['id'], row['version']):
                raise RunnerError(f'status path does not match execution record: {path}')
            rows[row['id'], row['version']] = row
        return rows

    def save(self, row):
        validate_record(row, self.status_dir)
        path = self.path(row['id'], row['version'])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)

    def claim(self, task, digest, source, results, worker):
        task_id, version = task['id'], task['version']
        row = {'id': task_id, 'name': task['name'], 'version': version, 'kind': task['kind'],
               'task_sha256': digest, 'source_commit': source, 'worker': worker,
               'host_id': host_identity(), 'claim': uuid.uuid4().hex,
               'state': 'running', 'claimed_at': timestamp()}
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        guard = self.guard(task_id)
        try:
            guard.mkdir()
        except FileExistsError:
            return None
        keep_guard = False
        try:
            write_json(guard / 'owner.json', row)
            prior = list(self.read(task_id).values())
            for path in self.lock_dir.glob(f'{task_id}.v*.lock'):
                try:
                    prior.append(validate_record(read_json(path / 'owner.json'), path))
                except FileNotFoundError:
                    return None  # A partial claim requires owner-confirmed recovery.
            for previous in prior:
                if previous['version'] == version and previous['task_sha256'] != digest:
                    raise RunnerError(f'task {task_id} changed without increasing version {version}')
            if any(previous['version'] >= version or previous['state'] == 'running' for previous in prior):
                return None
            output = run_path(results, row)
            if output.exists():
                raise RunnerError(f'result path already exists: {output}')
            reservation = self.reservation(task_id, version)
            try:
                reservation.mkdir()
            except FileExistsError:
                return None
            keep_guard = True
            write_json(reservation / 'owner.json', row)
            self.save(row)
            return row
        finally:
            if not keep_guard:
                remove_owned_directory(guard, row)

    def finish(self, row):
        validate_record(row, self.status_dir)
        if row['host_id'] != host_identity():
            raise RunnerError('completion recovery requires the original host')
        if row['state'] not in TERMINAL:
            raise RunnerError('completion must be terminal')
        reservation = self.reservation(row['id'], row['version'])
        if not owner_matches(reservation, row):
            raise RunnerError('result does not match the shared version reservation')
        current = self.read(row['id']).get((row['id'], row['version']))
        if current and (current['claim'] != row['claim'] or current['host_id'] != row['host_id']):
            raise RunnerError('result does not match shared status')
        if current and current['state'] in TERMINAL and current['state'] != row['state']:
            raise RunnerError('terminal result and status disagree')
        self.save(row)
        # Keep this reservation forever, including on failure. Even a stale
        # cached status listing cannot cause the same version to execute again.
        write_json(reservation / 'owner.json', row)
        remove_owned_directory(self.guard(row['id']), row)

    def reconcile(self, results, active_claim=None):
        for path in sorted(results.glob('[0-9][0-9][0-9]/v*/result.json')):
            row = validate_record(read_json(path), path)
            if path.parent != run_path(results, row) or row['state'] not in TERMINAL:
                raise RunnerError(f'invalid terminal result path: {path}')
            if row['host_id'] == host_identity() and row['claim'] != active_claim:
                self.finish(row)
        return self.read()

    def abandon(self, results, task_id, version):
        self.reconcile(results)
        reservation = self.reservation(task_id, version)
        try:
            row = read_json(reservation / 'owner.json')
        except FileNotFoundError:
            try:
                row = read_json(self.guard(task_id) / 'owner.json')
            except FileNotFoundError as exc:
                raise RunnerError('ownerless or absent task claim; confirm all workers stopped before manual recovery') from exc
        validate_record(row, reservation)
        if row['id'] != task_id or row['version'] != version:
            raise RunnerError('recovery task/version does not match the task guard')
        if row['host_id'] != host_identity():
            raise RunnerError('abandonment requires the original execution host')
        if row['state'] != 'running':
            raise RunnerError(f'task {task_id} v{version} is already terminal')
        if not owner_matches(self.guard(task_id), row):
            raise RunnerError('task guard ownership is incomplete or inconsistent; manual recovery required')
        # Recover a crash between claiming the task guard and journaling its version.
        reservation.mkdir(exist_ok=True)
        write_json(reservation / 'owner.json', row)
        self.finish({**row, 'state': 'abandoned', 'finished_at': timestamp()})
        print(f'abandoned task {task_id} v{version}; increase its version to rerun', flush=True)


def run_path(results, row):
    return results / row['id'] / f"v{row['version']}"


def prepare_archive(output, record):
    tarpath = output.with_suffix('.tar.gz')
    tmp_tar = tarpath.with_suffix('.tmp')
    metadata = output / '.archive-metadata.json'
    selected, total = [], 0
    for path in sorted(output.iterdir()):
        if (path.name.startswith('.') or path.name in {'result.json', 'run.json'}
                or path.suffix not in RESULT_SUFFIXES or path.is_symlink() or not path.is_file()):
            continue
        size = path.stat().st_size
        if size <= FILE_LIMIT and total + size <= ARCHIVE_LIMIT:
            total += size
            selected.append(path)
    record['archive_files'] = ['result.json', *[path.name for path in selected]]
    write_json(metadata, record)
    try:
        with tarfile.open(tmp_tar, 'w:gz') as archive:
            archive.add(metadata, arcname='result.json')
            for path in selected:
                archive.add(path, arcname=path.name)
        tmp_tar.replace(tarpath)
    finally:
        metadata.unlink(missing_ok=True)
        tmp_tar.unlink(missing_ok=True)


def execute(queue, output):
    run = read_json(output / 'run.json')
    record = dict(run)
    try:
        with tempfile.TemporaryDirectory(prefix='gpu-network-simulator-source-') as temp:
            source = Path(temp) / 'source'
            queue.export(run['source_commit'], source)
            pinned = source / 'tools' / 'experiment_handlers.py'
            if not pinned.is_file():
                raise RunnerError('pinned source lacks tools/experiment_handlers.py')
            record['pinned_handler_sha256'] = hashlib.sha256(pinned.read_bytes()).hexdigest()
            spec = importlib.util.spec_from_file_location('pinned_experiment_handlers', pinned)
            handlers = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(handlers)
            handlers.prepare(run['task']['kind'], run['task']['params'])(source, output)
        record.update(state='succeeded', exit_code=0)
    except Exception as exc:
        traceback.print_exc()
        record.update(state='failed', exit_code=getattr(exc, 'returncode', 1), error=str(exc))
    record['finished_at'] = timestamp()
    try:
        prepare_archive(output, record)
    except (OSError, tarfile.TarError) as exc:
        traceback.print_exc()
        record.update(state='failed', exit_code=1, archive_error=str(exc), archive_files=[])
    write_json(output / 'result.json', record)
    return int(record['state'] != 'succeeded')


def gpu_count():
    try:
        proc = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                              text=True, capture_output=True, timeout=10, pass_fds=inherited_fds())
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode:
        return 0
    physical = {}
    for line in proc.stdout.splitlines():
        fields = [field.strip() for field in line.split(',')]
        if len(fields) == 2 and fields[0].isdigit() and fields[1].startswith('GPU-'):
            physical[fields[0]] = fields[1]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        return len(physical)
    if visible.strip() in {'', '-1'}:
        return 0
    selected = []
    for selector in visible.split(','):
        selector = selector.strip()
        resolved = physical.get(selector) if selector.isdigit() else selector
        if resolved not in physical.values() or resolved in selected:
            raise RunnerError(f'unsupported CUDA_VISIBLE_DEVICES selector: {selector!r}')
        selected.append(resolved)
    return len(selected)


def start_task(queue, source, store, results, worker, fd, available_gpus):
    for task, digest in queue.tasks(source):
        if not task.get('enabled', True) or task.get('resources', {}).get('gpus', 0) > available_gpus:
            continue
        row = store.claim(task, digest, source, results, worker)
        if row is None:
            continue
        output = run_path(results, row)
        output.mkdir(parents=True)
        script = Path(__file__).resolve()
        write_json(output / 'run.json', {**row, 'task': task, 'lock_dir': str(store.lock_dir),
                                      'runner_sha256': hashlib.sha256(script.read_bytes()).hexdigest()})
        env = {**os.environ, 'GPU_NET_SIM_LOCK_FD': str(fd)}
        with (output / 'worker.log').open('wb') as log:
            child = subprocess.Popen([sys.executable, str(script), '--_execute', str(output)],
                                     env=env, start_new_session=True, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(fd,))
        print(f"running task {row['id']} v{row['version']} on {worker} in {output}", flush=True)
        return child, output, row['claim']
    return None


def poll(queue, store, args, fd):
    active = None
    while True:
        result = None
        if active is not None and active[0].poll() is not None:
            child, output, _ = active
            result = child.returncode
            if not (output / 'result.json').is_file():
                print(f'worker exited without terminal result; retain shared claim: {output}', file=sys.stderr)
                return 1
            active = None
        next_poll = time.monotonic() + POLL_SECONDS
        try:
            rows = store.reconcile(args.results_dir, active[2] if active else None)
            if active is None and any(row['host_id'] == host_identity() and row['state'] == 'running'
                                      for row in rows.values()):
                raise RunnerError('unfinished execution on this host; confirm it stopped, then use --abandon ID VERSION')
            if args.once and result is not None:
                return int(result != 0)
            source = queue.source()
            if active is None:
                active = start_task(queue, source, store, args.results_dir, args.worker_id, fd, args.gpu_count)
                if active is None and args.once:
                    return 0
        except (RunnerError, OSError, ValueError) as exc:
            print(f'poll failed: {exc}', file=sys.stderr)
            if args.once and active is None:
                return 1
        remaining = max(0, next_poll - time.monotonic())
        if active is None:
            time.sleep(remaining)
        else:
            try:
                active[0].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 3 and sys.argv[1] == '--_execute':
        fd = int(os.environ['GPU_NET_SIM_LOCK_FD'])
        if not os.path.samestat(os.fstat(fd), LOCK_PATH.stat()):
            raise RunnerError('worker has no inherited host lock')
        output = Path(sys.argv[2])
        return execute(Queue(root, lock_dir=Path(read_json(output / 'run.json')['lock_dir'])), output)

    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='finish at most one available task')
    mode.add_argument('--abandon', nargs=2, metavar=('ID', 'VERSION'), help='recover a stopped task on its original host')
    mode.add_argument('--recover-git-lock', action='store_true', help='recover a stale Git mutex on its original host')
    cli.add_argument('--source-branch', default='main')
    cli.add_argument('--lock-dir', type=Path, default=root / '.lock')
    cli.add_argument('--status-dir', type=Path, default=root / 'status')
    cli.add_argument('--results-dir', type=Path, default=root / 'results')
    cli.add_argument('--worker-id', default=socket.gethostname(), help='owner label for status, never a task assignment')
    args = cli.parse_args()
    for name in ('lock_dir', 'status_dir', 'results_dir'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        with device_lock(LOCK_PATH) as fd:
            os.environ['GPU_NET_SIM_LOCK_FD'] = str(fd)
            queue = Queue(root, args.source_branch, args.lock_dir)
            if args.recover_git_lock:
                queue.recover_git_lock()
                return 0
            store = ClaimStore(args.lock_dir, args.status_dir)
            if args.abandon:
                task_id, version = args.abandon
                if not re.fullmatch(ID_PATTERN, task_id) or int(version) < 1:
                    raise RunnerError('recovery requires a three-digit task ID and positive version')
                store.abandon(args.results_dir, task_id, int(version))
                return 0
            args.gpu_count = gpu_count()
            print(f'CPFS worker {args.worker_id}, {args.gpu_count} GPUs; shared locks {args.lock_dir}', flush=True)
            return poll(queue, store, args, fd)
    except (RunnerError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
