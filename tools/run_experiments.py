#!/usr/bin/env python3
"""Poll public Git code and claim experiments through shared CPFS directories."""

import argparse
import contextlib
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone

import task_catalog

LOCK_PATH = Path('/tmp/gpu-network-simulator.lock')
POLL_SECONDS = 10
TERMINAL = {'succeeded', 'failed', 'abandoned'}
NSYS_REPORT_SUFFIX = '.nsys-rep'
NSYS_REPORT_ARCHIVE_LIMIT = 8 * 1024**3
RESULT_SUFFIXES = {'.json', '.csv', '.txt', NSYS_REPORT_SUFFIX}
FILE_LIMIT = 2 * 1024**2
ARCHIVE_LIMIT = 16 * 1024**2
# The first classified catalog imported historical experiments. Its publications
# live in the backup and do not allocate slots in the reviewed queue.
ARCHIVED_CATALOG_COMMIT = '8d027d9f20839fa53992aabb81d00cb1cfc20c3c'
ID_PATTERN = r'[0-9]{3}'
CONTROLLER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SLUG_PATTERN = task_catalog.SHORT_NAME_PATTERN
CHILD_ID_PATTERN = task_catalog.CHILD_PATTERN
EXECUTION_ID_PATTERN = rf'(?:{ID_PATTERN}|{CHILD_ID_PATTERN})'
VERSION_NAME_PATTERN = rf'(?:v[1-9][0-9]*|{task_catalog.VERSION_LABEL_PATTERN})'
RELEASE_FIELDS = ('parent_id', 'subtask_id', 'task_status', 'subtask_status', 'legacy_id', 'version_label')
LEGACY_SLUGS = {'gpu_inventory': 'system-info', 'gedes_smoke': 'gedes-smoke',
                'days_cuda_smoke': 'days-cuda'}


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


def effective_slug(row):
    task = row.get('task') if isinstance(row.get('task'), dict) else {}
    slug = row.get('slug') or task.get('slug') or LEGACY_SLUGS.get(row.get('kind'))
    if slug is None:
        slug = re.sub('[^a-z0-9]+', '-', str(row.get('kind', 'task')).lower()).strip('-') or 'task'
    if not isinstance(slug, str) or len(slug) > 63 or not re.fullmatch(SLUG_PATTERN, slug):
        raise RunnerError('invalid task slug')
    return slug


def task_label(row):
    if re.fullmatch(CHILD_ID_PATTERN, row['id']):
        return row['id']
    return f"{row['id']}-{effective_slug(row)}"


def version_name(row):
    if 'version_label' not in row:
        return f"v{row['version']}"
    label = row['version_label']
    if not isinstance(label, str) or not re.fullmatch(task_catalog.VERSION_LABEL_PATTERN, label):
        raise RunnerError(f'invalid version label: {label!r}')
    return label


def archive_name(row):
    if (not isinstance(row.get('id'), str) or not re.fullmatch(EXECUTION_ID_PATTERN, row['id'])
            or type(row.get('version')) is not int or row['version'] < 1):
        raise RunnerError('archive name requires a valid execution ID and positive version')
    return f"{task_label(row)}.{version_name(row)}.tar.gz"


def archive_path(results, row):
    return Path(results) / task_label(row) / archive_name(row)


def archive_candidates(results, row):
    root = Path(results)
    if re.fullmatch(CHILD_ID_PATTERN, row['id']):
        return [archive_path(root, row)]
    return [archive_path(root, row),
            root / task_label(row) / f"v{row['version']}.tar.gz",
            root / row['id'] / f"v{row['version']}.tar.gz",
            root / row['id'] / archive_name(row),
            root / archive_name(row),
            root / task_label(row) / f"{task_label(row)}-v{row['version']}.tar.gz",
            root / row['id'] / f"{task_label(row)}-v{row['version']}.tar.gz",
            root / f"{task_label(row)}-v{row['version']}.tar.gz"]


def execution_metadata(row):
    row = dict(row)
    row['slug'] = effective_slug(row)
    task = row.get('task') if isinstance(row.get('task'), dict) else {}
    row.setdefault('name', task.get('name', row['slug']))
    row.setdefault('command', task.get('command') or
                   f"bash tools/runner.sh --task {row['id']} --once")
    row['archive_name'] = archive_name(row)
    return row


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_python_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RunnerError(f'cannot load Python module: {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def runtime_engine():
    path = Path(__file__).with_name('incremental.py')
    if not path.is_file():
        raise RunnerError('incremental runtime is missing; update the shared checkout and restart the runner')
    return load_python_module(path, 'runner_incremental_protocol')


def validate_release_fields(row, path):
    numeric = isinstance(row['id'], str) and re.fullmatch(ID_PATTERN, row['id'])
    if numeric and not any(key in row for key in RELEASE_FIELDS):
        return  # Numeric executions made before the catalog have no release metadata.
    parent = row.get('parent_id')
    subtask = row.get('subtask_id')
    alias = row.get('legacy_id')
    if (not isinstance(parent, str) or not re.fullmatch(task_catalog.PARENT_PATTERN, parent)
            or not isinstance(subtask, str) or not re.fullmatch(r'[0-9]{2}', subtask)
            or alias is not None and (subtask != '00' or not isinstance(alias, str)
                                      or not re.fullmatch(ID_PATTERN, alias))
            or row['id'] not in {f'{parent}-{subtask}', alias}):
        raise RunnerError(f'invalid catalog identity: {path}')
    try:
        parent_version = task_catalog.release_version(row.get('task_status'))
        child_version = task_catalog.release_version(row.get('subtask_status'))
    except ValueError as error:
        raise RunnerError(f'invalid publication timestamp: {path}: {error}') from error
    if parent_version is None or child_version is None or parent_version > child_version:
        raise RunnerError(f'invalid publication order or ignored execution: {path}')
    if not numeric and row['version'] != child_version:
        raise RunnerError(f'version does not match the subtask publication timestamp: {path}')
    if 'version_label' in row:
        suffix = task_catalog.publication_label(1, row['subtask_status'])[2:]
        if numeric or not version_name(row).endswith(suffix):
            raise RunnerError(f'version label does not match the subtask publication timestamp: {path}')


def validate_receipt(row, path=None):
    if (not isinstance(row, dict) or row.get('schema_version') != 1
            or not isinstance(row.get('id'), str) or not re.fullmatch(EXECUTION_ID_PATTERN, row['id'])
            or type(row.get('version')) is not int or row['version'] < 1
            or row.get('state') not in TERMINAL
            or any(not isinstance(row.get(key), str) or not row[key].strip()
                   for key in ('slug', 'name', 'command', 'claim', 'archived_at'))
            or not isinstance(row.get('source_commit'), str)
            or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', row['source_commit'])
            or any(not isinstance(row.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', row[key])
                   for key in ('task_sha256', 'archive_sha256'))):
        raise RunnerError(f'invalid archival receipt: {path}')
    effective_slug(row)
    validate_release_fields(row, path)
    if 'archive_name' in row and row['archive_name'] != archive_name(row):
        raise RunnerError(f'archive name does not match task identity: {path}')
    return row


def validate_record(row, path):
    if (not isinstance(row, dict)
            or not isinstance(row.get('id'), str) or not re.fullmatch(EXECUTION_ID_PATTERN, row['id'])
            or type(row.get('version')) is not int or row['version'] < 1
            or row.get('state') not in TERMINAL | {'running'}
            or any(not isinstance(row.get(key), str) for key in
                   ('claim', 'host_id', 'worker', 'task_sha256', 'source_commit'))):
        raise RunnerError(f'invalid execution record: {path}')
    validate_release_fields(row, path)
    return execution_metadata(row)


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


def unique_fields(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise RunnerError(f'duplicate JSON field: {key}')
        record[key] = value
    return record


class Queue:
    def __init__(self, root, source_branch='main', lock_dir=None):
        self.root = root
        self.source_ref = f'refs/heads/{source_branch}'
        self.lock_dir = lock_dir or root / '.lock'
        self._tasks_source = None
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

    def catalog_at(self, commit, previous):
        # Reuse unchanged blobs. A new runner can verify releases even if it
        # missed every poll between a publication and its retirement.
        tree = self.git('ls-tree', '-rz', commit, '--', 'tasks/').stdout
        catalog = {}
        slots = set()
        last_index = {}
        for published in previous:
            category, index, _ = re.fullmatch(task_catalog.PARENT_PATTERN, published).groups()
            last_index[category] = max(last_index.get(category, ''), index)
        for entry in filter(None, tree.split('\0')):
            header, path = entry.split('\t', 1)
            if not path.endswith('.json') or re.fullmatch(r'tasks/[0-9]{6}\.json', path):
                continue
            key = path[len('tasks/'):-len('.json')]
            parent_match = re.fullmatch(task_catalog.PARENT_PATTERN, key)
            if not parent_match:
                raise RunnerError(f'invalid classified task path: {path}')
            blob = header.split()[-1]
            prior = previous.get(key)
            if prior and prior[0] == blob:
                item = prior
            else:
                manifest = json.loads(self.git('cat-file', 'blob', blob).stdout,
                                      object_pairs_hook=unique_fields)
                if isinstance(manifest, dict) and 'legacy' in manifest:
                    raise RunnerError(f'historical execution aliases cannot be published in the reviewed catalog: {path}')
                releases = task_catalog.expand(manifest, path)
                if prior:
                    task_catalog.validate_transition(prior[1], manifest)
                releases, publications = task_catalog.label_releases(releases, prior[3] if prior else {})
                item = blob, manifest, releases, publications
            category, index, _ = parent_match.groups()
            slot = category, index
            if slot in slots:
                raise RunnerError(f'duplicate task index: {key}')
            if prior is None and index <= last_index.get(category, ''):
                raise RunnerError(f'new tasks must follow the published {category} indices: {key}')
            slots.add(slot)
            catalog[key] = item
        if previous.keys() - catalog.keys():
            raise RunnerError('published tasks must remain in the catalog; set status to ignore')
        return catalog

    def classified_tasks(self, source, *, reviewed):
        commits = self.git('log', '--first-parent', '--reverse', '--format=%H', source,
                           '--', ':(glob)tasks/*-*.json').stdout.splitlines()
        if not commits and not reviewed:
            return None
        if ARCHIVED_CATALOG_COMMIT in commits:
            commits = commits[commits.index(ARCHIVED_CATALOG_COMMIT) + 1:]
        elif commits and self.git('rev-parse', '--is-shallow-repository').stdout.strip() == 'true':
            raise RunnerError('publication labels require the complete task history; unshallow the shared checkout')
        cached = getattr(self, '_catalog_commit', None)
        if cached in commits:
            commits = commits[commits.index(cached) + 1:]
        else:
            self._catalog_commit, self._catalog = None, {}
        for index, commit in enumerate(commits):
            try:
                catalog = self.catalog_at(commit, self._catalog)
            except (ValueError, RunnerError) as error:
                # A later fix can restore the last valid catalog.
                if index == len(commits) - 1:
                    raise RunnerError(f'invalid task catalog at {commit}: {error}') from error
                continue
            self._catalog, self._catalog_commit = catalog, commit
        return [release for key in sorted(self._catalog)
                for release in self._catalog[key][2]]

    def tasks(self, source):
        # Polling an unchanged immutable Git commit needs no history traversal.
        if self._tasks_source != source:
            self._tasks = self.load_tasks(source)
            self._tasks_source = source
        return self._tasks

    def load_tasks(self, source):
        paths = self.git('ls-tree', '-rz', '--name-only', source, '--', 'tasks/').stdout.split('\0')
        # The tracked marker also identifies an empty reviewed queue in a shallow clone.
        releases = self.classified_tasks(source, reviewed='tasks/README.md' in paths)
        if releases is not None:
            known = {f'tasks/{key}.json' for key in self._catalog}
            for path in paths:
                if path.endswith('.json') and path not in known and not re.fullmatch(r'tasks/[0-9]{6}\.json', path):
                    raise RunnerError(f'invalid classified task path: {path}')
            return releases
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
                    or not isinstance(task.get('slug'), str) or len(task['slug']) > 63
                    or not re.fullmatch(SLUG_PATTERN, task['slug'])
                    or not isinstance(task.get('command'), str) or not task['command'].strip()
                    or type(task.get('version')) is not int or task['version'] < 1
                    or not isinstance(task.get('kind'), str) or not task['kind']
                    or not isinstance(task.get('params'), dict)
                    or type(task.get('enabled', True)) is not bool
                    or not isinstance(task.get('description', ''), str)
                    or not isinstance(resources, dict) or set(resources) - {'gpus'}
                    or type(resources.get('gpus', 0)) is not int or resources.get('gpus', 0) < 0
                    or set(task) - {'id', 'name', 'slug', 'command', 'version', 'kind', 'params', 'resources', 'enabled', 'description', 'execution'}):
                raise RunnerError(f'invalid task manifest: {path}')
            if 'execution' in task:
                try:
                    runtime_engine().validate_execution(task)
                except ValueError as error:
                    raise RunnerError(f'invalid execution configuration: {path}: {error}') from error
            definition = {key: value for key, value in task.items() if key not in {'description', 'enabled', 'slug', 'command'}}
            digest = hashlib.sha256(encode(definition).encode()).hexdigest()
            tasks.append((task, digest))
        return sorted(tasks, key=lambda item: item[0]['id'])


    def archived_receipts(self, source):
        paths = self.git('ls-tree', '-rz', '--name-only', source, '--', 'status/').stdout.split('\0')
        receipts = []
        identities = set()
        for path in paths:
            if not re.fullmatch(rf'status/(?:{ID_PATTERN}-{SLUG_PATTERN}|{CHILD_ID_PATTERN})/{VERSION_NAME_PATTERN}\.archived\.json', path):
                continue
            row = validate_receipt(json.loads(self.git('show', f'{source}:{path}').stdout), path)
            if path != f"status/{task_label(row)}/{version_name(row)}.archived.json":
                raise RunnerError(f'archival receipt path does not match record: {path}')
            identity = row['id'], row['version']
            if identity in identities:
                raise RunnerError(f'multiple archival receipts for execution: {identity}')
            identities.add(identity)
            receipts.append(row)
        return receipts


class ClaimStore:
    def __init__(self, lock_dir, status_dir):
        self.lock_dir = lock_dir
        self.status_dir = status_dir

    def guard(self, task_id):
        return self.lock_dir / f'{task_id}.task.lock'

    def guard_for(self, row):
        return self.guard(row.get('legacy_id', row['id']))

    def reservation(self, task_id, version):
        return self.lock_dir / f'{task_id}.v{version}.lock'

    def path(self, task_id, version, row=None):
        if row is not None:
            return self.status_dir / task_label(row) / f'{version_name(row)}.json'
        if re.fullmatch(ID_PATTERN, task_id):
            named = next(self.status_dir.glob(f'{task_id}-*/v{version}.json'), None)
            if named:
                return named
        return self.status_dir / task_id / f'v{version}.json'

    def read(self, task_id=None):
        pattern = '*' if task_id is None else task_id + ('*' if re.fullmatch(ID_PATTERN, task_id) else '')
        paths = self.status_dir.glob(f'{pattern}/v*.json')
        rows = {}
        for path in paths:
            if not re.fullmatch(rf'{VERSION_NAME_PATTERN}\.json', path.name):
                continue
            row = validate_record(read_json(path), path)
            if (path.parent.name not in {row['id'], task_label(row)}
                    or path.name != f"{version_name(row)}.json"):
                raise RunnerError(f'status path does not match execution record: {path}')
            key = row['id'], row['version']
            previous = rows.get(key)
            if previous and (version_name(previous) != version_name(row) or any(
                    previous[field] != row[field] for field in ('claim', 'source_commit', 'task_sha256'))):
                raise RunnerError(f'conflicting legacy and named status: {path}')
            if previous and previous['state'] in TERMINAL and row['state'] in TERMINAL and previous['state'] != row['state']:
                raise RunnerError(f'conflicting terminal status: {path}')
            if (previous is None or row['state'] in TERMINAL and previous['state'] == 'running'
                    or previous['state'] == row['state'] and path.parent.name == task_label(row)):
                rows[key] = row
        return rows

    def save(self, row):
        validate_record(row, self.status_dir)
        row = execution_metadata(row)
        path = self.path(row['id'], row['version'], row)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)

    def claim(self, task, digest, source, results, worker, prepared_base=None):
        if task.get('task_status') == 'ignore' or task.get('subtask_status') == 'ignore':
            return None
        task_id, version = task['id'], task['version']
        execution = task.get('execution', {'mode': 'full'})
        if execution.get('mode') == 'incremental':
            provenance = prepared_base.get('provenance', {}) if isinstance(prepared_base, dict) else {}
            if any(provenance.get(key) != value for key, value in execution['base'].items()):
                raise RunnerError('incremental task must prepare its verified baseline before claiming')
        row = {'id': task_id, 'name': task['name'], 'slug': task['slug'],
               'command': task['command'], 'version': version, 'kind': task['kind'],
               'execution': execution,
               'task_sha256': digest, 'source_commit': source, 'worker': worker,
               'host_id': host_identity(), 'claim': uuid.uuid4().hex,
               'state': 'running', 'delivery_state': 'awaiting_download', 'claimed_at': timestamp()}
        row.update({field: task[field] for field in RELEASE_FIELDS if field in task})
        validate_release_fields(row, 'task claim')
        row['archive_name'] = archive_name(row)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        guard = self.guard_for(row)
        try:
            guard.mkdir()
        except FileExistsError:
            return None
        keep_guard = False
        try:
            write_json(guard / 'owner.json', row)
            release_ids = {task_id}
            markers = {task_id}
            if 'parent_id' in row:
                child_id = f"{row['parent_id']}-{row['subtask_id']}"
                release_ids.add(child_id)
                if 'legacy_id' in row:
                    release_ids.add(row['legacy_id'])
                markers.update(release_ids | {row['parent_id']})
            if any((self.status_dir / key / 'ignored.json').exists() for key in markers):
                return None
            prior = []
            for release_id in release_ids:
                prior.extend(self.read(release_id).values())
                for path in self.lock_dir.glob(f'{release_id}.v*.lock'):
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
            legacy_output = results / task_id / f'v{version}'
            if output.exists() or legacy_output.exists():
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
        if current and (current['claim'] != row['claim'] or current['host_id'] != row['host_id']
                        or version_name(current) != version_name(row)):
            raise RunnerError('result does not match shared status')
        if current and current['state'] in TERMINAL and current['state'] != row['state']:
            raise RunnerError('terminal result and status disagree')
        if current and current['state'] in TERMINAL:
            # Archival metadata is written by another CPFS worker. Replaying an
            # old result must never replace its delivery confirmation.
            row = current
        else:
            self.save(row)
        # Keep this reservation forever, including on failure. Even a stale
        # cached status listing cannot cause the same version to execute again.
        write_json(reservation / 'owner.json', row)
        remove_owned_directory(self.guard_for(row), row)

    def reconcile(self, results, active_claim=None):
        for path in sorted(results.glob('*/v[0-9]*/result.json')):
            try:
                row = validate_record(read_json(path), path)
            except FileNotFoundError:
                continue  # Another worker already moved this terminal result to the archive.
            if path.parent not in run_candidates(results, row) or row['state'] not in TERMINAL:
                raise RunnerError(f'invalid terminal result path: {path}')
            if row['host_id'] == host_identity() and row['claim'] != active_claim:
                self.finish(row)
        return self.read()

    def archive_receipts(self, receipts, results, archive_dir, active_claim=None, *, retire=False):
        failures = []
        for receipt in receipts:
            try:
                validate_receipt(receipt)
            except (RunnerError, ValueError) as error:
                failures.append({'state': 'archive_error', 'reason': str(error)})
                print(f'archive pending: {error}', file=sys.stderr)
                continue
            error_path = self.status_dir / task_label(receipt) / f"{version_name(receipt)}.archive-error.json"
            try:
                completed = self._archive_receipt(receipt, results, archive_dir, active_claim, retire=retire)
                if completed:
                    error_path.unlink(missing_ok=True)
            except (RunnerError, OSError, ValueError) as error:
                failure = {
                    'id': receipt['id'], 'slug': receipt['slug'], 'name': receipt['name'],
                    'version': receipt['version'], 'state': 'archive_error',
                    'execution_state': receipt['state'], 'claim': receipt['claim'],
                    'archive_name': archive_name(receipt), 'archive_sha256': receipt['archive_sha256'],
                    'reason': str(error), 'updated_at': timestamp(),
                    'controller_sha256': CONTROLLER_SHA256,
                    'expected_package_paths': [str(path) for root in (results, archive_dir)
                                               for path in archive_candidates(root, receipt)],
                }
                if 'version_label' in receipt:
                    failure['version_label'] = receipt['version_label']
                failures.append(failure)
                try:
                    error_path.parent.mkdir(parents=True, exist_ok=True)
                    write_json(error_path, failure)
                except OSError as status_error:
                    print(f'cannot save archive error for {task_label(receipt)}: {status_error}', file=sys.stderr)
                print(f"archive pending for {task_label(receipt)} {version_name(receipt)}: {error}", file=sys.stderr)
        return failures

    def _archive_receipt(self, receipt, results, archive_dir, active_claim=None, *, retire=False):
        validate_receipt(receipt)
        key = receipt['id'], receipt['version']
        current = self.read(receipt['id']).get(key)
        if current is None or current['state'] not in TERMINAL or current['claim'] == active_claim:
            return False
        identity = ('id', 'version', 'claim', 'source_commit', 'task_sha256', 'state')
        if (version_name(current) != version_name(receipt)
                or any(current[field] != receipt[field] for field in identity)):
            raise RunnerError(f'archival receipt does not match execution {key}')
        if effective_slug(current) != effective_slug(receipt):
            raise RunnerError(f'archival receipt slug does not match execution {key}')
        if current.get('delivery_state') == 'archived':
            if current.get('archive_sha256') != receipt['archive_sha256']:
                raise RunnerError(f'archival receipt changed for completed delivery {key}')
            pending = run_candidates(results, receipt)
            pending_packages = archive_candidates(results, receipt)
            archived_aliases = [path for path in archive_candidates(archive_dir, receipt)
                                if path != archive_path(archive_dir, receipt)]
            if ((not retire or current.get('disposition') == 'ignored')
                    and not any(path.exists() for path in (*pending, *pending_packages, *archived_aliases))):
                return True
        with self.archive_operation(current) as acquired:
            if not acquired:
                return False
            self._archive_one(receipt, current, results, archive_dir, identity, retire=retire)
        return True

    @contextlib.contextmanager
    def archive_operation(self, row):
        guard = self.guard_for(row)
        if guard.exists():
            try:
                guard_owner = read_json(guard / 'owner.json')
            except FileNotFoundError:
                guard_owner = {}
            if not guard_owner or guard_owner.get('claim') == row['claim']:
                yield False
                return
        operation = self.lock_dir / f"archive-{row['id']}-v{row['version']}.lock"
        owner = {'claim': uuid.uuid4().hex, 'host_id': host_identity(), 'pid': os.getpid()}
        try:
            operation.mkdir()
        except FileExistsError:
            try:
                prior_owner = read_json(operation / 'owner.json')
            except FileNotFoundError:
                prior_owner = {}
            # main holds this host's /tmp lock, so its prior dispatcher has
            # exited. Another host's unfinished move remains untouched.
            if prior_owner.get('host_id') != host_identity():
                yield False
                return
            owner = prior_owner
        else:
            write_json(operation / 'owner.json', owner)
        try:
            yield True
        finally:
            remove_owned_directory(operation, owner)

    def _archive_one(self, receipt, current, results, archive_dir, identity, *, retire=False):
        destination = run_path(archive_dir, receipt)
        candidates = run_candidates(results, receipt)
        sources = [path for path in candidates if path.exists()]
        if len(sources) > 1:
            raise RunnerError(f'both legacy and named results exist for {task_label(receipt)}')
        source = sources[0] if sources else None
        source_candidates = archive_candidates(results, receipt)
        archived_candidates = archive_candidates(archive_dir, receipt)
        for package in (*source_candidates, *archived_candidates):
            if package.is_symlink() or package.exists() and not package.is_file():
                raise RunnerError(f'archival package is not a regular file: {package}')
        source_packages = [path for path in source_candidates if path.is_file()]
        archived_packages = [path for path in archived_candidates if path.is_file()]
        destination_package = archive_path(archive_dir, receipt)
        packages = source_packages + archived_packages
        for package in packages:
            if sha256_file(package) != receipt['archive_sha256']:
                raise RunnerError(f'archival package checksum mismatch: {package}')
        if not packages:
            raise RunnerError(f'archival package is missing for {task_label(receipt)}')
        if source is not None and destination.exists():
            raise RunnerError(f'both pending and archived raw directories exist: {destination}')
        raw = source or destination
        if not raw.is_dir():
            raise RunnerError(f'raw result directory is missing for {task_label(receipt)}')
        raw_record = validate_record(read_json(raw / 'result.json'), raw)
        if (version_name(raw_record) != version_name(receipt)
                or any(raw_record[field] != receipt[field] for field in identity)):
            raise RunnerError(f'archival raw result identity mismatch: {raw}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            source.rename(destination)
        if not destination_package.exists():
            # Only names change. Receipts continue to identify the original
            # compressed bytes, including packages written by older runners.
            packages[0].rename(destination_package)
        for package in packages:
            if package != destination_package and package.exists():
                package.unlink()  # Every alias matched the receipt checksum.
        # Re-read after the move; an old result replay never writes delivery
        # metadata over a terminal shared status.
        latest = self.read(receipt['id']).get((receipt['id'], receipt['version']), current)
        if (version_name(latest) != version_name(receipt)
                or any(latest[field] != receipt[field] for field in identity)):
            raise RunnerError('shared status changed while archiving completed data')
        if retire:
            latest = {**latest, 'disposition': 'ignored'}
            if latest.get('delivery_state') != 'archived':
                latest['archive_reason'] = 'ignored'
        self.save({**latest, 'slug': receipt['slug'], 'command': receipt['command'],
                   'archive_name': archive_name(receipt),
                   'delivery_state': 'archived', 'archived_at': latest.get('archived_at', receipt['archived_at']),
                   'archive_sha256': receipt['archive_sha256']})
        for parent in {path.parent for path in candidates}:
            try:
                parent.rmdir()  # Remove only empty pending task folders.
            except OSError:
                pass
        print(f"archived results {task_label(receipt)} {version_name(receipt)}", flush=True)

    def abandon(self, results, task_id, version):
        self.reconcile(results)
        guards = [self.guard(task_id)]
        if re.fullmatch(CHILD_ID_PATTERN, task_id):
            # A migrated child keeps its numeric guard to interoperate with existing workers.
            guards.extend(self.lock_dir.glob('[0-9][0-9][0-9].task.lock'))
        if isinstance(version, str):
            claims = list(self.lock_dir.glob(f'{task_id}.v*.lock')) + guards
            versions = set()
            for claim in claims:
                try:
                    candidate = read_json(claim / 'owner.json')
                except FileNotFoundError:
                    continue
                if candidate.get('id') == task_id and candidate.get('version_label') == version:
                    versions.add(validate_record(candidate, claim)['version'])
            if len(versions) != 1:
                raise RunnerError(f'no unique version reservation for {task_id} {version}')
            version = versions.pop()
        reservation = self.reservation(task_id, version)
        try:
            row = read_json(reservation / 'owner.json')
        except FileNotFoundError:
            for guard in guards:
                try:
                    candidate = read_json(guard / 'owner.json')
                except FileNotFoundError:
                    continue
                if candidate.get('id') == task_id and candidate.get('version') == version:
                    row = candidate
                    break
            else:
                raise RunnerError('ownerless or absent task claim; confirm all workers stopped before manual recovery')
        validate_record(row, reservation)
        if row['id'] != task_id or row['version'] != version:
            raise RunnerError('recovery task/version does not match the task guard')
        if row['host_id'] != host_identity():
            raise RunnerError('abandonment requires the original execution host')
        if row['state'] != 'running':
            raise RunnerError(f'task {task_id} {version_name(row)} is already terminal')
        if not owner_matches(self.guard_for(row), row):
            raise RunnerError('task guard ownership is incomplete or inconsistent; manual recovery required')
        # Recover a crash between claiming the task guard and journaling its version.
        reservation.mkdir(exist_ok=True)
        write_json(reservation / 'owner.json', row)
        self.finish({**row, 'state': 'abandoned', 'finished_at': timestamp()})
        print(f'abandoned task {task_id} {version_name(row)}; publish a new version to rerun', flush=True)


def run_path(results, row):
    return results / task_label(row) / version_name(row)


def run_candidates(results, row):
    path = run_path(results, row)
    if re.fullmatch(ID_PATTERN, row['id']):
        return [path, results / row['id'] / version_name(row)]
    return [path]


def prepare_archive(output, record):
    record['archive_name'] = archive_name(record)
    tarpath = archive_path(output.parent.parent, record)
    tarpath.parent.mkdir(parents=True, exist_ok=True)
    tmp_tar = tarpath.with_suffix('.tmp')
    metadata = output / '.archive-metadata.json'
    selected, omitted, total, trace_total = [], [], 0, 0
    candidates = []
    for path in output.rglob('*'):
        relative = path.relative_to(output)
        if (any(part.startswith('.') for part in relative.parts)
                or relative.as_posix() in {'result.json', 'run.json'}
                or path.suffix not in RESULT_SUFFIXES or path.is_symlink() or not path.is_file()
                or any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(output))):
            continue
        candidates.append(path)
    # Keep provenance and the case manifest ahead of optional per-case data.
    priority = {'artifacts-manifest.json', 'summary.json', 'commands.json'}
    candidates.sort(key=lambda path: (path.relative_to(output).as_posix() not in priority,
                                     path.relative_to(output).as_posix()))
    for path in candidates:
        size = path.stat().st_size
        if path.suffix == NSYS_REPORT_SUFFIX:
            if trace_total + size <= NSYS_REPORT_ARCHIVE_LIMIT:
                trace_total += size
                selected.append(path)
            else:
                omitted.append(path.relative_to(output).as_posix())
        elif size <= FILE_LIMIT and total + size <= ARCHIVE_LIMIT:
            total += size
            selected.append(path)
        else:
            omitted.append(path.relative_to(output).as_posix())
    record['archive_files'] = ['result.json', *[path.relative_to(output).as_posix() for path in selected]]
    if omitted:
        record['archive_omitted'] = omitted
    write_json(metadata, record)
    try:
        with tarfile.open(tmp_tar, 'w:gz') as archive:
            archive.add(metadata, arcname='result.json')
            for path in selected:
                archive.add(path, arcname=path.relative_to(output).as_posix())
        tmp_tar.replace(tarpath)
    finally:
        metadata.unlink(missing_ok=True)
        tmp_tar.unlink(missing_ok=True)


def execute(queue, output):
    run = execution_metadata(read_json(output / 'run.json'))
    record = {key: value for key, value in run.items() if key != 'prepared_base'}
    prepared = run.get('prepared_base')
    try:
        with tempfile.TemporaryDirectory(prefix='gpu-network-simulator-source-') as temp:
            source = Path(temp) / 'source'
            queue.export(run['source_commit'], source)
            pinned = source / 'tools' / 'experiment_handlers.py'
            if not pinned.is_file():
                raise RunnerError('pinned source lacks tools/experiment_handlers.py')
            record['pinned_handler_sha256'] = sha256_file(pinned)
            handlers = load_python_module(pinned, 'pinned_experiment_handlers')
            plan_factory = getattr(handlers, 'case_plan', None)
            plan = plan_factory(run['task'], source) if plan_factory else None
            if plan is not None:
                engine_path = source / 'tools' / 'incremental.py'
                if not engine_path.is_file():
                    raise RunnerError('case-based handler requires pinned tools/incremental.py')
                engine = load_python_module(engine_path, 'pinned_incremental_engine')
                manifest = engine.run_cases(run['task'], plan, output, record, prepared)
                if prepared:
                    record['base'] = prepared['provenance']
                finalize = getattr(handlers, 'finalize_cases', None)
                try:
                    if finalize:
                        finalize(run['task'], output, copy.deepcopy(manifest))
                finally:
                    manifest = engine.verify_outputs(output, manifest)
                    record['artifacts_manifest_sha256'] = sha256_file(output / 'artifacts-manifest.json')
                    record['case_counts'] = {key: manifest['summary'][key]
                                             for key in ('executed', 'reused', 'failed', 'skipped')}
                record.update(state=manifest['state'], exit_code=int(manifest['state'] != 'succeeded'))
                if manifest['state'] != 'succeeded':
                    record['error'] = 'one or more required cases did not pass'
            else:
                execution = run['task'].get('execution', {'mode': 'full'})
                if execution.get('mode') != 'full' or execution.get('targets') or execution.get('force_rerun'):
                    raise RunnerError('this handler does not expose incremental cases')
                handlers.prepare(run['task']['kind'], run['task']['params'])(source, output)
                record.update(state='succeeded', exit_code=0)
    except Exception as exc:
        traceback.print_exc()
        record.update(state='failed', exit_code=getattr(exc, 'returncode', 1), error=str(exc))
    finally:
        if prepared:
            snapshot = Path(prepared['root']).resolve()
            allowed = output.parents[1] / '.incremental-inputs'
            if snapshot.is_relative_to(allowed.resolve()):
                shutil.rmtree(snapshot, ignore_errors=True)
    record['finished_at'] = timestamp()
    command_log = output / 'commands.json'
    if command_log.is_file():
        record['commands'] = read_json(command_log)
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


def prepare_incremental_base(task, store, results, archive_dir, engine):
    execution = engine.validate_execution(task)
    base = execution['base']
    operation = store.lock_dir / f"archive-{base['id']}-v{base['version']}.lock"
    store.lock_dir.mkdir(parents=True, exist_ok=True)
    owner = {'claim': uuid.uuid4().hex, 'host_id': host_identity(), 'pid': os.getpid(),
             'operation': 'prepare-base'}
    try:
        operation.mkdir()
    except FileExistsError as error:
        try:
            prior = read_json(operation / 'owner.json')
        except FileNotFoundError:
            raise engine.BaseUnavailable('baseline is locked without complete owner metadata') from error
        if prior.get('host_id') != host_identity():
            raise engine.BaseUnavailable('baseline archive or preparation is in progress') from error
        # The original host lock is held and no experiment on this host is active.
        owner = prior
    else:
        write_json(operation / 'owner.json', owner)
    staging = None
    try:
        staging_parent = results / '.incremental-inputs'
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"{task['id']}-v{task['version']}-", dir=staging_parent))
        return engine.prepare_base(task, results, archive_dir, staging)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        remove_owned_directory(operation, owner)


def start_task(queue, source, store, results, worker, fd, available_gpus, task_id=None, archive_dir=None, releases=None):
    archive_dir = archive_dir or queue.root / 'results-archive'
    for task, digest in (queue.tasks(source) if releases is None else releases):
        selectors = {task['id'], task.get('parent_id'), task.get('legacy_id')}
        if 'parent_id' in task:
            selectors.add(f"{task['parent_id']}-{task['subtask_id']}")
        if task_id is not None and task_id not in selectors:
            continue
        if not task.get('enabled', True) or task.get('resources', {}).get('gpus', 0) > available_gpus:
            continue
        prepared = None
        waiting = store.status_dir / task_label(task) / f"{version_name(task)}.waiting.json"
        old_waiting = waiting.with_name(f"v{task['version']}.waiting.json")
        if task.get('execution', {}).get('mode', 'full') == 'incremental':
            # Cheap checks avoid repeatedly copying a baseline for an already
            # reserved version. The final claim still uses atomic mkdir.
            reservation = store.reservation(task['id'], task['version'])
            if reservation.exists():
                try:
                    previous = read_json(reservation / 'owner.json')
                except FileNotFoundError:
                    continue
                if previous.get('task_sha256') != digest:
                    raise RunnerError(f"task {task['id']} changed without increasing version {task['version']}")
                waiting.unlink(missing_ok=True)
                old_waiting.unlink(missing_ok=True)
                continue
            if store.guard_for(task).exists():
                continue
            engine = runtime_engine()
            try:
                prepared = prepare_incremental_base(task, store, results, archive_dir, engine)
            except engine.BaseUnavailable as error:
                if store.reservation(task['id'], task['version']).exists():
                    waiting.unlink(missing_ok=True)
                    continue
                waiting.parent.mkdir(parents=True, exist_ok=True)
                pending = {'id': task['id'], 'slug': task['slug'], 'name': task['name'],
                           'version': task['version'], 'command': task['command'],
                           'state': 'waiting_for_base', 'source_commit': source,
                           'reason': str(error), 'updated_at': timestamp()}
                if 'version_label' in task:
                    pending['version_label'] = task['version_label']
                write_json(waiting, pending)
                if old_waiting != waiting:
                    old_waiting.unlink(missing_ok=True)
                if store.reservation(task['id'], task['version']).exists():
                    waiting.unlink(missing_ok=True)
                    continue
                print(f"task {task['id']} {version_name(task)} waiting for baseline: {error}", flush=True)
                continue
        try:
            row = store.claim(task, digest, source, results, worker, prepared)
        except BaseException:
            if prepared:
                shutil.rmtree(prepared['root'], ignore_errors=True)
            raise
        if row is None:
            if prepared:
                shutil.rmtree(prepared['root'], ignore_errors=True)
            continue
        waiting.unlink(missing_ok=True)
        old_waiting.unlink(missing_ok=True)
        output = run_path(results, row)
        output.mkdir(parents=True)
        script = Path(__file__).resolve()
        run = {**row, 'task': task, 'lock_dir': str(store.lock_dir),
               'runner_sha256': CONTROLLER_SHA256}
        if prepared:
            run['prepared_base'] = prepared
        write_json(output / 'run.json', run)
        env = {**os.environ, 'GPU_NET_SIM_LOCK_FD': str(fd)}
        with (output / 'worker.log').open('wb') as log:
            child = subprocess.Popen([sys.executable, str(script), '--_execute', str(output)],
                                     env=env, start_new_session=True, stdout=log,
                                     stderr=subprocess.STDOUT, pass_fds=(fd,))
        print(f"running task {row['id']} {version_name(row)} on {worker} in {output}", flush=True)
        return child, output, row['claim']
    return None


def poll(queue, store, args, fd):
    from retired_experiments import reconcile_ignored

    active = None
    archive_dir = getattr(args, 'archive_dir', queue.root / 'results-archive')
    while True:
        result = None
        finished = active is not None and active[0].poll() is not None
        active_claim = active[2] if active and not finished else None
        next_poll = time.monotonic() + POLL_SECONDS
        try:
            rows = store.reconcile(args.results_dir, active_claim)
            if finished:
                child, output, claim = active
                row = next((row for row in rows.values() if row['claim'] == claim
                            and row['state'] in TERMINAL), None)
                if row is None:
                    print(f'worker exited without terminal result; retain shared claim: {output}', file=sys.stderr)
                    return 1
                archive = row['archive_name'] if any(
                    archive_path(root, row).is_file() for root in (args.results_dir, archive_dir)
                ) else '-'
                print(f"complete {row['id']} {version_name(row)} state={row['state']} archive_name={archive}",
                      flush=True)
                result = child.returncode
                active = None
                finished = False
            if active is None and any(row['host_id'] == host_identity() and row['state'] == 'running'
                                      for row in rows.values()):
                raise RunnerError('unfinished execution on this host; confirm it stopped, then use --abandon ID VERSION')
            source = queue.source()
            receipts = queue.archived_receipts(source)
            releases = queue.tasks(source)
            store.archive_receipts(receipts, args.results_dir, archive_dir, active_claim)
            reconcile_ignored(releases, store, args.results_dir, archive_dir, active_claim, receipts=receipts)
            if args.once and result is not None:
                return int(result != 0)
            if active is None:
                active = start_task(queue, source, store, args.results_dir, args.worker_id, fd, args.gpu_count, getattr(args, 'task', None), archive_dir, releases)
                if active is None and args.once:
                    return 0
        except (RunnerError, OSError, ValueError) as exc:
            print(f'poll failed: {exc}', file=sys.stderr)
            if args.once and (active is None or finished):
                return 1
        remaining = max(0, next_poll - time.monotonic())
        if active is None or finished:
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
    cli.add_argument('--task', help='select a parent, a two-digit subtask appended to its parent, or a historical numeric ID')
    cli.add_argument('--lock-dir', type=Path, default=root / '.lock')
    cli.add_argument('--status-dir', type=Path, default=root / 'status')
    cli.add_argument('--results-dir', type=Path, default=root / 'results')
    cli.add_argument('--archive-dir', type=Path, default=root / 'results-archive')
    cli.add_argument('--worker-id', default=socket.gethostname(), help='owner label for status, never a task assignment')
    args = cli.parse_args()
    for name in ('lock_dir', 'status_dir', 'results_dir', 'archive_dir'):
        setattr(args, name, getattr(args, name).resolve())
    try:
        if args.task is not None and not (re.fullmatch(EXECUTION_ID_PATTERN, args.task)
                                          or re.fullmatch(task_catalog.PARENT_PATTERN, args.task)):
            raise RunnerError('--task requires a parent, child or historical task ID')
        with device_lock(LOCK_PATH) as fd:
            os.environ['GPU_NET_SIM_LOCK_FD'] = str(fd)
            queue = Queue(root, args.source_branch, args.lock_dir)
            if args.recover_git_lock:
                queue.recover_git_lock()
                return 0
            store = ClaimStore(args.lock_dir, args.status_dir)
            if args.abandon:
                task_id, version = args.abandon
                if not re.fullmatch(EXECUTION_ID_PATTERN, task_id):
                    raise RunnerError('recovery requires an execution ID')
                if not re.fullmatch(task_catalog.VERSION_LABEL_PATTERN, version):
                    version = int(version)
                    if version < 1:
                        raise RunnerError('recovery requires a version label or positive integer version')
                store.abandon(args.results_dir, task_id, version)
                return 0
            args.gpu_count = gpu_count()
            print(f'CPFS worker {args.worker_id}, {args.gpu_count} GPUs; shared locks {args.lock_dir}; '
                  f'controller_sha256={CONTROLLER_SHA256} pid={os.getpid()}', flush=True)
            return poll(queue, store, args, fd)
    except (RunnerError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    # The archive and retirement helpers import this controller by name.
    sys.modules['run_experiments'] = sys.modules[__name__]
    sys.exit(main())
