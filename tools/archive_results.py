#!/usr/bin/env python3
"""Archive manually downloaded result packages and write Git-trackable receipts."""

import argparse
import contextlib
import gzip
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import uuid

import run_experiments as runner


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64 * 1024**2
MAX_MEMBERS = 4096
MAX_TAR_BYTES = MAX_BYTES + MAX_MEMBERS * 1024 + 1024**2
IDENTITY = ('id', 'version', 'claim', 'source_commit', 'task_sha256', 'state')


class ArchiveError(Exception):
    """Invalid incoming data or an existing archive with a different identity."""


@contextlib.contextmanager
def import_lock(lock_dir):
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / 'import-results.lock'
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise ArchiveError(f'result import already locked at {path}; inspect its owner before recovery') from exc
    token = uuid.uuid4().hex
    try:
        runner.write_json(path / 'owner.json', {'claim': token, 'host_id': runner.host_identity(), 'pid': os.getpid()})
        yield
    finally:
        owner = path / 'owner.json'
        if owner.is_file() and runner.read_json(owner).get('claim') == token:
            owner.unlink()
            path.rmdir()


def copy_limited(source, destination):
    total = 0
    while True:
        chunk = source.read(min(1024**2, MAX_BYTES - total + 1))
        if not chunk:
            return
        total += len(chunk)
        if total > MAX_BYTES:
            raise ArchiveError('result package exceeds the 64 MiB import limit')
        destination.write(chunk)


def inspect_package(package, extracted):
    files = {}
    directories = set()
    total = 0
    tar_end = 0
    with tarfile.open(package, 'r:gz') as archive:
        for member in archive:
            normalized = member.name.rstrip('/') if member.isdir() else member.name
            path = PurePosixPath(normalized)
            if (not normalized or path.is_absolute() or '..' in path.parts
                    or path.as_posix() != normalized or normalized in files or normalized in directories
                    or not (member.isfile() or member.isdir())):
                raise ArchiveError(f'unsupported or duplicate package path: {member.name!r}')
            if len(files) + len(directories) >= MAX_MEMBERS:
                raise ArchiveError('result package contains too many members')
            if member.isdir():
                directories.add(normalized)
                tar_end = max(tar_end, member.offset_data)
                continue
            total += member.size
            if member.size < 0 or total > MAX_BYTES:
                raise ArchiveError('uncompressed result package exceeds 64 MiB')
            tar_end = max(tar_end, member.offset_data + ((member.size + 511) // 512) * 512)
            target = extracted / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open('xb') as destination:
                copy_limited(source, destination)
                destination.flush()
                os.fsync(destination.fileno())
            if target.stat().st_size != member.size:
                raise ArchiveError(f'incomplete package member: {member.name}')
            files[normalized] = runner.sha256_file(target)
    # tarfile stops at its end marker. Drain gzip separately so incomplete or
    # CRC-invalid downloads cannot produce a receipt, including hidden tails.
    offset = 0
    with gzip.open(package, 'rb') as stream:
        while True:
            chunk = stream.read(1024**2)
            if not chunk:
                break
            if offset + len(chunk) > MAX_TAR_BYTES:
                raise ArchiveError('compressed package expands beyond the import limit')
            if offset + len(chunk) > tar_end and chunk[max(0, tar_end - offset):].strip(b'\0'):
                raise ArchiveError('unexpected data after the tar members')
            offset += len(chunk)
    result_file = extracted / 'result.json'
    if 'result.json' not in files or result_file.stat().st_size > 2 * 1024**2:
        raise ArchiveError('package must contain a root result.json no larger than 2 MiB')
    row = runner.read_json(result_file)
    if isinstance(row, dict) and 'task' in row and not isinstance(row['task'], dict):
        raise ArchiveError('result task metadata must be an object')
    # Validation enriches legacy rows for runtime use. Keep the original dict
    # here so inferred replay commands remain distinguishable from recorded ones.
    runner.validate_record(row, result_file)
    if row['state'] not in runner.TERMINAL:
        raise ArchiveError('incoming execution is not terminal')
    if (not row['claim'] or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', row['source_commit'])
            or not re.fullmatch(r'[0-9a-f]{64}', row['task_sha256'])):
        raise ArchiveError('result identity is missing a valid claim, source commit or task digest')
    declared = row.get('archive_files')
    if (not isinstance(declared, list) or any(not isinstance(name, str) for name in declared)
            or len(set(declared)) != len(declared) or set(declared) != set(files)):
        raise ArchiveError('archive_files does not match the package contents')
    return row, files


def receipt_for(row, package):
    command = row.get('command')
    if not command and isinstance(row.get('task'), dict):
        command = row['task'].get('command')
    recorded = bool(command)
    if not command:
        command = f"bash tools/runner.sh --task {row['id']} --once"
    receipt = {
        'schema_version': 1, 'id': row['id'], 'slug': runner.effective_slug(row),
        'name': row.get('name') or row.get('task', {}).get('name') or row.get('kind') or row['id'],
        'version': row['version'], 'claim': row['claim'], 'source_commit': row['source_commit'],
        'task_sha256': row['task_sha256'], 'state': row['state'], 'command': command,
        'command_source': 'recorded' if recorded else 'replay',
        'archive_sha256': runner.sha256_file(package), 'archive_size': package.stat().st_size,
        'archived_at': runner.timestamp(),
    }
    runner.validate_receipt(receipt, package)
    return receipt


def reject_symlink_path(path, root):
    for candidate in (path, *path.parents):
        if candidate == root:
            return
        if candidate.is_symlink():
            raise ArchiveError(f'archive target contains a symlink: {candidate}')
    raise ArchiveError(f'archive target escapes its root: {path}')


def validate_existing_files(destination, files, results):
    reject_symlink_path(destination, results)
    if destination.exists() and not destination.is_dir():
        raise ArchiveError(f'archive target is not a directory: {destination}')
    for name, digest in files.items():
        target = destination / name
        reject_symlink_path(target, results)
        if target.exists() and (not target.is_file() or runner.sha256_file(target) != digest):
            raise ArchiveError(f'archived result content differs: {target}')


def existing_receipt(status_dir, receipt):
    matches = sorted(status_dir.glob(f"{receipt['id']}-*/v{receipt['version']}.archived.json"))
    for path in matches:
        current = runner.validate_receipt(runner.read_json(path), path)
        if (any(current[key] != receipt[key] for key in (*IDENTITY, 'slug', 'command', 'archive_sha256'))):
            raise ArchiveError(f'an incompatible archival receipt already exists: {path}')
    if len(matches) > 1:
        raise ArchiveError('multiple archival receipts exist for the same task version')
    return runner.read_json(matches[0]) if matches else None


def archive_package(package, results_dir, status_dir):
    if package.is_symlink() or not package.is_file():
        raise ArchiveError(f'incoming package must be a regular file: {package}')
    original_stat = package.stat()
    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.import-', dir=results_dir) as temporary:
        staging = Path(temporary)
        frozen = staging / 'incoming.tar.gz'
        with package.open('rb') as source, frozen.open('xb') as destination:
            copy_limited(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        extracted = staging / 'files'
        extracted.mkdir()
        row, files = inspect_package(frozen, extracted)
        receipt = receipt_for(row, frozen)
        label = runner.task_label(receipt)
        destination = results_dir / label / f"v{row['version']}"
        tarpath = destination.with_suffix('.tar.gz')
        receipt_path = status_dir / label / f"v{row['version']}.archived.json"
        previous = existing_receipt(status_dir, receipt)
        reject_symlink_path(tarpath, results_dir)
        validate_existing_files(destination, files, results_dir)
        if tarpath.exists() and (not tarpath.is_file() or runner.sha256_file(tarpath) != receipt['archive_sha256']):
            raise ArchiveError(f'archived package differs: {tarpath}')
        # Validate every pre-existing destination before publishing any data.
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            for name in files:
                target = destination / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    (extracted / name).replace(target)
        else:
            extracted.replace(destination)
        if not tarpath.exists():
            frozen.replace(tarpath)
        # The receipt is the final durable import step. It is committed by the
        # authoring workflow only after the result directory and exact tar exist.
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        reject_symlink_path(receipt_path, status_dir)
        runner.write_json(receipt_path, previous or receipt)
    # A download still being written is retained for the next import attempt.
    if (not package.is_symlink() and package.is_file()
            and package.stat().st_ino == original_stat.st_ino
            and runner.sha256_file(package) == receipt['archive_sha256']):
        package.unlink()
    return receipt_path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--incoming-dir', type=Path, default=ROOT / 'results-remote')
    cli.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    cli.add_argument('--status-dir', type=Path, default=ROOT / 'status')
    cli.add_argument('--lock-dir', type=Path, default=ROOT / '.lock')
    args = cli.parse_args()
    for key in ('incoming_dir', 'results_dir', 'status_dir', 'lock_dir'):
        setattr(args, key, getattr(args, key).resolve())
    try:
        if (args.incoming_dir == args.results_dir or args.incoming_dir.is_relative_to(args.results_dir)
                or args.results_dir.is_relative_to(args.incoming_dir)):
            raise ArchiveError('incoming and archived result directories must be separate')
        args.incoming_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        with import_lock(args.lock_dir):
            packages = sorted(args.incoming_dir.rglob('*.tar.gz'))
            for package in packages:
                try:
                    receipt = archive_package(package, args.results_dir, args.status_dir)
                    print(f'archived {package.name}; receipt {receipt}')
                except (ArchiveError, runner.RunnerError, OSError, ValueError, EOFError, tarfile.TarError) as error:
                    failures += 1
                    print(f'cannot archive {package}: {error}', file=sys.stderr)
            loose = list(args.incoming_dir.rglob('result.json'))
            if loose:
                failures += 1
                print('extracted result directories remain untouched; place the original .tar.gz package in results-remote', file=sys.stderr)
            if not packages and not loose:
                print(f'no .tar.gz packages found under {args.incoming_dir}')
        return int(failures > 0)
    except (ArchiveError, runner.RunnerError, OSError, ValueError) as error:
        print(f'result import failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
