"""Apply published ignore markers to CPFS execution history."""

import sys
import tarfile
import tempfile
import zlib
from pathlib import Path

import archive_results
import run_experiments as runner


def mark_ignored(directory, **fields):
    marker = directory / 'ignored.json'
    if not marker.exists():
        directory.mkdir(parents=True, exist_ok=True)
        runner.write_json(marker, {**fields, 'status': 'ignore'})


def verified_retirement_receipt(row, results, archive_dir):
    # A previous attempt may have moved the raw directory or package before
    # checkpointing delivery. Validate the original package in either tree.
    for root in (results, archive_dir):
        for path in runner.archive_candidates(root, row):
            if path.is_symlink() or path.exists() and not path.is_file():
                raise archive_results.ArchiveError(f'package is not a regular file: {path}')
            if not path.is_file():
                continue
            with tempfile.TemporaryDirectory(prefix='gpu-retire-') as staging:
                packaged, _ = archive_results.inspect_package(path, Path(staging))
                if (any(packaged[key] != row[key] for key in archive_results.IDENTITY)
                        or runner.effective_slug(packaged) != runner.effective_slug(row)):
                    raise archive_results.ArchiveError(f'package execution identity mismatch: {path}')
                return archive_results.receipt_for(packaged, path)
    return None


def archive_abandoned_raw(store, row, results, archive_dir):
    """Move explicitly abandoned work with no final result or sealed package."""
    key = row['id'], row['version']
    identity = ('id', 'version', 'claim', 'source_commit', 'task_sha256', 'host_id')
    with store.archive_operation(row) as acquired:
        if not acquired:
            return False
        current = store.read(row['id']).get(key, {})
        owner = runner.read_json(store.reservation(*key) / 'owner.json')
        if (any(record.get('state') != 'abandoned' or not record.get('finished_at')
                or any(record.get(field) != row[field] for field in identity)
                for record in (current, owner)) or current.get('archive_sha256')):
            raise archive_results.ArchiveError('raw archival requires a confirmed abandoned reservation')
        for root in (results, archive_dir):
            if any(p.exists() or p.is_symlink() for p in runner.archive_candidates(root, row)):
                raise archive_results.ArchiveError('a package exists; retry sealed archive verification')
        dest = runner.run_path(archive_dir, row)
        candidates = {runner.run_path(results, row), results / row['id'] / f"v{row['version']}", dest}
        found = []
        for path in candidates:
            root = archive_dir if path == dest else results
            archive_results.reject_symlink_path(path, root)
            if path.exists():
                found.append(path)
        if len(found) != 1 or not found[0].is_dir():
            raise archive_results.ArchiveError('abandoned raw directory missing or ambiguous')
        raw = found[0]
        seal = raw / 'result.json'
        if seal.exists() or seal.is_symlink():
            raise archive_results.ArchiveError('raw result has a terminal record; its package is required')
        record_path = raw / 'run.json'
        archive_results.reject_symlink_path(record_path, archive_dir if raw == dest else results)
        run = runner.validate_record(runner.read_json(record_path), record_path)
        if run['state'] != 'running' or any(run[field] != row[field] for field in identity):
            raise archive_results.ArchiveError('abandoned raw run does not match reservation')
        if raw != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            raw.rename(dest)
            try:
                raw.parent.rmdir()
            except OSError:
                pass
        store.save({**current, 'delivery_state': 'archived', 'disposition': 'ignored',
                    'archive_reason': 'ignored', 'archive_contents': 'partial_raw_only',
                    'archived_at': current.get('archived_at') or runner.timestamp()})
        return True


def reconcile_ignored(tasks, store, results, archive_dir, active_claim=None, *, receipts=()):
    retired_ids = set()
    for task, _ in tasks:
        parent_ignored = task.get('task_status') == 'ignore'
        if not parent_ignored and task.get('subtask_status') != 'ignore':
            continue
        parent = task['parent_id']
        child = task['subtask_id']
        if parent_ignored:
            mark_ignored(store.status_dir / parent, parent_id=parent)
        ids = {f'{parent}-{child}'}
        if task.get('legacy_id'):
            ids.add(task['legacy_id'])
        for task_id in ids:
            mark_ignored(store.status_dir / task_id, parent_id=parent, subtask_id=child)
        retired_ids.update(ids)

    if not retired_ids:
        return []
    # Local CPFS retirement needs no manual download confirmation. When a Git
    # receipt exists its original checksum still governs the move.
    published = {(item['id'], item['version']): item for item in receipts}
    failures = []
    for key, row in store.read().items():
        if (row['id'] not in retired_ids or row['state'] not in runner.TERMINAL
                or row['claim'] == active_claim or row.get('disposition') == 'ignored'):
            continue
        path = store.status_dir / runner.task_label(row) / f"v{row['version']}.archive-error.json"
        receipt = published.get(key)
        if receipt is None:
            try:
                receipt = verified_retirement_receipt(row, results, archive_dir)
                if receipt is None:
                    if row['state'] != 'abandoned':
                        raise archive_results.ArchiveError('no completed result package available to archive')
                    if archive_abandoned_raw(store, row, results, archive_dir):
                        path.unlink(missing_ok=True)
                    continue
            except (OSError, EOFError, ValueError, tarfile.TarError, zlib.error,
                    archive_results.ArchiveError, runner.RunnerError) as error:
                failure = {'id': row['id'], 'version': row['version'], 'claim': row['claim'],
                           'state': 'archive_error', 'disposition': 'ignored',
                           'execution_state': row['state'], 'reason': str(error),
                           'updated_at': runner.timestamp()}
                path.parent.mkdir(parents=True, exist_ok=True)
                runner.write_json(path, failure)
                failures.append(failure)
                print(f"archive pending for ignored {row['id']} v{row['version']}: {error}", file=sys.stderr)
                continue
        failures.extend(store.archive_receipts([receipt], results, archive_dir, active_claim, retire=True))
    return failures
