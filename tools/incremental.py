"""Validated per-case reuse for versioned experiments using only the standard library.

Callers serialize baseline snapshotting with the baseline's archive operation.
Case callbacks receive their output directory and read-only dependency directories.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import uuid
from datetime import datetime, timezone


CASE_PATTERN = r'[a-z0-9]+(?:-[a-z0-9]+)*'
HEX_PATTERN = r'[0-9a-f]{64}'
IDENTITY_FIELDS = ('id', 'version', 'kind', 'claim', 'source_commit', 'task_sha256')


class IncrementalError(ValueError):
    """Invalid configuration or an invalid case implementation."""


class BaseUnavailable(IncrementalError):
    """The immutable baseline cannot currently be verified for execution."""


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(encoded(value).encode('utf-8')).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _case_ids(value, name):
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or len(item) > 63
                   or not re.fullmatch(CASE_PATTERN, item) for item in value)
            or len(set(value)) != len(value)):
        raise IncrementalError(f'{name} must be a nonempty list of unique case identifiers')
    return value


def _allowed_base_id(task, base_id):
    task_id = task.get('id')
    child_pattern = rf'[a-z][a-z0-9]*-[0-9]{{3}}-{CASE_PATTERN}-[0-9]{{2}}'
    if not isinstance(task_id, str) or not isinstance(base_id, str):
        return False
    if re.fullmatch(r'[0-9]{3}', task_id):
        return base_id == task_id
    if not re.fullmatch(child_pattern, task_id):
        return False
    if re.fullmatch(child_pattern, base_id):
        return base_id.rsplit('-', 1)[0] == task_id.rsplit('-', 1)[0]
    # The initial migrated subtask uses the original numeric execution identity.
    legacy = task.get('legacy_base', {})
    return (isinstance(legacy, dict) and re.fullmatch(r'[0-9]{3}', base_id) is not None
            and base_id == legacy.get('id'))


def validate_execution(task):
    execution = task.get('execution', {'mode': 'full'})
    if (not isinstance(execution, dict)
            or set(execution) - {'mode', 'base', 'targets', 'reuse', 'force_rerun'}):
        raise IncrementalError('invalid execution configuration')
    mode = execution.get('mode', 'full')
    if mode not in {'full', 'incremental'}:
        raise IncrementalError('execution.mode must be full or incremental')
    normalized = {'mode': mode}
    for name in ('targets', 'force_rerun'):
        if name in execution:
            normalized[name] = list(_case_ids(execution[name], f'execution.{name}'))
    if mode == 'full':
        if 'base' in execution or 'reuse' in execution:
            raise IncrementalError('full execution does not accept a base or reuse policy')
        return normalized
    base = execution.get('base')
    if (not isinstance(base, dict) or set(base) != {'id', 'version', 'artifacts_manifest_sha256'}
            or not _allowed_base_id(task, base.get('id'))
            or type(base.get('version')) is not int or base['version'] < 1
            or type(task.get('version')) is not int or base['version'] >= task['version']
            or not isinstance(base.get('artifacts_manifest_sha256'), str)
            or not re.fullmatch(HEX_PATTERN, base['artifacts_manifest_sha256'])):
        raise IncrementalError('incremental base must pin an earlier version within this task and its manifest SHA-256')
    if execution.get('reuse', 'verified-compatible') != 'verified-compatible':
        raise IncrementalError('execution.reuse must be verified-compatible')
    normalized.update(base=dict(base), reuse='verified-compatible')
    return normalized


def _regular_path(root, relative):
    if not isinstance(relative, str):
        raise IncrementalError('artifact path must be a string')
    name = PurePosixPath(relative)
    if (not relative or name.is_absolute() or '..' in name.parts
            or name.as_posix() != relative):
        raise IncrementalError(f'unsafe artifact path: {relative!r}')
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise IncrementalError(f'artifact contains a symbolic link: {relative}')
    return path


def _validate_artifacts(case_id, artifacts):
    if not isinstance(artifacts, list):
        raise IncrementalError(f'invalid artifact list for case {case_id}')
    paths = set()
    for item in artifacts:
        if (not isinstance(item, dict) or set(item) != {'path', 'size', 'sha256'}
                or not isinstance(item.get('path'), str)
                or not item['path'].startswith(f'cases/{case_id}/')
                or item['path'] in paths
                or type(item.get('size')) is not int or item['size'] < 0
                or not isinstance(item.get('sha256'), str) or not re.fullmatch(HEX_PATTERN, item['sha256'])):
            raise IncrementalError(f'invalid artifact record for case {case_id}')
        _regular_path(Path('/baseline'), item['path'])
        paths.add(item['path'])
    return artifacts


def _validate_manifest(manifest, result, base):
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or manifest.get('state') not in {'succeeded', 'failed'}
            or not isinstance(manifest.get('cases'), dict)
            or any(manifest.get(key) != result.get(key) for key in IDENTITY_FIELDS)
            or manifest.get('id') != base['id'] or manifest.get('version') != base['version']):
        raise BaseUnavailable('baseline manifest does not match the terminal execution identity')
    for case_id, case in manifest['cases'].items():
        if (not isinstance(case_id, str) or len(case_id) > 63 or not re.fullmatch(CASE_PATTERN, case_id)
                or not isinstance(case, dict) or case.get('id') != case_id
                or case.get('state') not in {'running', 'succeeded', 'failed', 'skipped'}):
            raise BaseUnavailable('baseline contains an invalid case record')
        if case['state'] == 'succeeded':
            try:
                _validate_artifacts(case_id, case.get('artifacts'))
            except IncrementalError as error:
                raise BaseUnavailable(str(error)) from error
            if (not isinstance(case.get('key'), str) or not re.fullmatch(HEX_PATTERN, case['key'])
                    or case.get('artifact_digest') != digest(case['artifacts'])):
                raise BaseUnavailable(f'baseline case {case_id} has invalid keys or artifact digest')
    return manifest


def prepare_base(task, results_dir, archive_dir, destination):
    """Copy verified baseline artifacts while caller holds its archive mutex.

    Missing or malformed manifests block a new reservation. Corrupt individual
    case artifacts remain eligible for recomputation and are never reused.
    """
    execution = validate_execution(task)
    if execution['mode'] == 'full':
        return None
    base = execution['base']
    labels = [base['id']]
    if re.fullmatch(r'[0-9]{3}', base['id']):
        slug = task.get('legacy_base', {}).get('slug', task.get('slug'))
        if not isinstance(slug, str) or not re.fullmatch(CASE_PATTERN, slug):
            raise IncrementalError('incremental task requires a safe slug')
        labels.append(f"{base['id']}-{slug}")
    candidates = [Path(root) / label / f"v{base['version']}"
                  for root in (results_dir, archive_dir) for label in labels]
    found = [path for path in candidates if path.is_dir()]
    if len(found) != 1:
        reason = 'missing' if not found else 'ambiguous'
        raise BaseUnavailable(f'{reason} baseline {base["id"]} v{base["version"]}')
    root = found[0]
    try:
        if root.is_symlink() or root.parent.is_symlink():
            raise BaseUnavailable('baseline result directory is a symbolic link')
        result_path = _regular_path(root, 'result.json')
        result = json.loads(result_path.read_text(encoding='utf-8'))
        if (not isinstance(result, dict) or result.get('state') not in {'succeeded', 'failed', 'abandoned'}
                or result.get('id') != base['id'] or result.get('version') != base['version']
                or result.get('kind') != task.get('kind')
                or any(not isinstance(result.get(key), str) or not result[key]
                       for key in ('claim', 'source_commit', 'task_sha256'))):
            raise BaseUnavailable('baseline terminal execution is missing or incompatible')
        manifest_path = _regular_path(root, 'artifacts-manifest.json')
        if manifest_path.stat().st_size > 2 * 1024**2:
            raise BaseUnavailable('baseline artifact manifest exceeds 2 MiB')
        if file_digest(manifest_path) != base['artifacts_manifest_sha256']:
            raise BaseUnavailable('baseline artifact manifest SHA-256 mismatch')
        manifest = _validate_manifest(json.loads(manifest_path.read_text(encoding='utf-8')), result, base)
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if any(destination.iterdir()):
            raise IncrementalError('baseline snapshot destination must be empty')
        unavailable = {}
        for case_id, case in manifest['cases'].items():
            if case['state'] != 'succeeded':
                continue
            try:
                for artifact in case['artifacts']:
                    source = _regular_path(root, artifact['path'])
                    target = _regular_path(destination, artifact['path'])
                    if not source.is_file() or source.stat().st_size != artifact['size']:
                        raise IncrementalError('missing or truncated artifact')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                    if file_digest(target) != artifact['sha256']:
                        raise IncrementalError('artifact SHA-256 mismatch')
            except (OSError, IncrementalError) as error:
                unavailable[case_id] = str(error)
                shutil.rmtree(destination / 'cases' / case_id, ignore_errors=True)
        shutil.copyfile(manifest_path, destination / 'artifacts-manifest.json')
        provenance = {key: result[key] for key in IDENTITY_FIELDS}
        provenance['artifacts_manifest_sha256'] = base['artifacts_manifest_sha256']
        snapshot = {'root': str(destination), 'manifest': manifest, 'provenance': provenance,
                    'unavailable': unavailable}
        write_json(destination / 'baseline.json', {key: value for key, value in snapshot.items() if key != 'root'})
        return snapshot
    except BaseUnavailable:
        raise
    except (OSError, ValueError, IncrementalError) as error:
        raise BaseUnavailable(f'baseline cannot be verified: {error}') from error


def _plan(cases, targets):
    if not isinstance(cases, list) or not cases:
        raise IncrementalError('case plan must be a nonempty list')
    by_id = {}
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('id'), str)
                or len(case['id']) > 63 or not re.fullmatch(CASE_PATTERN, case['id'])
                or case['id'] in by_id or not callable(case.get('run'))
                or not isinstance(case.get('implementation'), str) or not case['implementation']
                or type(case.get('reuse', True)) is not bool):
            raise IncrementalError('invalid case definition')
        dependencies = case.get('depends_on', [])
        if (not isinstance(dependencies, list) or any(not isinstance(item, str) for item in dependencies)
                or len(set(dependencies)) != len(dependencies)):
            raise IncrementalError(f'invalid dependencies for case {case["id"]}')
        try:
            encoded(case.get('inputs', {}))
        except (TypeError, ValueError) as error:
            raise IncrementalError(f'case {case["id"]} inputs are not finite JSON data') from error
        by_id[case['id']] = case
    selected = set(targets or by_id)
    if selected - by_id.keys():
        raise IncrementalError(f'unknown target cases: {sorted(selected - by_id.keys())}')
    ordered, visiting, visited = [], set(), set()

    def visit(case_id):
        if case_id in visiting:
            raise IncrementalError('case dependency graph contains a cycle')
        if case_id in visited:
            return
        if case_id not in by_id:
            raise IncrementalError(f'unknown dependency case: {case_id}')
        visiting.add(case_id)
        for parent in by_id[case_id].get('depends_on', []):
            visit(parent)
        visiting.remove(case_id)
        visited.add(case_id)
        ordered.append(case_id)

    for case_id in by_id:
        visit(case_id)
    closure = set()

    def include(case_id):
        if case_id in closure:
            return
        closure.add(case_id)
        for parent in by_id[case_id].get('depends_on', []):
            include(parent)

    for case_id in selected:
        include(case_id)
    return by_id, [case_id for case_id in ordered if case_id in closure], sorted(selected)


def _artifacts(output, case_id):
    directory = output / 'cases' / case_id
    result = []
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise IncrementalError(f'case {case_id} produced a symbolic link')
        if path.is_file():
            result.append({'path': path.relative_to(output).as_posix(),
                           'size': path.stat().st_size, 'sha256': file_digest(path)})
        elif not path.is_dir():
            raise IncrementalError(f'case {case_id} produced a non-regular artifact')
    return result


def _artifacts_match(root, artifacts):
    try:
        for artifact in artifacts:
            path = _regular_path(root, artifact['path'])
            if (not path.is_file() or path.stat().st_size != artifact['size']
                    or file_digest(path) != artifact['sha256']):
                return False
        return True
    except (OSError, IncrementalError):
        return False


def run_cases(task, cases, output, record, base=None):
    """Run a deterministic case DAG and commit an artifact manifest per case."""
    execution = validate_execution(task)
    if execution['mode'] == 'incremental' and base is None:
        raise BaseUnavailable('incremental execution requires a verified baseline snapshot')
    if execution['mode'] == 'full' and base is not None:
        raise IncrementalError('full execution received an unexpected baseline')
    by_id, ordered, targets = _plan(cases, execution.get('targets'))
    forced = set(execution.get('force_rerun', []))
    if forced - set(ordered):
        raise IncrementalError('force_rerun cases must belong to the selected target dependency graph')
    output = Path(output)
    manifest = {key: record[key] for key in IDENTITY_FIELDS}
    manifest.update(schema_version=1, slug=task.get('slug'), name=task.get('name'),
                    state='running', execution=execution, targets=targets, cases={}, started_at=timestamp())
    if base:
        if any(base.get('provenance', {}).get(key) != value for key, value in execution['base'].items()):
            raise BaseUnavailable('prepared baseline provenance does not match the requested base')
        try:
            frozen_manifest = _regular_path(Path(base['root']), 'artifacts-manifest.json')
            if file_digest(frozen_manifest) != execution['base']['artifacts_manifest_sha256']:
                raise BaseUnavailable('prepared baseline manifest SHA-256 mismatch')
            if json.loads(frozen_manifest.read_text(encoding='utf-8')) != base['manifest']:
                raise BaseUnavailable('prepared baseline metadata differs from its pinned manifest')
            _validate_manifest(base['manifest'], base['provenance'], execution['base'])
        except (OSError, ValueError) as error:
            raise BaseUnavailable(f'prepared baseline manifest cannot be verified: {error}') from error
        manifest['base'] = dict(base['provenance'])
    started = time.monotonic()

    def checkpoint():
        values = list(manifest['cases'].values())
        manifest['summary'] = {name: sum(row.get('origin') == name for row in values)
                               for name in ('executed', 'reused')}
        manifest['summary'].update(failed=sum(row['state'] == 'failed' for row in values),
                                   skipped=sum(row['state'] == 'skipped' for row in values),
                                   elapsed_seconds=time.monotonic() - started)
        write_json(output / 'artifacts-manifest.json', manifest)

    checkpoint()
    for case_id in ordered:
        definition = by_id[case_id]
        dependencies = definition.get('depends_on', [])
        dependency_rows = {name: manifest['cases'][name] for name in dependencies}
        row = {'id': case_id, 'state': 'running', 'origin': 'executed', 'depends_on': dependencies,
               'started_at': timestamp(), 'artifacts': []}
        manifest['cases'][case_id] = row
        checkpoint()
        if any(parent['state'] != 'succeeded' for parent in dependency_rows.values()):
            row.update(state='skipped', origin='skipped', error='dependency did not succeed', finished_at=timestamp())
            checkpoint()
            continue
        key = digest({'schema_version': 1, 'kind': task['kind'], 'id': case_id,
                      'implementation': definition['implementation'], 'inputs': definition.get('inputs', {}),
                      'dependencies': {name: {'key': parent['key'], 'artifact_digest': parent['artifact_digest']}
                                       for name, parent in dependency_rows.items()}})
        row['key'] = key
        previous = base['manifest']['cases'].get(case_id) if base else None
        reason = 'full execution' if not base else 'new case'
        reusable = False
        if base and previous:
            if not definition.get('reuse', True):
                reason = 'fresh measurement required'
            elif case_id in forced:
                reason = 'explicitly forced'
            elif previous['state'] != 'succeeded':
                reason = 'baseline case did not succeed'
            elif case_id in base.get('unavailable', {}):
                reason = 'baseline artifact unavailable: ' + base['unavailable'][case_id]
            elif previous['key'] != key:
                reason = 'implementation, inputs or dependency changed'
            elif not _artifacts_match(Path(base['root']), previous['artifacts']):
                reason = 'prepared baseline artifact changed'
            else:
                reusable = True
        row['reason'] = 'verified compatible baseline' if reusable else reason
        case_output = output / 'cases' / case_id
        case_output.mkdir(parents=True)
        case_started = time.monotonic()
        try:
            if reusable:
                for artifact in previous['artifacts']:
                    source = _regular_path(Path(base['root']), artifact['path'])
                    target = _regular_path(output, artifact['path'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
                if not _artifacts_match(output, previous['artifacts']):
                    raise IncrementalError('copied reused artifacts failed verification')
                row.update(origin='reused', reused_from={**base['provenance'], 'case_id': case_id})
            else:
                dependency_outputs = {name: output / 'cases' / name for name in dependencies}
                definition['run'](case_output, dependency_outputs)
            artifacts = _artifacts(output, case_id)
            row.update(state='succeeded', artifacts=artifacts, artifact_digest=digest(artifacts))
        except Exception as error:
            row.update(state='failed', error=f'{type(error).__name__}: {error}')
        row.update(finished_at=timestamp(), elapsed_seconds=time.monotonic() - case_started)
        checkpoint()
    manifest['summary']['elapsed_seconds'] = time.monotonic() - started
    return verify_outputs(output, manifest)


def verify_outputs(output, manifest):
    """Recheck committed case artifacts after callbacks or a finalization hook.

    Callers should keep this committed manifest separate from any mutable copy
    supplied to a handler's finalization hook.
    """
    output = Path(output)
    rows = manifest['cases']
    for row in rows.values():
        if row['state'] == 'succeeded' and not _artifacts_match(output, row['artifacts']):
            row.update(state='failed', error='case artifacts changed after their checkpoint')
    # JSON readers need not preserve topological ordering. Iterate until every
    # successful descendant of an invalidated dependency is marked failed.
    changed = True
    while changed:
        changed = False
        for row in rows.values():
            if row['state'] == 'succeeded' and any(rows[parent]['state'] != 'succeeded'
                                                  for parent in row.get('depends_on', [])):
                row.update(state='failed', error='dependency invalidated after its checkpoint')
                changed = True
    manifest['state'] = 'succeeded' if all(row['state'] == 'succeeded' for row in rows.values()) else 'failed'
    manifest['finished_at'] = timestamp()
    summary = manifest.setdefault('summary', {})
    for origin in ('executed', 'reused'):
        summary[origin] = sum(row.get('origin') == origin for row in rows.values())
    for state in ('failed', 'skipped'):
        summary[state] = sum(row['state'] == state for row in rows.values())
    write_json(output / 'artifacts-manifest.json', manifest)
    return manifest
