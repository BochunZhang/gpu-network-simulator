"""Expand the public experiment catalog into independently claimable releases."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from incremental import validate_execution


SHORT_NAME_PATTERN = r'[a-z0-9]+(?:-[a-z0-9]+)*'
CATEGORY_PATTERN = r'[a-z][a-z0-9]*(?:-[a-z][a-z0-9]*)*'
PARENT_PATTERN = rf'(?P<category>{CATEGORY_PATTERN})-(?P<index>[0-9]{{3}})-(?P<slug>{SHORT_NAME_PATTERN})'
CHILD_PATTERN = rf'{PARENT_PATTERN}-[0-9]{{2}}'
STATUS_PATTERN = r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z'
CHILD_FIELDS = {'status', 'name', 'kind', 'params', 'execution', 'resources', 'enabled', 'description'}


def release_version(status):
    """Return Unix seconds, reserving None for retired catalog entries."""
    if status == 'ignore':
        return None
    if not isinstance(status, str) or not re.fullmatch(STATUS_PATTERN, status):
        raise ValueError('status must be UTC YYYY-MM-DDTHH:MM:SSZ or ignore')
    value = datetime.strptime(status, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    seconds = int(value.timestamp())
    if seconds < 1:
        raise ValueError('release timestamp must be after the Unix epoch')
    return seconds


def execution_digest(task):
    # The same payload yields exactly the digest written by numeric-version controllers.
    metadata = {'description', 'enabled', 'slug', 'command', 'parent_id', 'subtask_id',
                'task_status', 'subtask_status', 'legacy_id', 'legacy_slug', 'legacy_base'}
    definition = {key: value for key, value in task.items() if key not in metadata}
    encoded = json.dumps(definition, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    return hashlib.sha256(encoded.encode()).hexdigest()


def expand(manifest, path):
    """Validate a parent file and yield every child, including disabled or retired ones."""
    if not isinstance(manifest, dict) or set(manifest) - {'id', 'name', 'status', 'subtasks', 'legacy'}:
        raise ValueError(f'invalid catalog manifest: {path}')
    parent = manifest.get('id')
    parent_match = re.fullmatch(PARENT_PATTERN, parent) if isinstance(parent, str) else None
    if (not parent_match or len(parent) > 96 or len(parent_match['slug']) > 63
            or Path(path).name != f'{parent}.json'
            or not isinstance(manifest.get('name'), str) or not manifest['name'].strip()
            or not isinstance(manifest.get('subtasks'), dict)):
        raise ValueError(f'invalid catalog parent: {path}')
    parent_version = release_version(manifest.get('status'))
    legacy = manifest.get('legacy')
    if 'legacy' in manifest:
        if (not isinstance(legacy, dict) or set(legacy) != {'id', 'slug', 'version', 'published_at'}
                or not isinstance(legacy['id'], str) or not re.fullmatch(r'[0-9]{3}', legacy['id'])
                or not isinstance(legacy['slug'], str) or len(legacy['slug']) > 63
                or not re.fullmatch(SHORT_NAME_PATTERN, legacy['slug'])
                or type(legacy['version']) is not int or legacy['version'] < 1
                or release_version(legacy['published_at']) is None or '00' not in manifest['subtasks']):
            raise ValueError(f'invalid legacy release mapping: {path}')

    tasks = []
    for index, child in sorted(manifest['subtasks'].items()):
        if (not isinstance(index, str) or not re.fullmatch(r'[0-9]{2}', index)
                or not isinstance(child, dict) or set(child) - CHILD_FIELDS
                or not isinstance(child.get('kind'), str) or not child['kind'].strip()
                or not isinstance(child.get('params'), dict)
                or type(child.get('enabled', True)) is not bool
                or not isinstance(child.get('description', ''), str)
                or not isinstance(child.get('name', ''), str)
                or 'name' in child and not child['name'].strip()):
            raise ValueError(f'invalid child definition: {path} {index}')
        resources = child.get('resources', {})
        if (not isinstance(resources, dict) or set(resources) - {'gpus'}
                or type(resources.get('gpus', 0)) is not int or resources.get('gpus', 0) < 0):
            raise ValueError(f'invalid child resources: {path} {index}')
        version = release_version(child.get('status'))
        if parent_version and version and version < parent_version:
            raise ValueError(f'child predates its parent: {path} {index}')
        child_id = f'{parent}-{index}'
        task = {key: value for key, value in child.items() if key != 'status'}
        task.update(id=child_id, name=child.get('name', manifest['name']),
                    slug=parent_match['slug'], version=version or 1,
                    command=f'bash tools/runner.sh --task {child_id} --once',
                    parent_id=parent, subtask_id=index, task_status=manifest['status'],
                    subtask_status=child['status'])
        if legacy:
            task['legacy_base'] = {'id': legacy['id'], 'slug': legacy['slug']}
            if index == '00':
                task.update(legacy_id=legacy['id'], legacy_slug=legacy['slug'])
                if child['status'] == legacy['published_at'] or version is None:
                    task.update(id=legacy['id'], slug=legacy['slug'], version=legacy['version'],
                                command=f"bash tools/runner.sh --task {legacy['id']} --once")
                elif version < release_version(legacy['published_at']):
                    raise ValueError(f'child predates its legacy release: {path} {index}')
        if parent_version is None or version is None:
            task['enabled'] = False
        if version is not None and 'execution' in task:
            validate_execution(task)
        tasks.append((task, execution_digest(task)))
    return tasks


def validate_transition(previous, current):
    """Validate the edit between two publications of the same parent manifest."""
    parent = previous['id']
    if current['id'] != parent or current.get('legacy') != previous.get('legacy'):
        raise ValueError(f'changed task identity or legacy mapping: {parent}')
    if current['status'] != previous['status'] and current['status'] != 'ignore':
        raise ValueError(f'parent publication timestamp is immutable: {parent}')
    before, after = previous['subtasks'], current['subtasks']
    if set(before) - set(after):
        raise ValueError(f'published children must remain in the catalog: {parent}')
    if before and any(index <= max(before) for index in set(after) - set(before)):
        raise ValueError(f'new children must follow previous indices: {parent}')
    if previous['status'] == 'ignore' and previous != current:
        raise ValueError(f'retired parent is immutable: {parent}')
    for index, old in before.items():
        new = after[index]
        if old['status'] == 'ignore' and new != old:
            raise ValueError(f'retired child is immutable: {parent}-{index}')
        if new['status'] == 'ignore':
            if {**old, 'status': 'ignore'} != new:
                raise ValueError(f'retirement must preserve the child definition: {parent}-{index}')
            continue
        if new['status'] < old['status']:
            raise ValueError(f'child publication timestamp moved backwards: {parent}-{index}')
        if new['status'] == old['status']:
            def payload(child, parent_name):
                return {**{k: v for k, v in child.items() if k not in {'enabled', 'description'}},
                        'name': child.get('name', parent_name)}
            if payload(new, current['name']) != payload(old, previous['name']):
                raise ValueError(f'changed child requires a new publication timestamp: {parent}-{index}')
