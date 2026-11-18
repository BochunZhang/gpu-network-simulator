import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import task_catalog as catalog


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.parent = 'gedes-000-multi-thread-test'
        self.first = '2026-11-18T10:00:00Z'
        self.later = '2026-11-18T11:00:00Z'
        self.path = f'tasks/{self.parent}.json'
        self.manifest = {'id': self.parent, 'name': 'GeDES multithread',
                         'status': self.first, 'subtasks': {'00': {
                             'status': self.first, 'kind': 'fixture', 'params': {'topos': ['A', 'B', 'C']}}}}

    def test_independent_release_and_ignore(self):
        [(original, digest)] = catalog.expand(self.manifest, self.path)
        current = copy.deepcopy(self.manifest)
        current['subtasks']['01'] = {'status': self.later, 'kind': 'fixture', 'params': {'topos': ['D']}}
        catalog.validate_transition(self.manifest, current)
        retained, (added, _) = catalog.expand(current, self.path)
        self.assertEqual(retained, (original, digest))
        self.assertEqual(added['id'], self.parent + '-01')
        self.assertEqual(added['subtask_status'], self.later)
        self.assertEqual(added['command'], f'bash tools/runner.sh --task {self.parent}-01 --once')
        rerun = copy.deepcopy(current)
        rerun['subtasks']['00']['status'] = self.later
        rerun['subtasks']['00']['params']['debug'] = True
        catalog.validate_transition(current, rerun)
        self.assertGreater(catalog.expand(rerun, self.path)[0][0]['version'], original['version'])
        current['subtasks']['00']['status'] = 'ignore'
        tasks = catalog.expand(current, self.path)
        self.assertFalse(tasks[0][0]['enabled'])
        self.assertTrue(tasks[1][0].get('enabled', True))
        current['status'] = 'ignore'
        self.assertTrue(all(not t['enabled'] for t, _ in catalog.expand(current, self.path)))

    def test_migrated_release_keeps_identity_and_payload_digest(self):
        self.manifest['legacy'] = {'id': '001', 'slug': 'gedes-smoke', 'version': 3,
                                   'published_at': self.first}
        [(existing, digest)] = catalog.expand(self.manifest, self.path)
        self.assertEqual((existing['id'], existing['slug'], existing['version']), ('001', 'gedes-smoke', 3))
        old_payload = {'id': '001', 'version': 3, 'kind': 'fixture',
                       'name': self.manifest['name'], 'params': {'topos': ['A', 'B', 'C']}}
        encoded = json.dumps(old_payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
        self.assertEqual(digest, hashlib.sha256(encoded.encode()).hexdigest())
        self.manifest['subtasks']['00']['status'] = self.later
        self.manifest['subtasks']['01'] = dict(self.manifest['subtasks']['00'])
        (updated, _), (new, _) = catalog.expand(self.manifest, self.path)
        self.assertEqual(updated['id'], self.parent + '-00')
        self.assertEqual(updated['legacy_id'], '001')
        self.assertNotIn('legacy_id', new)
        self.assertEqual(new['legacy_base'], {'id': '001', 'slug': 'gedes-smoke'})
        self.manifest['subtasks']['00']['status'] = 'ignore'
        self.assertEqual(catalog.expand(self.manifest, self.path)[0][0]['legacy_id'], '001')

    def test_invalid_transitions(self):
        changed = copy.deepcopy(self.manifest)
        changed['status'] = self.later
        with self.assertRaisesRegex(ValueError, 'parent publication'):
            catalog.validate_transition(self.manifest, changed)
        changed = copy.deepcopy(self.manifest)
        changed['subtasks']['00']['params'] = {}
        with self.assertRaisesRegex(ValueError, 'requires a new'):
            catalog.validate_transition(self.manifest, changed)
        changed['subtasks'].clear()
        with self.assertRaisesRegex(ValueError, 'remain in the catalog'):
            catalog.validate_transition(self.manifest, changed)
        changed = copy.deepcopy(self.manifest)
        changed['subtasks']['00']['status'] = 'ignore'
        with self.assertRaisesRegex(ValueError, 'retired child'):
            catalog.validate_transition(changed, self.manifest)
        changed = copy.deepcopy(self.manifest)
        changed['subtasks']['00']['status'] = self.later
        with self.assertRaisesRegex(ValueError, 'backwards'):
            catalog.validate_transition(changed, self.manifest)
        # Pausing publication preserves both the release and its digest.
        changed = copy.deepcopy(self.manifest)
        changed['subtasks']['00']['enabled'] = False
        catalog.validate_transition(self.manifest, changed)
        self.assertEqual(catalog.expand(self.manifest, self.path)[0][1],
                         catalog.expand(changed, self.path)[0][1])

    def test_invalid_manifest_and_timestamp(self):
        for status in ('2026-11-18T10:00:00', '2026-02-30T00:00:00Z', None, 1, 'ignored'):
            with self.subTest(status=status), self.assertRaises(ValueError):
                catalog.release_version(status)
        with self.assertRaises(ValueError):
            catalog.expand(self.manifest, 'tasks/gedes-001-other.json')
        self.manifest['subtasks']['0'] = self.manifest['subtasks'].pop('00')
        with self.assertRaises(ValueError):
            catalog.expand(self.manifest, self.path)
