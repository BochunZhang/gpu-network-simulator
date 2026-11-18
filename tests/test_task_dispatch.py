from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import experiment_handlers as handlers


class TaskDispatchTest(unittest.TestCase):
    def test_new_transfer_handler_uses_pinned_file_and_helpers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'tools').mkdir()
            (root / 'output').mkdir()
            (root / 'tools/gpu_transfer.py').write_text(
                'def run(source, output, helpers):\n'
                '    (output / "dispatch.txt").write_text(str(source) + helpers.GEDES[1])\n')
            with patch.dict(sys.modules, {'gpu_transfer': types.ModuleType('old_gpu_transfer')}):
                handlers.prepare('gpu_transfer', {})(root, root / 'output')
            self.assertEqual((root / 'output/dispatch.txt').read_text(), str(root) + handlers.GEDES[1])

    def test_audit_finalize_keeps_its_task_version_support(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for version in (1, 2):
                root = Path(temporary) / str(version)
                (root / 'tools').mkdir(parents=True)
                (root / 'output').mkdir()
                (root / 'tools/gedes_experiments.py').write_text(
                    'def case_plan(task, source, helpers):\n    return []\n'
                    'def finalize_cases(task, output, manifest, helpers):\n'
                    f'    (output / "dispatch.txt").write_text("source-{version}")\n')
                task = {'id': '001', 'version': version, 'kind': 'gedes_audit', 'params': {}}
                handlers.case_plan(task, root)
                paths.append((task, root))
            for task, root in paths:
                handlers.finalize_cases(task, root / 'output', {})
                self.assertEqual((root / 'output/dispatch.txt').read_text(), f'source-{task["version"]}')
