import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import experiment_handlers as handlers


class GPUVisibilityTest(unittest.TestCase):
    rows = [[str(index), f'GPU-test{index}', 'GB200', '10.0', '190000', '580', f'PCI{index}']
            for index in range(4)]

    def test_index_and_uuid_visibility(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,0'}, clear=True):
            self.assertEqual(handlers.visible_gpus(self.rows), [self.rows[2], self.rows[0]])
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-test3'}, clear=True):
            self.assertEqual(handlers.visible_gpus(self.rows), [self.rows[3]])
        for mask in ('', '-1'):
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': mask}, clear=True):
                self.assertEqual(handlers.visible_gpus(self.rows), [])

    def test_single_gpu_selection_respects_visible_pool(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(handlers, 'gpu_rows', return_value=self.rows):
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,2'}, clear=True):
                    capability, env = handlers.selected_gpu(Path(temp))
                    self.assertEqual(capability, '10.0')
                    self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test3')
                with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '3,2',
                                             'GPU_NET_SIM_GPU_UUID': 'GPU-test0'}, clear=True):
                    with self.assertRaises(ValueError):
                        handlers.selected_gpu(Path(temp))
