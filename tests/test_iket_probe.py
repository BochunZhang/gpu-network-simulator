import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import iket_probe


def sample_trace(**launch_overrides):
    launch = {
        'kernelName': 'iket_probe_kernel',
        'gridDimX': 1, 'gridDimY': 1, 'gridDimZ': 1,
        'blockDimX': 32, 'blockDimY': 1, 'blockDimZ': 1,
        'warpLifetimes': [{'locIdx': 0, 'startTs': 2, 'endTs': 5}],
        **launch_overrides,
    }
    return {'launches': [launch], 'locationTable': [{'ctaId': [0, 0, 0], 'warpId': 0, 'smId': 2}]}


class IketProbeTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'capture.json'

    def validate(self, trace, mode='direct'):
        self.path.write_text(json.dumps(trace), encoding='utf-8')
        return iket_probe.validate_trace(self.path, mode)

    def test_one_direct_or_flat_graph_launch(self):
        trace = sample_trace()
        self.assertIsInstance(self.validate(trace), dict)
        self.assertIsInstance(self.validate(trace, 'graph-early'), dict)

        trace['graphLaunches'] = {'graph_exec_1:0': trace.pop('launches')}
        self.assertIsInstance(self.validate(trace, 'graph-late'), dict)
        with self.assertRaises(ValueError):
            self.validate(trace)

    def test_rejects_incomplete_or_inconsistent_trace(self):
        duplicate_launch = sample_trace()
        duplicate_launch['graphLaunches'] = {'graph_exec_1:0': duplicate_launch['launches']}
        duplicate_warp = sample_trace()
        duplicate_warp['launches'][0]['warpLifetimes'] *= 2
        wrong_origin = sample_trace()
        wrong_origin['locationTable'][0]['ctaId'] = [1, 0, 0]
        wrong_warp = sample_trace()
        wrong_warp['locationTable'][0]['warpId'] = 1
        invalid_sm = sample_trace()
        invalid_sm['locationTable'][0]['smId'] = -1
        for label, trace in {
            'duplicate launch': duplicate_launch,
            'duplicate warp': duplicate_warp,
            'unexpected CTA': wrong_origin,
            'unexpected warp': wrong_warp,
            'invalid SM': invalid_sm,
            'no warp lifetimes': sample_trace(warpLifetimes=[]),
            'different grid': sample_trace(gridDimY=2),
            'different block size': sample_trace(blockDimX=64),
            'other kernel': sample_trace(kernelName='unrelated_kernel'),
            'invalid location index': sample_trace(warpLifetimes=[{'locIdx': 1, 'startTs': 2, 'endTs': 5}]),
            'backwards interval': sample_trace(warpLifetimes=[{'locIdx': 0, 'startTs': 5, 'endTs': 2}]),
        }.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.validate(trace, 'graph-early')

    def test_outcome_requires_early_graph_and_direct_controls(self):
        modes = ('direct', 'graph-late', 'graph-early')
        for states, expected in (
            (('valid', 'graph-state-abort', 'valid'), 'timing-sensitive'),
            (('valid', 'valid', 'valid'), 'not-reproduced'),
            (('valid', 'failed', 'valid'), 'inconclusive'),
            (('failed', 'graph-state-abort', 'valid'), 'inconclusive'),
            (('valid', 'graph-state-abort', 'graph-state-abort'), 'inconclusive'),
        ):
            with self.subTest(states=states):
                cases = {mode: {'state': state} for mode, state in zip(modes, states)}
                self.assertEqual(iket_probe.outcome(cases), expected)


if __name__ == '__main__':
    unittest.main()
