import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import gpu_transfer


class GPUTransferTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.devices = [{'logical_index': index, 'uuid': f'GPU-test{index}',
                         'name': 'GB200', 'compute_capability': '10.0', 'total_memory_bytes': 190000 * 1024**2}
                        for index in range(4)]

    def fixture(self, devices=None):
        devices = devices or self.devices
        context = {'selected_device_count': 4, 'visible_device_count': 4, 'devices': devices,
                   'cuda_runtime_version': 13000, 'cuda_driver_version': 13000,
                   'compiled_cudart_version': 13000, 'cuda_visible_devices': None}
        rows = []
        for size in gpu_transfer.SIZES:
            for source in range(4):
                for destination in range(4):
                    rows.append({'kind': 'd2d' if source == destination else 'p2p',
                                 'src': source, 'dst': destination, 'bytes': size, 'iterations': 10,
                                 'warmup': 3, 'peer_access': 1, 'cuda_ms': 2.0, 'wall_ms': 2.5,
                                 'gb_per_s': size * 10 / 2e6, 'mismatched_bytes': 0,
                                 'src_uuid': devices[source]['uuid'], 'dst_uuid': devices[destination]['uuid']})
        return context, rows

    def write_fixture(self, context, rows):
        (self.output / 'cuda-context.json').write_text(json.dumps(context))
        with (self.output / 'transfer-matrix.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_complete_matrix_and_payload_rate_gate(self):
        self.write_fixture(*self.fixture())
        summary = gpu_transfer.parse_results(self.output)
        self.assertTrue(summary['all_copies_verified'])
        self.assertEqual(len(summary['rows']), 32)
        self.assertEqual(sum(row['kind'] == 'p2p' for row in summary['rows']), 24)
        self.assertEqual(sum(row['kind'] == 'd2d' for row in summary['rows']), 8)
        self.assertIn('payload', summary['bandwidth_unit'])
        self.assertEqual(summary['sizes_bytes'], [1048576, 67108864])

    def test_rejects_missing_duplicate_corrupt_or_nonfinite_measurements(self):
        context, original = self.fixture()
        bad = [original[:-1], [*original[:-1], original[0]]]
        for field, value in [('mismatched_bytes', 1), ('cuda_ms', float('nan')),
                             ('gb_per_s', -1), ('src_uuid', 'GPU-wrong'), ('iterations', 1)]:
            rows = copy.deepcopy(original)
            rows[0][field] = value
            bad.append(rows)
        for rows in bad:
            with self.subTest(row_count=len(rows), first=rows[0]):
                self.write_fixture(context, rows)
                with self.assertRaises(RuntimeError):
                    gpu_transfer.parse_results(self.output)

    def test_unsupported_peer_capability_is_reported(self):
        context, rows = self.fixture()
        row = next(item for item in rows if item['src'] == 0 and item['dst'] == 1)
        row.update(peer_access=0, cuda_ms=0, wall_ms=0, gb_per_s=0, mismatched_bytes=-1)
        self.write_fixture(context, rows)
        summary = gpu_transfer.parse_results(self.output)
        self.assertFalse(summary['all_copies_verified'])
        self.assertEqual(summary['unsupported_pairs'], [{'src': 0, 'dst': 1, 'bytes': 1048576}])

    def helpers(self, reordered=None, fail_build=False, unsupported=False):
        test = self
        physical = [[str(index), f'GPU-test{index}', 'GB200', '10.0', '190000', '580', f'PCI{index}']
                    for index in range(4)]
        visible = reordered or physical
        identities = [{**self.devices[index], 'logical_index': logical}
                      for logical, index in enumerate([int(row[0]) for row in visible])]
        calls = []

        class Helpers:
            @staticmethod
            def gpu_rows(log):
                log.write_text('physical fixture')
                return physical

            @staticmethod
            def visible_gpus(rows):
                return visible

            @staticmethod
            def write_json(path, value):
                path.write_text(json.dumps(value))

            @staticmethod
            def command(args, log, env):
                calls.append((list(args), dict(env)))
                log.write_text('fixture command output')
                if '-o' in args:
                    if fail_build:
                        log.write_text('unsupported CUDA compiler architecture')
                        raise RuntimeError('nvcc returned 1')
                    Path(args[args.index('-o') + 1]).write_bytes(b'compiled fixture')
                elif args[0].endswith('/gpu-transfer'):
                    context, rows = test.fixture(identities)
                    if unsupported:
                        rows[1].update(peer_access=0, cuda_ms=0, wall_ms=0, gb_per_s=0, mismatched_bytes=-1)
                    test.write_fixture(context, rows)
                    if unsupported:
                        raise RuntimeError('benchmark returned 2')
                return 0.02
        return Helpers, calls

    def test_handler_preserves_visible_mask_and_pins_compiled_source(self):
        physical = [[str(index), f'GPU-test{index}', 'GB200', '10.0', '190000', '580', f'PCI{index}']
                    for index in (2, 0, 1, 3)]
        helpers, calls = self.helpers(reordered=physical)
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,0,1,3'}, clear=False):
            gpu_transfer.run(REPO, self.output, helpers)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertTrue(summary['all_copies_verified'])
        self.assertEqual(summary['selected_physical_uuids'], ['GPU-test2', 'GPU-test0', 'GPU-test1', 'GPU-test3'])
        self.assertEqual(summary['program_sha256'], gpu_transfer.digest(REPO / 'benchmarks/gpu_transfer.cu'))
        self.assertEqual(summary['build_architectures'], ['100'])
        self.assertEqual(json.loads((self.output / 'benchmark.json').read_text())['device_buffer_bytes_per_gpu'], 128 * 1024**2)
        self.assertTrue(all(env['CUDA_VISIBLE_DEVICES'] == '2,0,1,3' for _, env in calls))

    def test_native_failure_keeps_matrix_and_essential_error(self):
        helpers, _ = self.helpers(unsupported=True)
        with self.assertRaisesRegex(RuntimeError, 'returned 2'):
            gpu_transfer.run(REPO, self.output, helpers)
        summary = json.loads((self.output / 'summary.json').read_text())
        self.assertFalse(summary['all_copies_verified'])
        self.assertTrue((self.output / 'transfer-matrix.csv').is_file())
        self.assertIn('returned 2', (self.output / 'failure.txt').read_text())

    def test_compile_failure_preserves_important_feedback(self):
        helpers, calls = self.helpers(fail_build=True)
        with self.assertRaisesRegex(RuntimeError, 'nvcc returned'):
            gpu_transfer.run(REPO, self.output, helpers)
        self.assertIn('unsupported CUDA compiler architecture', (self.output / 'failure.txt').read_text())
        self.assertFalse(any(args[0].endswith('/gpu-transfer') for args, _ in calls))

    def test_cuda_host_source_syntax_with_runtime_api_declarations(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('C++ compiler unavailable')
        include = self.root / 'headers'
        include.mkdir()
        (include / 'cuda_runtime.h').write_text(r'''
#pragma once
#include <cstddef>
#define CUDART_VERSION 13000
struct cudaUUID_t { char bytes[16]; };
struct cudaDeviceProp { char name[256]; cudaUUID_t uuid; int major; int minor; size_t totalGlobalMem; };
using cudaStream_t = struct Stream*;
using cudaEvent_t = struct Event*;
using cudaError_t = int;
constexpr int cudaSuccess = 0;
constexpr int cudaErrorPeerAccessAlreadyEnabled = 1;
constexpr unsigned cudaStreamNonBlocking = 1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp*, int);
cudaError_t cudaGetDeviceCount(int*);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaMalloc(void**, size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaEventCreate(cudaEvent_t*);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float*, cudaEvent_t, cudaEvent_t);
cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemcpyPeerAsync(void*, int, const void*, int, size_t, cudaStream_t);
cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t);
cudaError_t cudaDeviceCanAccessPeer(int*, int, int);
cudaError_t cudaDeviceEnablePeerAccess(int, unsigned);
cudaError_t cudaDeviceDisablePeerAccess(int);
cudaError_t cudaGetLastError();
cudaError_t cudaRuntimeGetVersion(int*);
cudaError_t cudaDriverGetVersion(int*);
''')
        checked = subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-pedantic',
                                  '-x', 'c++', '-fsyntax-only', '-I', str(include),
                                  str(REPO / 'benchmarks/gpu_transfer.cu')],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
