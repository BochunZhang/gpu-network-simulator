import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import copy
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "profiling/warp_trace/trace.py"
spec = importlib.util.spec_from_file_location("ft48_lifetimes", MODULE)
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def form_window():
    locations, indices, launches = [], {}, []
    for (name, grid, block), repeats in trace.GEDES.items():
        for _ in range(repeats):
            sequence = len(launches)
            lifetimes = []
            for cta in range(grid):
                for warp in range((block + 31) // 32):
                    key = (cta, warp)
                    if key not in indices:
                        indices[key] = len(locations)
                        locations.append(dict(smId=cta % 16, ctaId=[cta, 0, 0], warpId=warp))
                    lifetimes.append(dict(locIdx=indices[key], startTs=10 * sequence + 3, endTs=10 * sequence + 9))
            launches.append(dict(kernelName=name, contextId=1, gridId=sequence, gridDimX=grid, gridDimY=1, gridDimZ=1,
                                 blockDimX=block, blockDimY=1, blockDimZ=1, warpLifetimes=lifetimes))
    return dict(graphLaunches={"graph_exec_4:3": launches}, launches=[], locationTable=locations)


class WarpLifetimeTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_two_serial_replays_in_one_graph_exec_or_flat_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            window = form_window()
            group = window["graphLaunches"]["graph_exec_4:3"]
            following = copy.deepcopy(group)
            for launch in following:
                for warp in launch["warpLifetimes"]:
                    warp["startTs"] += 1000
                    warp["endTs"] += 1000
            group.extend(following)
            for flat in (False, True):
                directory = Path(tmp) / str(flat)
                directory.mkdir()
                if flat:
                    window["launches"] = group
                    window["graphLaunches"] = {}
                (directory / "pending.iket.json").write_text(json.dumps(window))
                trace.ingest(directory, "gedes", "graph", 0, 2)
                self.assertEqual(trace.verify(directory, "gedes", 2)["replays"], 2)

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_nested_graph_records_retain_original_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            window = form_window()
            launches = window["graphLaunches"]["graph_exec_4:3"]
            window["graphLaunches"] = {"graph_exec_2:0": launches[::2], "graph_exec_3:0": launches[1::2]}
            (directory / "pending.iket.json").write_text(json.dumps(window))
            trace.ingest(directory, "gedes", "graph", 0, 1)
            trace.verify(directory, "gedes", 1)
            with sqlite3.connect(directory / "index.sqlite") as db:
                keys = [r[0] for r in db.execute("select graph_key from launches order by node")]
            self.assertEqual(keys, ["graph_exec_2:0", "graph_exec_3:0"] * 22 + ["graph_exec_2:0"])

    def test_duplicate_warp_rejected(self):
        row = dict(locIdx=0, startTs=2, endTs=3)
        launch = dict(kernelName="x", gridDimX=1, gridDimY=1, gridDimZ=1,
                      blockDimX=64, blockDimY=1, blockDimZ=1, warpLifetimes=[row, row])
        with self.assertRaisesRegex(ValueError, "duplicate CTA"):
            trace.store_launch(io.BytesIO(), ([0], [0]), launch)

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_complete_gedes_replay_and_short_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            raw = directory / "pending.iket.json"
            sample = form_window()
            raw.write_text(json.dumps(sample))
            trace.ingest(directory, "gedes", "graph", 0, 1)
            summary = trace.verify(directory, "gedes", 1)
            self.assertEqual(summary["warp_lifetimes"], 23983)
            self.assertEqual(summary["binary_bytes"], 575592)
            self.assertEqual(sum(s["summed_warp_ns"] for s in summary["per_sm"]), 23983 * 6)
            self.assertLess((directory / "preview.perfetto.json").stat().st_size, 1024 * 1024)
            with sqlite3.connect(directory / "index.sqlite") as db:
                offset, = db.execute("select byte_offset from launches where replay=0 and node=0").fetchone()
            with (directory / "lifetimes.bin").open("rb") as data:
                data.seek(offset)
                self.assertEqual(trace.RECORD.unpack(data.read(24)), (0, 0, 0, 3, 9))
            with self.assertRaisesRegex(ValueError, "expected 2 graph"):
                trace.verify(directory, "gedes", 2)
            launch = sample["graphLaunches"]["graph_exec_4:3"][0]
            # Same node count and valid individual geometry, incorrect graph configuration.
            launch["kernelName"] = "update_timer"
            raw.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "expected nodes"):
                trace.ingest(directory, "gedes", "graph", 1, 1)
            launch["kernelName"] = "DeviceScanInitKernel"
            sample["graphLaunches"]["graph_exec_4:3"].pop()
            raw.write_text(json.dumps(sample))
            with self.assertRaisesRegex(ValueError, "expected 45 graph/direct nodes"):
                trace.ingest(directory, "gedes", "graph", 1, 1)
            self.assertEqual((directory / "lifetimes.bin").stat().st_size, 575592)
            with self.assertRaisesRegex(ValueError, "expected 2 graph"):
                trace.verify(directory, "gedes", 2)


if __name__ == "__main__":
    unittest.main()
