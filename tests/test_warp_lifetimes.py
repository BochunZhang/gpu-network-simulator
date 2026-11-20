import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import copy
import tempfile
import unittest
from unittest.mock import patch

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

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_empty_invalid_graph_key_copies_preserve_complete_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            window = form_window()
            launches = window["graphLaunches"].pop("graph_exec_4:3")
            window["graphLaunches"] = {
                "graph_exec_2:4294967295": [dict(launch, warpLifetimes=[]) for launch in launches],
                "graph_exec_2:0": launches,
            }
            (directory / "pending.iket.json").write_text(json.dumps(window))
            trace.ingest(directory, "gedes", "graph", 0, 1)
            summary = trace.verify(directory, "gedes", 1)
            self.assertEqual(summary["kernel_launches"], 45)
            self.assertEqual(summary["warp_lifetimes"], 23983)
            with sqlite3.connect(directory / "index.sqlite") as db:
                self.assertEqual(db.execute("select distinct graph_key from launches").fetchall(), [("graph_exec_2:0",)])

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_invalid_graph_key_does_not_hide_capture_errors(self):
        window = form_window()
        launches = window["graphLaunches"]["graph_exec_4:3"]
        empty = [dict(launch, warpLifetimes=[]) for launch in launches]
        invalid_key = "graph_exec_4:4294967295"
        cases = {
            "unmatched grid": (invalid_key, [dict(empty[0], gridId=-1), *empty[1:]], "empty IKET graph group"),
            "different executable": ("graph_exec_5:4294967295", empty, "empty IKET graph group"),
            "mismatched metadata": (invalid_key, [dict(empty[0], enabledCluster={"x": 2, "y": 1, "z": 1}),
                                                  *empty[1:]], "empty IKET graph group"),
            "duplicate placeholder": (invalid_key, [*empty, empty[0]], "empty IKET graph group"),
            "incomplete placeholder group": (invalid_key, empty[:-1], "empty IKET graph group"),
            "mixed populated and empty rows": (invalid_key, [launches[0], *empty[1:]], "empty IKET graph group"),
            "partial lifetimes": (invalid_key, [dict(launches[0], warpLifetimes=launches[0]["warpLifetimes"][:1])],
                                  "expected 4 warp lifetimes, got 1"),
            "empty regular launch": ("graph_exec_4:0", empty, "expected 4 warp lifetimes, got 0"),
        }
        for name, (key, rows, error) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                invalid = dict(window, graphLaunches={**window["graphLaunches"], key: rows})
                (directory / "pending.iket.json").write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, error):
                    trace.ingest(directory, "gedes", "graph", 0, 1)
                self.assertEqual((directory / "lifetimes.bin").stat().st_size, 0)
                with sqlite3.connect(directory / "index.sqlite") as db:
                    self.assertEqual(db.execute("select count(*) from launches").fetchone(), (0,))

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
            self.assertNotIn("requested_graph_replays", summary)
            self.assertNotIn("simulation_graph_replays", summary)
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

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "install the capture dependency to test streaming")
    def test_gedes_graph_prefix_requires_exact_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "pending.iket.json").write_text(json.dumps(form_window()))
            trace.ingest(directory, "gedes", "graph", 0, 1)
            summary = trace.verify(directory, "gedes", 3, graph_replays=1)
            self.assertEqual(summary["simulation_rounds"], 3)
            self.assertEqual(summary["simulation_graph_replays"], 3)
            self.assertEqual(summary["requested_graph_replays"], 1)
            self.assertEqual(summary["replays"], 1)
            self.assertEqual(summary["kernel_launches"], 45)
            self.assertTrue(summary["coverage_passed"])
            self.assertEqual(trace.verify(directory, "gedes", 1, graph_replays=5)["replays"], 1)
            with self.assertRaisesRegex(ValueError, "expected 2 graph"):
                trace.verify(directory, "gedes", 3, graph_replays=2)
            trace.ingest(directory, "gedes", "graph", 1, 1)
            with self.assertRaisesRegex(ValueError, "expected 1 graph"):
                trace.verify(directory, "gedes", 3, graph_replays=1)

    def test_days_graph_prefix_keeps_all_direct_gathers(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(trace, "export"):
            directory = Path(tmp)
            index = directory / "index.sqlite"
            binary = directory / "lifetimes.bin"
            with sqlite3.connect(index) as db:
                db.executescript(trace.SCHEMA)

            def append_chunk(phase, replay, geometry):
                warp_counts = [grid * ((block + 31) // 32) for _, grid, block in geometry]
                warps = sum(warp_counts)
                with binary.open("ab") as data, sqlite3.connect(index) as db:
                    start = data.tell()
                    end = start + warps * trace.RECORD.size
                    data.truncate(end)
                    chunk = db.execute("insert into chunks values(null,?,?,1,?,?,'days',1)",
                                       (phase, replay, start, end)).lastrowid
                    db.execute("insert into sm_totals values(?,0,?,0)", (chunk, warps))
                    for node, ((_, grid, block), count) in enumerate(zip(geometry, warp_counts)):
                        db.execute("insert into launches values(?,?,?,?,'',0,'1','0',1,?,?,?,0,0,0,null)",
                                   (phase, replay, node, chunk, grid, block, count))

            append_chunk("graph", 0, trace.DAYS * 64)
            append_chunk("direct", 0, [(trace.GATHER, 576, 256)])
            append_chunk("direct", 1, [(trace.GATHER, 93, 256)])
            with self.assertRaisesRegex(ValueError, "'direct': 2"):
                trace.verify(directory, "days", 129, graph_replays=1)
            append_chunk("direct", 2, [(trace.GATHER, 576, 256)])
            summary = trace.verify(directory, "days", 129, graph_replays=1)
            self.assertEqual(summary["simulation_rounds"], 129)
            self.assertEqual(summary["simulation_graph_replays"], 3)
            self.assertEqual(summary["requested_graph_replays"], 1)
            self.assertEqual(summary["replays"], 1)
            self.assertEqual(summary["direct_launches"], 3)
            self.assertEqual(summary["kernel_launches"], 771)
            with sqlite3.connect(index) as db:
                db.execute("update launches set grid_x=94 where phase='direct' and replay=1")
            with self.assertRaisesRegex(ValueError, "missing Days compact gathers"):
                trace.verify(directory, "days", 129, graph_replays=1)

    def test_graph_replay_limit_requires_a_positive_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            for graph_replays in (0, -1, True, 1.5):
                with self.subTest(graph_replays=graph_replays):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        trace.verify(tmp, "gedes", 2, graph_replays=graph_replays)


if __name__ == "__main__":
    unittest.main()
