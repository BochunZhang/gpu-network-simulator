"""Exercise the Nsight Systems SQLite parser without a GPU."""

import csv
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import nsys_ft48


SCHEMA = """
CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
    (start, end, globalTid, correlationId, nameId, returnValue);
CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
    (start, end, deviceId, correlationId, graphNodeId, graphId, shortName,
     gridX, gridY, gridZ, blockX, blockY, blockZ);
CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
    (start, end, deviceId, correlationId, graphNodeId, graphId, bytes, copyKind);
CREATE TABLE ENUM_CUDA_MEMCPY_OPER (id, label);
INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES (2, 'Device to Host');
CREATE TABLE GPU_METRICS (timestamp, typeId, metricId, value);
CREATE TABLE TARGET_INFO_GPU_METRICS (typeId, metricId, metricName);
CREATE TABLE DIAGNOSTIC_EVENT (text TEXT);
"""


class NsysFT48Tests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript(SCHEMA)
        self.report = {"errors": [], "warnings": []}
        self.db.executemany("INSERT INTO StringIds VALUES (?, ?)", enumerate(
            ("cudaGraphLaunch_v10000", "cuGraphLaunch_v10000", "cudaStreamSynchronize_v3020",
             "forward_frames", "days_horizon_sweep", "days_round",
             "days_round_finalize", "other_days_stage"), 1))
        for i, name in enumerate(("SMs Active", "SM Issue", "DRAM Read Bandwidth", "DRAM Write Bandwidth"), 1):
            self._add_metric(i, name)

    def _add_metric(self, metric_id, name):
        self.db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)", (500, metric_id, name))
        self.db.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)",
                            [(30, 500, metric_id, 0), (110, 500, metric_id, 80)])

    def test_capture_switch_preserves_full_run_prefix(self):
        full = nsys_ft48.prefix("nsys", self.output)
        self.assertIn("--gpu-metrics-set=gb10x", full)
        self.assertIn("--capture-range=none", full)
        self.assertNotIn("--capture-range-end=stop", full)
        window = nsys_ft48.prefix("nsys", self.output, capture_range=True)
        position = full.index("--capture-range=none")
        self.assertEqual(window, full[:position] + [
            "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
        ] + full[position + 1:])
        triage = nsys_ft48.prefix("nsys", self.output, capture_range=True, metric_set="gb10x-ct")
        self.assertEqual(triage, ["--gpu-metrics-set=gb10x-ct" if arg == "--gpu-metrics-set=gb10x"
                                  else arg for arg in window])

    def test_preflight_requires_the_requested_alias_on_the_selected_gpu(self):
        nsys = self.output / f"NsightSystems-{nsys_ft48.VERSION}/bin/nsys"
        nsys.parent.mkdir(parents=True)
        nsys.touch(mode=0o755)
        env = {"GPU_NET_SIM_REPO_ROOT": str(self.output), "CUDA_VISIBLE_DEVICES": "GPU-abc"}
        sets = "    gb10x : General metrics\n"

        def command(args, log, **kwargs):
            log.write_text(nsys_ft48.VERSION if "--version" in args else sets)

        helpers = Mock()
        helpers.command.side_effect = command
        with self.assertRaisesRegex(RuntimeError, "gb10x-ct"):
            nsys_ft48.preflight(self.output, env, helpers, metric_set="gb10x-ct")
        sets += "    gb10x-ct : Compute triage metrics\n"
        self.assertEqual(nsys_ft48.preflight(self.output, env, helpers, metric_set="gb10x-ct"), nsys.resolve())

    def _add_window_replays(self):
        self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?, ?, 0)", (
            (10, 20, 150, 1), (30, 90, 152, 3), (100, 112, 160, 1), (113, 180, 161, 3),
        ))
        self.db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, 673, 71, 4, 2, 1, 1, 64, 1, 1)",
            [(24, 78, 150), (115, 175, 160)])

    def _collect_window(self, window, *, lost=0, simulator="gedes", rounds=95851, **profile_options):
        self.db.commit()
        source = self.db
        output = self.output
        (output / "profile.nsys-rep").write_bytes(b"native report stub")

        class Helpers:
            @staticmethod
            def command(args, log, **_kwargs):
                if args[1] == "survey":
                    log.write_text("".join(json.dumps({
                        "type": "capture", "name": name, "value": value,
                    }) + "\n" for name, value in (("Duration (s)", 10.002), ("Lost Events", lost))))
                else:
                    with sqlite3.connect(output / "profile.sqlite") as target:
                        source.backup(target)

            @staticmethod
            def write_json(path, value):
                path.write_text(json.dumps(value))

        return nsys_ft48.collect("nsys", output, {}, Helpers, simulator=simulator,
                                rounds=rounds, capture_window=window, **profile_options)

    def test_compute_triage_exports_distinct_percent_series_and_requires_sol(self):
        self._add_window_replays()
        self.db.execute("DELETE FROM TARGET_INFO_GPU_METRICS")
        self.db.execute("DELETE FROM GPU_METRICS")
        labels = ("SMs Active", "SM Issue Active", "SM Throughput", "DRAM Read Bandwidth",
                  "DRAM Write Bandwidth", "SM ALU Pipe Throughput", "SM Warps Stalled on L1 Operation")
        for i, name in enumerate(labels, 1):
            self._add_metric(i, name + " [Throughput %]")
        self._add_metric(8, "L2 Sector Hit Rate [Ratio %]")
        self._add_metric(9, "SM ALU Pipe Throughput [Avg]")
        window = {"first_round": 0, "end_round": 2, "wall_ns": 10_000_002_200, "requested_seconds": 10}
        report = self._collect_window(window, metric_set="gb10x-ct")
        self.assertEqual(report["metric_set"], "gb10x-ct")
        for name in (*labels, "L2 Sector Hit Rate"):
            self.assertEqual(report["gpu_metrics"][name]["sample_point_mean_pct"], 40)
        with (self.output / "profile-metric-windows.csv").open(newline="") as stream:
            recorded = {row["metric"] for row in csv.DictReader(stream)}
        self.assertIn("SM Throughput [Throughput %]", recorded)
        self.assertIn("SM Issue Active [Throughput %]", recorded)
        self.assertNotIn("SM ALU Pipe Throughput [Avg]", recorded)

        self.db.execute("DELETE FROM TARGET_INFO_GPU_METRICS WHERE metricName='SM Throughput [Throughput %]'")
        with self.assertRaisesRegex(RuntimeError, "missing usable GPU series: SM Throughput"):
            self._collect_window(window, metric_set="gb10x-ct")

    def test_capture_local_graph_replay_count_differs_from_full_run(self):
        self._add_window_replays()
        window = {"first_round": 0, "end_round": 2, "wall_ns": 10_000_002_200,
                  "requested_seconds": 10}
        report = self._collect_window(window)
        self.assertTrue(report["passed"])
        self.assertEqual(report["capture_window"], window)
        self.assertEqual(report["capture_rounds_reported"], 2)
        self.assertEqual(report["simulation_rounds_reported"], 95851)
        self.assertEqual(report["host_graph_replays"], 2)
        self.assertEqual(report["capture_survey"], {"duration_seconds": 10.002, "lost_events": 0})
        with (self.output / "profile-replays.csv").open(newline="") as f:
            replays = list(csv.DictReader(f))
        self.assertEqual([row["capture_local_round"] for row in replays], ["0", "1"])

    def test_failed_capture_still_exports_and_preserves_raw_evidence(self):
        self._add_window_replays()
        window = {"first_round": 0, "end_round": 3, "wall_ns": 9_999_999_999,
                  "requested_seconds": 10}
        with patch.object(nsys_ft48, "MAX_WINDOW_REPORT_BYTES", 8):
            with self.assertRaisesRegex(RuntimeError, "native report exceeds 64 MiB"):
                self._collect_window(window, lost=1)
        report = json.loads((self.output / "profile-summary.json").read_text())
        self.assertFalse(report["passed"])
        self.assertEqual(set(report["raw_files"]), {"profile.nsys-rep", "profile.sqlite"})
        self.assertEqual(report["capture_survey"]["lost_events"], 1)
        self.assertIn("GeDES capture window ended before its requested wall duration", report["errors"])
        self.assertIn("GeDES CUDA graph launch count does not match captured simulation rounds",
                      report["errors"])
        self.assertIn("Nsight Systems survey reports lost events", report["errors"])

    def test_nested_host_launch_and_synchronized_node_fallback(self):
        self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?, ?, 0)", (
            (10, 20, 150, 1),
            (12, 18, 151, 2),
            (30, 90, 152, 3),
            (100, 112, 160, 1),
            (113, 180, 161, 3),
        ))
        self.db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, 71, 4, 2, 1, 1, 64, 1, 1)",
            [(24, 78, 151, 673), (26, 60, 999, 674), (115, 175, 160, 673)])
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (75, 85, 0, 150, 675, 71, 8, 2)")
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES ('Using software trace')")
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES (NULL)")
        nsys_ft48._analyze(self.db, self.output, "gedes", 2, self.report)
        self.assertFalse(self.report["errors"])
        self.assertEqual(self.report["host_graph_replays"], 2)
        self.assertEqual(self.report["graph_nodes_assigned_by_synchronized_time_window"], 1)
        self.assertEqual(self.report["graph_kernel_nodes_per_replay"], {2: 1, 1: 1})
        self.assertEqual(self.report["graph_kernel_duration_sum_ns_by_name"], {"forward_frames": 148})
        self.assertEqual(self.report["graph_device_union_ns"], 121)
        self.assertEqual(self.report["device_memory_copies"], [{
            "copy_kind": "Device to Host", "within_graph_loop": True,
            "records": 1, "bytes": 8, "duration_sum_ns": 10,
        }])
        self.assertEqual(self.report["gpu_metrics"]["SMs Active"]["sample_point_mean_pct"], 40)
        self.assertEqual(self.report["trace_diagnostics"], ["Using software trace"])

    def _add_days_wave(self, start, correlation):
        self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?, ?, 0)",
                            [(start, start + 10, correlation, 2),
                             (start + 12, start + 8990, correlation + 1, 3)])
        nodes = (5, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 7) * 64
        self.db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, ?, ?, 71, ?, 2, 1, 1, 64, 1, 1)",
            [(start + 15 + i * 2, start + 16 + i * 2, correlation, i + 1, name_id)
             for i, name_id in enumerate(nodes)])

    def test_days_attempts_are_separate_from_committed_rounds(self):
        self._add_days_wave(10, 150)
        nsys_ft48._analyze(self.db, self.output, "days-ago", 57, self.report)
        self.assertFalse(self.report["errors"])
        self.assertEqual(self.report["days_graph_encoded_attempts"], 64)
        self.assertEqual(self.report["simulation_rounds_reported"], 57)
        self.assertEqual(self.report["days_encoded_attempt_kernel_node_counts"], {12: 64})
        self.assertEqual(len((self.output / "profile-days-attempts.csv").read_text().splitlines()), 65)
        self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId=3")
        self.report = {"errors": [], "warnings": []}
        nsys_ft48._analyze(self.db, self.output, "days-ago", 57, self.report)
        self.assertIn("Days graph attempt nodes do not match the pinned attempt DAG and waves",
                      self.report["errors"])

    def test_days_ct_window_counts_waves_and_committed_rounds_independently(self):
        self._add_days_wave(10, 150)
        self._add_days_wave(10_000, 160)
        self.db.execute("UPDATE TARGET_INFO_GPU_METRICS SET metricName='SM Issue Active' WHERE metricId=2")
        self._add_metric(5, "SM Throughput [Throughput %]")
        self.db.execute("INSERT INTO GPU_METRICS VALUES (10100, 500, 5, 25)")
        window = {"first_wave": 0, "end_wave": 2, "first_round": 0, "end_round": 103,
                  "wall_ns": 10_020_000_000, "requested_seconds": 10}
        report = self._collect_window(window, simulator="days-ago", rounds=1002, metric_set="gb10x-ct")
        self.assertTrue(report["passed"])
        self.assertEqual(report["capture_waves_reported"], 2)
        self.assertEqual(report["host_graph_replays"], 2)
        self.assertEqual(report["days_graph_encoded_attempts"], 128)
        self.assertEqual(report["capture_rounds_reported"], 103)
        self.assertEqual(report["simulation_rounds_reported"], 1002)
        with (self.output / "profile-replays.csv").open() as f:
            self.assertEqual([row["graph_wave"] for row in csv.DictReader(f)], ["0", "1"])
        with (self.output / "profile-metric-windows.csv").open() as f:
            self.assertEqual([(r["first_replay"], r["last_replay"], r["mean_pct"]) for r in csv.DictReader(f)
                              if r["metric"] == "SM Throughput [Throughput %]"], [("0", "0", "40.0"),
                                                                                 ("1", "1", "25.0")])
        for change, error in (
            ({"end_wave": 1}, "launch count does not match captured graph waves"),
            ({"end_round": 129}, "attempt nodes do not match"),
            ({"wall_ns": 9_000_000_000}, "Days capture window ended before its requested wall duration"),
        ):
            with self.subTest(change=change), self.assertRaisesRegex(RuntimeError, error):
                self._collect_window({**window, **change}, simulator="days-ago", rounds=1002,
                                     metric_set="gb10x-ct")
            failed = json.loads((self.output / "profile-summary.json").read_text())
            self.assertFalse(failed["passed"])
            self.assertTrue(failed["raw_files"]["profile.sqlite"]["bytes"])


if __name__ == "__main__":
    unittest.main()
