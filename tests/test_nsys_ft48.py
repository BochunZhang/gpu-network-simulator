"""Exercise the Nsight Systems SQLite parser without a GPU."""

from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

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
        for i, metric in enumerate(nsys_ft48.METRICS, 1):
            self.db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?, ?)",
                            (500, i, metric))
            self.db.executemany("INSERT INTO GPU_METRICS VALUES (?, ?, ?, ?)",
                                [(30, 500, i, 0), (110, 500, i, 80)])

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

    def test_days_attempts_are_separate_from_committed_rounds(self):
        self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?, ?, 0)",
                            [(10, 20, 150, 2), (22, 9000, 152, 3)])
        nodes = (5, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 7) * 64
        self.db.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 0, 150, ?, 71, ?, 2, 1, 1, 64, 1, 1)",
            [(25 + i * 2, 26 + i * 2, i + 1, name_id) for i, name_id in enumerate(nodes)])
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


if __name__ == "__main__":
    unittest.main()
