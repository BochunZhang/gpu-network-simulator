from pathlib import Path
import sqlite3
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import gedes_timeline as profile


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("""
            CREATE TABLE StringIds(id INTEGER, value TEXT);
            INSERT INTO StringIds VALUES(1,'cudaGraphLaunch_v10000'),(2,'cudaStreamSynchronize_v3020'),(3,'kernel');
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(nameId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(1,10,15,9),(2,16,90,9);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,streamId INTEGER,shortName INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(18,50,0,2,3),(30,60,0,3,3);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,deviceId INTEGER);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(64,80,0);
        """)

    def test_overlapping_kernels_and_dma(self):
        events = []
        report = profile.timeline(self.db, 1, events)
        self.assertEqual(report["device_activity_union_ns"], 58)
        self.assertEqual(report["span_without_observed_device_activity_ns"], 22)
        self.assertEqual(report["host_stream_sync_ns"], 74)
        self.assertEqual(len(events), 4)

    def test_rejects_incomplete_capture(self):
        with self.assertRaisesRegex(ValueError, "graph launches"):
            profile.timeline(self.db, 2)


if __name__ == "__main__":
    unittest.main()
