import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from training_roce_profile import device_trace


class DeviceTraceTest(unittest.TestCase):
    def test_physical_origin_and_virtual_events_are_independent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "stages.tsv"
            path.write_text("time_ns\tdevice_tick_begin\tdevice_tick_end\tsm\tcta\tqp\tsrc\tdst\tlane\tbatch_width\tstage\tframe\top_id\n"
                            "500\t1000005\t1001005\t2\t0\t0\t0\t1\t0\t1\tendpoint\tack\t7\n")
            event, = device_trace(path)["traceEvents"]
            self.assertEqual((event["ts"], event["dur"], event["args"]["time_ns"]), (0, 1, 500))
            self.assertEqual((event["args"]["op_id"], event["args"]["frame"]), (7, "ack"))
