import copy
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import training_owner_sweep as sweep


class TrainingOwnerSweepTest(unittest.TestCase):
    def samples(self):
        timings = [dict(repetition=r, order=i, variant=(r+i) % 4, owners=sweep.OWNERS,
                        workers=10, events=sweep.EVENTS, payload_bytes=sweep.PAYLOAD_BYTES, resident_loop_us=10+r+i)
                   for r in range(5) for i in range(4)]
        progress = []
        for v in range(4):
            for wave in range(sweep.WAVES):
                early = 1 if v >= 2 else 0
                n = early if wave == 0 else sweep.EVENTS-early if wave == sweep.WAVES-1 else 0
                progress.append(dict(variant=v, wave=wave, events=n, max_owner=n, max_cta=n,
                                     active_owners=int(bool(n)), active_ctas=int(bool(n))))
        return timings, progress

    def test_paired_order_and_progress_validation(self):
        timings, progress = self.samples()
        self.assertEqual(set(sweep.inspect_shape(timings, progress)["median_us"]), set(sweep.VARIANTS))
        for changed, key, value in (("timings", "variant", 3), ("timings", "resident_loop_us", "nan"),
                                    ("progress", "events", 1), ("progress", "wave", 8)):
            t, p = copy.deepcopy(timings), copy.deepcopy(progress)
            (t if changed == "timings" else p)[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                sweep.inspect_shape(t, p)

    @unittest.skipUnless(shutil.which("c++"), "requires a C++17 compiler")
    def test_host_equal_boundary_counterexample_and_scalar_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "owner-sweep"
            subprocess.run(["c++", "-std=c++17", "-O2", "-x", "c++",
                            str(REPO / "benchmarks/training_owner_sweep.cu"), "-o", str(exe)], check=True)
            subprocess.run([str(exe), "--self-test"], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
