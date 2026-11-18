import csv
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rc_stage_pipeline as pipeline


def sample_output(depth):
    binary = bytearray()
    for owner in range(2):
        sequence = (0xffffff - 15 + owner) & 0xffffff
        delivered = completed = end = next_unique = 0
        for i in range(depth):
            new = i % 13 != 12
            if new:
                psn = sequence
                sequence = (sequence + 1) & 0xffffff
                length = 64 + (owner * 17 + next_unique * 113) % 4033
                next_unique += 1
                delivered += length
                completed += next_unique % 4 == 0
            end = max(end, i * (3 + owner)) + (length + 129) // 50
            binary.extend(pipeline.RECORD.pack(owner, psn, new, delivered, completed, end))
    return binary


class StageResults(unittest.TestCase):
    def test_exact_ledger_and_all_paired_samples(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(pipeline, "OWNERS", 2), \
                patch.object(pipeline, "DEPTHS", (32,)):
            output = Path(temporary)
            ledger = sample_output(32)
            self.assertEqual(hashlib.sha256(ledger).hexdigest(), pipeline.reference_hash(2, 32))
            (output / "records-32.bin").write_bytes(ledger)
            with (output / "samples.csv").open("w", newline="") as f:
                rows = csv.writer(f)
                rows.writerow(("depth", "repeat", "variant", "records", "wall_seconds"))
                for v in range(8):
                    for rep in range(5):
                        rows.writerow((32, rep, v, 64, 0.5))
            self.assertEqual(len(pipeline.inspect_results(output)["timings"]), 8)
            path = output / "records-32.bin"
            path.write_bytes(ledger[:-1])
            with self.assertRaisesRegex(ValueError, "ordered RC output"):
                pipeline.inspect_results(output)
            path.write_bytes(ledger)
            samples = output / "samples.csv"
            samples.write_text("\n".join(samples.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "missing samples"):
                pipeline.inspect_results(output)


if __name__ == "__main__":
    unittest.main()
