"""Finite-batch RC descriptor processing, CTA role, and result copy experiments."""
import csv
import hashlib
import math
from pathlib import Path
import shutil
import statistics
import struct
from tempfile import TemporaryDirectory

OWNERS = 512
DEPTHS = (32, 1024)
REPETITIONS = 5
RECORD = struct.Struct("<IIIIII")
PSN_MASK = (1 << 24) - 1


def reference_hash(owners, depth):
    """Generate expected receipts directly from unique-message ordinals."""
    digest = hashlib.sha256()
    for owner in range(owners):
        buffer = bytearray()
        finish = delivered = 0
        for i in range(depth):
            accepted = i % 13 != 12
            # The thirteenth descriptor repeats the preceding sequence number.
            unique = i - (i + 1) // 13
            psn = (PSN_MASK - 15 + owner % 7 + unique) & PSN_MASK
            length = 64 + (17 * owner + 113 * unique) % 4033
            finish = max(finish, i * (3 + owner % 37)) + math.ceil((length + 80) / 50)
            delivered += length if accepted else 0
            buffer.extend(RECORD.pack(owner, psn, accepted, delivered, (unique + 1) // 4, finish))
        digest.update(buffer)
    return digest.hexdigest()


def inspect_results(output):
    groups = {}
    with (output / "samples.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            depth, variant, repeat = (int(row[k]) for k in ("depth", "variant", "repeat"))
            wall = float(row["wall_seconds"])
            if (depth not in DEPTHS or variant not in range(8) or repeat not in range(REPETITIONS)
                    or int(row["records"]) != OWNERS * depth or not math.isfinite(wall) or wall <= 0):
                raise ValueError(f"invalid timing row: {row}")
            samples = groups.setdefault((depth, variant), {})
            if repeat in samples:
                raise ValueError(f"duplicate repeat: {row}")
            samples[repeat] = wall
    outputs = {}
    timings = []
    for depth in DEPTHS:
        path = output / f"records-{depth}.bin"
        expected = reference_hash(OWNERS, depth)
        if path.stat().st_size != OWNERS * depth * RECORD.size or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"ordered RC output differs at depth {depth}")
        outputs[depth] = {"sha256": expected, "bytes": path.stat().st_size}
        for variant in range(8):
            samples = groups.get((depth, variant), {})
            if set(samples) != set(range(REPETITIONS)):
                raise ValueError(f"missing samples for depth {depth}, variant {variant}")
            times = [samples[i] for i in range(REPETITIONS)]
            timings.append({"depth": depth, "variant": variant, "stages": "fused" if variant & 1 else "split",
                            "loads": "producer-warp" if variant & 2 else "cooperative",
                            "output": "double-buffer" if variant & 4 else "synchronous",
                            "seconds": times, "median_seconds": statistics.median(times)})
    return {"owners": OWNERS, "verified_outputs": outputs, "timings": timings}


def run(source: Path, output: Path, helpers):
    program = source / "benchmarks/rc_stage_pipeline.cu"
    report = {"passed": False, "source_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
              "scope": "Ordered BTH-like descriptor fixture on independent directed links; includes D2H and file close.",
              "timing_excludes": ["fixture generation", "initial H2D", "allocation", "post-run verification"]}
    try:
        capability, env = helpers.selected_gpu(output)
        arch = capability.replace(".", "")
        helpers.command(["nvcc", "--version"], output / "nvcc.txt", env=env)
        with TemporaryDirectory(prefix=".rc-stage-pipeline-", dir=output) as temp:
            binary = Path(temp) / "rc-stage-pipeline"
            helpers.command(["nvcc", "-O3", "-std=c++17", "-lineinfo", "-Xptxas=-v",
                             "-gencode", f"arch=compute_{arch},code=sm_{arch}", program, "-o", binary],
                            output / "build.txt", env=env)
            report["architecture"] = arch
            report["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            helpers.command([binary, output], output / "normal.txt", env=env)
            report.update(inspect_results(output))
            dump = shutil.which("cuobjdump", path=env.get("PATH"))
            if dump:
                try:
                    helpers.command([dump, "--dump-sass", binary], output / "sass.txt", env=env)
                    report["sass_collected"] = True
                except (OSError, RuntimeError) as error:
                    report["sass_collected"] = False
                    report["sass_error"] = str(error)
            else:
                report["sass_collected"] = False
            report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", report)
