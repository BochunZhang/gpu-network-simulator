"""Matched unroll variants of one ordered RC kernel; no assumed cache capacity."""

import csv
import hashlib
import math
import re
import shutil
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory


def functions(path):
    """Count emitted SASS instructions per function (static code, not cache misses)."""
    counts = {}
    name = None
    for line in path.read_text().splitlines():
        match = re.search(r"Function\s*:\s*(\S+)", line)
        if match:
            name = match[1]
            counts[name] = 0
        elif name and re.match(r"\s*/\*[0-9a-fA-F]+\*/\s+(?!/\*)\S", line):
            counts[name] += 1
    if not any("decode_tiles" in key and value > 0 for key, value in counts.items()):
        raise ValueError("no decoded pipeline kernels in cuobjdump output")
    return counts


def run(source, output, helpers):
    shape = helpers._support_module(source, "rc_stage_pipeline")
    variants = (1, 4, 8)
    cap, env = helpers.selected_gpu(output)
    arch = cap.replace(".", "")
    reference = shape.reference_hash(shape.OWNERS, 1024)
    report = {"passed": False, "scope": "cooperative fused receive descriptors; full ordered output through close",
              "depth": 1024, "expected_sha256": reference, "samples": [], "builds": {}}
    try:
        with TemporaryDirectory(prefix=".instruction-", dir=output) as tmp:
            for unroll in variants:
                folder = output / f"u{unroll}"
                folder.mkdir()
                binary = Path(tmp) / f"pipeline-{unroll}"
                helpers.command(["nvcc", "-std=c++17", "-O3", "-lineinfo", "-Xptxas=-v",
                                 "-gencode", f"arch=compute_{arch},code=sm_{arch}", f"-DRC_STAGE_UNROLL={unroll}",
                                 source / "benchmarks/rc_stage_pipeline.cu", "-o", binary],
                                folder / "ptxas.txt", env=env, journal_dir=output)
                helpers.command(["cuobjdump", "--dump-sass", binary], folder / "sass.txt", env=env, journal_dir=output)
                report["builds"][unroll] = {"sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                                            "static_sass_instructions": functions(folder / "sass.txt")}
            for rep in range(5):
                order = variants[rep % 3:] + variants[:rep % 3]
                for unroll in order:
                    folder = output / f"u{unroll}"
                    binary = Path(tmp) / f"pipeline-{unroll}"
                    helpers.command([binary, folder, "--icache"], folder / f"rep{rep}.txt", env=env,
                                    journal_dir=output)
                    with (folder / "samples.csv").open(newline="") as f:
                        row, = csv.DictReader(f)
                    seconds = float(row["wall_seconds"])
                    if (not math.isfinite(seconds) or seconds <= 0 or int(row["variant"]) != 1
                            or int(row["records"]) != shape.OWNERS * 1024 or int(row["depth"]) != 1024
                            or hashlib.sha256((folder / "records-1024.bin").read_bytes()).hexdigest() != reference):
                        raise ValueError(f"invalid ordered work for unroll={unroll}, repetition={rep}")
                    report["samples"].append({"repetition": rep, "unroll": unroll, "seconds": seconds})
            report["median_seconds"] = {u: median(r["seconds"] for r in report["samples"] if r["unroll"] == u)
                                        for u in variants}
            ncu = shutil.which("ncu", path=env.get("PATH"))
            if ncu:
                try:
                    helpers.command([ncu, "--query-metrics", "--query-metrics-mode", "all"],
                                    output / "available-metrics.txt", env=env)
                    available = (output / "available-metrics.txt").read_text()
                    candidates = ("smsp__pcsamp_warps_issue_stalled_no_instructions",
                                  "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
                                  "sm__icc_requests.sum", "sm__icc_requests_hit.sum", "sm__icc_requests_miss.sum")
                    metrics = [m for m in candidates if re.search(r"(?<![\w.])" + re.escape(m) + r"(?![\w.])", available)]
                    report["available_instruction_metrics"] = metrics
                    for u in variants:
                        folder = output / f"u{u}"
                        cmd = [ncu, "--section", "LaunchStats", "--section", "SchedulerStats",
                               "--kernel-name", "regex:decode_tiles", "--launch-count", "1"]
                        if metrics:
                            cmd.extend(["--metrics", ",".join(metrics)])
                        helpers.command([*cmd, "--export", folder / "instructions", Path(tmp) / f"pipeline-{u}",
                                         folder, "--icache"], folder / "ncu.txt", env=env, journal_dir=output)
                except (OSError, RuntimeError) as error:
                    report["ncu_error"] = str(error)
            report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", report)
