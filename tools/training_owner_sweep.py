"""Verify matched resident-loop timing and causal progress for the single-GPU owner sweep."""
import csv
import hashlib
import math
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory

VARIANTS = ("global-static", "global-dynamic", "local-static", "local-dynamic")
SHAPES = ("single-hot", "multi-hot")
OWNERS = 32768
EVENTS = 2 * (24 + 8) * OWNERS
WAVES = 12


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def inspect_shape(timings, progress):
    samples = {variant: [] for variant in VARIANTS}
    workers = {int(row["workers"]) for row in timings}
    if len(workers) != 1 or min(workers) <= 0 or len(timings) != 20:
        raise ValueError("incomplete paired samples or inconsistent worker count")
    for pos, row in enumerate(timings):
        rep, order = divmod(pos, 4)
        variant = (rep + order) % 4
        latency = float(row["resident_loop_us"])
        if (int(row["variant"]) != variant or int(row["repetition"]) != rep or int(row["order"]) != order
                or int(row["events"]) != EVENTS or int(row["owners"]) != OWNERS
                or not math.isfinite(latency) or latency <= 0):
            raise ValueError("incorrect paired order, event count or interval")
        samples[VARIANTS[variant]].append(latency)

    work = {variant: [] for variant in VARIANTS}
    if len(progress) != 4 * WAVES:
        raise ValueError("incomplete wave diagnostics")
    for pos, row in enumerate(progress):
        variant, wave = divmod(pos, WAVES)
        if int(row["variant"]) != variant or int(row["wave"]) != wave:
            raise ValueError("unexpected wave identity")
        metrics = {k: int(row[k]) for k in ("events", "max_owner", "active_owners", "max_cta", "active_ctas")}
        if (not 0 <= metrics["max_owner"] <= metrics["max_cta"] <= metrics["events"]
                or not 0 <= metrics["active_ctas"] <= min(min(workers), metrics["active_owners"])
                or metrics["active_owners"] > OWNERS
                or metrics["events"] > metrics["active_owners"] * metrics["max_owner"]
                or metrics["events"] > metrics["active_ctas"] * metrics["max_cta"]
                or bool(metrics["events"]) != bool(metrics["active_owners"])):
            raise ValueError("inconsistent committed-work counters")
        work[VARIANTS[variant]].append(metrics)

    for variant in VARIANTS:
        if sum(row["events"] for row in work[variant]) != EVENTS:
            raise ValueError("workload was not fully committed")
    for left, right in (VARIANTS[:2], VARIANTS[2:]):
        for a, b in zip(work[left], work[right]):
            if any(a[k] != b[k] for k in ("events", "max_owner", "active_owners")):
                raise ValueError("claim policy changed the safe work committed")
    if work["local-static"][0]["events"] <= work["global-static"][0]["events"]:
        raise ValueError("local MET did not expose the known independent first-wave work")
    return {"samples_us": samples, "median_us": {k: median(v) for k, v in samples.items()},
            "wave_work": work}


def run(source: Path, output: Path, helpers):
    cap, env = helpers.selected_gpu(output)
    arch = cap.replace(".", "")
    program = source / "benchmarks/training_owner_sweep.cu"
    summary = {"scope": "single-card independent FIFO owners with staged synthetic fan-in streams",
               "timing": "wall time of all resident waves, includes frontier/ready queue kernels and synchronization",
               "timing_excludes": "allocation, precomputed publication upload, per-run reset, audit D2H, oracle and files",
               "method": "one uninstrumented warmup per variant then five paired cyclic-order trials; separate diagnostics",
               "mapping": "one warp CTA per tile of 32 owners, static grid stride or atomic ready-tile queue",
               "owners": OWNERS, "committed_events_per_run": EVENTS, "waves": WAVES,
               "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(), "passed": False}
    try:
        with TemporaryDirectory(prefix=".owner-sweep-", dir=output) as tmp:
            exe = Path(tmp) / "training-owner"
            helpers.command(["nvcc", "--version"], output / "nvcc.txt", env=env)
            summary["build_seconds"] = helpers.command(
                ["nvcc", "-O3", "-std=c++17", "-lineinfo", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
                 program, "-o", exe], output / "build.txt", env=env)
            summary["binary_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
            helpers.command([exe, "--self-test"], output / "scalar.txt", env=env)
            summary["process_seconds"] = helpers.command([exe, output], output / "run.txt", env=env)
            summary["shapes"] = {s: inspect_shape(rows(output / f"{s}-timings.csv"),
                                                  rows(output / f"{s}-progress.csv")) for s in SHAPES}
            summary["passed"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", summary)
