"""Capture and summarize single-GPU FT48 CUDA graphs and device samples."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
from heapq import merge
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys

VERSION = "2026.6.1"
FREQUENCY_HZ = 10_000
METRIC_SET = "gb10x"
METRICS = ("SMs Active", "SM Issue", "DRAM Read Bandwidth", "DRAM Write Bandwidth")
PERIOD_NS = 1_000_000_000 // FREQUENCY_HZ
WINDOW_ROUNDS = 128
MAX_WINDOW_REPORT_BYTES = 64 * 1024**2
ACTIVITY_TABLE = "CUPTI_ACTIVITY_KIND_"
LAUNCH_NAMES = (
    "cudaGraphLaunch", "cuGraphLaunch", "cudaLaunchKernel", "cuLaunchKernel",
    "cuLaunchCooperativeKernel", "cudaLaunchCooperativeKernel",
)
SYNC_NAMES = ("cudaStreamSynchronize", "cuStreamSynchronize")


def preflight(output, env, helpers):
    runner = Path(sys.argv[0]).resolve()
    nsys = runner.parent.parent / f"NsightSystems-{VERSION}" / "bin" / "nsys"
    if not nsys.is_file() or not os.access(nsys, os.X_OK):
        raise RuntimeError(f"missing executable Nsight Systems {VERSION} CLI at {nsys}")
    if not re.fullmatch(r"GPU-[a-fA-F0-9-]+", env.get("CUDA_VISIBLE_DEVICES", "")):
        raise RuntimeError("Nsight Systems requires exactly one selected GPU UUID")
    version = output / "nsys-version.txt"
    helpers.command([nsys, "--version"], version, env=env)
    if not re.search(rf"\b{re.escape(VERSION)}(?:\.|\s|$)", version.read_text()):
        raise RuntimeError(f"expected Nsight Systems {VERSION}; see {version.name}")
    helpers.command([nsys, "profile", "--gpu-metrics-devices=help"],
                    output / "nsys-devices.txt", env=env)
    sets = output / "nsys-metric-sets.txt"
    helpers.command([nsys, "profile", "--gpu-metrics-devices=cuda-visible",
                     "--gpu-metrics-set=help"], sets, env=env)
    if not re.search(rf"\b{METRIC_SET}\b(?!-)", sets.read_text()):
        raise RuntimeError(f"selected GPU does not expose {METRIC_SET}; see {sets.name}")
    return nsys


def prefix(nsys, output, *, capture_range=False):
    capture = (["--capture-range=cudaProfilerApi", "--capture-range-end=stop"]
               if capture_range else ["--capture-range=none"])
    return [str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none", "--cpuctxsw=none",
            *capture, "--cuda-graph-trace=node", "--gpu-metrics-devices=cuda-visible",
            f"--gpu-metrics-set={METRIC_SET}", f"--gpu-metrics-frequency={FREQUENCY_HZ}",
            "-o", str(output / "profile")]


@dataclass
class IntervalUnion:
    # Input is ordered by start time, which permits an exact union in constant memory.
    end: int = 0
    total: int = 0

    def add(self, start, end):
        self.total += max(0, end - max(start, self.end))
        self.end = max(end, self.end)


@dataclass
class Attempt:
    start: int
    activity: IntervalUnion = field(default_factory=IntervalUnion)
    kernels: int = 0
    last_name: str = ""

    def add(self, start, end, name):
        self.activity.add(start, end)
        self.kernels += 1
        self.last_name = name


@dataclass
class Replay:
    start: int
    api_end: int
    thread: int
    correlations: set
    sync_start: int = 0
    sync_end: int = 0
    kernel: IntervalUnion = field(default_factory=IntervalUnion)
    device: IntervalUnion = field(default_factory=IntervalUnion)
    counts: Counter = field(default_factory=Counter)
    attempts: list = field(default_factory=list)
    first_node: int | None = None
    last_node: int | None = None

    def add(self, start, end, kind, name):
        self.device.add(start, end)
        self.counts[kind] += 1
        if kind == "kernel":
            self.kernel.add(start, end)
            if name in ("days_round", "days_round_mechanisms"):
                self.counts["days_attempt_nodes"] += 1
            if name == "days_horizon_sweep":
                self.attempts.append(Attempt(start))
            if self.attempts:
                self.attempts[-1].add(start, end, name)
        if self.first_node is None:
            self.first_node = start
        self.last_node = max(self.last_node or end, end)


def _read_apis(db, tables):
    names = LAUNCH_NAMES + SYNC_NAMES
    # Nsight exports driver and runtime events together in the documented runtime
    # table. Include the separate driver table when the export provides it.
    api_tables = [ACTIVITY_TABLE + kind for kind in ("RUNTIME", "DRIVER")
                  if ACTIVITY_TABLE + kind in tables]
    rows = []
    for table in api_tables:
        predicates = " OR ".join("s.value GLOB ?" for _ in names)
        rows.extend(db.execute(
            f"SELECT a.start, a.end, a.globalTid, a.correlationId, s.value, a.returnValue "
            f"FROM {table} a JOIN StringIds s ON s.id=a.nameId WHERE {predicates}",
            [name + "*" for name in names]))
    replays = []
    direct_launches = set()
    for start, end, thread, correlation, name, result in sorted(
            set(rows), key=lambda row: (row[0], -row[1])):
        if result != 0:
            continue
        if name.startswith(SYNC_NAMES):
            if replays and replays[-1].thread == thread and not replays[-1].sync_end:
                replays[-1].sync_start, replays[-1].sync_end = start, end
        elif name.startswith(("cudaGraphLaunch", "cuGraphLaunch")):
            if (replays and thread == replays[-1].thread
                    and start < replays[-1].api_end and end <= replays[-1].api_end):
                replays[-1].correlations.add(correlation)
            else:
                replays.append(Replay(start, end, thread, {correlation}))
        elif correlation:
            direct_launches.add(correlation)
    return replays, direct_launches


def _activity(db, tables):
    def read(kind):
        table = ACTIVITY_TABLE + kind.upper()
        if table not in tables:
            return
        fields = "a.start, a.end, a.deviceId, a.correlationId, a.graphNodeId, a.graphId"
        if kind == "kernel":
            fields += ", s.value, a.gridX, a.gridY, a.gridZ, a.blockX, a.blockY, a.blockZ"
            join = " LEFT JOIN StringIds s ON s.id=a.shortName"
        else:
            fields += ", NULL, NULL, NULL, NULL, NULL, NULL, NULL"
            join = ""
        for row in db.execute(f"SELECT {fields} FROM {table} a{join} ORDER BY a.start, a.end"):
            yield (*row, kind)

    return merge(*(read(kind) for kind in ("kernel", "memcpy", "memset")),
                 key=lambda row: row[:2])


def _copy_summary(db, tables, start, end):
    if ACTIVITY_TABLE + "MEMCPY" not in tables:
        return []
    return [
        {"copy_kind": kind, "within_graph_loop": bool(in_loop), "records": count,
         "bytes": copied, "duration_sum_ns": duration}
        for kind, in_loop, count, copied, duration in db.execute(
            "SELECT COALESCE(e.label, CAST(c.copyKind AS TEXT)), "
            "(c.start >= ? AND c.end <= ?) AS in_loop, COUNT(*), "
            "SUM(c.bytes), SUM(c.end - c.start) FROM CUPTI_ACTIVITY_KIND_MEMCPY c "
            "LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id = c.copyKind "
            "GROUP BY c.copyKind, in_loop ORDER BY c.copyKind, in_loop", (start, end))
    ]


def _bin_metrics(db, first, last, windows, output, tables):
    if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        return {}
    names = db.execute("SELECT DISTINCT typeId, metricId, metricName FROM TARGET_INFO_GPU_METRICS")
    selected = defaultdict(list)
    for type_id, metric_id, name in names:
        for metric in METRICS:
            if name and name.startswith(metric):
                selected[metric].append((type_id, metric_id, name))
    ends = [window[1] for window in windows]
    summary = {}
    fields = ["first_replay", "last_replay", "start_ns", "end_ns", "metric", "samples",
              "missing_or_invalid_values", "mean_pct", "min_pct", "max_pct"]
    with (output / "profile-metric-windows.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for metric, records in selected.items():
            if len(records) != 1:
                summary[metric] = {"error": f"ambiguous metric: {len(records)} sample streams"}
                continue
            type_id, metric_id, name = records[0]
            bins = [[0, 0, 0.0, None, None] for _ in windows]
            valid = invalid = gaps = 0
            total = maximum_gap = 0
            previous = earliest = None
            for timestamp, value in db.execute(
                    "SELECT timestamp, value FROM GPU_METRICS "
                    "WHERE typeId=? AND metricId=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (type_id, metric_id, first, last)):
                if value is None or not 0 <= value <= 100:
                    invalid += 1
                    index = min(bisect_right(ends, timestamp), len(bins) - 1)
                    bins[index][1] += 1
                    continue
                if previous is not None:
                    delta = timestamp - previous
                    maximum_gap = max(maximum_gap, delta)
                    gaps += int(delta > 2 * PERIOD_NS)
                else:
                    earliest = timestamp
                previous = timestamp
                valid += 1
                total += value
                index = min(bisect_right(ends, timestamp), len(bins) - 1)
                stats = bins[index]
                stats[0] += 1
                stats[2] += value
                stats[3] = value if stats[3] is None else min(stats[3], value)
                stats[4] = value if stats[4] is None else max(stats[4], value)
            summary[metric] = {
                "metric_name": name, "type_id": type_id, "samples": valid,
                "invalid_samples": invalid, "sample_point_mean_pct": total / valid if valid else None,
                "first_sample_ns": earliest, "last_sample_ns": previous,
                "first_sample_after_window_start_ns": earliest - first if valid else None,
                "window_end_after_last_sample_ns": last - previous if valid else None,
                "max_valid_sample_gap_ns": maximum_gap, "gaps_above_two_nominal_periods": gaps,
            }
            for (start, end, i, j), (count, bad, summed, minimum, maximum) in zip(windows, bins):
                writer.writerow([i, j, start, end, name, count, bad,
                                 round(summed / count, 3) if count else None, minimum, maximum])
    return summary


def _write_replays(replays, output, simulator, capture_window):
    index_column = "round" if simulator == "gedes" else "graph_wave"
    if capture_window is not None:
        index_column = "capture_local_round"
    with (output / "profile-replays.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([index_column, "api_start_ns",
                         "api_end_ns", "sync_start_ns", "sync_end_ns", "next_launch_start_ns",
                         "first_graph_node_ns", "last_graph_node_ns", "kernel_nodes",
                         "memcpy_nodes", "memset_nodes", "days_attempt_kernel_nodes",
                         "graph_kernel_union_ns", "graph_device_union_ns",
                         "graph_span_without_observed_node_activity_ns",
                         "api_return_to_first_node_if_positive_ns", "last_node_to_sync_return_ns",
                         "last_graph_node_to_next_first_graph_node_ns"])
        for i, replay in enumerate(replays):
            start, end = replay.first_node, replay.last_node
            following = replays[i + 1] if i + 1 < len(replays) else None
            writer.writerow([i, replay.start, replay.api_end, replay.sync_start or None,
                             replay.sync_end or None,
                             following.start if following else None, start, end,
                             replay.counts["kernel"], replay.counts["memcpy"], replay.counts["memset"],
                             replay.counts["days_attempt_nodes"] if simulator == "days-ago" else None,
                             replay.kernel.total, replay.device.total,
                             end - start - replay.device.total if start is not None else None,
                             max(0, start - replay.api_end) if start is not None else None,
                             replay.sync_end - end if end is not None and replay.sync_end else None,
                             following.first_node - end if following and following.first_node is not None
                             and end is not None else None])


def _write_days_attempts(replays, output):
    attempt_sizes = Counter()
    invalid_attempts = 0
    with (output / "profile-days-attempts.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["graph_wave", "encoded_attempt", "first_kernel_start_ns",
                         "last_kernel_end_ns", "kernel_nodes", "kernel_union_ns"])
        for wave, replay in enumerate(replays):
            for encoded, attempt in enumerate(replay.attempts):
                attempt_sizes[attempt.kernels] += 1
                invalid_attempts += int(attempt.kernels not in (12, 13)
                                        or attempt.last_name != "days_round_finalize")
                writer.writerow([wave, encoded, attempt.start, attempt.activity.end,
                                 attempt.kernels, attempt.activity.total])
    return attempt_sizes, invalid_attempts


def _analyze(db, output, simulator, rounds, report, *, capture_window=None):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if ACTIVITY_TABLE + "KERNEL" not in tables:
        raise RuntimeError("CUDA node trace contains no kernel activity table")
    replays, host_direct_correlations = _read_apis(db, tables)
    if not replays:
        raise RuntimeError("CUDA node trace contains no successful host graph launch")
    starts = [replay.start for replay in replays]
    by_correlation = {correlation: i for i, replay in enumerate(replays)
                      for correlation in replay.correlations if correlation}
    graph_kernel_names = Counter()
    graph_kernel_duration = Counter()
    kernel_configs = Counter()
    device_ids = set()
    device_activity = IntervalUnion()
    kinds = Counter()
    unmapped = 0
    mapped_by_time = 0
    direct_correlations = set()
    for (start, end, device, correlation, node, graph, name, gx, gy, gz, bx, by, bz,
         kind) in _activity(db, tables):
        if end < start:
            raise RuntimeError("CUDA activity export contains a reversed interval")
        device_ids.add(device)
        kinds[kind] += 1
        if kind == "kernel":
            kernel_configs[(name, (gx, gy, gz), (bx, by, bz))] += 1
        replay_id = by_correlation.get(correlation)
        if node or graph or replay_id is not None:
            if replay_id is None:
                replay_id = bisect_right(starts, start) - 1
                if (replay_id < 0 or not replays[replay_id].sync_end
                        or end > replays[replay_id].sync_end):
                    unmapped += 1
                    continue
                mapped_by_time += 1
            replay = replays[replay_id]
            if start < replay.start or (replay.sync_end and end > replay.sync_end):
                unmapped += 1
                continue
            replay.add(start, end, kind, name)
            if kind == "kernel":
                graph_kernel_names[name] += 1
                graph_kernel_duration[name] += end - start
        elif kind == "kernel" and correlation:
            direct_correlations.add(correlation)
        if starts[0] <= start <= replays[-1].sync_end and end <= replays[-1].sync_end:
            device_activity.add(start, end)

    _write_replays(replays, output, simulator, capture_window)
    batch_groups = [replays[i:i + WINDOW_ROUNDS] for i in range(0, len(replays), WINDOW_ROUNDS)]
    windows = [(group[0].start, (batch_groups[n + 1][0].start if n + 1 < len(batch_groups)
                                else group[-1].sync_end), n * WINDOW_ROUNDS,
                n * WINDOW_ROUNDS + len(group) - 1) for n, group in enumerate(batch_groups)]
    metrics = _bin_metrics(db, starts[0], replays[-1].sync_end, windows, output, tables)
    diagnostics = []
    if "DIAGNOSTIC_EVENT" in tables:
        diagnostics = [text for text, in db.execute("SELECT DISTINCT text FROM DIAGNOSTIC_EVENT")
                       if text and re.search(r"(?i)\b(HES|software trace|hardware trace|overflow|"
                                    r"dropped|missing data|inconsistent data)\b", text)]
    counts = Counter()
    for replay in replays:
        counts.update(replay.counts)
    loop_span = replays[-1].sync_end - starts[0] if replays[-1].sync_end else None
    report.update({
        "cuda_activity_records": kinds, "device_ids": sorted(device_ids),
        "host_graph_replays": len(replays), "simulation_rounds_reported": rounds,
        "host_direct_kernel_launches_correlated_to_execution":
            len(host_direct_correlations & direct_correlations),
        "device_kernel_records_outside_graph": kinds["kernel"] - sum(graph_kernel_names.values()),
        "graph_activity_records": counts,
        "graph_kernel_nodes_per_replay": Counter(replay.counts["kernel"] for replay in replays),
        "days_attempt_nodes_per_wave": (Counter(replay.counts["days_attempt_nodes"] for replay in replays)
                                        if simulator == "days-ago" else None),
        "graph_nodes_assigned_by_synchronized_time_window": mapped_by_time,
        "graph_nodes_unmapped": unmapped,
        "replays_without_terminal_stream_sync": sum(not replay.sync_end for replay in replays),
        "first_graph_submit_to_last_sync_ns": loop_span,
        "device_activity_union_in_graph_loop_ns": device_activity.total,
        "graph_loop_without_observed_device_activity_ns":
            loop_span - device_activity.total if loop_span is not None else None,
        "graph_kernel_union_ns": sum(replay.kernel.total for replay in replays),
        "graph_device_union_ns": sum(replay.device.total for replay in replays),
        "host_graph_launch_api_ns": sum(replay.api_end - replay.start for replay in replays),
        "host_wave_stream_sync_api_ns": sum(replay.sync_end - replay.sync_start for replay in replays),
        "device_memory_copies": _copy_summary(db, tables, starts[0], replays[-1].sync_end),
        "graph_kernel_names": graph_kernel_names,
        "graph_kernel_duration_sum_ns_by_name": graph_kernel_duration,
        "kernel_launch_configurations": [
            {"name": name, "grid": grid, "block": block, "instances": count}
            for (name, grid, block), count in kernel_configs.most_common()],
        "sample_scope": "Device-wide samples by graph replay groups. Zero is a real sample. "
                        "Periods spanning stages are not assigned to individual kernel nodes.",
        "gpu_metrics": metrics,
        "trace_diagnostics": [text[:1000] for text in diagnostics[:20]],
    })
    errors = report["errors"]
    expected_replays = rounds
    if capture_window is not None:
        expected_replays = capture_window["end_round"] - capture_window["first_round"]
        report["capture_rounds_reported"] = expected_replays
        report["capture_csv_scope"] = (
            "CSV replay indices are zero-based within the capture. Add first_round from "
            "capture_window to obtain a simulation round. Timestamps are relative to the "
            "Nsight Systems report, and metric-window bounds are inclusive replay indices."
        )
    if len(device_ids) != 1:
        errors.append("expected CUDA activity on one GPU")
    if not counts["kernel"] or any(not replay.counts["kernel"] for replay in replays):
        errors.append("some graph replays lack CUDA kernel nodes")
    if unmapped or report["replays_without_terminal_stream_sync"]:
        errors.append("graph activity could not be matched completely to synchronized replay windows")
    if simulator == "gedes" and len(replays) != expected_replays:
        scope = "captured simulation rounds" if capture_window is not None else "simulation rounds"
        errors.append(f"GeDES CUDA graph launch count does not match {scope}")
    if simulator == "days-ago":
        attempt_sizes, invalid = _write_days_attempts(replays, output)
        encoded = sum(attempt_sizes.values())
        report.update({
            "days_graph_encoded_attempts": encoded,
            "days_encoded_attempt_kernel_node_counts": attempt_sizes,
            "days_attempt_scope": "Graph attempts include continuation and no-op nodes. "
                                  "The simulator reports the separate committed-round count.",
        })
        if (invalid or encoded != counts["days_attempt_nodes"]
                or encoded < rounds or not all(len(replay.attempts) == 64 for replay in replays)):
            errors.append("Days graph attempt nodes do not match the pinned attempt DAG and waves")
    for metric in METRICS:
        samples = metrics.get(metric, {})
        if not samples.get("samples"):
            errors.append(f"missing usable GPU series: {metric}")
        elif samples["invalid_samples"] or samples["gaps_above_two_nominal_periods"]:
            report["warnings"].append(f"{metric} has invalid samples or gaps; consult metric coverage")
    if any(re.search(r"(?i)buffer overflow|dropped .* (?:records|events|samples)", text)
           for text in diagnostics):
        errors.append("report contains a capture overflow/drop diagnostic")


def _capture_rounds(window, rounds):
    keys = ("first_round", "end_round", "wall_ns", "requested_seconds")
    if not isinstance(window, dict) or any(type(window.get(key)) is not int for key in keys):
        raise ValueError("invalid GeDES capture window marker")
    if not (0 <= window["first_round"] < window["end_round"] <= rounds
            and window["wall_ns"] > 0 and window["requested_seconds"] > 0):
        raise ValueError("GeDES capture window marker is outside the simulation run")
    return window["end_round"] - window["first_round"]


def _survey(nsys, native, output, env, helpers):
    log = output / "nsys-survey.txt"
    helpers.command([nsys, "survey", "--format=jsonl", native], log, env=env)
    capture = {}
    for line in log.read_text().splitlines():
        # The helper also captures stderr. Non-JSON tool messages remain in the log.
        if line.lstrip().startswith("{"):
            row = json.loads(line)
            if row.get("type") == "capture":
                capture[row["name"]] = row.get("value")
    duration = capture.get("Duration (s)")
    lost = capture.get("Lost Events")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("Nsight Systems survey did not report a positive capture duration")
    if type(lost) is not int or lost < 0:
        raise RuntimeError("Nsight Systems survey did not report a valid lost event count")
    return {"duration_seconds": duration, "lost_events": lost}


def collect(nsys, output, env, helpers, *, simulator, rounds, capture_window=None):
    if simulator not in {"days-ago", "gedes"} or not isinstance(rounds, int) or rounds <= 0:
        raise ValueError("expected simulator days-ago/gedes and positive simulation round count")
    native = output / "profile.nsys-rep"
    database = output / "profile.sqlite"
    report = {
        "simulator": simulator, "nsys_version": VERSION, "metric_set": METRIC_SET,
        "simulation_rounds_reported": rounds, "requested_sampling_hz": FREQUENCY_HZ,
        "passed": False, "warnings": [], "errors": [],
        "raw_files": {}, "profiling_scope": "One full simulator process with CUDA graph-node trace "
                                             "and device-system-wide performance sampling.",
    }
    window_is_usable = capture_window is None
    if capture_window is not None:
        report["capture_window"] = capture_window
        report["profiling_scope"] = (
            "CUDA graph-node trace and device-system-wide performance sampling during the marker "
            "window within the full GeDES simulation."
        )
        if simulator != "gedes":
            report["errors"].append("only GeDES uses the profile capture window")
        else:
            try:
                report["capture_rounds_reported"] = _capture_rounds(capture_window, rounds)
                window_is_usable = True
                if capture_window["wall_ns"] < capture_window["requested_seconds"] * 1_000_000_000:
                    report["errors"].append("GeDES capture window ended before its requested wall duration")
            except ValueError as exc:
                report["errors"].append(str(exc))
    try:
        if not native.is_file() or not native.stat().st_size:
            raise RuntimeError(f"missing nonempty {native.name}")
        if capture_window is not None:
            if native.stat().st_size > MAX_WINDOW_REPORT_BYTES:
                report["errors"].append("native report exceeds 64 MiB")
            try:
                report["capture_survey"] = _survey(nsys, native, output, env, helpers)
                if report["capture_survey"]["lost_events"]:
                    report["errors"].append("Nsight Systems survey reports lost events")
            except Exception as exc:
                report["errors"].append(f"Nsight Systems survey failed: {exc}")
        helpers.command([nsys, "export", "--type=sqlite", "-o", database, native],
                        output / "nsys-export.txt", env=env)
        if window_is_usable:
            with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
                _analyze(db, output, simulator, rounds, report, capture_window=capture_window)
    except Exception as exc:
        report["errors"].append(str(exc))
    finally:
        for path in (native, database):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                report["raw_files"][path.name] = {"bytes": path.stat().st_size,
                                                  "sha256": digest.hexdigest()}
        report["passed"] = not report["errors"]
        helpers.write_json(output / "profile-summary.json", report)
    if report["errors"]:
        raise RuntimeError(f"Nsight Systems profile gate failed: {'; '.join(report['errors'])}")
    return report
