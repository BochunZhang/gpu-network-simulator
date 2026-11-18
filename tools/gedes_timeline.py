"""Paired short Nsight captures on the task 004 GeDES k8 drain fixture."""

import hashlib
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory

# In the fixed seed-41 FCT ledger, 53 connections are live at 2 ms and one at 6.9 ms.
WINDOWS = {"busy": 2000, "tail": 6900}
WINDOW_ROUNDS = 16


def instrument_window(source):
    declaration = "    int audit_rounds = 0;"
    open_range = "#if ENABLE_HUGE_GRAPH\n        if (audit_detail)"
    close_range = "        ++audit_rounds;"
    for anchor in (declaration, open_range, close_range):
        if source.count(anchor) != 1:
            raise ValueError(f"unexpected audited main: {anchor}")
    return ("#include <cuda_profiler_api.h>\n#include <cstdlib>\n" + source
            .replace(declaration, declaration + """
    const char* window = std::getenv("GEDES_PROFILE_BEGIN");
    const int first_profile_round = window ? std::stoi(window) : -1;
""").replace(open_range, """
        if (i == first_profile_round && cudaProfilerStart() != cudaSuccess) {
            std::cerr << "could not start CUDA capture" << std::endl;
            return 1;
        }
""" + open_range).replace(close_range, f"""
        if (window && i + 1 == first_profile_round + {WINDOW_ROUNDS} &&
            cudaProfilerStop() != cudaSuccess) {{
            std::cerr << "could not stop CUDA capture" << std::endl;
            return 1;
        }}
""" + close_range))


def timeline(db, rounds=WINDOW_ROUNDS, events=None):
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    activities = []
    device_ids = set()
    kernel_count = 0
    for kind in ("KERNEL", "MEMCPY", "MEMSET"):
        table = "CUPTI_ACTIVITY_KIND_" + kind
        if table in tables:
            for start, end, device in db.execute(f"SELECT start, end, deviceId FROM {table}"):
                if end < start:
                    raise ValueError(f"invalid interval in {table}")
                activities.append((start, end))
                device_ids.add(device)
                kernel_count += int(kind == "KERNEL")
    if kernel_count == 0 or len(device_ids) != 1:
        raise ValueError("capture must have kernel node activity on exactly one device")
    runtime = list(db.execute("""SELECT s.value,r.start,r.end,r.globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                                JOIN StringIds s ON s.id=r.nameId ORDER BY r.start"""))
    launches = [(s,e) for name,s,e,_tid in runtime if name.startswith("cudaGraphLaunch")]
    syncs = [(s,e) for name,s,e,_tid in runtime if name.startswith("cudaStreamSynchronize")]
    if len(launches) != rounds or len(syncs) != rounds:
        raise ValueError(f"expected {rounds} graph launches and synchronizations, saw {len(launches)} and {len(syncs)}")
    start, end = launches[0][0], syncs[-1][1]
    if start >= end:
        raise ValueError("invalid host round span")
    cursor = start
    busy_ns = 0
    for a,b in sorted(activities):
        lo, hi = max(a,cursor), min(b,end)
        if hi > lo:
            busy_ns += hi - lo
            cursor = hi
    if events is not None:
        for name, a, b, tid in runtime:
            events.append({"name": name, "cat": "CUDA API", "ph": "X", "pid": 1, "tid": tid,
                           "ts": (a - start) / 1000, "dur": (b - a) / 1000})
        for name, stream, a, b in db.execute("""
                SELECT s.value, k.streamId, k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON s.id=k.shortName ORDER BY k.start"""):
            events.append({"name": name, "cat": "CUDA kernel", "ph": "X", "pid": 2, "tid": stream,
                           "ts": (a - start) / 1000, "dur": (b - a) / 1000})
    return {"observed_rounds": rounds, "device": next(iter(device_ids)), "kernel_nodes": kernel_count,
            "host_round_span_ns": end-start, "device_activity_union_ns": busy_ns,
            "span_without_observed_device_activity_ns": end-start-busy_ns,
            "host_graph_launch_ns": sum(b-a for a,b in launches),
            "host_stream_sync_ns": sum(b-a for a,b in syncs)}


def run(source, output, helpers):
    audit = helpers._support_module(source, "gedes_experiments")
    drain = helpers._support_module(source, "gedes_drain")
    report = {"passed": False, "scope": "GeDES k8/seed41 short CUDA graph windows; host waits overlap GPU activity",
              "windows": {}, "completion_gate": "task 004 k8 byte, queue drain, input and FCT checks"}
    try:
        if helpers.GEDES[1] != audit.UPSTREAM_REVISION:
            raise ValueError("unexpected GeDES source pin")
        capability, env = helpers.selected_gpu(output)
        env.pop("GEDES_PROFILE_BEGIN", None)
        helpers.command(["nsys", "--version"], output / "nsys.txt", env=env)
        helpers.command(["nvcc", "--version"], output / "nvcc.txt", env=env)
        with TemporaryDirectory(prefix=".timeline-", dir=output) as temporary:
            binary = drain.build_audit(source, output, helpers, audit, Path(temporary), capability, env,
                                       instrument_window)
            normal = drain._case(binary, "normal", 8, False, output, env, audit, helpers)
            report["normal"] = normal
            if not normal["passed"] or normal["measurements"]["rounds"] != 7001:
                raise ValueError("normal k8 reference failed")
            for name, first in WINDOWS.items():
                folder = output / name
                stem = folder / "window"
                prefix = ["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                          "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi",
                          "--capture-range-end=stop", "-o", str(stem)]
                case = drain._case(binary, name, 8, False, output, {**env, "GEDES_PROFILE_BEGIN": str(first)},
                                   audit, helpers, prefix=prefix)
                report["windows"][name] = case
                if not case["passed"] or case["measurements"]["rounds"] != 7001:
                    raise ValueError(f"{name}: profiled run failed drain or round count")
                native = stem.with_suffix(".nsys-rep")
                database = stem.with_suffix(".sqlite")
                helpers.command(["nsys", "export", "--type=sqlite", "-o", database, native],
                                folder / "export.txt", env=env, journal_dir=output)
                with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
                    events = []
                    case["timeline"] = timeline(db, events=events)
                    helpers.write_json(folder / "timeline.json", {"traceEvents": events})
                case.update(first_round=first, trace_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", report)
