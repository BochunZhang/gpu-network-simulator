"""Single-device differential checks for the public RC training mechanism fixture."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def read_events(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        events = {}
        for row in reader:
            key = (int(row["id"]), row["kind"])
            if key in events:
                raise ValueError(f"duplicate {key} in {path.name}")
            events[key] = (int(row["src"]), int(row["dst"]), int(row["bytes"]), int(row["time_ns"]))
    return events


def validate_case(report, events, workload, ranks, size, fault):
    operations = 5 if workload == "closed-loop" else ranks * (ranks - 1)
    if (report["passed"] is not True or report["matches_cpu"] is not True
            or report["workload"] != workload or report["ranks"] != ranks
            or report["bytes_per_wr"] != size or report["fault"] != fault
            or report["operations"] != operations or len(events) != 3 * operations):
        raise ValueError("incomplete CPU/CUDA operation comparison")
    stats = report["stats"]
    if (stats["posts"] != operations or stats["cqes"] != operations or stats["recv_cqes"] != operations
            or stats["qps_created"] <= 0 or stats["qps_created"] != stats["qps_closed"]
            or stats["wire_bytes"] <= operations * size or stats["payload_bytes"] != operations * size):
        raise ValueError("RC byte, completion or lifetime gate failed")
    for op in {op for op, _ in events}:
        post, recv, send = (events[op, kind] for kind in ("post", "recv", "send"))
        if (post[:3] != recv[:3] or post[:3] != send[:3] or post[2] != size
                or not 0 <= post[3] < recv[3] < send[3]):
            raise ValueError(f"invalid local CQE order or operation identity for {op}")
    counter = {"drop-data": "dropped_data", "drop-ack": "dropped_ack", "rnr": "rnr_packets"}
    for variant, name in counter.items():
        if (stats[name] > 0) != (fault == variant):
            raise ValueError(f"{name}: fault controls did not exercise the requested path")
    if (stats["retransmits"] > 0) != (fault != "none"):
        raise ValueError("unexpected retransmission count")


def check_dependency(clean, delayed):
    for events in (clean, delayed):
        for op, parent, kind, compute_ns in ((3, 1, "recv", 7), (4, 1, "send", 31), (5, 2, "send", 31)):
            if events[op, "post"][3] != events[parent, kind][3] + compute_ns:
                raise ValueError(f"operation {op} failed its local {kind} CQE dependency")
    if delayed[4, "post"][3] <= clean[4, "post"][3]:
        raise ValueError("delayed ACK did not propagate to the sender's dependent work")
    if delayed[5, "post"] != clean[5, "post"]:
        raise ValueError("independent rank progress changed in the disjoint control pair")


def build(source, binary, output, helpers, arch, env):
    sources = [source / "benchmarks" / name for name in ("training_roce_engine.cu", "training_roce_main.cpp")]
    helpers.command(["nvcc", "--version"], output / "nvcc.txt", env=env)
    seconds = helpers.command(
        ["nvcc", "-O2", "-std=c++17", "-lineinfo", "-gencode", f"arch=compute_{arch},code=sm_{arch}",
         *sources, "-o", binary], output / "build.txt", env=env)
    return {"build_seconds": seconds, "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}


def run(source, output, helpers):
    cap, env = helpers.selected_gpu(output)
    arch = cap.replace(".", "")
    summary = {"scope": "single GPU public synthetic RC and collective mechanism fixtures",
               "passed": False, "cases": {}}
    try:
        with TemporaryDirectory(prefix=".training-roce-", dir=output) as tmp:
            exe = Path(tmp) / "training-roce"
            summary.update(build(source, exe, output, helpers, arch, env))
            cases = [("closed-loop", 4, fault) for fault in ("none", "drop-data", "drop-ack", "rnr")]
            cases.extend([("ring", 8, "none"), ("all-to-all", 8, "none")])
            for workload, ranks, fault in cases:
                name = f"{workload}-{fault}"
                folder = output / name
                folder.mkdir()
                result = {"passed": False}
                summary["cases"][name] = result
                try:
                    trace = folder / "ledger.tsv"
                    args = [exe, "--backend", "compare", "--workload", workload, "--ranks", str(ranks),
                            "--bytes", "8193", "--fault", fault, "--trace", trace]
                    wall = helpers.command(args, folder / "run.txt", env=env, journal_dir=output)
                    data = json.loads((folder / "run.txt").read_text())
                    events = read_events(trace)
                    validate_case(data, events, workload, ranks, 8193, fault)
                    # CSV format is included by the runner's standard summary archive.
                    with (folder / "ledger.csv").open("w", newline="") as dest, trace.open() as src:
                        csv.writer(dest).writerows(csv.reader(src, delimiter="\t"))
                    result.update(passed=True, comparison=data, process_seconds=wall)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    result["error"] = str(error)
            failures = [name for name, case in summary["cases"].items() if not case["passed"]]
            if failures:
                raise RuntimeError("failed RC cases: " + ", ".join(failures))
            check_dependency(read_events(output / "closed-loop-none/ledger.tsv"),
                             read_events(output / "closed-loop-drop-ack/ledger.tsv"))
            summary.update(passed=True, causal_delay_control=True)
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", summary)
