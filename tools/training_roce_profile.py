"""Perfetto-compatible hardware-stage trace and separate virtual completion ledger."""

import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


def device_trace(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if not rows:
        raise ValueError("CUDA profiler returned no transitions")
    first = min(int(row["device_tick_begin"]) for row in rows)
    out = []
    tracks = {}
    for row in rows:
        stage = row.pop("stage")
        frame = row.pop("frame")
        fields = {key: int(value) for key, value in row.items()}
        a = fields.pop("device_tick_begin")
        b = fields.pop("device_tick_end")
        if (a <= 0 or b < a or min(fields["sm"], fields["cta"], fields["qp"]) < 0
                or fields["batch_width"] <= 0):
            raise ValueError("invalid CUDA time or owner in device trace")
        key = (fields["cta"], fields["qp"])
        track = tracks.setdefault(key, len(tracks) + 1)
        out.append({"name": stage, "cat": "RC transition", "ph": "X",
                    "ts": (a-first)/1000, "dur": (b-a)/1000,
                    "pid": fields["sm"], "tid": track, "args": {**fields, "frame": frame}})
    return {"traceEvents": out, "displayTimeUnit": "ns"}


def run(source, output, helpers):
    checks = helpers._support_module(source, "training_roce_checks")
    cap, env = helpers.selected_gpu(output)
    report = {"passed": False, "scope": "single CTA RC baseline; short synthetic closed-loop control",
              "device_clock": "globaltimer nanoseconds, relative to first sampled transition",
              "cases": {}}
    try:
        with TemporaryDirectory(prefix=".roce-profile-", dir=output) as tmp:
            exe = Path(tmp) / "training-roce"
            report.update(checks.build(source, exe, output, helpers, cap.replace(".", ""), env))
            args = [exe, "--workload", "closed-loop", "--ranks", "4", "--bytes", "8193", "--fault", "drop-ack"]
            for name in ("reference", "normal", "instrumented"):
                folder = output / name
                folder.mkdir()
                ledger = folder / "ledger.tsv"
                cmd = [*args, "--backend", "compare" if name == "reference" else "cuda", "--trace", ledger]
                stages = folder / "stages.tsv"
                if name == "instrumented":
                    cmd += ["--profile", stages, "--trace-limit", "8192"]
                wall = helpers.command(cmd, folder / "run.txt", env=env, journal_dir=output)
                data = json.loads((folder / "run.txt").read_text())
                if name == "reference":
                    checks.validate_case(data, checks.read_events(ledger), "closed-loop", 4, 8193, "drop-ack")
                if name == "instrumented":
                    trace = device_trace(stages)
                    if data["profile_truncated"] or len(trace["traceEvents"]) != data["profile_rows"]:
                        raise ValueError("incomplete device transition trace")
                    helpers.write_json(folder / "perfetto.json", trace)
                    with stages.open() as src, (folder / "stages.csv").open("w", newline="") as dst:
                        csv.writer(dst).writerows(csv.reader(src, delimiter="\t"))
                report["cases"][name] = {"summary": data, "process_seconds": wall}
            if len({(output / name / "ledger.tsv").read_bytes() for name in report["cases"]}) != 1:
                raise ValueError("profiling changed the frontend ledger")
            # Device time and simulated time have different origins; preserve the latter separately.
            with (output / "normal/ledger.tsv").open() as src, (output / "ledger.csv").open("w", newline="") as dst:
                csv.writer(dst).writerows(csv.reader(src, delimiter="\t"))
            report["ncu_available"] = bool(shutil.which("ncu", path=env.get("PATH")))
            if report["ncu_available"]:
                try:
                    helpers.command(["ncu", "--query-metrics"], output / "ncu-metrics.txt", env=env)
                    helpers.command(["ncu", "--kernel-name", "regex:run_cuda", "--launch-count", "1",
                                     "--section", "LaunchStats", "--section", "SchedulerStats",
                                     "--export", str(output / "rc-kernel"), *args, "--backend", "cuda"],
                                    output / "ncu-launch.txt", cwd=output, env=env)
                    report["ncu_collected"] = (output / "rc-kernel.ncu-rep").is_file()
                except (RuntimeError, OSError) as error:
                    report["ncu_error"] = str(error)
            report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        helpers.write_json(output / "summary.json", report)
