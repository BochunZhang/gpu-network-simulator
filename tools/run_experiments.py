#!/usr/bin/env python3
"""Run assigned scientific experiments from a pull-only Git queue."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone

LOCK_PATH = Path("/tmp/gpu-network-simulator.lock")
POLL_SECONDS = 10
TERMINAL = {"succeeded", "failed", "abandoned"}
RESULT_SUFFIXES = {".json", ".csv", ".txt"}
FILE_LIMIT = 2 * 1024**2
ARCHIVE_LIMIT = 16 * 1024**2
WORKER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"


class RunnerError(Exception):
    """Failure the polling loop can report without losing its local journal."""


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def host_identity():
    try:
        machine = Path("/etc/machine-id").read_text(encoding="ascii").strip()
    except (OSError, ValueError):
        machine = ""
    if not re.fullmatch(r"[0-9a-fA-F]{32}", machine):
        machine = socket.gethostname()
    return hashlib.sha256(f"gpu-network-simulator:{machine}".encode()).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        out.write(encode(value))
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def validate_worker(worker):
    if not isinstance(worker, str) or not re.fullmatch(WORKER_PATTERN, worker):
        raise RunnerError("worker must be a 1-64 character identifier using letters, digits, _, . or -")
    return worker


def validate_record(row, path):
    if (
        not isinstance(row, dict)
        or type(row.get("id")) is not int or row["id"] < 1
        or type(row.get("version")) is not int or row["version"] < 1
        or row.get("state") not in TERMINAL | {"running"}
        or not isinstance(row.get("claim"), str)
        or not isinstance(row.get("host_id"), str)
        or not isinstance(row.get("task_sha256"), str)
        or not isinstance(row.get("source_commit"), str)
    ):
        raise RunnerError(f"invalid local execution record: {path}")
    validate_worker(row.get("worker"))
    return row


@contextlib.contextmanager
def device_lock(path):
    # Keep one inode for all checkouts. The child and GPU commands inherit the FD.
    fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunnerError(f"device already locked: {path}") from exc
        yield fd
    finally:
        os.close(fd)


class Queue:
    def __init__(self, root, source_branch="main"):
        self.root = root
        self.source_ref = f"refs/heads/{source_branch}"
        self.git("check-ref-format", self.source_ref)

    def git(self, *args, timeout=30):
        try:
            proc = subprocess.run(
                ["git", *args], cwd=self.root,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                text=True, encoding="utf-8", errors="replace", capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"git {args[0]} timed out after {timeout} seconds") from exc
        if proc.returncode:
            raise RunnerError(f"git {args[0]}: {proc.stderr.strip() or proc.stdout.strip()}")
        return proc

    def source(self):
        local_ref = "refs/gpu-network-simulator/source"
        self.git("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "origin",
                 f"+{self.source_ref}:{local_ref}")
        return self.git("rev-parse", local_ref).stdout.strip()

    def tasks(self, source):
        paths = self.git("ls-tree", "-rz", "--name-only", source, "--", "tasks/").stdout.split("\0")
        tasks = []
        for path in paths:
            if not path.endswith(".json"):
                continue
            task = json.loads(self.git("show", f"{source}:{path}").stdout)
            if not isinstance(task, dict):
                raise RunnerError(f"expected task object: {path}")
            task_id, version = task.get("id"), task.get("version")
            if "worker" not in task:
                raise RunnerError(f"task requires explicit worker assignment: {path}")
            validate_worker(task["worker"])
            if (
                type(task_id) is not int or task_id < 1
                or path != f"tasks/{task_id:06d}.json"
                or type(version) is not int or version < 1
                or not isinstance(task.get("kind"), str)
                or not isinstance(task.get("params"), dict)
                or not isinstance(task.get("description", ""), str)
                or set(task) - {"id", "version", "worker", "kind", "params", "description"}
            ):
                raise RunnerError(f"invalid task manifest: {path}")
            definition = {key: value for key, value in task.items() if key != "description"}
            try:
                digest = hashlib.sha256(encode(definition).encode("utf-8")).hexdigest()
            except ValueError as exc:
                raise RunnerError(f"invalid task values: {path}: {exc}") from exc
            tasks.append((task, digest))
        return sorted(tasks, key=lambda item: item[0]["id"])


class StatusStore:
    def __init__(self, root, worker):
        self.root = root / validate_worker(worker)
        self.worker = worker

    def path(self, task_id, version):
        return self.root / f"{task_id:06d}" / f"v{version}.json"

    def read(self):
        rows = {}
        for path in sorted(self.root.glob("*/v*.json")):
            row = validate_record(json.loads(path.read_text(encoding="utf-8")), path)
            if row["worker"] != self.worker or path != self.path(row["id"], row["version"]):
                raise RunnerError(f"local status path does not match execution record: {path}")
            rows[row["id"], row["version"]] = row
        return rows

    def save(self, row):
        validate_record(row, self.root)
        if row["worker"] != self.worker:
            raise RunnerError("execution worker does not match local status directory")
        path = self.path(row["id"], row["version"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, row)

    def reconcile(self, results):
        rows = self.read()
        for path in sorted(results.glob("*/v*/result.json")):
            row = validate_record(json.loads(path.read_text(encoding="utf-8")), path)
            if row["worker"] != self.worker or path.parent != run_path(results, row):
                raise RunnerError(f"result path or worker does not match local queue: {path}")
            if row["state"] not in TERMINAL:
                raise RunnerError(f"result is not terminal: {path}")
            current = rows.get((row["id"], row["version"]))
            if current:
                immutable = ("claim", "host_id", "worker", "source_commit", "task_sha256")
                if any(current[key] != row[key] for key in immutable):
                    raise RunnerError(f"result does not match local execution record: {path}")
                if current["state"] in TERMINAL:
                    if current["state"] != row["state"]:
                        raise RunnerError(f"result and local status disagree: {path}")
                    continue
            self.save(row)
            rows[row["id"], row["version"]] = row
        return rows


def validate_task(task, handlers):
    try:
        return handlers.prepare(task["kind"], task["params"])
    except ValueError as exc:
        raise RunnerError(f"task {task['id']} v{task['version']}: {exc}") from exc


def run_path(results, row):
    return results / f"{row['id']:06d}" / f"v{row['version']}"


def prepare_archive(output, record):
    tarpath = output.with_suffix(".tar.gz")
    tmp_tar = tarpath.with_suffix(".tmp")
    metadata = output / ".archive-metadata.json"
    selected = []
    total = 0
    for path in sorted(output.iterdir()):
        if (
            path.name.startswith(".")
            or path.name in {"result.json", "run.json"}
            or path.suffix not in RESULT_SUFFIXES
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        size = path.stat().st_size
        if size <= FILE_LIMIT and total + size <= ARCHIVE_LIMIT:
            total += size
            selected.append(path)
    record["archive_files"] = ["result.json", *[path.name for path in selected]]
    write_json(metadata, record)
    try:
        with tarfile.open(tmp_tar, "w:gz") as archive:
            archive.add(metadata, arcname="result.json")
            for path in selected:
                archive.add(path, arcname=path.name)
        tmp_tar.replace(tarpath)
    finally:
        metadata.unlink(missing_ok=True)
        tmp_tar.unlink(missing_ok=True)


def execute(queue, output):
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    record = dict(run)
    try:
        with tempfile.TemporaryDirectory(prefix="gpu-network-simulator-source-") as temp:
            source = Path(temp) / "source"
            queue.git("worktree", "add", "--detach", str(source), run["source_commit"], timeout=180)
            try:
                pinned = source / "tools" / "experiment_handlers.py"
                if not pinned.resolve().is_relative_to(source.resolve()) or not pinned.is_file():
                    raise RunnerError("pinned source is missing tools/experiment_handlers.py")
                record["pinned_handler_sha256"] = hashlib.sha256(pinned.read_bytes()).hexdigest()
                spec = importlib.util.spec_from_file_location("pinned_experiment_handlers", pinned)
                handlers = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(handlers)
                validate_task(run["task"], handlers)(source, output)
            finally:
                queue.git("worktree", "remove", "--force", str(source), timeout=180)
        record.update(state="succeeded", exit_code=0)
    except Exception as exc:
        traceback.print_exc()
        record.update(state="failed", exit_code=getattr(exc, "returncode", 1), error=str(exc))
    record["finished_at"] = timestamp()
    try:
        prepare_archive(output, record)
    except (OSError, tarfile.TarError) as exc:
        traceback.print_exc()
        record.update(state="failed", exit_code=1, archive_error=str(exc), archive_files=[])
    write_json(output / "result.json", record)
    return int(record["state"] != "succeeded")


def start_task(queue, source, store, statuses, results, fd):
    unfinished = next((row for row in statuses.values() if row["state"] == "running"), None)
    if unfinished:
        raise RunnerError(
            f"unfinished local execution {unfinished['id']} v{unfinished['version']}; "
            f"after confirming it stopped, use --abandon {unfinished['id']} {unfinished['version']}"
        )
    for task, digest in queue.tasks(source):
        if task["worker"] != store.worker:
            continue
        task_id, version = task["id"], task["version"]
        prior = [row for (i, _), row in statuses.items() if i == task_id]
        current = statuses.get((task_id, version))
        if current and current["task_sha256"] != digest:
            raise RunnerError(f"task {task_id} changed without increasing version {version}")
        if any(row["version"] >= version for row in prior):
            continue
        row = {
            "id": task_id, "version": version, "kind": task["kind"],
            "task_sha256": digest, "source_commit": source,
            "worker": store.worker, "host_id": host_identity(),
            "claim": uuid.uuid4().hex, "state": "running", "claimed_at": timestamp(),
        }
        output = run_path(results, row)
        if output.exists():
            raise RunnerError(f"local result path already exists: {output}")
        # Persist before launching. A crash here requires local abandonment.
        store.save(row)
        output.mkdir(parents=True)
        script = Path(__file__).resolve()
        runner_sha256 = {script.name: hashlib.sha256(script.read_bytes()).hexdigest()}
        write_json(output / "run.json", {**row, "task": task, "runner_sha256": runner_sha256})
        env = {**os.environ, "GPU_NET_SIM_LOCK_FD": str(fd)}
        with (output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                [sys.executable, str(script), "--_execute", str(output)],
                env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(fd,),
            )
        print(f"running task {task_id} v{version} at {source[:12]} in {output}", flush=True)
        return child, output


def abandon(store, results, task_id, version):
    row = store.reconcile(results).get((task_id, version))
    if not row or row["state"] != "running":
        raise RunnerError(f"task {task_id} v{version} has no running local execution")
    if row["host_id"] != host_identity():
        raise RunnerError("abandonment requires the original execution host")
    store.save({**row, "state": "abandoned", "finished_at": timestamp()})
    print(f"abandoned task {task_id} v{version}; raise its version to rerun")


def poll(queue, store, args, fd):
    active = None
    while True:
        result = None
        if active is not None and active[0].poll() is not None:
            child, output = active
            result = child.returncode
            if not (output / "result.json").is_file():
                print(f"worker exited without a terminal record: {output}", file=sys.stderr)
                return 1
            active = None
        next_poll = time.monotonic() + POLL_SECONDS
        try:
            # Finish local bookkeeping even when the public origin is offline.
            statuses = store.reconcile(args.results_dir)
            if args.once and result is not None:
                return int(result != 0)
            source = queue.source()
            if active is None:
                active = start_task(queue, source, store, statuses, args.results_dir, fd)
                if active is None and args.once:
                    return 0
        except (RunnerError, OSError, ValueError) as exc:
            print(f"poll failed: {exc}", file=sys.stderr)
            if args.once and active is None:
                return 1
        remaining = max(0, next_poll - time.monotonic())
        if active is None:
            time.sleep(remaining)
        else:
            try:
                active[0].wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass


def main():
    root = Path(__file__).resolve().parents[1]
    if len(sys.argv) == 3 and sys.argv[1] == "--_execute":
        # An internal subprocess only runs through the inherited host lock.
        fd = int(os.environ["GPU_NET_SIM_LOCK_FD"])
        if not os.path.samestat(os.fstat(fd), LOCK_PATH.stat()):
            raise RunnerError("worker has no inherited device lock")
        return execute(Queue(root), Path(sys.argv[2]))

    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="finish at most one assigned task")
    mode.add_argument(
        "--abandon", nargs=2, metavar=("ID", "VERSION"), type=int,
        help="locally mark a stopped experiment after confirming its process has exited",
    )
    cli.add_argument("--source-branch", default="main")
    cli.add_argument("--status-dir", type=Path, default=root / "status")
    cli.add_argument("--results-dir", type=Path, default=root / "results")
    cli.add_argument("--worker-id", default="gpu01")
    args = cli.parse_args()
    args.status_dir = args.status_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    try:
        with device_lock(LOCK_PATH) as fd:
            store = StatusStore(args.status_dir, args.worker_id)
            if args.abandon:
                abandon(store, args.results_dir, *args.abandon)
                return 0
            print(f"pull-only worker {store.worker}; local status {store.root}", flush=True)
            return poll(Queue(root, args.source_branch), store, args, fd)
    except (RunnerError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
