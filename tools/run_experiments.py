#!/usr/bin/env python3
"""Run one allowlisted experiment at a time from a Git-backed queue."""

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
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

import experiment_handlers


LOCK_PATH = Path("/tmp/gpu-network-simulator.lock")
POLL_SECONDS = 10
TERMINAL = {"succeeded", "failed", "abandoned"}
RESULT_SUFFIXES = {".json", ".csv", ".txt"}
FILE_LIMIT = 2 * 1024**2
ARCHIVE_LIMIT = 16 * 1024**2


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


def status_path(task_id, version):
    return f"status/{task_id:06d}/v{version}.json"


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
    def __init__(self, root, source_branch, status_branch):
        self.root = root
        self.source_ref = f"refs/heads/{source_branch}"
        self.status_ref = f"refs/heads/{status_branch}"
        self.status_snapshot = ("", {})
        if self.source_ref == self.status_ref:
            raise RunnerError("source and status branches must be different")
        for ref in (self.source_ref, self.status_ref):
            self.git("check-ref-format", ref)

    def git(self, *args, input_text=None, env=None, check=True, timeout=30):
        try:
            proc = subprocess.run(
                ["git", *args],
                cwd=self.root,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0", **(env or {})},
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"git {args[0]} timed out after {timeout} seconds") from exc
        if proc.returncode and check:
            raise RunnerError(f"git {args[0]}: {proc.stderr.strip() or proc.stdout.strip()}")
        return proc

    def fetch(self, ref, local_ref):
        self.git("fetch", "--quiet", "--no-tags", "--no-write-fetch-head", "origin", f"+{ref}:{local_ref}")
        return self.git("rev-parse", local_ref).stdout.strip()

    def source(self):
        return self.fetch(self.source_ref, "refs/gpu-network-simulator/source")

    def statuses(self):
        remote = self.git("ls-remote", "--heads", "origin", self.status_ref).stdout.strip()
        if not remote:
            return "", {}
        head = remote.split()[0]
        if head == self.status_snapshot[0]:
            return self.status_snapshot
        local_ref = "refs/gpu-network-simulator/status"
        local = self.git("rev-parse", "--verify", local_ref, check=False).stdout.strip()
        if head != local:
            head = self.fetch(self.status_ref, local_ref)
        rows = {}
        for path in self.git("ls-tree", "-rz", "--name-only", head, "--", "status/").stdout.split("\0"):
            if re.fullmatch(r"status/\d{6,}/v[1-9]\d*\.json", path):
                row = self.json_at(head, path)
                if (
                    type(row.get("id")) is not int
                    or type(row.get("version")) is not int
                    or path != status_path(row["id"], row["version"])
                    or row.get("state") not in TERMINAL | {"running"}
                    or not isinstance(row.get("claim"), str)
                ):
                    raise RunnerError(f"invalid status: {path}")
                rows[row["id"], row["version"]] = row
        self.status_snapshot = head, rows
        return self.status_snapshot

    def json_at(self, head, path):
        try:
            value = json.loads(self.git("show", f"{head}:{path}").stdout)
        except ValueError as exc:
            raise RunnerError(f"invalid JSON at {head}:{path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RunnerError(f"expected object at {head}:{path}")
        return value

    def tasks(self, source):
        paths = self.git("ls-tree", "-rz", "--name-only", source, "--", "tasks/").stdout.split("\0")
        tasks = []
        for path in paths:
            if not path.endswith(".json"):
                continue
            task = self.json_at(source, path)
            task_id, version = task.get("id"), task.get("version")
            if (
                type(task_id) is not int or task_id < 1
                or path != f"tasks/{task_id:06d}.json"
                or type(version) is not int or version < 1
                or not isinstance(task.get("kind"), str)
                or not isinstance(task.get("params"), dict)
                or not isinstance(task.get("description", ""), str)
                or set(task) - {"id", "version", "kind", "params", "description"}
            ):
                raise RunnerError(f"invalid task manifest: {path}")
            definition = {k: v for k, v in task.items() if k != "description"}
            try:
                digest = hashlib.sha256(encode(definition).encode("utf-8")).hexdigest()
            except ValueError as exc:
                raise RunnerError(f"invalid task values: {path}: {exc}") from exc
            tasks.append((task, digest))
        return sorted(tasks, key=lambda item: item[0]["id"])

    def publish(self, parent, row):
        task_id, version = row["id"], row["version"]
        with tempfile.TemporaryDirectory(prefix="gpu-network-simulator-index-") as temp:
            env = {
                "GIT_INDEX_FILE": str(Path(temp) / "index"),
                "GIT_AUTHOR_NAME": "GPU Network Simulator",
                "GIT_AUTHOR_EMAIL": "gpu-network-simulator@localhost",
                "GIT_COMMITTER_NAME": "GPU Network Simulator",
                "GIT_COMMITTER_EMAIL": "gpu-network-simulator@localhost",
            }
            self.git("read-tree", parent or "--empty", env=env)
            blob = self.git("hash-object", "-w", "--stdin", input_text=encode(row)).stdout.strip()
            self.git("update-index", "--add", "--cacheinfo", f"100644,{blob},{status_path(task_id, version)}", env=env)
            tree = self.git("write-tree", env=env).stdout.strip()
            parents = ["-p", parent] if parent else []
            commit = self.git(
                "commit-tree", tree, *parents, "-m", f"Task {task_id} v{version}: {row['state']}", env=env
            ).stdout.strip()
        # Every write is a child of the observed head. The lease also covers
        # initialization and a remote reset where a normal push could fast-forward.
        push = self.git(
            "push", "--porcelain", f"--force-with-lease={self.status_ref}:{parent}",
            "origin", f"{commit}:{self.status_ref}", check=False,
        )
        if push.returncode:
            print(f"status push not confirmed: {push.stderr.strip() or push.stdout.strip()}", file=sys.stderr)
        return push.returncode == 0


def validate_task(task, handlers=experiment_handlers):
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
        validate_task(run["task"])
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
                handler = validate_task(run["task"], handlers)
                handler(source, output)
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


def flush_terminal(queue, results, head, statuses):
    for path in sorted(results.glob("*/v*/result.json")):
        ack = path.with_name(".status-published")
        if ack.exists():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        remote = statuses.get((row["id"], row["version"]))
        if not remote or remote["claim"] != row["claim"]:
            raise RunnerError(f"no matching remote claim for {path}")
        if remote["state"] == "running":
            terminal = {**remote, **{key: row[key] for key in ("state", "finished_at", "exit_code")}}
            if not queue.publish(head, terminal):
                raise RunnerError(f"status update pending for {path}")
            head, statuses = queue.statuses()
        elif remote["state"] != row["state"]:
            print(f"retaining local result for {path}; remote state is {remote['state']}", file=sys.stderr)
        ack.touch()
    return head, statuses


def start_task(queue, source, head, statuses, results, worker, fd):
    for task, digest in queue.tasks(source):
        task_id, version = task["id"], task["version"]
        prior = [row for (i, _), row in statuses.items() if i == task_id]
        current = statuses.get((task_id, version))
        if current and current.get("task_sha256") != digest:
            raise RunnerError(f"task {task_id} changed without increasing version {version}")
        if any(row["version"] >= version or row["state"] == "running" for row in prior):
            continue
        validate_task(task)
        row = {
            "id": task_id,
            "version": version,
            "kind": task["kind"],
            "task_sha256": digest,
            "source_commit": source,
            "worker": worker,
            "host_id": host_identity(),
            "claim": uuid.uuid4().hex,
            "state": "running",
            "claimed_at": timestamp(),
        }
        output = run_path(results, row)
        if output.exists():
            raise RunnerError(f"local result path already exists: {output}")
        if not queue.publish(head, row):
            raise RunnerError(f"claim not confirmed for task {task_id} v{version}")
        output.mkdir(parents=True)
        script = Path(__file__).resolve()
        runner_sha256 = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (script, Path(experiment_handlers.__file__))
        }
        write_json(output / "run.json", {**row, "task": task, "runner_sha256": runner_sha256})
        env = {**os.environ, "GPU_NET_SIM_LOCK_FD": str(fd)}
        with (output / "worker.log").open("wb") as log:
            child = subprocess.Popen(
                [sys.executable, str(script), "--_execute", str(output)],
                env=env,
                start_new_session=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                pass_fds=(fd,),
            )
        print(f"running task {task_id} v{version} at {source[:12]} in {output}", flush=True)
        return child, output


def abandon(queue, results, task_id, version, worker):
    head, statuses = queue.statuses()
    head, statuses = flush_terminal(queue, results, head, statuses)
    row = statuses.get((task_id, version))
    if not row or row["state"] != "running":
        raise RunnerError(f"task {task_id} v{version} has no running claim")
    # The caller holds the same device lock as the original dispatcher and
    # experiment. Only that host can attest that they no longer hold it.
    if row.get("host_id") != host_identity():
        raise RunnerError("abandonment requires the claiming host")
    if not queue.publish(head, {**row, "state": "abandoned", "finished_at": timestamp(), "abandoned_by": worker}):
        raise RunnerError("abandonment not confirmed; inspect remote status")
    print(f"abandoned task {task_id} v{version}; raise its version to rerun")


def poll(queue, args, fd):
    active = None
    while True:
        result = None
        if active is not None and active[0].poll() is not None:
            child, output = active
            result = child.returncode
            if not (output / "result.json").is_file():
                print(f"worker exited without a terminal record: {output}", file=sys.stderr)
                result = 1
            active = None
        next_poll = time.monotonic() + POLL_SECONDS
        try:
            head, statuses = queue.statuses()
            head, statuses = flush_terminal(queue, args.results_dir, head, statuses)
            source = queue.source()
            if args.once and result is not None:
                return max(1, result) if result else 0
            if active is None:
                active = start_task(queue, source, head, statuses, args.results_dir, args.worker_id, fd)
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
        return execute(Queue(root, "main", "experiment-status"), Path(sys.argv[2]))

    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="claim and finish at most one task")
    mode.add_argument(
        "--abandon", nargs=2, metavar=("ID", "VERSION"), type=int,
        help="mark a running claim after confirming its experiment process has stopped",
    )
    cli.add_argument("--source-branch", default="main")
    cli.add_argument("--status-branch", default="experiment-status")
    cli.add_argument("--results-dir", type=Path, default=root / "results")
    cli.add_argument("--worker-id", default=socket.gethostname())
    args = cli.parse_args()
    args.results_dir = args.results_dir.resolve()
    try:
        with device_lock(LOCK_PATH) as fd:
            queue = Queue(root, args.source_branch, args.status_branch)
            if args.abandon:
                abandon(queue, args.results_dir, *args.abandon, args.worker_id)
                return 0
            return poll(queue, args, fd)
    except (RunnerError, OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
