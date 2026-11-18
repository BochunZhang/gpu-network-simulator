import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import run_experiments as runner


class RunnerTest(unittest.TestCase):
    def git(self, cwd, *args):
        return subprocess.run(
            [self.real_git, *args], cwd=cwd, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def setUp(self):
        self.real_git = shutil.which("git")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.author = self.root / "author"
        self.checkout = self.root / "worker"
        self.git(self.root, "init", "--bare", "--initial-branch=main", str(self.remote))
        hook = self.remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho 'uploads forbidden' >&2\nexit 1\n")
        hook.chmod(0o755)
        self.git(self.root, "init", "--initial-branch=main", str(self.author))
        self.git(self.author, "config", "user.name", "Test")
        self.git(self.author, "config", "user.email", "test@localhost")
        self.git(self.author, "config", "commit.gpgsign", "false")
        (self.author / "tasks").mkdir()
        (self.author / "tools").mkdir()
        shutil.copyfile(REPO / "tools/run_experiments.py", self.author / "tools/run_experiments.py")
        self.commit_task()
        self.git(self.root, "clone", str(self.remote), str(self.checkout))
        self.results = self.root / "results"
        self.queue = runner.Queue(self.checkout)
        self.store = runner.StatusStore(self.root / "status", "gpu01")

    def commit_task(self, version=1, marker="original", params=None, task_id=1, worker="gpu01", kind="new_kind"):
        task = {"id": task_id, "version": version, "worker": worker, "kind": kind, "params": params or {}}
        runner.write_json(self.author / f"tasks/{task_id:06d}.json", task)
        (self.author / "tools/experiment_handlers.py").write_text(
            "def prepare(kind, params):\n"
            "    def run(source, output):\n"
            f"        (output / 'summary.txt').write_text({marker!r})\n"
            "        (output / 'raw.log').write_text('excluded')\n"
            "    return run\n",
            encoding="utf-8",
        )
        self.git(self.author, "add", ".")
        self.git(self.author, "commit", "-m", f"task {task_id} v{version} {marker}")
        # Publish the developer fixture locally. Workers have no receiving access.
        self.git(self.remote, "fetch", str(self.author), "main:main")
        return task

    def run_record(self, task_id=1):
        source = self.queue.source()
        task, digest = next(pair for pair in self.queue.tasks(source) if pair[0]["id"] == task_id)
        row = {
            "id": task_id, "version": task["version"], "kind": task["kind"], "worker": task["worker"],
            "host_id": runner.host_identity(), "task_sha256": digest, "source_commit": source,
            "state": "running", "claim": f"attempt-{task_id}-{task['version']}",
        }
        return task, row

    @staticmethod
    def faked_popen(args, *positional, **kwargs):
        if "--_execute" in args:
            return object()
        return REAL_POPEN(args, *positional, **kwargs)

    def start(self, store=None):
        store = store or self.store
        with patch.object(runner.subprocess, "Popen", wraps=self.faked_popen):
            return runner.start_task(self.queue, self.queue.source(), store, store.reconcile(self.results), self.results, 0)

    def test_static_assignment_between_two_workers(self):
        self.commit_task(task_id=2, worker="gpu02", marker="second worker")
        other_results = self.root / "results2"
        other_store = runner.StatusStore(self.root / "status2", "gpu02")
        second_checkout = self.root / "worker2"
        self.git(self.root, "clone", str(self.remote), str(second_checkout))
        second_queue = runner.Queue(second_checkout)
        first = self.start()
        self.assertEqual(first[1], self.results / "000001/v1")
        with patch.object(runner.subprocess, "Popen", wraps=self.faked_popen):
            second = runner.start_task(second_queue, second_queue.source(), other_store, {}, other_results, 0)
        self.assertEqual(second[1], other_results / "000002/v1")
        self.assertEqual(set(self.store.read()), {(1, 1)})
        self.assertEqual(set(other_store.read()), {(2, 1)})

    def test_local_terminal_resume_version_and_manifest_change(self):
        task, row = self.run_record()
        self.store.save(row)
        self.commit_task(version=2)
        with self.assertRaisesRegex(runner.RunnerError, "unfinished local execution"):
            self.start()
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / "result.json", {**row, "state": "failed", "finished_at": "now", "exit_code": 1})
        rows = self.store.reconcile(self.results)
        self.assertEqual(rows[1, 1]["state"], "failed")
        self.assertEqual(runner.StatusStore(self.root / "status", "gpu01").read()[1, 1]["state"], "failed")
        active = self.start()
        self.assertEqual(active[1].name, "v2")
        row2 = self.store.read()[1, 2]
        runner.write_json(active[1] / "result.json", {**row2, "state": "succeeded", "finished_at": "now", "exit_code": 0})
        self.assertIsNone(self.start())
        self.commit_task(version=2, params={"changed": True})
        with self.assertRaisesRegex(runner.RunnerError, "changed without increasing version"):
            self.start()

    def test_running_record_blocks_all_tasks_and_recovery_is_local(self):
        _, row = self.run_record()
        self.store.save(row)  # Simulate a crash before even creating the output directory.
        self.commit_task(task_id=2, marker="must wait")
        with self.assertRaisesRegex(runner.RunnerError, "unfinished local execution"):
            self.start()
        with patch.object(runner, "host_identity", return_value="different host"):
            with self.assertRaisesRegex(runner.RunnerError, "requires the original execution host"):
                runner.abandon(self.store, self.results, 1, 1)
        runner.abandon(self.store, self.results, 1, 1)
        self.assertEqual(self.store.read()[1, 1]["state"], "abandoned")
        self.assertEqual(self.start()[1], self.results / "000002/v1")

    def test_executes_pinned_new_handler_and_packs_small_results(self):
        self.commit_task(marker="from pinned source", kind="new_kind")
        task, row = self.run_record()
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / "run.json", {**row, "task": task})
        self.commit_task(version=2, marker="newer source")
        self.assertEqual(runner.execute(self.queue, output), 0)
        self.assertEqual((output / "summary.txt").read_text(), "from pinned source")
        self.assertTrue((output / "raw.log").exists())
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["source_commit"], row["source_commit"])
        self.assertEqual(result["state"], "succeeded")
        self.assertIn("pinned_handler_sha256", result)
        with tarfile.open(output.with_suffix(".tar.gz")) as archive:
            self.assertEqual(archive.getnames(), ["result.json", "summary.txt"])
            self.assertEqual(archive.extractfile("summary.txt").read(), b"from pinned source")

    def test_reconcile_precedes_network_failure(self):
        _, row = self.run_record()
        self.store.save(row)
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        runner.write_json(output / "result.json", {**row, "state": "succeeded", "finished_at": "now", "exit_code": 0})
        args = SimpleNamespace(once=True, results_dir=self.results)
        with patch.object(self.queue, "source", side_effect=runner.RunnerError("offline")):
            self.assertEqual(runner.poll(self.queue, self.store, args, 0), 1)
        self.assertEqual(self.store.read()[1, 1]["state"], "succeeded")

    def test_worker_exit_without_terminal_stops_queue(self):
        class ExitedWorker:
            returncode = 9
            def poll(self):
                return self.returncode
            def wait(self, timeout):
                return self.returncode
        output = self.results / "000001/v1"
        args = SimpleNamespace(once=False, results_dir=self.results)
        with patch.object(runner, "start_task", return_value=(ExitedWorker(), output)) as start:
            self.assertEqual(runner.poll(self.queue, self.store, args, 0), 1)
        self.assertEqual(start.call_count, 1)

    def test_invalid_assignment_and_result_collisions_are_rejected(self):
        with self.assertRaises(runner.RunnerError):
            runner.StatusStore(self.root / "status", "../escape")
        task = json.loads((self.author / "tasks/000001.json").read_text())
        del task["worker"]
        runner.write_json(self.author / "tasks/000001.json", task)
        self.git(self.author, "add", ".")
        self.git(self.author, "commit", "-m", "unassigned task")
        self.git(self.remote, "fetch", str(self.author), "main:main")
        with self.assertRaisesRegex(runner.RunnerError, "explicit worker assignment"):
            self.queue.tasks(self.queue.source())
        self.commit_task(version=2)
        _, row = self.run_record()
        output = runner.run_path(self.results, row)
        output.mkdir(parents=True)
        with self.assertRaisesRegex(runner.RunnerError, "result path already exists"):
            self.start()
        self.store.save(row)
        runner.write_json(output / "result.json", {**row, "claim": "different", "state": "failed"})
        with self.assertRaisesRegex(runner.RunnerError, "does not match"):
            self.store.reconcile(self.results)

    def test_cli_pull_only_origin_and_version_rerun(self):
        guard_dir = self.root / "bin"
        guard_dir.mkdir()
        command_log = self.root / "git-calls.txt"
        guard = guard_dir / "git"
        guard.write_text(
            f"#!{sys.executable}\n"
            "import os, sys\n"
            f"with open({str(command_log)!r}, 'a') as log: log.write(' '.join(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1] in {'push', 'send-pack', 'credential'}: sys.exit(98)\n"
            f"os.execv({self.real_git!r}, [{self.real_git!r}, *sys.argv[1:]])\n"
        )
        guard.chmod(0o755)
        env = {**os.environ, "PATH": str(guard_dir) + os.pathsep + os.environ["PATH"], "PYTHONDONTWRITEBYTECODE": "1"}
        args = [sys.executable, "-B", str(self.checkout / "tools/run_experiments.py"), "--once",
                "--status-dir", str(self.root / "cli-status"), "--results-dir", str(self.results)]
        first = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        store = runner.StatusStore(self.root / "cli-status", "gpu01")
        self.assertEqual(store.read()[1, 1]["state"], "succeeded")
        first_result = self.results / "000001/v1/result.json"
        first_mtime = first_result.stat().st_mtime_ns
        same = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(same.returncode, 0, same.stdout + same.stderr)
        self.assertEqual(first_result.stat().st_mtime_ns, first_mtime)
        self.commit_task(version=2, marker="second version")
        updated = subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(updated.returncode, 0, updated.stdout + updated.stderr)
        self.assertEqual((self.results / "000001/v2/summary.txt").read_text(), "second version")
        self.assertEqual(store.read()[1, 2]["state"], "succeeded")
        self.assertEqual(self.git(self.remote, "for-each-ref", "--format=%(refname)"), "refs/heads/main")
        commands = command_log.read_text().splitlines()
        self.assertTrue(any(line.startswith("fetch ") for line in commands))
        self.assertFalse(any(line.split()[0] in {"push", "send-pack", "credential"} for line in commands))

    def test_device_lock_survives_parent_fd_close(self):
        path = self.root / "device.lock"
        with runner.device_lock(path) as fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.read()"],
                stdin=subprocess.PIPE, pass_fds=(fd,),
            )
        try:
            with self.assertRaisesRegex(runner.RunnerError, "already locked"):
                with runner.device_lock(path):
                    self.fail("second worker obtained a held device lock")
        finally:
            child.communicate(timeout=10)
        with runner.device_lock(path):
            pass


REAL_POPEN = subprocess.Popen
