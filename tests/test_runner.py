import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import run_experiments as runner


class RunnerTest(unittest.TestCase):
    def git(self, cwd, *args):
        return subprocess.run(
            ["git", *args], cwd=cwd, text=True, capture_output=True, check=True,
        ).stdout.strip()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.checkout = self.root / "worker"
        self.git(self.root, "init", "--bare", "--initial-branch=main", str(self.remote))
        self.git(self.root, "clone", str(self.remote), str(self.checkout))
        self.git(self.checkout, "config", "user.name", "Test")
        self.git(self.checkout, "config", "user.email", "test@localhost")
        self.git(self.checkout, "config", "commit.gpgsign", "false")
        (self.checkout / "tasks").mkdir()
        (self.checkout / "tools").mkdir()
        self.results = self.root / "results"
        self.queue = runner.Queue(self.checkout, "main", "experiment-status")

    def commit_task(self, version=1, marker="original", params=None):
        task = {"id": 1, "version": version, "kind": "gpu_inventory", "params": params or {}}
        runner.write_json(self.checkout / "tasks/000001.json", task)
        (self.checkout / "tools/experiment_handlers.py").write_text(
            "def prepare(kind, params):\n"
            "    def run(source, output):\n"
            f"        (output / 'summary.txt').write_text({marker!r})\n"
            "        (output / 'raw.log').write_text('excluded')\n"
            "    return run\n",
            encoding="utf-8",
        )
        self.git(self.checkout, "add", ".")
        self.git(self.checkout, "commit", "-m", f"task v{version} {marker}")
        self.git(self.checkout, "push", "origin", "main")
        return task

    def claim(self, token="worker-a"):
        source = self.queue.source()
        task, digest = self.queue.tasks(source)[0]
        claim = {
            "id": 1, "version": task["version"], "kind": task["kind"],
            "task_sha256": digest, "source_commit": source, "state": "running", "claim": token,
        }
        return task, claim

    def test_two_devices_cannot_claim_same_task(self):
        self.commit_task()
        _, first = self.claim()
        second_checkout = self.root / "worker2"
        self.git(self.root, "clone", str(self.remote), str(second_checkout))
        second_queue = runner.Queue(second_checkout, "main", "experiment-status")
        self.assertEqual(self.queue.statuses(), ("", {}))
        self.assertEqual(second_queue.statuses(), ("", {}))
        with concurrent.futures.ThreadPoolExecutor() as pool:
            a = pool.submit(self.queue.publish, "", first)
            b = pool.submit(second_queue.publish, "", {**first, "claim": "worker-b"})
            outcomes = [a.result(), b.result()]
        self.assertEqual(outcomes.count(True), 1)
        _, rows = self.queue.statuses()
        self.assertEqual(rows[1, 1]["claim"], "worker-a" if outcomes[0] else "worker-b")

    def test_terminal_retry_and_increasing_version(self):
        self.commit_task()
        _, claim = self.claim()
        self.assertTrue(self.queue.publish("", claim))
        self.commit_task(version=2)
        head, rows = self.queue.statuses()
        self.assertIsNone(runner.start_task(self.queue, self.queue.source(), head, rows, self.results, "worker", 0))
        output = runner.run_path(self.results, claim)
        output.mkdir(parents=True)
        runner.write_json(output / "result.json", {**claim, "state": "failed", "finished_at": "now", "exit_code": 1})
        with patch.object(self.queue, "publish", return_value=False):
            with self.assertRaisesRegex(runner.RunnerError, "status update pending"):
                runner.flush_terminal(self.queue, self.results, head, rows)
        self.assertFalse((output / ".status-published").exists())
        head, rows = runner.flush_terminal(self.queue, self.results, head, rows)
        self.assertEqual(rows[1, 1]["state"], "failed")
        self.assertTrue((output / ".status-published").exists())
        with patch.object(runner.subprocess, "Popen", wraps=self.faked_popen):
            active = runner.start_task(self.queue, self.queue.source(), head, rows, self.results, "worker", 0)
        self.assertEqual(active[1].name, "v2")
        self.assertEqual(self.queue.statuses()[1][1, 2]["state"], "running")

        self.commit_task(version=2, params={"changed": True})
        head, rows = self.queue.statuses()
        with self.assertRaisesRegex(runner.RunnerError, "changed without increasing version"):
            runner.start_task(self.queue, self.queue.source(), head, rows, self.results, "worker", 0)

    @staticmethod
    def faked_popen(args, *positional, **kwargs):
        if "--_execute" in args:
            return object()
        return REAL_POPEN(args, *positional, **kwargs)

    def test_executes_handler_at_claimed_commit_and_packs_small_results(self):
        self.commit_task(marker="from claimed source")
        task, claim = self.claim()
        output = runner.run_path(self.results, claim)
        output.mkdir(parents=True)
        runner.write_json(output / "run.json", {**claim, "task": task})
        self.commit_task(version=2, marker="newer source")
        self.assertEqual(runner.execute(self.queue, output), 0)
        self.assertEqual((output / "summary.txt").read_text(), "from claimed source")
        self.assertTrue((output / "raw.log").exists())
        result = json.loads((output / "result.json").read_text())
        self.assertEqual(result["source_commit"], claim["source_commit"])
        self.assertEqual(result["state"], "succeeded")
        with tarfile.open(output.with_suffix(".tar.gz")) as archive:
            self.assertEqual(archive.getnames(), ["result.json", "summary.txt"])
            self.assertEqual(archive.extractfile("summary.txt").read(), b"from claimed source")

    def test_only_claiming_host_can_abandon(self):
        self.commit_task()
        _, claim = self.claim()
        self.assertTrue(self.queue.publish("", {**claim, "host_id": "original host"}))
        with patch.object(runner, "host_identity", return_value="different host"):
            with self.assertRaisesRegex(runner.RunnerError, "requires the claiming host"):
                runner.abandon(self.queue, self.results, 1, 1, "worker")
        self.assertEqual(self.queue.statuses()[1][1, 1]["state"], "running")
        with patch.object(runner, "host_identity", return_value="original host"):
            runner.abandon(self.queue, self.results, 1, 1, "worker")
        self.assertEqual(self.queue.statuses()[1][1, 1]["state"], "abandoned")

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
