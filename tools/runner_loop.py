#!/usr/bin/env python3
"""Keep the local host lock while swapping idle runner controllers."""

import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile

from run_experiments import (
    CONTROLLER_COMMIT_ENV, CONTROLLER_HANDOFF_ENV, CONTROLLER_LOCK_ENV, CONTROLLER_RELOAD,
    CONTROLLER_TREE_ENV, LOCK_PATH, REPO_ROOT_ENV, RunnerError, device_lock, read_json,
)


OBJECT_ID = re.compile(r'(?:[0-9a-f]{40}|[0-9a-f]{64})')


def next_controller(handoff, current_tree):
    update = read_json(handoff)
    if not isinstance(update, dict) or any(not isinstance(update.get(key), str)
                                          for key in ('script', 'tree', 'commit')):
        raise RunnerError('invalid controller handoff')
    script = Path(update['script'])
    tree, commit = update['tree'], update['commit']
    if (not OBJECT_ID.fullmatch(tree) or not OBJECT_ID.fullmatch(commit) or tree == current_tree
            or not script.is_absolute() or script.name != 'run_experiments.py'
            or script.parent.name != 'tools' or script.parent.parent.name != tree or not script.is_file()):
        raise RunnerError('invalid controller snapshot in handoff')
    handoff.unlink()
    return script, tree, commit


def supervise(root, argv):
    script = root / 'tools' / 'run_experiments.py'
    if any(arg in ('-h', '--help') for arg in argv):
        return subprocess.call([sys.executable, '-B', str(script), *argv])

    child = None
    stopping = None
    forwarded = False

    def interrupt_child():
        nonlocal forwarded
        if child is not None and not forwarded:
            forwarded = True
            # Let the controller unwind its short Git lock. Experiment workers
            # are in their own sessions and keep the host lock until they exit.
            child.send_signal(signal.SIGINT)

    def stop(signum, _frame):
        nonlocal stopping
        if stopping is None:
            stopping = signum
            interrupt_child()

    handlers = {signum: signal.signal(signum, stop) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        with device_lock(LOCK_PATH) as fd, tempfile.TemporaryDirectory(prefix='gpu-runner-handoff-') as temp:
            handoff = Path(temp) / 'next.json'
            env = {**os.environ, REPO_ROOT_ENV: str(root), CONTROLLER_LOCK_ENV: str(fd),
                   CONTROLLER_HANDOFF_ENV: str(handoff)}
            for name in ('GPU_NET_SIM_LOCK_FD', CONTROLLER_TREE_ENV, CONTROLLER_COMMIT_ENV):
                env.pop(name, None)
            tree = None
            while stopping is None:
                child = subprocess.Popen([sys.executable, '-B', '-u', str(script), *argv],
                                         env=env, pass_fds=(fd,), start_new_session=True)
                if stopping is not None:
                    interrupt_child()
                status = child.wait()
                child = None
                if stopping is not None:
                    break
                if status != CONTROLLER_RELOAD:
                    return status if status >= 0 else 128 - status
                script, tree, commit = next_controller(handoff, tree)
                env.update({CONTROLLER_TREE_ENV: tree, CONTROLLER_COMMIT_ENV: commit})
                print(f'runner controller switched to {commit} (tools {tree})', flush=True)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return 128 + stopping


if __name__ == '__main__':
    try:
        sys.exit(supervise(Path(__file__).resolve().parents[1], sys.argv[1:]))
    except (RunnerError, OSError, ValueError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
