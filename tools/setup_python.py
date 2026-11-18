#!/usr/bin/env python3
"""Prepare a platform-specific Python environment under .python and .venv."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid


PYTHON_VERSION = "3.12"
BOOTSTRAP_SCHEMA = 2
ROOT = Path(__file__).resolve().parents[1]
INFO_CODE = (
    "import encodings,json,os,sys,venv; print(json.dumps({"
    "'implementation':sys.implementation.name,"
    "'version':list(sys.version_info[:3]),"
    "'executable':os.path.realpath(getattr(sys,'_base_executable',sys.executable)),"
    "'base_prefix':sys.base_prefix,'prefix':sys.prefix}))"
)


def platform_key():
    machine = platform.machine().lower()
    machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
    return f"{sys.platform}-{machine}"


def command(args, env=None):
    process = subprocess.Popen(
        [str(arg) for arg in args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate()
    except BaseException:
        # Bootstrap children must stop writing before their shared lock is removed.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"{args[0]} exited {process.returncode}: {stderr.strip()}")
    return stdout.strip()


def python_info(executable):
    info = json.loads(command([executable, "-I", "-c", INFO_CODE]))
    if info["version"][:2] < [3, 9]:
        raise RuntimeError("Python 3.9 or newer is required")
    return info


def runtime_fingerprint(info):
    signature = {key: info[key] for key in ("implementation", "version", "executable", "base_prefix")}
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]


def write_metadata(path, value):
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextlib.contextmanager
def bootstrap_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    last_notice = 0
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError:
            now = time.monotonic()
            if now - last_notice >= 30:
                print(
                    f"Waiting for Python setup lock {path}. If its owner stopped, "
                    "confirm its bootstrap children have exited before removing the lock.",
                    file=sys.stderr, flush=True,
                )
                last_notice = now
            time.sleep(1)
    token = uuid.uuid4().hex
    owner = path / "owner.json"
    try:
        write_metadata(owner, {"host": socket.gethostname(), "pid": os.getpid(), "token": token})
        yield
    finally:
        # Never remove an ownership record belonging to another setup attempt.
        if owner.is_file() and json.loads(owner.read_text())["token"] == token:
            owner.unlink()
            path.rmdir()


def managed_runtime(runtime_root, uv):
    managed = runtime_root / "managed"
    env = {**os.environ, "UV_PYTHON_INSTALL_DIR": str(managed),
           "UV_CACHE_DIR": str(runtime_root / "cache"), "UV_NO_CONFIG": "1"}
    # Corporate CA roots may be installed in the host trust store. This option
    # keeps certificate verification enabled and also works with older uv.
    if "UV_NATIVE_TLS" not in env and "UV_SYSTEM_CERTS" not in env:
        env["UV_NATIVE_TLS"] = "true"
    find = [uv, "python", "find", "--managed-python", "--no-project", "--no-python-downloads", PYTHON_VERSION]
    try:
        executable = Path(command(find, env))
        if not executable.resolve().is_relative_to(managed.resolve()):
            raise RuntimeError("uv selected a Python outside the repository-managed directory")
    except RuntimeError:
        print(f"Installing Python {PYTHON_VERSION} under {managed}", file=sys.stderr, flush=True)
        command([uv, "python", "install", "--install-dir", managed, "--no-bin", PYTHON_VERSION], env)
        executable = Path(command(find, env))
    if not executable.resolve().is_relative_to(managed.resolve()):
        raise RuntimeError("uv selected a Python outside the repository-managed directory")
    info = python_info(executable)
    metadata = {"provider": "uv", "python": str(executable), "runtime": info}
    write_metadata(runtime_root / "managed-runtime.json", metadata)
    return executable, info, metadata


def existing_runtime(runtime_root, selected):
    # A Python found in an active virtual environment may be a copied binary.
    # Probe its base executable again so creation always runs from the original
    # installation, where relocatable Python can locate its standard library.
    info = python_info(selected)
    info = python_info(info["executable"])
    system_root = runtime_root / f"system-{runtime_fingerprint(info)}"
    system_root.mkdir(parents=True, exist_ok=True)
    executable = system_root / "python"
    if executable.is_symlink():
        if executable.resolve() != Path(info["executable"]):
            raise RuntimeError(f"Recorded Python symlink changed: {executable}")
    elif executable.exists():
        raise RuntimeError(f"Expected a Python symlink: {executable}")
    else:
        executable.symlink_to(info["executable"])
    metadata = {"provider": "system-symlink", "python": str(executable), "runtime": info}
    write_metadata(system_root / "runtime.json", metadata)
    return executable, info, metadata


def selected_runtime(runtime_root):
    override = os.environ.get("GPU_NET_SIM_PYTHON")
    if override:
        return existing_runtime(runtime_root, override)
    try:
        # The runner needs only the standard library. Reuse the interpreter
        # already available on this device before downloading another runtime.
        return existing_runtime(runtime_root, sys.executable)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Existing Python runtime is unusable ({error}); trying managed Python.", file=sys.stderr)
    uv = shutil.which("uv")
    if uv:
        return managed_runtime(runtime_root, uv)
    raise RuntimeError("No usable Python 3.9+ with the venv module was found; set GPU_NET_SIM_PYTHON")


def environment_python(root=ROOT):
    if sys.version_info < (3, 9):
        raise RuntimeError("Python 3.9 or newer is required to start the runner")
    key = platform_key()
    with bootstrap_lock(root / ".lock" / f"python-{key}.lock"):
        runtime_root = root / ".python" / key
        runtime_root.mkdir(parents=True, exist_ok=True)
        executable, info, metadata = selected_runtime(runtime_root)
        major, minor = info["version"][:2]
        environment = root / ".venv" / f"{key}-py{major}.{minor}-{runtime_fingerprint(info)}-b{BOOTSTRAP_SCHEMA}"
        python = environment / "bin" / "python"
        marker = environment / ".runner-environment.json"
        if environment.exists():
            if not marker.is_file() or not python.is_file():
                raise RuntimeError(f"Incomplete Python environment {environment}; inspect it before removing or repairing it")
            actual = python_info(python)
            if runtime_fingerprint(actual) != runtime_fingerprint(info) or actual["prefix"] != str(environment):
                raise RuntimeError(f"Python environment no longer matches its recorded runtime: {environment}")
            return python
        environment.parent.mkdir(parents=True, exist_ok=True)
        try:
            # The runner uses the standard library, so it does not require pip.
            # Invoking the recorded .python symlink here makes Python 3.12
            # write that symlink's directory as pyvenv.cfg home. Standalone
            # builds can then fall back to /install and fail to import encodings.
            command([info["executable"], "-I", "-m", "venv", "--without-pip", environment])
            actual = python_info(python)
            if runtime_fingerprint(actual) != runtime_fingerprint(info) or actual["prefix"] != str(environment):
                raise RuntimeError(f"Created Python environment has an unexpected runtime: {environment}")
            write_metadata(marker, {**metadata, "bootstrap_schema": BOOTSTRAP_SCHEMA})
        except BaseException:
            shutil.rmtree(environment, ignore_errors=True)
            raise
        return python


if __name__ == "__main__":
    try:
        print(environment_python())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Python environment setup failed: {error}", file=sys.stderr)
        sys.exit(1)
