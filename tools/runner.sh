#!/bin/sh
# Start the shared-filesystem experiment runner from its managed environment.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
bootstrap_python=${GPU_NET_SIM_PYTHON:-python3}

if [ "$#" -eq 1 ] && { [ "$1" = "--help" ] || [ "$1" = "-h" ]; }; then
    exec "$bootstrap_python" -B "$script_dir/run_experiments.py" "$@"
fi

environment_python=$("$bootstrap_python" -B "$script_dir/setup_python.py")
exec "$environment_python" -B -u "$repo_root/tools/run_experiments.py" "$@"
