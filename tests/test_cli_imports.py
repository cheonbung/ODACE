"""P1 gate: CLI entrypoints must run --help from the REPO ROOT after the common-module
refactor (plan P1.10). Heavy (imports torch/diffusers per subprocess) but definitive."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

ENTRYPOINTS = [
    "models/odace/train_odace.py",
    "models/odace/experiments/regression_compare.py",
    "models/odace/experiments/compare_smoke_runs.py",
    "models/odace/experiments/xodace/run_pilot.py",
    "models/odace/experiments/xodace/weight_transplant.py",
]


@pytest.mark.parametrize("script", ENTRYPOINTS)
def test_cli_help_from_repo_root(script):
    proc = subprocess.run([sys.executable, script, "--help"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, (
        f"{script} --help failed (rc={proc.returncode})\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}")
    assert "usage" in (proc.stdout + proc.stderr).lower()
