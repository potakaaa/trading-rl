"""Round-6 CNN tests, run in a separate process: PyTorch and LightGBM must not share one on macOS
(two OpenMP runtimes -> segfault). The checks themselves live in tests/torch_isolated/cnn_checks.py."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_cnn_checks_in_isolated_process():
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(ROOT / "tests" / "torch_isolated" / "cnn_checks.py")],
                       cwd=ROOT, capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert "2 passed" in r.stdout
