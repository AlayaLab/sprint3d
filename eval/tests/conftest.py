"""Test configuration: CPU only, no metric models, no data.  The GPU smoke test is marked `gpu` and skipped without a
CUDA device or nvdiffrast."""
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent))
sys.path.insert(0, str(TESTS))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA device and nvdiffrast")


@pytest.fixture()
def synthetic(tmp_path):
    from synthetic import build_dataset
    return build_dataset(tmp_path)
