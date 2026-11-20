"""Test bootstrap.

On a machine with CUDA and real Triton (Colab) the whole suite runs against the real
stack. Elsewhere the fake Triton in `tests/stubs` is put on `sys.path` (spawned sandbox
children inherit it) and CPU evaluation is enabled; `gpu`-marked tests are skipped.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "environments" / "triton_kernel_env"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _real_stack() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import triton

        return not getattr(triton, "IS_TEST_STUB", False)
    except ImportError:
        return False


REAL = _real_stack()
if not REAL:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
    os.environ["TRITON_ENV_ALLOW_CPU"] = "1"


def pytest_collection_modifyitems(config, items):
    if REAL:
        return
    skip = pytest.mark.skip(reason="needs CUDA + real Triton (run notebooks/01 in Colab)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_leaked_children():
    """Every test must leave no sandbox child behind."""
    import multiprocessing as mp

    yield
    assert mp.active_children() == []
