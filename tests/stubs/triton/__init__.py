"""Fake Triton for CPU-only unit tests (never installed, only put on PYTHONPATH by tests).

It mirrors just enough of the real API surface for the sandbox to exercise its control
flow: `@triton.jit` returns a `triton.runtime.jit.JITFunction`, `kernel[grid](...)` goes
through `JITFunction.run(grid=..., warmup=False)`. The "kernel" body runs as plain Python
on NumPy views of the tensor arguments, so — like real Triton — its work is invisible to
the PyTorch dispatcher. It says nothing about real Triton codegen; GPU behaviour is
covered by the `gpu`-marked tests in the Colab notebook.
"""

import torch.utils._triton as _torch_triton

from . import language
from .runtime.jit import JITFunction

IS_TEST_STUB = True

# torch (dynamo / inductor) treats any importable `triton` as the real compiler and
# reaches into its internals. Tell torch there is no Triton so it leaves the stub alone.
_torch_triton.has_triton_package = lambda: False


def jit(fn=None, **_kwargs):
    if fn is None:
        return lambda f: JITFunction(f)
    return JITFunction(fn)


def autotune(configs=None, key=None, **_kwargs):
    return lambda fn: fn


class Config:
    def __init__(self, kwargs=None, **_meta):
        self.kwargs = kwargs or {}


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(n):
    return 1 << (max(int(n), 1) - 1).bit_length()
