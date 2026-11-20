"""Triton-only KernelBench environment for verifiers (v1).

The taskset / env classes are resolved lazily: the guard, sandbox and reward modules
stay importable without verifiers, and spawned sandbox children never pay for it.
"""

__all__ = ["TritonKernelEnv", "TritonKernelTaskset"]


def __getattr__(name: str):
    if name in __all__:
        from triton_kernel_env import taskset

        return getattr(taskset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
