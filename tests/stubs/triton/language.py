"""Fake `triton.language`: the kernel-signature annotation, plus what torch probes for."""

constexpr = int


class dtype:  # torch._dynamo references `triton.language.dtype` when triton is importable
    pass
