"""Prompt text, reply parsing and the built-in smoke tasks (with gold Triton solutions)."""

from __future__ import annotations

import re

SYSTEM_PROMPT = """You are an expert GPU kernel engineer. You are given a PyTorch reference \
architecture `Model`. Rewrite it as `ModelNew`, a drop-in replacement whose computation is \
implemented with custom Triton kernels and which is as fast as possible on an NVIDIA GPU.

Hard rules — any violation scores 0:
- Triton only. All computation happens inside `@triton.jit` kernels launched from \
`ModelNew.forward` (`kernel[grid](...)`). No raw CUDA / C++.
- No PyTorch compute fallback anywhere outside the kernels: no `torch.nn.functional` / `F.*`, \
no `torch.matmul/mm/bmm/einsum/conv*/softmax/...`, no tensor math methods or operators \
(`x.sum()`, `a @ b`, `a + b`, ...), no calling `nn.Linear/Conv*/LayerNorm/...` layers. \
PyTorch may only allocate tensors (`torch.empty`, `torch.zeros`, `*_like`) and change \
shape / layout / dtype (`view`, `reshape`, `permute`, `contiguous`, `to`, slicing).
- `nn.*` layers may be constructed in `__init__` only as parameter containers, in the same \
order as the reference so that weights initialise identically; read their `.weight` / \
`.bias` and compute in Triton.
- Only `torch`, `torch.nn`, `triton`, `triton.language`, `math` may be imported. No file, \
network or process I/O, no `eval/exec/getattr`, no monkey-patching.
- Never modify the input tensors. Same output shape, dtype and device as the reference, \
float32 precision.

Your code is checked on several hidden random inputs, then timed against the reference.

Reply with the complete new file in a single ```python code block: imports, kernels, \
helpers and `class ModelNew(nn.Module)` with the same `__init__` and `forward` signatures \
as `Model`. Do not include `get_inputs`, tests or usage code."""

EXAMPLE_REFERENCE = '''import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b
'''

EXAMPLE_SOLUTION = '''import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return triton_add(a, b)
'''

_RELU_REFERENCE = '''import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def get_inputs():
    return [torch.randn(2048, 4096)]


def get_init_inputs():
    return []
'''

_RELU_SOLUTION = '''import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=1024)
        return out
'''

_ADD_INPUTS = '''

def get_inputs():
    return [torch.randn(2048, 4096), torch.randn(2048, 4096)]


def get_init_inputs():
    return []
'''

# (name, reference source, gold solution). Not KernelBench problems: these exist so the
# package can be validated end to end without downloading the benchmark.
SMOKE_TASKS = [
    ("smoke_add", EXAMPLE_REFERENCE + _ADD_INPUTS, EXAMPLE_SOLUTION),
    ("smoke_relu", _RELU_REFERENCE, _RELU_SOLUTION),
]


def build_prompt(ref_src: str) -> str:
    return (
        "Here is an example of a reference architecture and its Triton rewrite.\n\n"
        f"Reference:\n```python\n{EXAMPLE_REFERENCE}```\n\n"
        f"Triton rewrite:\n```python\n{EXAMPLE_SOLUTION}```\n\n"
        "Now rewrite this reference architecture. `get_inputs()` / `get_init_inputs()` "
        "show the shapes it is evaluated with.\n\n"
        f"```python\n{ref_src.strip()}\n```\n\n"
        "Reply with the full new file in one ```python block."
    )


_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(reply: str | None) -> str:
    """The last fenced code block of a reply (or the raw text if there is none)."""
    text = reply or ""
    blocks = _CODE_BLOCK.findall(text)
    return (blocks[-1] if blocks else text).strip()
