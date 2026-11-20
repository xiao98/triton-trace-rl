"""Reference problems and candidate sources shared by the tests.

Kernel *bodies* come in two flavours: real Triton (used when CUDA + Triton are present)
and the NumPy semantics of the fake Triton in `tests/stubs`. Everything around the body
— wrappers, `ModelNew`, and every cheat — is identical in both, so the guards are tested
on the same Python either way.
"""

from conftest import REAL

HEADER = """import torch
import torch.nn as nn
import triton
import triton.language as tl
"""

_REAL_BINARY = """
@triton.jit
def {name}(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x {op} y, mask=mask)
"""

_STUB_BINARY = """
@triton.jit
def {name}(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    out_ptr[...] = x_ptr {op} y_ptr
"""


def binary_kernel(name: str = "add_kernel", op: str = "+") -> str:
    return (_REAL_BINARY if REAL else _STUB_BINARY).format(name=name, op=op)


LAUNCH = """
def launch(x, y, out):
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=1024)
"""

PREAMBLE = HEADER + binary_kernel() + LAUNCH

# ---------------------------------------------------------------- reference problems
REF_ADD = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a + b

def get_inputs():
    return [torch.randn(256, 512), torch.randn(256, 512)]

def get_init_inputs():
    return []
"""

REF_PARAM = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.w = nn.Parameter(torch.randn(rows, cols))

    def forward(self, x):
        return x + self.w

def get_inputs():
    return [torch.randn(128, 256)]

def get_init_inputs():
    return [128, 256]
"""

REF_MATMUL = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.matmul(a, b)

def get_inputs():
    return [torch.randn(64, 64), torch.randn(64, 64)]

def get_init_inputs():
    return []
"""


def model(forward_body: str, extra: str = "") -> str:
    """PREAMBLE + a ModelNew whose forward(self, a, b) has the given body."""
    body = "\n".join("        " + line for line in forward_body.strip().splitlines())
    return f"""{PREAMBLE}{extra}

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
{body}
"""


# ---------------------------------------------------------------- honest candidates
HONEST_ADD = model("""
a = a.contiguous()
b = b.contiguous()
out = torch.empty_like(a)
launch(a, b, out)
return out
""")

HONEST_PARAM = PREAMBLE + """

class ModelNew(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.w = nn.Parameter(torch.randn(rows, cols))

    def forward(self, x):
        out = torch.empty_like(x)
        launch(x.contiguous(), self.w, out)
        return out
"""

# Honest but wrong (subtracts): must score "compiled" (0.2), not 0.0.
WRONG_SUB = HEADER + binary_kernel(op="-") + LAUNCH + """

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        out = torch.empty_like(a)
        launch(a, b, out)
        return out
"""

NAME_ERROR = model("""
out = torch.empty_like(a)
launch(a, b, undefined_name)
return out
""")
