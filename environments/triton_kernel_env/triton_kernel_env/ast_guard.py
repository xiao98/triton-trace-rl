"""Static (AST) anti-reward-hacking guard for Triton-only kernel submissions.

A submission is one Python source file that must define `ModelNew(nn.Module)` whose
`forward` reaches at least one `@triton.jit` kernel launch. Everything PyTorch could
compute on the model's behalf is rejected: only an allow-list of tensor allocation and
shape operations may be referenced. The guard is deliberately strict and purely
syntactic; `sandbox_runner` adds the runtime checks that syntax cannot express
(operator overloads, dead code, in-place input edits, output caching).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

ALLOWED_IMPORTS = {
    "torch",
    "torch.nn",
    "torch.nn.functional",  # importing is tolerated (LLM habit); any *use* is rejected
    "triton",
    "triton.language",
    "math",
    "typing",
    "__future__",
}

# `torch.<name>` references that are allowed: allocation, shape, dtype, bookkeeping.
TORCH_ALLOWED = {
    "Tensor", "Size", "device", "dtype", "no_grad", "inference_mode", "is_tensor",
    "float32", "float", "float64", "double", "int8", "int16", "int32", "int", "int64",
    "long", "uint8", "bool", "contiguous_format",
    "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like",
    "full_like", "tensor", "arange",
    "reshape", "permute", "transpose", "flatten", "squeeze", "unsqueeze", "cat", "stack",
    "broadcast_to", "movedim", "numel",
}  # fmt: skip

# Random initialisers: only inside `__init__`, where no input is in scope (KernelBench
# models build parameters with them and rely on same-seed construction).
TORCH_INIT_ONLY = {"randn", "rand", "randint", "randn_like", "rand_like", "normal"}

# `torch.nn.<name>` usable anywhere.
NN_ALWAYS_ALLOWED = {"Module", "Parameter", "ParameterList", "ParameterDict"}

TRITON_ALLOWED = {"jit", "autotune", "Config", "cdiv", "next_power_of_2", "language"}

# Tensor methods that compute (or leak data to Python / lower precision). Checked on any
# attribute call outside `@triton.jit` bodies that is not rooted at an imported module.
DENIED_METHODS = {
    "matmul", "mm", "bmm", "mv", "dot", "addmm", "addbmm", "baddbmm", "addmv", "einsum",
    "softmax", "log_softmax", "sum", "mean", "prod", "max", "min", "amax", "amin",
    "argmax", "argmin", "cumsum", "cumprod", "logsumexp", "norm", "var", "std", "median",
    "exp", "log", "log2", "log10", "sqrt", "rsqrt", "tanh", "sigmoid", "relu", "abs",
    "pow", "square", "mul", "add", "sub", "div", "neg", "clamp", "clip", "where",
    "masked_fill", "gather", "scatter", "index_select", "sort", "topk", "erf", "sin",
    "cos", "floor", "ceil", "round", "sign", "reciprocal", "maximum", "minimum",
    "mul_", "add_", "sub_", "div_", "clamp_", "exp_", "relu_", "sigmoid_", "tanh_",
    "numpy", "tolist", "item", "half", "bfloat16", "apply_", "map_", "backward",
    "register_forward_hook", "register_forward_pre_hook", "register_buffer_hook",
    "forward",
}  # fmt: skip

DENIED_BUILTINS = {
    "eval", "exec", "compile", "open", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "input", "breakpoint", "memoryview", "exit", "quit",
}  # fmt: skip

# Attribute names that expose the raw Python function behind a JIT kernel.
DENIED_ATTRS = {"fn", "py_fn"}


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    lineno: int = 0

    def __str__(self) -> str:
        return f"[{self.rule}] line {self.lineno}: {self.message}"


@dataclass
class GuardResult:
    violations: list[Violation] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [
                {"rule": v.rule, "message": v.message, "lineno": v.lineno}
                for v in self.violations
            ],
        }


def _is_layer_name(name: str) -> bool:
    return name[:1].isupper()


class _Guard(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.tree = tree
        self.violations: list[Violation] = []
        self.aliases: dict[str, str] = {}  # local name -> dotted import path
        self.local_classes = {
            n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
        }
        self._in_jit = 0
        self._in_init = 0

    # -- helpers ---------------------------------------------------------------
    def flag(self, rule: str, message: str, node: ast.AST) -> None:
        self.violations.append(Violation(rule, message, getattr(node, "lineno", 0)))

    def dotted(self, node: ast.AST) -> str | None:
        """Dotted import path of a Name/Attribute chain rooted at an imported name."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in self.aliases:
            return ".".join([self.aliases[node.id], *reversed(parts)])
        return None

    def is_jit_decorator(self, node: ast.AST) -> bool:
        target = node.func if isinstance(node, ast.Call) else node
        return self.dotted(target) in {"triton.jit", "triton.autotune"}

    def check_path(self, path: str, node: ast.AST) -> None:
        root = path.split(".")[0]
        if root in {"math", "typing", "__future__"}:
            return
        if root == "triton":
            parts = path.split(".")
            if len(parts) >= 2 and parts[1] == "language":
                return
            if len(parts) == 2 and parts[1] in TRITON_ALLOWED:
                return
            self.flag("triton_api_not_allowed", f"`{path}` is not allowed", node)
            return
        if self._in_jit:
            self.flag("torch_inside_kernel", f"`{path}` used inside a @triton.jit body", node)
            return
        if path == "torch.nn.functional" or path.startswith("torch.nn.functional."):
            self.flag("functional_fallback", f"`{path}`: torch.nn.functional is forbidden", node)
            return
        parts = path.split(".")
        if parts[:2] == ["torch", "nn"]:
            if len(parts) == 3 and parts[2] in NN_ALWAYS_ALLOWED:
                return
            if len(parts) == 4 and parts[2] == "init":
                if not self._in_init:
                    self.flag("nn_init_outside_init", f"`{path}` outside __init__", node)
                return
            if len(parts) == 3 and _is_layer_name(parts[2]):
                # Layers are tolerated only as parameter containers built in __init__
                # (KernelBench relies on same-seed construction for weight parity).
                if not self._in_init:
                    self.flag("nn_layer_outside_init", f"`{path}` used outside __init__", node)
                return
            self.flag("torch_op_not_allowed", f"`{path}` is not on the allow-list", node)
            return
        if len(parts) == 2 and parts[1] in TORCH_ALLOWED:
            return
        if len(parts) == 2 and parts[1] in TORCH_INIT_ONLY and self._in_init:
            return
        self.flag("torch_op_not_allowed", f"`{path}` is not on the allow-list", node)

    # -- imports ---------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name not in ALLOWED_IMPORTS:
                self.flag("forbidden_import", f"import {a.name}", node)
                continue
            if a.asname:
                self.aliases[a.asname] = a.name
            else:
                root = a.name.split(".")[0]
                self.aliases[root] = root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level or mod not in ALLOWED_IMPORTS:
            self.flag("forbidden_import", f"from {'.' * node.level}{mod} import ...", node)
            return
        for a in node.names:
            if a.name == "*":
                self.flag("forbidden_import", f"from {mod} import *", node)
                continue
            path = f"{mod}.{a.name}"
            self.aliases[a.asname or a.name] = path
            if path not in ALLOWED_IMPORTS:  # an object, not a module: check it now
                self.check_path(path, node)

    # -- scopes ----------------------------------------------------------------
    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for dec in node.decorator_list:
            self.visit(dec)
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        is_jit = any(self.is_jit_decorator(d) for d in node.decorator_list)
        is_init = node.name == "__init__"
        self._in_jit += is_jit
        self._in_init += is_init
        for stmt in node.body:
            self.visit(stmt)
        self._in_jit -= is_jit
        self._in_init -= is_init

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.keywords:
            self.flag("bad_base_class", "class keywords (metaclass) are forbidden", node)
        for base in node.bases:
            path = self.dotted(base)
            local = isinstance(base, ast.Name) and base.id in self.local_classes
            if path != "torch.nn.Module" and not local:
                self.flag(
                    "bad_base_class",
                    f"class {node.name} may only inherit nn.Module or a local class",
                    node,
                )
        for dec in node.decorator_list:
            self.flag("class_decorator", "class decorators are forbidden", dec)
        for stmt in node.body:
            self.visit(stmt)

    # -- expressions -----------------------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr.endswith("__") and node.attr != "__init__":
            self.flag("dunder_access", f"dunder attribute `{node.attr}`", node)
        if node.attr in DENIED_ATTRS:
            self.flag("raw_kernel_fn", f"attribute `.{node.attr}` is forbidden", node)
        path = self.dotted(node)
        if path is None:
            self.generic_visit(node)
            return
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.flag("monkey_patch", f"assignment to `{path}`", node)
            return
        self.check_path(path, node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in self.aliases:
                self.flag("monkey_patch", f"rebinding imported name `{node.id}`", node)
            return
        if node.id in DENIED_BUILTINS:
            self.flag("forbidden_builtin", f"`{node.id}` is forbidden", node)
        path = self.aliases.get(node.id)
        if path is None:
            return
        if path in ALLOWED_IMPORTS and path not in {"math", "typing"}:
            # A bare module object escaping into a variable defeats path checks.
            self.flag("module_alias_escape", f"module `{node.id}` used as a value", node)
        else:
            self.check_path(path, node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and self.dotted(func) is None
            and not self._in_jit
            and func.attr in DENIED_METHODS
        ):
            self.flag("tensor_compute_method", f"`.{func.attr}()` computes in PyTorch", node)
        if isinstance(func, ast.Call):
            inner = self.dotted(func.func)
            if inner and inner.startswith("torch.nn."):
                self.flag("nn_module_call", f"calling `{inner}(...)` instance", node)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult) and not self._in_jit:
            self.flag("matmul_operator", "`@` operator falls back to torch.matmul", node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.op, ast.MatMult) and not self._in_jit:
            self.flag("matmul_operator", "`@=` operator falls back to torch.matmul", node)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        for name in node.names:
            if name in self.aliases:
                self.flag("monkey_patch", f"global rebinding of `{name}`", node)


def _nn_attrs(guard: _Guard) -> set[str]:
    """Names X with `self.X = nn.<Layer>(...)` (or a container of layers) anywhere."""
    found: set[str] = set()
    for node in ast.walk(guard.tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        builds_layer = any(
            isinstance(c, ast.Call)
            and (p := guard.dotted(c.func)) is not None
            and p.startswith("torch.nn.")
            and p.split(".")[2] not in {"Parameter", "init"}
            for c in ast.walk(node.value)
        )
        if not builds_layer:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for t in targets:
            if (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
            ):
                found.add(t.attr)
    return found


def _check_nn_calls(guard: _Guard) -> None:
    """`self.layer(x)`: invoking a torch.nn layer is a PyTorch fallback."""
    layers = _nn_attrs(guard)
    for node in ast.walk(guard.tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        while isinstance(f, ast.Subscript):  # self.layers[0](x)
            f = f.value
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
            and f.attr in layers
        ):
            guard.flag("nn_module_call", f"`self.{f.attr}(...)` runs a torch.nn layer", node)


def _check_structure(guard: _Guard) -> None:
    """ModelNew.forward must reach a `kernel[grid](...)` launch of a @triton.jit function."""
    tree = guard.tree
    funcs: dict[str, ast.AST] = {}
    jit: set[str] = set()
    methods_of: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
            if any(guard.is_jit_decorator(d) for d in node.decorator_list):
                jit.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{node.name}.{item.name}"
                    funcs[key] = item
                    methods_of.setdefault(node.name, []).append(key)

    if "ModelNew" not in methods_of and not any(
        isinstance(n, ast.ClassDef) and n.name == "ModelNew" for n in tree.body
    ):
        guard.flag("missing_model_new", "no top-level `class ModelNew`", tree)
        return
    if "ModelNew.forward" not in funcs:
        guard.flag("missing_forward", "`ModelNew` must define `forward`", tree)
        return
    if not jit:
        guard.flag("no_triton_kernel", "no top-level @triton.jit function", tree)
        return

    def edges(key: str) -> set[str]:
        out: set[str] = set()
        cls = key.split(".")[0] if "." in key else None
        for n in ast.walk(funcs[key]):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                if n.id in funcs:
                    out.add(n.id)
                out.update(methods_of.get(n.id, []))
            elif (
                isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
                and cls
            ):
                out.update(k for k in funcs if k.endswith(f".{n.attr}") and "." in k)
        return out

    def launches(key: str) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Subscript)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id in jit
            for n in ast.walk(funcs[key])
        )

    seen, stack = set(), ["ModelNew.forward"]
    while stack:
        key = stack.pop()
        if key in seen or key in jit:
            continue
        seen.add(key)
        if launches(key):
            return
        stack.extend(edges(key))
    guard.flag(
        "zombie_kernel",
        "no @triton.jit kernel launch (`kernel[grid](...)`) is reachable from ModelNew.forward",
        tree,
    )


def check_source(source: str) -> GuardResult:
    """Statically validate a submission. `result.ok` is False on any violation."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as e:
        return GuardResult([Violation("syntax_error", str(e), getattr(e, "lineno", 0) or 0)])
    guard = _Guard(tree)
    # Imports first so aliases are known regardless of statement order.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            guard.visit_Import(node)
        elif isinstance(node, ast.ImportFrom):
            guard.visit_ImportFrom(node)
    import_violations = guard.violations
    guard.violations = []
    for stmt in tree.body:
        if not isinstance(stmt, (ast.Import, ast.ImportFrom)):
            guard.visit(stmt)
    # Nested imports were already validated above; drop duplicate reports.
    seen = {(v.rule, v.lineno, v.message) for v in import_violations}
    guard.violations = import_violations + [
        v for v in guard.violations if (v.rule, v.lineno, v.message) not in seen
    ]
    _check_nn_calls(guard)
    _check_structure(guard)
    return GuardResult(guard.violations)
