"""Generate notebooks/01_colab_smoke_and_gate1.ipynb (kept as a script so the notebook
is reviewable as plain text and regenerated, never hand-edited)."""

from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/<your-user>/triton-trace-rl.git"

cells: list[tuple[str, str]] = [
    ("md", """# 01 — Colab smoke test and Gate 1 (real GPU + real Triton)

Runs the whole Gate 1 test suite against the **real** stack on the Colab A100. Locally the
suite runs on CPU with a fake Triton, so this notebook is what actually verifies:

1. honest Triton kernels pass the runtime guard (no false positives from Triton internals),
2. every red-team cheat / crash scores `0.0` with real kernels and the parent survives,
3. CUDA-event timing and `torch.profiler` output look sane,
4. the package loads through `verifiers` (`vf-validate`), gold solutions included.

Run all cells top to bottom. Raw outputs are written to Google Drive."""),
    ("code", """# 1. Mount Drive (all logs go here; safe to re-run after a disconnect)
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/triton-trace-rl'
OUT = f'{DRIVE}/gate1'
os.makedirs(OUT, exist_ok=True)
print('logs ->', OUT)"""),
    ("code", f"""# 2. Get the code. Either set REPO_URL to your pushed repo, or upload the repo folder to
#    Drive at MyDrive/triton-trace-rl/repo (used when it exists).
REPO_URL = '{REPO_URL}'
import os, subprocess
if os.path.isdir(f'{{DRIVE}}/repo/tests'):
    subprocess.run(['rm', '-rf', '/content/triton-trace-rl'])
    subprocess.run(['cp', '-r', f'{{DRIVE}}/repo', '/content/triton-trace-rl'], check=True)
elif not os.path.isdir('/content/triton-trace-rl'):
    subprocess.run(['git', 'clone', REPO_URL, '/content/triton-trace-rl'], check=True)
%cd /content/triton-trace-rl
!git rev-parse HEAD 2>/dev/null | tee {{OUT}}/commit.txt || echo 'no git metadata (Drive copy)'"""),
    ("code", """# 3. Install (no Docker; Colab already ships torch + CUDA; triton is kept if present).
!pip -q install triton pytest verifiers datasets
!pip -q install --no-deps -e environments/triton_kernel_env
import json, sys, torch, triton
env = {'python': sys.version.split()[0], 'torch': torch.__version__,
       'triton': triton.__version__, 'cuda': torch.cuda.is_available(),
       'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(env)
assert env['cuda'], 'Switch the runtime to GPU (A100).'
json.dump(env, open(f'{OUT}/env.json', 'w'), indent=2)"""),
    ("code", """# 4. Positive control first: the built-in gold solutions on the real GPU.
#    Prints the raw result (status, speedup, timing, profiler rows) and saves it.
import json, sys
sys.path.insert(0, 'environments/triton_kernel_env')
from triton_kernel_env.prompts import SMOKE_TASKS
from triton_kernel_env.sandbox_runner import evaluate_kernel
from triton_kernel_env.reward import shaped_reward
from triton_kernel_env.profiler_feedback import build_feedback

records = []
for name, ref, gold in SMOKE_TASKS:
    res = evaluate_kernel(ref, gold, profile=True)
    rec = dict(res.to_dict(), task=name, reward=shaped_reward(res))
    records.append(rec)
    print(f"== {name}: status={res.status} reward={rec['reward']:.3f} speedup={res.speedup} "
          f"cheats={res.cheats}")
    print(res.timing)
    print(build_feedback(res, 'profile'))
    print()
json.dump(records, open(f'{OUT}/smoke_gold_results.json', 'w'), indent=2)
assert all(r['status'] == 'correct' for r in records), 'gold solution failed: see output above'"""),
    ("code", """# 5. Full Gate 1 suite on the real stack (red team + units + verifiers + gpu-marked tests).
!python -m pytest -q -rs --junitxml={OUT}/pytest_gate1.xml 2>&1 | tee {OUT}/pytest_gate1.log"""),
    ("code", """# 6. Package check through the verifiers CLI: setup, gold and noop on the smoke tasks.
!vf-validate triton-kernel-env --taskset.source smoke -n 2 --runtime.type subprocess 2>&1 | tail -25 | tee {OUT}/vf_validate.log"""),
    ("code", """# 7. The real benchmark loads (no model calls, no GPU work): Level 1 + 2 = 200 tasks.
from verifiers.v1.utils import loaders
from triton_kernel_env.taskset import TritonKernelConfig
tasks = list(loaders.load_taskset(TritonKernelConfig(id='triton-kernel-env')).load())
from collections import Counter
counts = dict(Counter(t.data.level for t in tasks))
print(counts, tasks[0].data.name)
json.dump({'levels': counts, 'n': len(tasks)}, open(f'{OUT}/kernelbench_load.json', 'w'))
assert len(tasks) == 200"""),
    ("md", """## What to send back for the Gate 1 review

`MyDrive/triton-trace-rl/gate1/` now holds `env.json`, `smoke_gold_results.json`,
`pytest_gate1.log`, `vf_validate.log`, `kernelbench_load.json`. The pass criteria:

- cell 4: both gold kernels `status=correct`, `cheats=[]`, a plausible speedup;
- cell 5: `0 failed`, and nothing skipped for lack of a GPU;
- cell 6: gold and noop checks valid;
- cell 7: 200 tasks."""),
]

nb = nbf.v4.new_notebook()
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
}
for kind, src in cells:
    nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
out = Path(__file__).resolve().parent.parent / "notebooks" / "01_colab_smoke_and_gate1.ipynb"
nbf.write(nb, out)
print("wrote", out)
