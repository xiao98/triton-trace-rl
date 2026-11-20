"""Generate notebooks/02_colab_phase1_eval.ipynb (regenerated from here, never hand-edited)."""

from pathlib import Path

import nbformat as nbf

REPO_URL = "https://github.com/xiao98/triton-trace-rl.git"

cells: list[tuple[str, str]] = [
    ("md", """# 02 — Phase 1: fair-budget multi-turn evaluation (Colab GPU + Claude API)

Runs `scripts/phase1_eval.py` in the background and writes one raw JSON record per
(arm, problem) to Google Drive. **Safe to re-run after a disconnect**: finished records are
skipped, so run every cell again from the top and it resumes.

Default configuration is the **Gate 2 pilot**: 5 `train_source` problems, 3 arms, 4
submissions each (60 model calls). Do not change `SPLIT` to `heldout_test` before Gate 3
is approved.

Needs a Colab secret named `ANTHROPIC_API_KEY` (key icon in the left sidebar, notebook
access enabled)."""),
    ("code", """# 1. Mount Drive: every record and log lives here.
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE = '/content/drive/MyDrive/triton-trace-rl'
os.makedirs(DRIVE, exist_ok=True)"""),
    ("code", f"""# 2. Code + dependencies (no Docker; Colab ships torch + CUDA + triton).
import subprocess
if os.path.isdir('/content/triton-trace-rl/.git'):
    subprocess.run(['git', '-C', '/content/triton-trace-rl', 'pull', '-q'], check=True)
else:
    subprocess.run(['git', 'clone', '-q', '{REPO_URL}', '/content/triton-trace-rl'], check=True)
os.chdir('/content/triton-trace-rl')
subprocess.run('pip -q install triton anthropic datasets pytest verifiers && '
               'pip -q install --no-deps -e environments/triton_kernel_env', shell=True, check=True)
import torch
print(subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
      torch.cuda.get_device_name(0))"""),
    ("code", """# 3. API key from Colab secrets -> environment of the background run (never printed).
from google.colab import userdata
os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
print('key loaded')"""),
    ("code", """# 4. Launch (or resume) the run in the background.
RUN = 'pilot_train5'          # one Drive folder per run
SPLIT = 'train_source'        # heldout_test only for the final evaluation, with PURPOSE = 'heldout_eval'
PURPOSE = 'pilot'
N_PROBLEMS = 5                # None = the whole split
BUDGET = 4
MODEL, EFFORT, WORKERS = 'claude-opus-5-5', 'medium', 4

OUT = f'{DRIVE}/phase1/{RUN}'
os.makedirs(OUT, exist_ok=True)
cmd = (f'python scripts/phase1_eval.py --out {OUT} --split {SPLIT} --purpose {PURPOSE} '
       f'--budget {BUDGET} --model {MODEL} --effort {EFFORT} --workers {WORKERS}'
       + (f' --n-problems {N_PROBLEMS}' if N_PROBLEMS else ''))
subprocess.Popen(f'nohup {cmd} >> {OUT}/run.log 2>&1 &', shell=True)
print(cmd)"""),
    ("code", """# 5. Progress (re-run this cell; the run is over when PHASE1_EVAL_FINISHED appears).
import glob
print(len(glob.glob(f'{OUT}/*/*.json')), 'episodes recorded')
print(open(f'{OUT}/run.log').read()[-2500:])"""),
    ("code", """# 6. Aggregate raw records -> metrics / cost / wall-time tables (also saved to Drive).
!python scripts/phase1_report.py --runs {OUT} --out {OUT}/summary --project 120 80"""),
]

nb = nbf.v4.new_notebook()
nb.metadata = {
    "accelerator": "GPU",
    "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
}
for kind, src in cells:
    nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
out = Path(__file__).resolve().parent.parent / "notebooks" / "02_colab_phase1_eval.ipynb"
nbf.write(nb, out)
print("wrote", out)
