# Gate 1 审查报告

代码提交：`09df728`（本报告在其后单独提交）。日期：2026-11-20。

## 结论

Gate 1 的代码、测试、打包、notebook 全部完成，**本地测试 100% 通过**。但有一条必须先说清：
**本机没有 GPU 也装不了 Triton，所以"真 Triton + 真 GPU"这条路径一次都没跑过**。本地通过的是
"守卫与沙箱的控制流"，用的是 `tests/stubs/` 里的假 Triton。真实验证要靠你在 Colab 跑
`notebooks/01_colab_smoke_and_gate1.ipynb`。建议看完 notebook 结果再回复"通过"。

## 测试结果（均为实际运行输出）

| 环境 | 结果 |
| --- | --- |
| Windows 原生，Python 3.11，torch 2.15.0+cpu | 96 passed, 1 skipped |
| WSL Ubuntu 24.04，Python 3.11，torch 2.15.0+cpu，verifiers 0.4.1 | 102 passed, 1 skipped |
| `vf-validate triton-kernel-env --taskset.source smoke`（WSL，subprocess runtime） | setup 2/2 valid，noop 2/2 valid |

- Windows 跳过的是整个 `test_verifiers_env.py`（见下方"调研发现 2"）；WSL 跳过的 1 个是仅 GPU 可跑的金标准解测试。
- 红队文件 `tests/test_anti_cheat_redteam.py` 共 57 个用例，全部打出 0.0 且父进程存活：

| 类别 | 用例数 | 拦截层 |
| --- | --- | --- |
| 1. PyTorch 回退（`F.*`、`torch.matmul`、`@`、张量方法、裸运算符 `a + b`、调用 `nn` 层） | 12 | 静态为主；`a + b` 和容器里的 `nn` 层靠运行时 |
| 2. 僵尸 `@triton.jit`（从不调用、只在死代码里调用、取 `.fn` 原函数） | 4 | 静态 + 运行时启动计数 |
| 3. 原地修改输入 | 2 | 运行时（克隆后比对） |
| 4. 硬编码 / 缓存回放（缓存首次答案、常量输出、重设种子、校验后才作弊、按值缓存） | 6 | 运行时 |
| 5. 禁用 import / monkey-patch（`os`、`subprocess`、`getattr` 拼字符串、改 `torch.allclose`、改计时函数等） | 19 | 静态 |
| 6. 死循环 / 崩溃（import 时死循环、forward 死循环、仅计时阶段死循环、段错误、`os._exit`、`abort`、无限递归） | 10 | 硬超时 + 进程隔离 |
| 正向对照（诚实内核≥0.5、参数权重对齐、算错得 0.2 而非 0） | 3 | — |
| 连续恶意提交后父进程仍可正常评测 | 1 | — |

## 调研发现（影响需求原文，请确认）

1. **`verifiers` 接口已整体换代。** PyPI 最新版 0.4.1 与 `main` 只有 `verifiers.v1`，
   `vf.MultiTurnEnv`、`vf.Rubric`、`load_environment()`、`vf-eval` 都不存在了。新接口是
   `Taskset / Task / TaskData` + `@vf.reward / @vf.metric` + 自定义 `vf.Env.run()`（用
   `agents.solver.interaction(task)` 做多轮），CLI 是 `vf-validate / vf-run`。我按 v1 实现，
   并把这条写进了 `CLAUDE.md`。`prime env push` 本地未验证（需要 Prime 账号）。
2. **verifiers v1 只支持 POSIX**（import `fcntl`），Windows 原生跑不了，所以 verifiers 相关测试在 WSL 里跑。
3. **KernelBench**：GitHub `main` 为 `423217d`（2026-03-05），已原生支持 Triton 后端并自带静态检查器；
   HF 数据集 `ScalingIntelligence/KernelBench` 修订版 `ca1464e`（2025-07-21），Level 1/2 各 100 题，
   字段 `code / level / name / problem_id`。我固定了 HF 修订版。**两处题目内容是否一致没有比对**。
4. KernelBench 当前 fp32 容差是 `1e-4`（早期论文口径是 `1e-2`）。我用了 `1e-4`。

## 我做的设计决定（与需求字面有出入之处）

1. **`nn.Linear / Conv*` 允许在 `__init__` 里构造、禁止调用。** KernelBench 靠"同种子构造"让参考模型与
   候选模型权重一致；完全禁止构造的话 Level 2 几乎无法做对。调用（`self.fc(x)`）静态和运行时都会拦。
2. **超时从"可信阶段结束"起算。** import torch/triton 和运行参考模型不计入 25 s，否则 Colab 冷启动就会误判超时。
3. **运行时守卫是主防线。** 静态 AST 无法区分 `a + b` 是整数还是张量，所以在子进程里挂了 PyTorch
   dispatcher 钩子：候选 forward 期间只允许"分配 / 形状 / 拷贝"类算子，其余一律判作弊。
4. **计时用候选从未见过的随机输入池，且每次输出都核对。** 这是自查时发现并修掉的漏洞：原先计时
   反复用同一份输入，"前 3 次老实算、之后回放缓存"可以刷出任意加速比。现在 warmup 次数随机（5–20 次）
   且全程开严格守卫，计时段之后再做一次严格调用。
5. **以下两种情况按作弊判 0.0（偏保守）**：三个种子输出完全相同而参考输出不同；校验通过后在计时阶段输出变错。
   代价是"内核没写输出、返回全零"这类老实 bug 也得 0.0 而不是 0.2。
6. 评测在**评测机本机**的子进程里跑（需要 GPU），不在 verifiers 的 runtime 容器里。

## 需要你裁决的一个问题：奖励在 speedup = 1 处不连续

按需求字面实现：正确 = 0.5，正确且加速 = `0.5 + min(1.5, speedup)`。于是 speedup 从 0.99 到 1.01，
奖励从 0.5 跳到 1.51。大量 Level 1 题的真实加速比就在 1.0 附近，测速噪声会让奖励在 0.5 和 1.5 之间随机跳，
GRPO 的优势估计会被这个噪声主导。建议改成连续形式，例如 `0.5 + min(1.5, max(0, speedup - 1))`
（上限仍是 2.0）。**我没有擅自改**，等你定。

## 未验证项与残余风险

- **真 Triton / 真 GPU 全部未验证**，具体三处最可能出问题：
  ① Triton 内核启动时内部若触发了白名单外的 torch 算子，诚实内核会被误判作弊；
  ② 启动计数靠包装 `triton.runtime.jit.JITFunction.run`（按 Triton `main` 源码确认过签名，未实跑）；
  ③ Triton 内核在 `torch.profiler` 里显示成什么名字未知，瓶颈提示的关键词可能需要调整。
  notebook 第 4 格就是针对这三点的正向对照。
- **按值缓存作弊仍有约 1/16 的碰运气成功率**（需要猜中随机的 warmup 次数），期望收益低于老实做；未做到零。
- 计时段（20 次）不开严格守卫（开了会拖慢候选、测速失真），此时候选可用裸运算符，但最多与参考持平。
- **一次未复现的偶发问题**：某次 Windows 全量测试里 1 个用例返回了 `infra_error`，单独重复 8 次、
  全量再跑 2 次均未复现，原因未查明。已加"基础设施错误自动重试一次并保留错误详情"。
- 数据划分保护（红线 4）按计划在 Gate 2 冻结 `manifest.json` 时实现，本阶段没有任何代码读取题目划分。

## 交付物

```
CLAUDE.md                                         项目规则（目标 / 硬件 / 6 红线 / 5 Gate）
environments/triton_kernel_env/triton_kernel_env/
  ast_guard.py  sandbox_runner.py  _sandbox_child.py
  profiler_feedback.py  reward.py  prompts.py  taskset.py
tests/test_anti_cheat_redteam.py  test_core_units.py  test_verifiers_env.py
notebooks/01_colab_smoke_and_gate1.ipynb          由 scripts/build_notebook_01.py 生成
```

## 请你做的事

1. 把仓库放到 Colab 能拿到的地方：推到 GitHub 后改 notebook 第 2 格的 `REPO_URL`，或把整个目录
   拷到 Drive 的 `MyDrive/triton-trace-rl/repo`。（我没有替你推送到任何远端。）
2. 运行 notebook 01，把 `MyDrive/triton-trace-rl/gate1/` 下的日志发我，或直接告诉我第 4、5 格的结果。
3. 对"奖励不连续"和上面 6 条设计决定给出意见。
4. 回复"通过"后我再开始 Gate 2。
