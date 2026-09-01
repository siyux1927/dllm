"""生成 notebooks/colab_p1_p2.ipynb。

手写 ipynb 的 JSON 容易出错，用脚本生成能保证格式合法、也便于以后改。
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md(
    """
# dLLM 上的 diffu-GRPO：P1 耗时拆解 + P2 log-prob 误差量化

本 notebook 对应 `docs/plan-diffu-grpo.md` 的 P1 与 P2 两个阶段，产出故事线前两幕的数据。

**P1 要回答**：单步训练的时间花在哪儿？采样到底占多少？
**P2 要回答**：把 GRPO 搬到扩散模型上，最大的障碍是 log-prob 没有链式法则可用。
单步近似到底差多少？为什么 clip 范围 ε 必须放宽到 0.5？

运行前请确认：**运行时类型 → 硬件加速器 → A100 GPU**。

> 全程约 40 分钟，其中依赖安装与权重下载约 25 分钟（首次；之后走 Drive 缓存）。
"""
)

md("## 0. 环境确认\n\n先确认拿到的是 A100。T4 显存 16GB 装不下 8B 的 bf16 权重。")

code(
    """
!nvidia-smi

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}  显存 {props.total_memory/1e9:.1f} GB")
    assert props.total_memory > 30e9, "显存不足 30GB，8B 模型的 bf16 权重加激活放不下，请换 A100"
"""
)

md(
    """
## 1. 挂载 Drive

`/content` 下的一切随会话消失，所以三类东西都要落到 Drive：

- **HF 缓存**：LLaDA-8B 权重约 16GB，重连后不必重下。
- **结果 JSON 与逐步指标 CSV**：只有几十 KB，直接写 Drive。
  脚本每完成一个阶段就落一次盘，不是等全部跑完——那样在第二档断掉会把第一档一起赔进去。
- **checkpoint**：约 4GB，情况复杂，见第 8 节。
"""
)

code(
    """
import os
from pathlib import Path

from google.colab import drive

drive.mount('/content/drive')

DRIVE_ROOT = Path('/content/drive/MyDrive/dllm-grpo')
RESULTS = DRIVE_ROOT / 'results'
for sub in ('hf-cache', 'results', 'checkpoints'):
    (DRIVE_ROOT / sub).mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(DRIVE_ROOT / 'hf-cache')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
# tokenizers 的并行分词在 fork 后会告警刷屏，且本项目分词不是瓶颈
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print(f"HF_HOME = {os.environ['HF_HOME']}")
print(f"结果落点 = {RESULTS}")
"""
)

md(
    """
## 2. 取代码

两种方式二选一：

- 跑远端的已推送版本 → 直接跑本单元，`GIT_URL` 已填好
- 跑本地未推送的改动 → 把项目文件夹上传到 Drive 的 `MyDrive/dllm-grpo/diff`，并把 `GIT_URL` 清空
"""
)

code(
    """
import shutil
import subprocess

GIT_URL = 'https://github.com/siyux1927/dllm.git'  # 留空则改从 Drive 复制
PROJECT_DIR = Path('/content/diff')

if PROJECT_DIR.exists():
    shutil.rmtree(PROJECT_DIR)

if GIT_URL:
    subprocess.run(['git', 'clone', '--depth', '1', GIT_URL, str(PROJECT_DIR)], check=True)
else:
    source = DRIVE_ROOT / 'diff'
    assert source.exists(), (
        f"{source} 不存在。请把本地项目文件夹上传到 Drive 的 MyDrive/dllm-grpo/diff，"
        "或在上面填入 GIT_URL。"
    )
    # 复制到本地磁盘再跑：Drive 是网络挂载，随机读写慢，且 import 期间断连会直接报错
    shutil.copytree(source, PROJECT_DIR, ignore=shutil.ignore_patterns(
        '__pycache__', '.git', '.pytest_cache', '*.pyc', 'checkpoints', 'results'))

os.chdir(PROJECT_DIR)
print(f"工作目录 {Path.cwd()}")
print(sorted(p.name for p in PROJECT_DIR.iterdir()))
"""
)

md(
    """
## 3. 装依赖

约 5-10 分钟，主要花在 torch 上。

版本是钉死的，有两处容易踩：

- **`transformers` 必须是 4.x**。LLaDA 的 `trust_remote_code` 建模代码写于 4.49 时代，
  5.x 改掉了它依赖的若干内部接口。这类不兼容往往不是干净的报错，
  而是加载到一半出些莫名其妙的属性错误。
- **换 torch 就必须一起换 torchvision**。二者的 C++ 扩展是配对编译的。
  只把 torch 钉到 2.6.0、留着 Colab 预装的 torchvision，`torchvision::nms` 就注册不上；
  而 `transformers.image_utils` 会无条件 import torchvision，
  于是连 `AutoModel` 都加载不了，报错却指向 transformers 内部，完全看不出病因。

下面**不截断 pip 的输出**。装依赖失败如果被 `| tail` 吞掉，
你会带着一个半好不坏的环境往下跑，到加载模型时才炸。
"""
)

code(
    """
import subprocess
import sys

result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-colab.txt'],
    capture_output=True, text=True)

tail = result.stdout.strip().split('\\n')[-15:]
print('\\n'.join(tail))
if result.returncode != 0:
    print('\\n--- pip 报错 ---')
    print(result.stderr[-4000:])
    raise SystemExit('依赖安装失败，先修好再往下走')
print('\\n依赖安装完成')
"""
)

code(
    """
# 装完必须重启运行时：torch 被换了版本，已经 import 的旧版还留在内存里
import IPython
IPython.Application.instance().kernel.do_shutdown(True)
"""
)

md(
    """
> **重启后从这里继续**，不用重跑上面的单元格（Drive 挂载和文件都还在）。
"""
)

code(
    """
import os
from pathlib import Path

DRIVE_ROOT = Path('/content/drive/MyDrive/dllm-grpo')
RESULTS = DRIVE_ROOT / 'results'
PROJECT_DIR = Path('/content/diff')
os.environ['HF_HOME'] = str(DRIVE_ROOT / 'hf-cache')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.chdir(PROJECT_DIR)

# check_env 会先验 torchvision 的算子再去 import transformers / trl。
# 顺序是有意的：后者的导入链会踩到 torchvision，一旦踩爆，
# 报出来的是一条指向 transformers 内部的六十行 traceback，看不出真正的病因。
!python -m scripts.check_env
"""
)

md(
    """
## 4. 先跑单测

在碰 GPU 之前先跑一遍 CPU 单测。这一步花 30 秒，能挡掉「代码传上来时缺了文件」
或「装依赖时版本冲突改了行为」这类问题——这些如果留到加载完 8B 模型之后才发现，
浪费的是十几分钟的下载和加载。

单测里的 LoRA `target_modules` 和真实配置用的是同一份，所以「适配器挂不上」在这里就会暴露。
"""
)

code(
    """
!python -m pytest -q 2>&1 | tail -n 15
"""
)

md(
    """
## 5. P1：加载模型 + 耗时拆解

脚本依次做四件事：

1. **装配自检**。确认 LoRA 目标模块真的匹配上了，且改动适配器权重确实改变输出。
   LLaDA 派生自 OLMo，用的是 `attn_out` / `ff_proj` / `ff_out` 而非 Llama 的
   `o_proj` / `gate_proj` / `down_proj`；如果这里报错，照它打印出来的真实模块名改配置即可。
2. **padding 不变性自检**。LLaDA 官方的 generate 是按单条 prompt 写的，从没验证过
   带 padding 的批处理。如果 padding 会泄漏进注意力，同一条 prompt 会因为批内其他
   样本的长度不同而得到不同结果——这种错误不报错，只是让结果不可复现。
3. **生成抽查**。看一眼模型到底输出了什么。
4. **分阶段计时**，并把实测占比和理论前向次数预算对照。

这里在 `diffusion_steps` 的两个取值上各测一遍，因为它有个待定的取舍：

- **64**（当前配置）：completion 长 128，等于每步解 2 个 token。单步便宜，
  但采样占比降到约 52%，P4 的端到端上限只剩 2.1x——而且基线本身已经在并行解码了。
- **128**（对齐 d1 官方）：每步解 1 个 token，是干净的基线。采样占比约 68%，
  P4 上限 3.1x，代价是单步慢约 1.5 倍。

脚本会把两者的单步耗时、采样占比、P4 上限和「跑满 200 步要多久」列成一张表。
拿着这张表再定，比凭直觉定可靠。

首次运行要下载约 16GB 权重（15-20 分钟），之后走 Drive 缓存。两个设定合计约 15 分钟。
"""
)

code(
    """
!python -m scripts.run_p1_profile \\
    --config configs/countdown_base.yaml \\
    --warmup 1 --steps 2 \\
    --sweep-diffusion-steps 64 128 \\
    --out {RESULTS}/p1_profile.json \\
    --metrics-csv {RESULTS}/p1_metrics.csv
"""
)

md("### P1 结果可视化")

code(
    """
import json

import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False

report = json.loads((RESULTS / 'p1_profile.json').read_text(encoding='utf-8'))
timing = report['timing']
budget = report['budget']

phases = ['generation', 'reward', 'logprob', 'forward_backward']
labels = ['sampling', 'reward', 'logprob', 'fwd+bwd']
measured = [timing[f'{p}_frac'] for p in phases]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].bar(labels, [timing[f'{p}_s'] for p in phases], color='#4C72B0')
axes[0].set_ylabel('seconds per step')
axes[0].set_title(f"Phase breakdown (total {timing['total_s']:.1f}s/step)")
for i, p in enumerate(phases):
    axes[0].text(i, timing[f'{p}_s'], f"{timing[f'{p}_frac']:.0%}",
                 ha='center', va='bottom')

ceiling = report['amdahl']['ceiling']
speedups = sorted(float(k[:-1]) for k in report['amdahl']['projections'])
end_to_end = [report['amdahl']['projections'][f'{s:g}x'] for s in speedups]
axes[1].plot(speedups, end_to_end, 'o-', color='#C44E52')
axes[1].axhline(ceiling, ls='--', c='gray')
axes[1].text(speedups[0], ceiling, f' Amdahl ceiling {ceiling:.2f}x', va='bottom')
axes[1].set_xlabel('sampling speedup (P4 target)')
axes[1].set_ylabel('end-to-end speedup')
axes[1].set_title(f"Payoff of accelerating sampling ({measured[0]:.0%} of step)")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'p1_breakdown.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"实测采样占比 {timing['generation_frac']:.1%}   "
      f"理论预算 {budget['budget/generation_share']:.1%}")
print(f"P4 端到端收益上限 {ceiling:.2f}x")
"""
)

md(
    """
### 读这张图

左图是第 1 幕的事实：单步时间的分布。右图是从它推出的结论——**P4 到底值不值得做**。

Amdahl 上限是 `1 / (1 - 采样占比)`。如果采样只占一半，那么把采样加速到无穷倍，
端到端也只有 2 倍。这个上限应该在花 A100 之前就算出来，而不是做完 P4 才发现。
"""
)

md("### 决策：diffusion_steps 取 64 还是 128")

code(
    """
sweep = report['sweep']
completion = report['config']['sampling']['max_completion_length']
keys = sorted(sweep, key=int)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x = range(len(keys))
labels = [f"{k} steps\\n({completion/int(k):.0f} tok/step)" for k in keys]

axes[0].bar(labels, [sweep[k]['timing']['total_s'] for k in keys], color='#4C72B0')
axes[0].set_ylabel('seconds per step')
axes[0].set_title('Cost: wall-clock per training step')

axes[1].bar(labels, [sweep[k]['timing']['generation_frac'] for k in keys], color='#DD8452')
axes[1].set_ylabel('sampling share of step time')
axes[1].set_title('Where the time goes')

axes[2].bar(labels, [sweep[k]['amdahl']['ceiling'] for k in keys], color='#55A868')
axes[2].set_ylabel('max end-to-end speedup')
axes[2].set_title('Ceiling on what P4 can deliver')

for ax in axes:
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width() / 2, patch.get_height(),
                f'{patch.get_height():.2f}', ha='center', va='bottom')
    ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS / 'p1_diffusion_steps.png', dpi=140, bbox_inches='tight')
plt.show()

print(f"{'steps':>7}{'tok/step':>10}{'sec/step':>10}{'sampling':>10}"
      f"{'P4 ceiling':>12}{'200 steps':>12}")
for k in keys:
    t = sweep[k]['timing']['total_s']
    print(f"{k:>7}{completion/int(k):>10.1f}{t:>10.1f}"
          f"{sweep[k]['timing']['generation_frac']:>9.1%}"
          f"{sweep[k]['amdahl']['ceiling']:>11.2f}x{t*200/3600:>11.1f}h")
"""
)

md(
    """
### 怎么读这张表

这是个真实的取舍，不是纯优化：

- **每步解码 token 数**大于 1，说明基线本身已经在并行解码了。
  Fast-dLLM 的并行解码收益正是「每步多解几个 token」，
  所以 P4 的加速会有一部分只是把这份预支的收益兑现一次，而不是新增的。
- **P4 上限**是 Amdahl 算出来的天花板。采样占比越低，第 3 幕能讲的东西越少。
- **200 步耗时**决定 P3 能不能跑完。如果这一列超过 5 小时，
  该调的是 `max_steps` 或 `num_prompts_per_step`，而不是硬扛。

三列凑齐了才好定。128 步让第 1、3 幕都更有分量，但如果它把 P3 顶到 8 小时以上，
那就是拿第 5 幕（同等墙钟时间下的 reward 曲线对比）去换第 3 幕，不划算。
"""
)

md(
    """
## 6. P2：log-prob 近似的误差

自回归模型靠链式法则一次前向拿到序列 log-prob，扩散模型没有这个分解。
这是把 GRPO 搬到 dLLM 上最核心的障碍，d1 的做法是单步近似：
把补全全部置为掩码、做一次前向，用单步去噪的输出当作 log π。

三个实验：

- **A** 单步近似 vs 蒙特卡洛真值（128 样本，LLaDA 官方口径）。
- **B** 同一份权重、两个不同的 prompt 掩码模式，算出的 ratio 本该恒等于 1。
  实测的离散程度就是估计器的噪声底噪。
- **C** 噪声随 `p_mask_prompt` 的变化。

B 是关键：它给 ε=0.5 提供实证依据。clip 要挡的是策略跑偏，
如果 ε 收到常见的 0.2，挡掉的绝大部分其实是估计噪声。
"""
)

code(
    """
!python -m scripts.run_p2_logprob \\
    --config configs/countdown_base.yaml \\
    --num-prompts 2 --mc-samples 128 \\
    --out {RESULTS}/p2_logprob.json \\
    --pairs-csv {RESULTS}/p2_pairs.csv
"""
)

md("### P2 结果可视化")

code(
    """
import csv

import numpy as np

p2 = json.loads((RESULTS / 'p2_logprob.json').read_text(encoding='utf-8'))

with open(RESULTS / 'p2_pairs.csv', encoding='utf-8') as f:
    rows = list(csv.reader(f))[1:]
pairs = np.array([[float(a), float(b)] for a, b in rows])

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

# A: 单步 vs 蒙特卡洛
axes[0].scatter(pairs[:, 1], pairs[:, 0], s=4, alpha=0.25, color='#4C72B0')
lims = [min(pairs.min(), -0.5), pairs.max()]
axes[0].plot(lims, lims, 'k--', lw=1, label='y = x')
axes[0].set_xlabel('Monte Carlo logprob (128 samples)')
axes[0].set_ylabel('one-step approximation')
axes[0].set_title(
    f"A. one-step vs MC\\n"
    f"Pearson {p2['experiment_a']['pearson']:.3f}, "
    f"bias {p2['experiment_a']['bias_one_step_minus_mc']:+.2f}")
axes[0].legend()
axes[0].grid(alpha=0.3)

# B: 噪声底噪 vs clip 区间
b = p2['experiment_b']
eps_keys = [k for k in b if k.startswith('out_of_range_frac_eps_')]
eps_values = sorted((float(k.rsplit('_', 1)[1]), b[k]) for k in eps_keys)
axes[1].bar([f"eps={e:g}" for e, _ in eps_values],
            [v for _, v in eps_values], color=['#C44E52', '#55A868'])
for i, (_, v) in enumerate(eps_values):
    axes[1].text(i, v, f"{v:.1%}", ha='center', va='bottom')
axes[1].set_ylabel('fraction of tokens outside clip range')
axes[1].set_title("B. Noise floor at unchanged policy\\n(ratio should be exactly 1)")
axes[1].grid(alpha=0.3, axis='y')

# C: 噪声随 p_mask 变化
c = p2['experiment_c']
p_masks = sorted(float(k) for k in c)
axes[2].plot(p_masks, [c[f'{p:g}']['log_ratio_std'] for p in p_masks],
             'o-', color='#8172B2')
axes[2].axvline(0.15, ls='--', c='gray')
axes[2].text(0.15, axes[2].get_ylim()[1] * 0.9, ' p_mask=0.15 (used)', va='top')
axes[2].set_xlabel('p_mask_prompt')
axes[2].set_ylabel('std of log-ratio')
axes[2].set_title('C. More masking = more regularization,\\nbut also more ratio noise')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS / 'p2_logprob.png', dpi=140, bbox_inches='tight')
plt.show()
"""
)

md(
    """
### 读这三张图

**A** 里单步估计通常系统性偏低——全掩码是信息最少的上下文，模型给不出高置信度。
但 GRPO 只用 ratio 的相对关系，绝对值偏低不致命，**相关性**才是关键。

**B** 检验的是 ε=0.5 这个乍看很奇怪的超参有没有实证依据。策略一步都没更新，
ratio 本该恒等于 1，实测的离散程度就是估计器的噪声底噪。

两种结果都要如实接受：

- 噪声大到 ε=0.2 会截掉可观比例的 token → ε 放宽有实证支持，clip 挡的应是策略跑偏而非估计噪声。
- 噪声很小 → 这组测量**不足以**支持「ε 必须放宽」。别硬往结论上靠。
  更可能的解释是噪声随 θ 真正偏离 θ_old 而放大，那要到 P3 看
  `grpo/ratio_out_of_range_frac` 随步数的变化才能验证。

**C** 说明 `p_mask_prompt` 是条权衡曲线：掩得越多正则越强、μ 才能提到 12，
但 ratio 噪声也跟着涨。`p_mask=0` 处噪声应当为 0（两次前向的输入完全相同），
这可以当作整套测量的自检。
"""
)

md(
    """
## 7. 确认结果都在 Drive 上

P1、P2 的输出是直接写到 Drive 的，而且每完成一个阶段就落一次盘，
所以即便刚才中途断过，已经跑完的部分也还在。这里核对一遍。
"""
)

code(
    """
expected = ['p1_profile.json', 'p1_metrics.csv', 'p2_logprob.json', 'p2_pairs.csv']
for name in expected:
    path = RESULTS / name
    mark = 'OK ' if path.exists() else '缺失'
    size = f"{path.stat().st_size/1024:.1f} KB" if path.exists() else '-'
    print(f"  [{mark}] {name:<20}{size:>12}")

missing = [n for n in expected if not (RESULTS / n).exists()]
if missing:
    print(f"\\n缺 {missing}，把对应的单元格重跑一次即可，已有结果不会丢。")
"""
)

md(
    """
## 8. P3 之前：checkpoint 该怎么存

P1、P2 的输出只有几十 KB，直接写 Drive 没问题。**P3 的训练 checkpoint 不一样。**

LoRA `r=128` 挂在 LLaDA-8B 的七个投影上是 3.36 亿参数，fp32 约 1.34 GB；
加上 AdamW 的一阶、二阶矩，一个完整 checkpoint 约 **4 GB**。
Drive 是网络挂载，写入约 10-20 MB/s——存一次要 **3 到 7 分钟**，比一个训练步还慢。
每 25 步存一次的话，光存盘就能吃掉三成以上的训练时间。

所以分两级，因为两种故障的代价不同：

| | 落点 | 频率 | 救得了什么 |
|---|---|---|---|
| 本地 | `/content/checkpoints` | 每 `save_steps` 步 | 进程崩、OOM、手滑打断 |
| 镜像 | Drive | 每 `mirror_every` 次本地保存 | 会话断开、换机器 |

`trainer.resume()` 先找本地，本地没有（重连后换了机器就是这种情况）再回落到 Drive 镜像。

下面先量一次真实的存盘耗时再定频率——4GB 写 Drive 到底多慢，各人的网络不一样，
猜不如测。
"""
)

code(
    """
import time

import torch

CKPT_LOCAL = Path('/content/checkpoints/countdown_base')
CKPT_DRIVE = DRIVE_ROOT / 'checkpoints' / 'countdown_base'
CKPT_LOCAL.mkdir(parents=True, exist_ok=True)
CKPT_DRIVE.mkdir(parents=True, exist_ok=True)

# 用一个和真实 checkpoint 同量级的张量实测写入速度
probe = torch.zeros(int(1e9 / 4), dtype=torch.float32)  # 1 GB
for label, directory in (('本地 /content', CKPT_LOCAL), ('Drive', CKPT_DRIVE)):
    target = directory / 'probe.pt'
    start = time.perf_counter()
    torch.save(probe, target)
    elapsed = time.perf_counter() - start
    size_gb = target.stat().st_size / 1e9
    target.unlink()
    print(f"  {label:<14}{size_gb:.2f} GB  用时 {elapsed:6.1f}s  "
          f"({size_gb*1000/elapsed:.0f} MB/s)  →  4GB 约需 {elapsed*4:.0f}s")
del probe
"""
)

md(
    """
拿上面的实测数字定 `configs/countdown_base.yaml` 里的两个值：

```yaml
run:
  save_steps: 25          # 本地保存频率
  mirror_dir: /content/drive/MyDrive/dllm-grpo/checkpoints/countdown_base
  mirror_every: 4         # 每 4 次本地保存镜像一次，即每 100 步
```

判据是**镜像耗时别超过它所保护的那段训练时间的一成**。
若 4GB 写 Drive 要 300 秒，而 100 步训练是 3 小时，那 300 秒占 2.8%，可以接受；
若单步只要 10 秒，100 步才 17 分钟，300 秒就占了 30%，得把 `mirror_every` 再放大。

### 恢复演练

**在正式开跑前务必演练一次。**「存了但读不回来」这类问题不会报错，
只会在你真的断线之后才发现——那时已经没有第二次机会了。

P3 的训练脚本会以 `trainer.resume()` 开头：先找本地，再回落到 Drive 镜像，
都没有就从头开始。这条路径在 CPU 上有测试覆盖（`tests/test_trainer.py` 里
`test_resume_falls_back_to_the_drive_mirror` 会删掉本地目录来模拟换机器），
但真实模型上还要再验一次，因为 LoRA 的 `state_dict` 键名在 peft 版本之间变过。
"""
)

md(
    """
## 9. 下一次开机要跑的两件事

第一轮 P1/P2 已经跑完，数据在 `figures/`。这一节是第二轮，只有两件事，
**上面第 1 到 4 节的环境准备照跑，第 6 节的 P2 不用重跑**——改动只涉及采样的选点方式，
log-prob 估计器没动，P2 的偏差与 ratio 噪声不会变。

1. **验证采样簿记的修复。** 第一轮实测采样占比比按前向次数算的预算高 5-6 个百分点，
   归因是重掩码逐行 topk 每步 48 次 GPU 到 CPU 同步。改成整批取点后这个差应该明显收窄。
2. **测截断影响。** 第一轮七到九成补全撞在 128 的长度上限上，`correct_mean` 只有 0.16，
   分不清是推理不行还是答案没写完。对照组必须固定每步解码的 token 数，否则并行度
   一起变了就说明不了长度的事：128 长度配 64 步、256 长度配 128 步，两边都是每步 2 个。

两段合计约 40 分钟。结果写成 `_v2` / `_trunc256` 后缀的新文件，不覆盖第一轮的数据。
"""
)

code(
    """
# 一、重跑两档，看采样占比是否向理论预算收敛
!python -m scripts.run_p1_profile \\
    --config configs/countdown_base.yaml \\
    --sweep-diffusion-steps 64 128 \\
    --warmup 1 --steps 2 \\
    --out {RESULTS}/p1_profile_v2.json \\
    --metrics-csv {RESULTS}/p1_metrics_v2.csv

# 二、截断探针：256 长度配 128 步，与上面 128 长度配 64 步同为每步 2 token
!python -m scripts.run_p1_profile \\
    --config configs/countdown_base.yaml \\
    --max-completion-length 256 \\
    --sweep-diffusion-steps 128 \\
    --warmup 1 --steps 3 \\
    --out {RESULTS}/p1_trunc256.json \\
    --metrics-csv {RESULTS}/p1_metrics_trunc256.csv
"""
)

code(
    """
import csv
import json

report = json.loads((RESULTS / 'p1_profile_v2.json').read_text(encoding='utf-8'))
print('采样占比与理论预算的差（第一轮为 64 步 +6.3、128 步 +5.3 个百分点）')
for key in sorted(report['sweep'], key=int):
    entry = report['sweep'][key]
    measured = entry['timing']['generation_frac']
    budget = entry['budget']['budget/generation_share']
    print(f"  {key:>4} 步   实测 {measured:6.1%}   预算 {budget:6.1%}   "
          f"差 {(measured - budget) * 100:+5.1f} 个百分点")


def summarize(path, label, forward_passes=None):
    with path.open(encoding='utf-8') as handle:
        rows = [r for r in csv.DictReader(handle)
                if forward_passes is None
                or int(r['generation/forward_passes']) == forward_passes]

    def avg(key):
        return sum(float(r[key]) for r in rows) / len(rows)

    print(f"  {label}  {len(rows)} 步   correct {avg('reward/correct_mean'):.3f}   "
          f"eos 命中 {avg('completion/eos_hit_frac'):.2f}   "
          f"长度 {avg('completion/mean_length'):5.1f}   "
          f"单步 {avg('time/total_s'):6.1f}s")


print('\\n截断影响（两行都是每步 2 token，只有补全长度不同）')
summarize(RESULTS / 'p1_metrics_v2.csv', '长度 128', forward_passes=64)
summarize(RESULTS / 'p1_metrics_trunc256.csv', '长度 256')
"""
)

md(
    """
### 怎么判读

采样占比那两行，差值收窄到 1-2 个百分点就说明归因正确、修复生效；若还是 5 个点以上，
那 5-6 个点不是同步开销，得回头加细粒度计时重新归因，别急着接 Fast-dLLM。

截断那两行，样本只有几十条补全，是方向性探针不是测量，别拿它算显著性。
`correct` 明显上去、`eos 命中` 明显上去，就说明 0.16 里有相当一部分是被长度掐掉的，
P3 该把 `max_completion_length` 调到 256，代价是单步耗时翻倍。
两个数都没动，那 0.16 就是模型的真实水平，长度维持 128，省下的时间留给训练步数。

## 下一步

上面两件事都清了才进 P3，也就是跑 vanilla diffu-GRPO 基线。开跑前还有两件：

1. 拿第 5 节那张表定 `diffusion_steps`——它同时决定单步成本和 P4 的收益上限。
   顺便看一眼「200 步耗时」那列，如果超过 5 小时，先调 `max_steps` 或 `num_prompts_per_step`。
2. 拿第 8 节的实测写入速度定 `mirror_every`，并**真的做一次恢复演练**。
   200 步乘 200 秒是 11 小时，远超一个 Colab 会话，续训跑不通 P3 就跑不完。
"""
)


def build() -> dict:
    cells = []
    for kind, text in CELLS:
        lines = text.split("\n")
        source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": source}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "A100"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def check(notebook: dict) -> None:
    """确认每个代码单元格都能编译。

    notebook 里的语法错误要等到在 Colab 上执行到那一格才暴露，那时已经加载完 16GB 权重了。
    IPython 的 ! 与 % 前缀不是合法 Python，编译前先剥掉。
    """
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        lines, in_shell = [], False
        for line in source.split("\n"):
            if in_shell or line.lstrip().startswith(("!", "%")):
                # 反斜杠续行的后续行不以 ! 开头，但同属这条 shell 命令
                in_shell = line.rstrip().endswith("\\")
                lines.append("")
            else:
                lines.append(line)
        stripped = "\n".join(lines)
        try:
            compile(stripped, f"<cell {index}>", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"第 {index} 个单元格语法错误: {exc}\n{source}") from exc


if __name__ == "__main__":
    notebook = build()
    check(notebook)
    out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_p1_p2.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    code_cells = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(f"已生成 {out}（{len(CELLS)} 个单元格，其中 {code_cells} 个代码格已通过语法检查）")
