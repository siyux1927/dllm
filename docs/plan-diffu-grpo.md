# 实施文档：把推理加速塞进扩散语言模型的强化学习循环

> 方案编号：候选方案 2（后训练 / RL），详见 `docs/proposals.md`
> 模型：`GSAI-ML/LLaDA-8B-Instruct`（dense，不用 MoE）
> 硬件：Colab 单卡 A100-40GB ｜ 本地 conda `myenv`（CPU，只写代码与冒烟测试）
> 预算：约 12–16 小时墙钟，$15–20 compute units

---

## 1. 项目要讲的故事

### 一句话

扩散语言模型的强化学习训练，七到八成时间耗在在线采样上；把 training-free 的推理加速接进 rollout 环节能大幅提速，但并行解码会改变采样分布、破坏 on-policy 假设——本项目量化这个权衡，并找出安全的加速配置。

### 五幕结构

```
第0幕  背景     dLLM 的 RL 后训练刚起步，d1 是唯一开源方案；它自己承认
                "在线生成开销过大，因此把生成长度限制在 256"
第1幕  测量     单步耗时拆解，证明 rollout 占 70–80%，瓶颈定位在采样而非反向
第2幕  质疑     diffu-GRPO 的地基是单步 log-prob 近似。它到底有多准？
                论文没报告过。我们对照 128 样本蒙特卡洛真值给出答案，
                并用它解释为什么官方要把 clip 范围 ε 从 0.2 放宽到 0.5
第3幕  提速     把 Fast-dLLM 的块级 KV cache + 置信度并行解码接进 rollout
第4幕  代价     加速改变采样分布 → 额外 off-policy 偏差。扫置信度阈值，
                画出「加速倍数 vs 分布偏移 vs reward 曲线」的三方权衡
第5幕  结论     同等墙钟时间下的 reward 曲线对比 + 安全加速配置建议
                + GSM8K 回归检查（有没有把模型训坏）
```

### 为什么这个故事站得住

- 第 2 幕补的是 d1 论文的真实空白：论文提到 LLaDA 官方用 128 样本蒙特卡洛估计 log-prob，并说这对在线 RL 太贵，但**从未报告单步近似与真值的偏差**。
- 第 3、4 幕是 Fast-dLLM（推理侧）与 d1（训练侧）的交叉，目前没有现成工作做过这个组合。
- 第 4 幕的矛盾是真实存在的，不是人为制造：GRPO 要求轨迹采自 `π_θold`，用被改动过的采样器产出轨迹必然引入偏差。

---

## 2. 关键决策与依据

所有偏离 d1 官方配置的改动，都必须有理由。下表是全部改动清单。

| 决策 | 官方做法 | 本项目 | 依据 |
|---|---|---|---|
| 是否做 SFT | s1k 训 20 epoch，seq 4096 | **砍掉，直接 RL** | 论文 Table 1：diffu-GRPO 在 12/12 配置赢过 baseline 与 SFT；SFT 仅 7/12 赢过 baseline。SFT 是增量而非主因，但要花数小时 |
| 任务数量 | GSM8K / MATH500 / Sudoku / Countdown 各训一个模型 | **只做 Countdown** | 增益差 6 倍：Countdown +26.2%、Sudoku +10.0%、MATH500 +4.0%、GSM8K +3.9%。论文解释为 base model 在数学上已饱和 |
| 量化 | `load_in_4bit: true` | **bf16** | 官方 4-bit 是为 8 卡挤显存。单卡 40GB 装得下 bf16（16GB 权重），且 bitsandbytes 4-bit 反量化开销让计算密集场景慢 2–3 倍 |
| 参考模型 | 独立 ref + 每 64 步同步 | **adapter-disable 复用底座** | LoRA 只训适配器，关掉适配器即得参考策略，省 16GB 显存 |
| 分布式 | DeepSpeed ZeRO-2，8 进程 | **单卡，无 accelerate** | 单卡不需要 ZeRO，去掉一整层调试复杂度 |
| prompt 长度 | 200 | **128** | 200 是为 GSM8K 留的余量；Countdown prompt 仅 60–80 token |
| completion 长度 | 256 | **128** | Countdown 答案是一个算式，不需要长推理链 |
| diffusion_steps | 128 | **64** | 配合 block_length=32，每块 16 步、每步解 2 token |
| checkpoint 频率 | 每 100 步 | **每 25 步** | Colab 会断线，100 步意味着可能丢 2.5 小时 |
| attention | flash_attention_2 | **sdpa** | Colab 上编译 flash-attn 要 20 分钟以上且易与 torch 版本冲突；LLaDA 是双向注意力，sdpa 完全够用 |

### 提速效果核算

按 A100 实测 100–150 TFLOPS 估算（序列 256 token、batch 24）：

| 项目 | 官方配置折算单卡 | 优化后 |
|---|---|---|
| rollout 生成 | ~150 s | ~42 s |
| 12 次内更新 | ~45 s | ~23 s |
| 单 step 合计 | ~200 s | **90–150 s** |
| 200 step 墙钟 | ~11 h | **5–8 h** |
| 200 step 花费 | ~$13 | **$6–9** |

---

## 3. 完整超参表

保留官方值的部分不要动，尤其是下面标注了「不要改」的几项。

### 模型与 LoRA

| 参数 | 值 | 备注 |
|---|---|---|
| `model_path` | `GSAI-ML/LLaDA-8B-Instruct` | 需 `trust_remote_code=True` |
| `torch_dtype` | `bfloat16` | |
| `attn_implementation` | `sdpa` | 偏离官方 |
| mask token id | `126336` | LLaDA 特有，自实现采样循环时勿写错 |
| `lora_r` | 128 | |
| `lora_alpha` | 64 | **不要改**。alpha < r 意味着缩放系数 0.5，不是常见的 alpha=2r。按习惯改会让有效学习率翻 4 倍 |
| `lora_dropout` | 0.05 | |
| target modules | 全部线性层（q/k/v/o/gate/up/down） | |

### 优化器

| 参数 | 值 | 备注 |
|---|---|---|
| `learning_rate` | `3e-6` | |
| `lr_scheduler_type` | `constant_with_warmup` | |
| `warmup_ratio` | `0.0001` | |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.99 | beta2 是 0.99 不是 0.999 |
| `weight_decay` | 0.1 | |
| `max_grad_norm` | 0.2 | 比常见的 1.0 严格得多 |

### GRPO 与扩散采样

| 参数 | 值 | 备注 |
|---|---|---|
| `num_generations` | 6 | 每个 prompt 的 group 大小 |
| `per_device_train_batch_size` | 6 | 即每设备 1 个 prompt |
| `gradient_accumulation_steps` | 4 | 偏离官方（官方 2×8卡），凑到每 step 4 prompt × 6 = 24 条 rollout |
| `num_iterations` (μ) | 12 | **不要改**。这是 d1 的核心卖点：随机 prompt masking 带来的正则效应让 μ 可以从常规的 2 提到 12 |
| `epsilon` | 0.5 | **不要改**。log-prob 是单步近似、ratio 噪声大，收紧到 0.2 会导致几乎每个 token 都被截断 |
| `beta` (KL) | 0.04 | |
| `p_mask_prompt` | 0.15 | 实验 C 的消融对象 |
| `random_masking` | `True` | |
| `max_prompt_length` | 128 | 偏离官方 |
| `max_completion_length` | 128 | 偏离官方 |
| `block_length` | 32 | |
| `diffusion_steps` | 64 | 偏离官方 |
| `remasking` | `low_confidence` | |
| `seed` | 42 | |
| `save_steps` | 25 | 偏离官方 |

---

## 4. 数据与评测

### 训练数据

Countdown-3：给定 3 个数与一个目标数，用四则运算凑出目标。可纯合成、无需下载，也可用 `Jiayi-Pan/Countdown-Tasks-3to4`。建议自己合成，理由是可以精确控制难度分布并留出干净的 held-out 集。

### 奖励函数

组合式，但**两项必须分开记录**：

- **格式奖励**：输出是否符合约定格式（如 `<answer>...</answer>` 包裹的算式）
- **正确性奖励**：算式是否合法（只用给定数字、每个用一次）且求值等于目标

分开记录的原因：Countdown 的格式奖励极易被刷——模型会先学会输出合法格式但不解题，此时总奖励上升而正确率不动。只看总奖励会被骗。

### 评测

| 项目 | 设置 |
|---|---|
| 主评测 | 500 道 held-out Countdown，gen_len 128 与 256 各测一次，贪心解码，0-shot |
| 回归检查 | GSM8K 200 条子集，检查通用数学能力有没有被 RL 训坏 |
| 评测频率 | 每 50 步一次（官方是 600 步后每 100 步，我们步数少所以加密） |

回归检查这一项不能省。「在目标任务上涨了，但有没有把模型训坏」是工程成熟度的分水岭。

---

## 5. 分阶段执行计划

每个阶段都有明确的验收标准，没达到就不要进下一阶段。

### P0 · 本地骨架（`dllm-dev`，CPU，零 GPU 成本）—— 已完成

- 数据合成与奖励函数，含单元测试（奖励函数必须有测试，它是整个 RL 的信号源）
- 采样循环、log-prob 估计器、GRPO loss 的完整实现
- 用一个随机初始化的极小模型在 CPU 上跑通完整链路
- checkpoint 保存与断点续训逻辑

**验收标准**：CPU 上完整跑通多个 outer step，loss 有限、无 NaN，杀掉进程后能从 checkpoint 精确恢复（含 optimizer state 与数据游标）。

**实际结果**：106 个测试全绿。断点续训以 `test_resume_reproduces_uninterrupted_run` 自动化断言——中断恢复后的第 3 步与不中断时逐位一致，无需手动 kill 验证。冒烟测试跑的是真实 PEFT 策略而非 plain 拷贝路径，因此适配器切换、θ_old 快照、LoRA 状态存取都已在本地覆盖。

P0 阶段暴露的问题（已修）：

| 问题 | 影响 | 处理 |
|---|---|---|
| `nn.TransformerEncoderLayer` 在 eval 下走融合快速路径，绕过 `linear1`/`linear2` | **静默**失效：LoRA 挂得上但完全不参与前向，训练看着正常跑 | 小模型改写为显式 Transformer 块，模块命名对齐 LLaDA；补回归测试同时断言「被包装」与「被调用」 |
| trl 版本 | PyPI 正式版的 `GRPOTrainer` 接口与 d1 对不上 | 锁定 d1 使用的 commit `0f88c179`，`scripts/check_env.py` 通过检查 `GRPOConfig` 字段来验证 |
| transformers 5.x 与 LLaDA 远程代码不兼容 | 加载即崩 | 另建 `dllm-dev` 环境锁 4.49.0 |

### P1 · 打通 Colab 与基线测量（约 1h）

- HF 缓存挂载到 Google Drive
- 加载 LLaDA-8B-Instruct，跑通一次生成
- **单步耗时拆解打点**：generation / logprob / backward 三段分别计时

**验收**：拿到第 1 幕的数据——rollout 占比确实在 70–80%。这个数字是后面所有论证的起点，必须先落地。

### P2 · 实验 A：log-prob 估计器验证（约 0.5h，极便宜）

离线跑一批 completion，对同一批数据同时计算：

- 单步 unmasking 估计（d1 的做法）
- 128 样本蒙特卡洛估计（LLaDA 官方做法，视为真值）

**产出**：散点图 + 相关系数；偏差随 mask 比例的变化；偏差随 token 位置的变化（前面的 token 是不是估得更准？）。

**验收**：能用数据回答「为什么 ε 要开到 0.5」。这是全项目性价比最高的实验——几分钟 GPU 换一个论文没做的分析。

### P3 · 基线 RL 训练（约 5–8h）

用第 3 节的配置跑 200 步 vanilla diffu-GRPO。

**验收**：正确性奖励（不是总奖励）出现肉眼可见的上升趋势。若 100 步后仍无动静，先停下来查奖励函数和 ratio 分布，不要盲目加步数。

### P4 · 实验 B：加速 rollout（约 3–5h）

- 把 Fast-dLLM 的块级近似 KV cache 与置信度并行解码接进采样环节
- 扫置信度阈值（如 0.9 / 0.95 / 0.99），每档跑同样墙钟时间
- 量化分布偏移：对同一 prompt，比较加速采样与原始采样产出 completion 的 log-prob 分布差异

**验收**：拿到「加速倍数 vs 分布偏移 vs reward 曲线」三方权衡图，并能给出一个推荐阈值。

### P5 · 实验 C 与收尾（约 2h）

- `p_mask_prompt` 消融：0 / 0.15 / 0.3 / 0.5，各跑 100 步。论文只做了 random vs fixed masking，没扫过 p 值
- GSM8K 回归检查
- 出图、写 README 故事线

---

## 6. 必须记录的指标

从第一步就打点，事后补不回来。

| 类别 | 指标 | 用途 |
|---|---|---|
| 奖励 | 总奖励、格式奖励、正确性奖励 **分三条线** | 识别奖励 hacking |
| 策略 | ratio 分布、**clip 触发比例** | 诊断 log-prob 估计质量，与实验 A 呼应 |
| 策略 | KL to reference | 监控偏离程度 |
| 生成 | completion 长度分布、EOS 命中率 | 看模型有没有学会提前收敛 |
| 性能 | 每步墙钟拆成 generation / logprob / backward | 实验 B 的全部论据来源 |
| 性能 | GPU 显存峰值、利用率 | 调 batch 的依据 |

---

## 7. 风险与预案

| 风险 | 影响 | 预案 |
|---|---|---|
| ~~transformers 版本不兼容~~ | **已解决**。原 `myenv` 是 5.15.1，与 LLaDA 远程代码（按 4.x 写）不兼容 | 已另建 `dllm-dev` 环境锁定 `transformers==4.49.0`，并对齐 d1 官方全部版本，见第 8 节 |
| **Colab 断线** | 高。5–8h 训练必然跨 session | `save_steps=25`，存 LoRA + optimizer + scheduler + 数据游标到 Drive；**正式开跑前手动 kill 一次验证 resume 真能接上** |
| **抢不到 A100** | 中 | 降级到 L4(24GB)：bf16 16GB 权重仍放得下，把 batch 降到 12、步数减半，结论方向不变 |
| **奖励 hacking** | 中 | 格式与正确性奖励分开记录；定期人工抽查生成样本 |
| **200 步不够看出趋势** | 中 | 优先保证 P2（离线、便宜、必出结果）；P3 曲线不动时先查 ratio 分布而非加步数 |
| **Fast-dLLM 接入有坑** | 中 | 双向注意力的近似 cache 有 mask 对齐与块边界问题；兜底是直接复用官方实现的 cache 部分，精力集中在第 4 幕的偏差量化上 |
| **flash-attn 编译失败** | 低 | 直接用 sdpa，已写进配置 |

---

## 8. 环境

### 本地 `dllm-dev`（conda，Python 3.10）—— 已建好

职责：写代码、跑 CPU 冒烟测试、出图、写文档。**不跑真实训练。**

原先的 `myenv` 是 Python 3.12 + transformers 5.15.1，与 LLaDA 的远程建模代码（按 4.x 编写）不兼容，因此另建 `dllm-dev`，版本逐项对齐 d1 官方 `env.yml`：

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.10.20 | 对齐官方 |
| torch | 2.6.0+cpu | 本地无 GPU，用 CPU 构建 |
| transformers | 4.49.0 | **关键**，5.x 会让 LLaDA 远程代码崩 |
| accelerate | 1.4.0 | |
| peft | 0.15.1 | |
| datasets | 3.3.2 | |
| trl | 0.16.0.dev0 | **锁 commit `0f88c179`**，否则 `GRPOTrainer` 接口对不上 |
| numpy | 1.26.4 | 官方锁 1.25，取 <2 的最近可用版 |

重建方式：

```powershell
conda env create -f environment.yml
conda activate dllm-dev
python scripts/check_env.py
```

`scripts/check_env.py` 会逐项校验版本，并确认 `GRPOConfig` 具备 `num_iterations` / `epsilon` / `beta` / `num_generations` 四个字段——这是 trl commit 是否正确的判据。

已确认状态：自检全部通过，`torch.cuda.is_available() == False`（本地无 NVIDIA 卡，符合预期）。

### Colab

```python
!pip install -q -r requirements-colab.txt
!python scripts/check_env.py   # 与本地跑同一个自检
```

HF 缓存指向 Drive，避免每次 session 重下 16GB 权重：

```
HF_HOME=/content/drive/MyDrive/hf_cache
```

明确不装的三样：`bitsandbytes`（已改用 bf16）、`deepspeed`（单卡不需要）、`flash-attn`（编译慢且易冲突，用 sdpa 替代）。

---

## 9. 目录结构

```
diff/
├── environment.yml           # 本地 dllm-dev 环境（CPU）
├── requirements-colab.txt    # Colab 环境（CUDA）
├── docs/
│   ├── proposals.md          # 六个候选方案
│   └── plan-diffu-grpo.md    # 本文档
├── src/
│   ├── data/                 # Countdown 合成与数据集
│   ├── rewards/              # 奖励函数（必须有单测）
│   ├── sampling/             # 扩散采样循环、Fast-dLLM 加速版
│   ├── logprob/              # 单步估计器与蒙特卡洛真值
│   └── train/                # diffu-GRPO 训练循环、checkpoint
├── tests/                    # CPU 单测，含小模型全链路冒烟
├── scripts/
│   └── check_env.py          # 环境自检，本地与 Colab 跑同一份
├── configs/                  # 超参 YAML
├── results/                  # 指标 CSV 与图
└── README.md                 # 最终故事线
```

---

## 10. 成本预算

| 阶段 | GPU 时长 | 花费 |
|---|---|---|
| P0 本地骨架 | 0 | $0 |
| P1 打通与测量 | 1 h | ~$1.2 |
| P2 log-prob 验证 | 0.5 h | ~$0.6 |
| P3 基线 RL | 5–8 h | $6–9 |
| P4 加速 rollout | 3–5 h | $4–6 |
| P5 消融与收尾 | 1–2 h | $1–2 |
| **合计** | **11–17 h** | **$13–19** |

按 Colab Pro 的 A100 约 11.8 compute units/小时、$9.99/100 units 折算。

---

## 11. 参考

- **d1**（UCLA）— masked SFT + diffu-GRPO，arXiv 2504.12216，代码 `dllm-reasoning/d1`
- **Fast-dLLM**（NVIDIA，ICLR'26）— KV cache + 置信度并行解码，arXiv 2505.22618
- **LLaDA** — arXiv 2502.09992，模型 `GSAI-ML/LLaDA-8B-Instruct`
- 官方超参来源：`d1/diffu-grpo/slurm_scripts/train.yaml`、`d1/diffu-grpo/run.sh`
