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
第1幕  测量     单步耗时拆解，定位瓶颈；并由采样占比推出加速的收益上限
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
| 单次前向/反向的批大小 | 整批（8 卡分摊，每卡 2 prompt） | **微批 4 条 + 梯度累积** | 单卡整批 24 条（4 prompt × 6 生成、序列 256）的激活要 40GB 以上：每层 MLP 中间态 24×256×12288×2B≈151MB，一层各类张量约 1.3GB，32 层即 42GB，而权重 17.4GB 加优化器状态 3.4GB 后只剩 18GB。选微批而非 activation checkpointing：后者反向要多跑一遍前向，会直接改掉 P1 要测的耗时占比；微批总计算量不变，且因 `grpo_loss` 按序列归一化，梯度与全批**逐位相等** |
| 采样期 Gumbel 噪声的作用范围 | 整条序列 `(B, 256, V)`，fp32 | **只在当前块 `(B, 32, V)`，且原地** | 块外位置的 `x0` 与 confidence 全被 `-inf` 屏蔽、从不采纳，算了就是白算。官方写法一步内会同时活着五六个同尺寸 fp32 临时张量（`clamp_min` 非原地、两次 `log`、`logits.float()`、`T*gumbel`、求和），单个 24×256×126464×4B≈2.89GB，峰值 12–15GB——这是采样阶段最大的显存项，比权重之外的一切都大。切块 8 倍 + 原地链后降到约 0.4GB，且与原式**逐位相等**（已验证 fp32/bf16 下 max diff 均为 0） |
| LoRA target modules | 写的是 `q/k/v/o_proj/up/down/gate_proj`，**实际只有 `q/k/v/up_proj` 生效** | **七类全挂**（`q/k/v/attn_out/ff_proj/up_proj/ff_out`） | 官方那份是 Llama 命名，`o_proj`/`gate_proj`/`down_proj` 在 LLaDA 上一个都匹配不上；PEFT 仅在全不命中时报错，部分命中静默跳过，于是注意力输出投影和整个 FFN 都没训到。本项目按其**意图**补齐，可训练参数因此约为官方实际值的两倍（1.68 亿 → 3.36 亿）。要严格对齐官方**实测**基线，改回 `["q_proj","k_proj","v_proj","up_proj"]` 即可 |
| batch 内的 padding | 未处理 | **同样不处理**（已查清并接受） | `LLaDAModel.forward` 把 `attention_mask` 算成加性 bias 后随即丢弃（紧跟 `attention_bias = None`），且外部无法注入掩码，所以真实 token 会注意到 padding。补齐到固定 `max_prompt_length` 使每行 padding 量恒定，**不破坏可复现性**，代价是补全质量被系统性拉低。修它有两条路——按 prompt 分组（只修得了生成阶段，loss 阶段仍要堆成单张量）、包一层 `block.forward` 注入 bias（两阶段一起修、不影响 GPU 利用率，但依赖远端代码结构）。**均不采用**：污染在 P3 与 P4 中完全一致，会在最终要测的差值里抵消；而多一处偏离会让「曲线对不上」更难归因 |
| attention | flash_attention_2 | **eager** | 不是取舍，是唯一选项：`LLaDAModelLM` 不参与 HF 的 attn 派发，非 eager 一律 `ValueError`。也没有性能损失，它内部本来就在调 `F.scaled_dot_product_attention`。顺带省掉 flash-attn 在 Colab 上 20 分钟的编译 |

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
| `attn_implementation` | `eager` | **只能填这个**。`LLaDAModelLM` 未声明 `_supports_sdpa`，填 `sdpa` / `flash_attention_2` 会抛 `ValueError`。不影响性能：LLaDA 远端代码自己就在调 `F.scaled_dot_product_attention`，HF 这个参数管不到它 |
| mask token id | `126336` | LLaDA 特有，自实现采样循环时勿写错 |
| `lora_r` | 128 | |
| `lora_alpha` | 64 | **不要改**。alpha < r 意味着缩放系数 0.5，不是常见的 alpha=2r。按习惯改会让有效学习率翻 4 倍 |
| `lora_dropout` | 0.05 | |
| target modules | `q_proj` `k_proj` `v_proj` `attn_out` `ff_proj` `up_proj` `ff_out` | LLaDA（OLMo 系）的命名，不是 Llama 那套：`attn_out`↔`o_proj`、`ff_proj`↔`gate_proj`、`ff_out`↔`down_proj`。**与 d1 实际行为有别**，见第 2 节 |
| LoRA 是否覆盖词表投影 | 否 | LLaDA 的词表投影也叫 `ff_out`，与块内 FFN 下投影重名。PEFT 按名字末段匹配，不排除就会把 `[4096, 126464]` 一起挂上（r=128 时多 1670 万参数）。`lm_head_exclusion` 用全匹配正则只排掉它 |

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
- **装配自检**：LoRA 目标模块真的匹配上、适配器真的参与前向、padding 不污染真实位置
- **单步耗时拆解打点**：generation / reward / logprob / forward_backward 四段分别计时
- 实测占比对照理论前向次数预算（`dllm.experiment.step_compute_budget`）

入口：`notebooks/colab_p1_p2.ipynb`，底层是 `scripts/run_p1_profile.py`。
脚本带 `--tiny`，可在本地 CPU 上把整条路径先跑通再上 Colab。

**验收**：拿到第 1 幕的数据——单步时间的实际分布，以及由它推出的加速收益上限。

> **写计划时的预期被推翻了。** 原本写的是「rollout 占 70–80%」，那个数字对应 d1 官方配置
> （diffusion_steps=128、μ=12）：128 / (128 + 24 + 36) ≈ 68%。
> 本项目为省成本把 diffusion_steps 砍到 64，采样占比随之掉到 **约 52%**：
> 64 / (64 + 24 + 36) ≈ 51.6%。
>
> 这直接改变第 3 幕的分量。按 Amdahl，采样占 52% 时即使把采样加速到无穷倍，
> 端到端上限也只有 2.1x；加速 3 倍则只有 1.5x。
> 砍 diffusion_steps 省下的，恰好是 P4 要攻的那部分。
>
> 还有一层：completion 长度 128 配 diffusion_steps 64，等于每步解 2 个 token，
> 而 Fast-dLLM 并行解码的收益正是「每步多解几个 token」——基线里已经预支了一部分。

**待定：`diffusion_steps` 取 64 还是 128。**

| | 64（当前） | 128（对齐官方） |
|---|---|---|
| 每步解码 token 数 | 2 | 1 |
| 采样占单步 | ~52% | ~68% |
| 单步成本 | 基准 | 约 1.5x |
| P4 端到端上限 | 2.1x | 3.1x |
| 基线是否已预支并行解码收益 | 是 | 否 |

**已决定：等 P1 实测再定。** `run_p1_profile.py --sweep-diffusion-steps 64 128`
会在同一个 Colab 会话里把两个取值各测一遍，输出单步耗时、采样占比、P4 上限
和「跑满 200 步要多久」四列。

判据：如果 128 步把 P3 顶到 8 小时以上，就是拿第 5 幕（同等墙钟时间下的 reward 曲线对比）
去换第 3 幕，不划算；否则取 128，让基线是干净的「每步一个 token」。

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
| **flash-attn 编译失败** | 低 | 不装，配置里用 eager；LLaDA 本就不接受 HF 的 attn 派发，装了也白装 |

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

明确不装的三样：`bitsandbytes`（已改用 bf16）、`deepspeed`（单卡不需要）、`flash-attn`（编译慢，且 LLaDA 根本不接受 HF 的 attn 派发，装了用不上）。

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
