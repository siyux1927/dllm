# 把推理加速塞进扩散语言模型的强化学习循环

## TLDR

扩散语言模型（dLLM）的强化学习训练，大半时间耗在在线采样上。把 training-free 的推理加速
接进 rollout 环节能提速，但并行解码会改变采样分布、破坏 on-policy 假设。本项目量化这个
权衡，并找出安全的加速配置。

模型 `LLaDA-8B-Instruct`，任务 Countdown，单卡 A100-40GB。P0 骨架与 P1/P2 的 Colab 入口
已就绪，164 个测试全绿且全部可在 CPU 上跑。下一步是把
[notebooks/colab_p1_p2.ipynb](notebooks/colab_p1_p2.ipynb) 拿到 A100 上，取第 1、2 幕的实测数据。

完整实施方案见 [docs/plan-diffu-grpo.md](docs/plan-diffu-grpo.md)，方向选型过程见
[docs/proposals.md](docs/proposals.md)，踩过的坑见 [docs/lessons.md](docs/lessons.md)。

## 项目进度

| 编号 | 问题 | 状态 |
|---|---|---|
| 0 | dLLM 的 RL 后训练现状：d1 自承「在线生成开销过大，因此把生成长度限制在 256」 | 已成文 |
| 1 | 耗时到底花在哪？分阶段拆解，定位瓶颈并推出加速的收益上限 | 代码就绪，待上 Colab |
| 2 | diffu-GRPO 的地基是单步 log-prob 近似。它有多准？论文没报告过 | 代码就绪，待上 Colab |
| 3 | 把 Fast-dLLM 的块级 KV cache 与置信度并行解码接进 rollout | 待做 |
| 4 | 加速改变采样分布，引入额外 off-policy 偏差。量化这个权衡 | 待做 |
| 5 | 同等墙钟时间下的 reward 曲线对比，加 GSM8K 回归检查 | 待做 |

第 2 幕补的是 d1 论文的真实空白：论文提到 LLaDA 官方用 128 样本蒙特卡洛估计 log-prob，
并说这对在线 RL 太贵，但从未报告单步近似与真值的偏差。第 3、4 幕是 Fast-dLLM（推理侧）
与 d1（训练侧）的交叉。

已实现：Countdown 合成与去重、格式/正确性分离的奖励函数、块级扩散采样、单步与蒙特卡洛
两种 log-prob 估计器、diffu-GRPO 损失、θ/θ_old/θ_ref 三策略封装、分阶段计时、指标打点、
断点续训、LLaDA 装配与三项自检、单步算力预算推导。

> **一个已经浮现的问题。** 算力预算显示，当前配置（`diffusion_steps=64`）下采样只占单步
> 约 52%，而非计划里写的 70–80%——后者对应的是 d1 官方的 128 步。按 Amdahl，采样占 52%
> 时端到端加速上限只有 2.1x，而且 64 步配 128 长度等于每步解 2 个 token，基线本身已经在
> 并行解码了。P1 会在两个取值上各测一遍再定，详见 plan 的 P1 一节。

## 设计和实验参数

**参数**（完整表与逐项偏离理由见 [docs/plan-diffu-grpo.md](docs/plan-diffu-grpo.md) 第 2、3 节）

- 模型 `LLaDA-8B-Instruct`，`bfloat16`，`attn_implementation=eager`——这是唯一可用值，
  `LLaDAModelLM` 不参与 HF 的 attn 派发，填 `sdpa` 会抛 `ValueError`，且不影响性能
- LoRA `r=128` / `alpha=64`。alpha < r 意味着缩放系数 0.5，按 alpha=2r 的习惯改会让有效学习率翻 4 倍
- 目标模块 `q_proj` `k_proj` `v_proj` `attn_out` `ff_proj` `up_proj` `ff_out`。这是 OLMo 系命名，
  不是 Llama 那套；d1 官方填的是后者，在 LLaDA 上实际只有 4/7 生效
- 采样 `block_length=32`、`diffusion_steps=64`、`max_completion_length=128`、`low_confidence` 重掩码
- GRPO `num_iterations=12`、`epsilon=0.5`、`beta=0.04`、`p_mask_prompt=0.15`、每步 4 prompt × 6 生成
- `micro_batch_size=4`。整批 24 条的激活要 40GB 以上，单卡放不下；拆微批累积，梯度与全批逐位相等。
  没选 activation checkpointing——它反向要多跑一遍前向，会改掉 P1 要测的耗时占比
- 优化器 `lr=3e-6`、`adam_beta2=0.99`（不是 0.999）、`max_grad_norm=0.2`、`max_steps=200`

**决定**

- **不复用 TRL 的 GRPOTrainer。** 它的采样是为自回归模型写的，套扩散采样要大幅覆写；
  而本项目的核心产出都要求对循环内部有完全控制权。
- **参考策略取自关掉 LoRA 的底座**，不额外占显存。独立 reference model 会再吃 16GB，
  单卡 40GB 放不下 8B 模型加两份权重。
- **每次内更新都重新采 prompt 掩码模式 q'**，且 θ、θ_old、θ_ref 三次估计共享同一个 q'——
  论文式 4 中 q' 位于期望之内，三者必须一致。
- **CPU 小模型的结构对齐 LLaDA**，线性层照抄它的命名，连词表投影与 FFN 下投影重名这点也保留。
  单测里的 LoRA `target_modules` 因此与真实配置是同一份，适配器挂不上会在本地就暴露。
- **每个自检都对应一种「不报错的失效」。** 这类问题在 8B 模型上要烧几小时 A100 才可能发现，
  所以尽量在 CPU 上、或在加载权重的第一分钟内就把它们变成显式报错。
- **按「Colab 一定会断」来设计存盘。** 阶段结果即时原子落盘（先写 `.tmp` 再 replace）；
  checkpoint 约 4GB 而 Drive 写入仅 10–20MB/s，故本地每 25 步存一次救进程崩溃、
  每 100 步镜像到 Drive 救会话断开，存盘耗时记进指标。

## 目录

```
src/dllm/
├── config.py          超参 dataclass，「不要改」的项在注释里写明理由
├── experiment.py      实验装配、单步算力预算、Amdahl 上限
├── data/              Countdown 合成
├── rewards/           格式与正确性奖励，AST 安全求值
├── sampling/          块级扩散采样（P4 在此接入 Fast-dLLM）
├── logprob/           单步近似与蒙特卡洛真值（P2 的对照对象）
├── train/             GRPO 损失、策略封装、训练循环、断点续训
├── models/            llada.py 真实装配与自检；tiny.py CPU 替身
└── utils/             分阶段计时、指标打点

scripts/
├── check_env.py       版本与依赖校验
├── run_p1_profile.py  P1 耗时拆解（--tiny 可在 CPU 上跑通路径）
└── run_p2_logprob.py  P2 log-prob 误差量化（同上）

notebooks/colab_p1_p2.ipynb   Colab 入口
```

## 快速开始

```bash
conda env create -f environment.yml
conda activate dllm-dev
python scripts/check_env.py     # 校验版本与依赖
python -m pytest                # 164 个 CPU 测试
python -m ruff check .
```

上 Colab 之前，先在本地 CPU 上把两个脚本的整条路径跑通（数字无意义，只验证接线）：

```bash
python scripts/run_p1_profile.py --config configs/countdown_base.yaml --tiny --steps 2
python scripts/run_p2_logprob.py --config configs/countdown_base.yaml --tiny --mc-samples 16
```

Colab：打开 `notebooks/colab_p1_p2.ipynb`，运行时选 A100。

## 参考

- **d1**（UCLA）— masked SFT + diffu-GRPO，[arXiv 2504.12216](https://arxiv.org/abs/2504.12216)
- **Fast-dLLM**（NVIDIA, ICLR'26）— KV cache + 置信度并行解码，[arXiv 2505.22618](https://arxiv.org/abs/2505.22618)
- **LLaDA** — [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
