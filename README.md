# 把推理加速塞进扩散语言模型的强化学习循环

扩散语言模型（dLLM）的强化学习训练，七到八成时间耗在在线采样上。把 training-free 的
推理加速接进 rollout 环节能大幅提速，但并行解码会改变采样分布、破坏 on-policy 假设。
本项目量化这个权衡，并找出安全的加速配置。

模型 `LLaDA-8B-Instruct`，任务 Countdown，单卡 A100-40GB。
完整实施方案见 [`docs/plan-diffu-grpo.md`](docs/plan-diffu-grpo.md)，
方向选型过程见 [`docs/proposals.md`](docs/proposals.md)。

## 故事线

| 幕 | 问题 | 状态 |
|---|---|---|
| 0 | dLLM 的 RL 后训练现状：d1 自己承认「在线生成开销过大，因此把生成长度限制在 256」 | — |
| 1 | 耗时到底花在哪？分阶段拆解，定位瓶颈在采样而非反向 | 待跑 |
| 2 | diffu-GRPO 的地基是单步 log-prob 近似。它有多准？论文没报告过 | 待跑 |
| 3 | 把 Fast-dLLM 的块级 KV cache 与置信度并行解码接进 rollout | 待做 |
| 4 | 加速改变采样分布，引入额外 off-policy 偏差。量化这个权衡 | 待做 |
| 5 | 同等墙钟时间下的 reward 曲线对比，加 GSM8K 回归检查 | 待做 |

第 2 幕补的是 d1 论文的真实空白：论文提到 LLaDA 官方用 128 样本蒙特卡洛估计 log-prob，
并说这对在线 RL 太贵，但从未报告单步近似与真值的偏差。第 3、4 幕是 Fast-dLLM（推理侧）
与 d1（训练侧）的交叉。

## 当前进度

**P0 本地骨架已完成**，106 个测试全绿，全部可在 CPU 上跑。

已实现：Countdown 合成与去重、格式/正确性分离的奖励函数、块级扩散采样、
单步与蒙特卡洛两种 log-prob 估计器、diffu-GRPO 损失、θ/θ_old/θ_ref 三策略封装、
分阶段计时、指标打点、断点续训。

下一步是 P1：在 Colab 上跑通 LLaDA-8B 并拿到第 1 幕的耗时拆解数据。

## 快速开始

```bash
conda env create -f environment.yml
conda activate dllm-dev
python scripts/check_env.py     # 校验版本与 trl commit 是否正确
python -m pytest                # 106 个 CPU 测试
```

Colab：

```python
!pip install -q -r requirements-colab.txt
!python scripts/check_env.py
```

## 设计上的几个决定

**不复用 TRL 的 GRPOTrainer。** 它的采样是为自回归模型写的，套扩散采样要大幅覆写；
而本项目的核心产出都要求对循环内部有完全控制权。trl 仍锁在 d1 使用的 commit 上作为对照基准。

**参考策略取自关掉 LoRA 的底座**，不额外占显存。官方那套独立 reference model 会再吃 16GB，
单卡 40GB 放不下 8B 模型加两份权重。

**每次内更新都重新采 prompt 掩码模式 q'**，且 θ、θ_old、θ_ref 三次估计共享同一个 q'——
论文式 4 中 q' 位于期望之内，三者必须一致。

**CPU 小模型的结构对齐 LLaDA**（`q_proj` / `gate_proj` 等命名），因此单测里的 LoRA
`target_modules` 与真实配置是同一份，适配器挂不上会在本地就暴露。

## 踩过的坑

`nn.TransformerEncoderLayer` 在 eval 模式下走融合快速路径，**绕过 `linear1` / `linear2`
子模块**，挂在上面的 LoRA 会被静默忽略——前向照常返回结果，训练看着也在跑，但适配器是死的。
小模型因此改成显式的 Transformer 块，并留了一条回归测试同时断言「被包装」和「被调用」。

## 目录

```
src/dllm/
├── config.py          超参 dataclass，「不要改」的项在注释里写明理由
├── data/              Countdown 合成
├── rewards/           格式与正确性奖励，AST 安全求值
├── sampling/          块级扩散采样（P4 在此接入 Fast-dLLM）
├── logprob/           单步近似与蒙特卡洛真值（P2 的对照对象）
├── train/             GRPO 损失、策略封装、训练循环、断点续训
├── models/            CPU 单测用的小模型
└── utils/             分阶段计时、指标打点
```

## 参考

- **d1**（UCLA）— masked SFT + diffu-GRPO，[arXiv 2504.12216](https://arxiv.org/abs/2504.12216)
- **Fast-dLLM**（NVIDIA, ICLR'26）— KV cache + 置信度并行解码，[arXiv 2505.22618](https://arxiv.org/abs/2505.22618)
- **LLaDA** — [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
