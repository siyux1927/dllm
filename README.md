# 把推理加速塞进扩散语言模型的强化学习循环

扩散语言模型（dLLM）的强化学习训练，大半时间耗在在线采样上。把 training-free 的
推理加速接进 rollout 环节能提速，但并行解码会改变采样分布、破坏 on-policy 假设。
本项目量化这个权衡，并找出安全的加速配置。

模型 `LLaDA-8B-Instruct`，任务 Countdown，单卡 A100-40GB。
完整实施方案见 [`docs/plan-diffu-grpo.md`](docs/plan-diffu-grpo.md)，
方向选型过程见 [`docs/proposals.md`](docs/proposals.md)。

## 故事线

| 幕 | 问题 | 状态 |
|---|---|---|
| 0 | dLLM 的 RL 后训练现状：d1 自己承认「在线生成开销过大，因此把生成长度限制在 256」 | — |
| 1 | 耗时到底花在哪？分阶段拆解，定位瓶颈并推出加速的收益上限 | 代码就绪，待上 Colab |
| 2 | diffu-GRPO 的地基是单步 log-prob 近似。它有多准？论文没报告过 | 代码就绪，待上 Colab |
| 3 | 把 Fast-dLLM 的块级 KV cache 与置信度并行解码接进 rollout | 待做 |
| 4 | 加速改变采样分布，引入额外 off-policy 偏差。量化这个权衡 | 待做 |
| 5 | 同等墙钟时间下的 reward 曲线对比，加 GSM8K 回归检查 | 待做 |

第 2 幕补的是 d1 论文的真实空白：论文提到 LLaDA 官方用 128 样本蒙特卡洛估计 log-prob，
并说这对在线 RL 太贵，但从未报告单步近似与真值的偏差。第 3、4 幕是 Fast-dLLM（推理侧）
与 d1（训练侧）的交叉。

## 当前进度

**P0 本地骨架 + P1/P2 的 Colab 入口已完成**，137 个测试全绿，全部可在 CPU 上跑。

已实现：Countdown 合成与去重、格式/正确性分离的奖励函数、块级扩散采样、
单步与蒙特卡洛两种 log-prob 估计器、diffu-GRPO 损失、θ/θ_old/θ_ref 三策略封装、
分阶段计时、指标打点、断点续训、LLaDA 装配与三项自检、单步算力预算推导。

下一步是把 [`notebooks/colab_p1_p2.ipynb`](notebooks/colab_p1_p2.ipynb) 拿到 A100 上跑，
取得第 1、2 幕的实测数据。

> **一个已经浮现的问题。** 算力预算显示，当前配置（`diffusion_steps=64`）下采样只占单步
> 约 52%，而非计划里写的 70–80%——后者对应的是 d1 官方的 128 步。按 Amdahl，
> 采样占 52% 时端到端加速上限只有 2.1x，而且 64 步配 128 长度等于每步解 2 个 token，
> 基线本身已经在并行解码了。P1 会在两个取值上各测一遍再定，详见
> [`docs/plan-diffu-grpo.md`](docs/plan-diffu-grpo.md) 的 P1 一节。

## 快速开始

```bash
conda env create -f environment.yml
conda activate dllm-dev
python scripts/check_env.py     # 校验版本与 trl commit 是否正确
python -m pytest                # 137 个 CPU 测试
python -m ruff check .
```

上 Colab 之前，先在本地 CPU 上把两个脚本的整条路径跑通（数字无意义，只验证接线）：

```bash
python scripts/run_p1_profile.py --config configs/countdown_base.yaml --tiny --steps 2
python scripts/run_p2_logprob.py --config configs/countdown_base.yaml --tiny --mc-samples 16
```

Colab：打开 `notebooks/colab_p1_p2.ipynb`，运行时选 A100。

## 设计上的几个决定

**不复用 TRL 的 GRPOTrainer。** 它的采样是为自回归模型写的，套扩散采样要大幅覆写；
而本项目的核心产出都要求对循环内部有完全控制权。trl 仍锁在 d1 使用的 commit 上作为对照基准。

**参考策略取自关掉 LoRA 的底座**，不额外占显存。官方那套独立 reference model 会再吃 16GB，
单卡 40GB 放不下 8B 模型加两份权重。

**每次内更新都重新采 prompt 掩码模式 q'**，且 θ、θ_old、θ_ref 三次估计共享同一个 q'——
论文式 4 中 q' 位于期望之内，三者必须一致。

**CPU 小模型的结构对齐 LLaDA**，线性层照抄它的命名（`q_proj` / `k_proj` / `v_proj` /
`attn_out` / `ff_proj` / `up_proj` / `ff_out`），连词表投影与 FFN 下投影重名这点也保留。
单测里的 LoRA `target_modules` 因此与真实配置是同一份，适配器挂不上会在本地就暴露。
这条对齐曾经是假的——替身用的是 Llama 命名，于是它什么也没能拦住，见「踩过的坑」。

**每个自检都对应一种「不报错的失效」。** 这类问题在 8B 模型上要烧几小时 A100 才可能发现，
所以尽量在 CPU 上、或在加载权重的第一分钟内就把它们变成显式报错。

**按「Colab 一定会断」来设计存盘。** 掉线是常态不是异常：

- P1/P2 每完成一个阶段就把结果落盘，不等全部跑完。原子写（先写 `.tmp` 再 replace），
  避免写到一半掉线留下截断的 JSON——那会把上一次成功的结果一起毁掉。
- checkpoint 分两级。LoRA `r=128` 在 LLaDA-8B 上是 3.36 亿参数，加 AdamW 两个矩约 **4GB**；
  Drive 写入约 10-20MB/s，存一次要 3-7 分钟，比一个训练步还慢。
  所以本地每 `save_steps` 步存一次（救进程崩溃），每 `mirror_every` 次镜像到 Drive
  （救会话断开）。存盘耗时记进指标，免得它悄悄吃掉大半训练时间。
- `trainer.resume()` 先找本地，本地没有（重连后换机器就是这种情况）再回落到 Drive 镜像。

## 踩过的坑

`nn.TransformerEncoderLayer` 在 eval 模式下走融合快速路径，**绕过 `linear1` / `linear2`
子模块**，挂在上面的 LoRA 会被静默忽略——前向照常返回结果，训练看着也在跑，但适配器是死的。
小模型因此改成显式的 Transformer 块，并留了一条回归测试同时断言「被包装」和「被调用」。
`load_llada` 里的 `probe_lora_is_live` 是同一条检查在真实模型上的版本。

**替身与真身对不上，等于没有替身。** 小模型的线性层原本用 Llama 命名，因为想当然地以为
LLaDA 也是那套。于是「单测里的 target_modules 与真实配置是同一份」这个前提一直不成立，
`resolve_target_modules` 在 CPU 上永远命中、永远不报警，直到租下 A100、下完 16GB 权重才
第一次真正生效。现在小模型照抄 LLaDA 的命名，包括词表投影与 FFN 下投影都叫 `ff_out`
这个坑——PEFT 按名字末段匹配，不显式排除就会把 `[d_model, vocab]` 那个大矩阵一起挂上。

**同一个模式咬了第二次：CPU 测不到「设备」这个维度。** 采样生成器曾写死
`torch.Generator(device="cpu")`，在 GPU 上 `torch.rand(device="cuda", generator=cpu_gen)`
直接报错。更糟的是当时的补法——`generator=self.generator if prompt_ids.is_cpu else None`：
CPU 上照常可复现，GPU 上悄悄改用全局 RNG，`run.seed` 对那段等于失效。崩溃看得见，
这种降级看不见。现在生成器跟随 `self.device` 构造，并用一条拦截构造调用的测试钉住——
不必真有 GPU 也能验证它请求的是哪个设备。

**这类「本地跑不到的维度」只能靠探针补。** 结构差异用替身对齐（模块命名），
行为差异用构造拦截或状态断言（设备、随机数流），两者都做不到时就在真机上留显式自检
（`probe_lora_is_live`、`probe_padding_invariance`）。

由此衍生出的三条自检：

- **`resolve_target_modules`**：LLaDA 派生自 OLMo，注意力输出叫 `attn_out` 而非 `o_proj`，
  FFN 是 `ff_proj` / `up_proj` / `ff_out` 而非 `gate/up/down_proj`。匹配不上就带着模型真实的
  模块名单报错；**只匹配上一部分同样报错**——部分命中比全不命中更危险，它会给你一个和预期
  不同却照常训练的模型。d1 官方正是照 Llama 命名配置的，实际只挂上了 q/k/v/up 四类。
- **`probe_padding_invariance`**：已经查出结论——**LLaDA 直接忽略 `attention_mask`**
  （`forward` 里算完加性 bias 随即 `attention_bias = None`），真实 token 会注意到 padding。
  补齐到固定长度使每行 padding 量恒定，可复现性不受影响；d1 官方同样如此，保持一致
  才可比，故不修（取舍见 plan 第 2 节）。探针留着的职责已经从「发现问题」变成
  「发现变化」：换模型、上游补上掩码、或 codec 改成按批内最大长度补齐，这个数字会动。
- **`Policy.trainable_parameters` 只返回 `default` 适配器**。peft 会把 `old` 适配器一并
  交出来，眼下靠 `as_old` 里的 `no_grad` 兜底才没出错——但那样「θ_old 会不会被优化」
  就取决于一个远处的上下文管理器。`grpo_loss` 里也对 θ_old、θ_ref 显式 `detach`。

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
├── check_env.py       版本与 trl commit 校验
├── run_p1_profile.py  P1 耗时拆解（--tiny 可在 CPU 上跑通路径）
└── run_p2_logprob.py  P2 log-prob 误差量化（同上）

notebooks/colab_p1_p2.ipynb   Colab 入口
```

## 参考

- **d1**（UCLA）— masked SFT + diffu-GRPO，[arXiv 2504.12216](https://arxiv.org/abs/2504.12216)
- **Fast-dLLM**（NVIDIA, ICLR'26）— KV cache + 置信度并行解码，[arXiv 2505.22618](https://arxiv.org/abs/2505.22618)
- **LLaDA** — [arXiv 2502.09992](https://arxiv.org/abs/2502.09992)
