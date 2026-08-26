# dLLM 项目候选方案

> 目标：5 小时内完成的校招作品集项目，方向为 diffusion（落在扩散语言模型 dLLM 上，而非图像生成）。
> 约束：Colab A100（40GB）、成本可控（单方案 GPU 花费 $2–5）、必须有完整故事线而非跑通 demo。
> 本地 `myenv`（conda, Python 3.12）只负责写代码与 CPU 单测，A100 只跑实验。

---

## 为什么是 dLLM 而不是文生图

岗位是**大模型**算法工程师，文生图项目在校招中同质化严重且与 JD 关联弱。dLLM 是 2025–2026 大厂真正投入的方向（Gemini Diffusion、Inception Mercury 宣称 1000+ tok/s），且开源侧模型、训练代码、评测栈都已齐备，5 小时内可以做出有结论的东西。

这个方向自带一个可以当故事钩子的矛盾：

> dLLM 的卖点是"并行解码所以快"，但开源 dLLM 的实测吞吐反而低于同规模自回归模型。

---

## 方案总览

| # | 方案 | 主战场 | 5h 可行性 | GPU 成本 | 差异化 |
|---|---|---|---|---|---|
| 1 | AR → dLLM 权重转换 | 预训练 / 训练 | 高 | ~$2 | 高 |
| 2 | masked SFT + diffu-GRPO | 后训练 / RL | 低 | ~$5 | 中 |
| 3 | 解码策略与重掩码机制 | 推理算法 | 高 | ~$3 | 中 |
| 4 | dLLM 护城河：填充与约束生成 | 业务场景 | 高 | ~$3 | 很高 |
| 5 | dLLM 当合成数据引擎 | 数据 | 中 | ~$4 | 中 |
| 6 | 推理加速与上线决策 | 部署 | 高 | ~$3 | 中 |

---

## 方案 1：把自回归模型改造成扩散模型

**定位**：唯一一个能把数据、训练、推理串成一条线的方案。

**要回答的问题**：AR 预训练权重迁移到 diffusion 目标，到底省了多少训练量？

**背景**：从头训 dLLM 要 LLaDA 2.3T token / Dream 580B token。DiffuLLaMA（ICLR'25）证明可以拿现成 AR 权重继续预训练，200B token 以内完成转换。

**改造三处**：

1. **目标函数**：next-token CE → masked denoising。每条样本采 mask 比例 `t ~ U(0,1)`，只在被掩位置算 CE，loss 按 `1/t` 加权（ELBO 推导而来）。
2. **注意力**：causal → bidirectional，用 **mask annealing** 逐步放开右侧可见上下文，避免模型花算力"忘掉"因果偏置。
3. **shift 操作**：继承 AR 的位置对齐，让适配初期训练动力学接近原模型。

**核心实验**：固定 5000 万 token 预算跑四条曲线 —— AR-init + annealing / AR-init + 直接双向 / 随机 init / AR-init + 冻结底层。

**预期结论**：AR 初始化在同预算下达到的 PPL，随机初始化需要 N 倍 token 才能追平；annealing 在训练前段有明显优势。

**成本**：Qwen2.5-0.5B 在 A100 上跑 5000 万 token 约 10–15 分钟，四条曲线一个多小时。

**风险**：5000 万 token 训出的 0.5B 模型生成质量必然很差。主图必须是 loss/PPL 对照曲线，生成样例只作附录。

---

## 方案 2：后训练 —— masked SFT + diffu-GRPO

**定位**：最热但 5 小时内大概率跑不完的方案。

**技术看点**：GRPO 需要序列 log-prob，但 dLLM 没有自回归分解，算不出来。d1（UCLA）用 mean-field 近似 + 随机 prompt masking 解决，副作用是随机性变成正则，允许每批数据做更多梯度更新，反而降低在线采样成本。

**问题**：GRPO 要在线采样，而 dLLM 采样本身就慢。5 小时内跑不出 reward 上升曲线。

**降级方案（推荐）**：
- 只做 masked SFT（LoRA + GSM8K 推理轨迹，1–2 小时可完成）
- diffu-GRPO 的 log-prob 估计器**实现出来并做正确性验证**：对比 mean-field 估计 vs 蒙特卡洛真值，画偏差散点图
- 不跑完整 RL 训练，但展示对算法的理解

---

## 方案 3：解码策略与重掩码机制

**定位**：推理侧的算法研究，不是部署工程。

**做什么**：
- 系统对比 unmasking policy：random / low-confidence / entropy / top-2 margin / 误差预算式自适应
- **误差预算法**（自己的改进点）：每步按置信度排序，选最大的 `n` 使 `Σ(1 - p_i) ≤ ε`，即"本步期望解错的 token 数不超过 ε"。相比 Fast-dLLM 的固定全局阈值 τ，它显式考虑了并行 token 数 n，而固定 τ 恰恰忽略了这一点。
- **生成顺序分析**：模型实际按什么顺序解码？与文本结构的关系，可视化
- **可重掩码（ReMDM 式）**：允许已解开的 token 被撤销重掩，这是 AR 没有的纠错机制

**产出**：质量-速度 Pareto 前沿图 + 解码顺序热力图。

---

## 方案 4：不在自回归的主场上打

**定位**：洞察最强、业务感最好的方案。

**核心主张（反直觉）**：所有人卷 dLLM 都在卷速度，但速度是 AR 的主场（vLLM 生态已优化到牙齿）。**dLLM 真正无法被替代的是双向注意力带来的任意顺序生成。**

**场景一 —— 中间填充（FIM）**：AR 做代码光标补全必须专门训 FIM special token 或把 prompt 重排（后缀在前），都是 hack。dLLM 直接把中间设成 mask 即可原生支持（DiffuLLaMA 论文明确提到 "filling in the middle without prompt re-ordering"），且能**精确指定填充长度**，AR 只能靠 stop token 碰运气。

**场景二 —— 结构化约束生成**：让 AR 输出合法 JSON，工业做法是 grammar-constrained decoding，代价是运行时开销 + **约束扭曲输出分布**。dLLM 只需把模板固定部分（括号、key、引号）锁成已知 token，只对 value 位置去噪 —— 格式合法率天然 100%，且不扭曲分布。

**故事线**：先用数据承认 dLLM 在通用生成上速度和质量都打不过 AR → 论证护城河在双向性 → 构造 FIM 与 JSON 抽取两个 benchmark → 证明在这两个场景 dLLM 用更低复杂度拿到更好结果 → 给出"哪类业务该考虑 dLLM"的判断。

**风险**：benchmark 自建，容易被质疑挑了对自己有利的题。AR 对照组必须用 Qwen2.5-Coder 的原生 FIM 能力，不能用弱 baseline。

---

## 方案 5：dLLM 当高吞吐合成数据引擎

**定位**：数据侧，业务映射是数据飞轮。

**做什么**：同等 GPU-hour 预算下，对比 dLLM 与 AR 生成合成指令数据的数量、质量、多样性；有余力再用生成的数据 SFT 一个小模型验证下游收益。

**风险**：下游验证环节大概率超时需裁掉；且结论可能是"dLLM 不划算"——虽然也是有效结论，但不好看。

---

## 方案 6：推理加速与上线决策

**定位**：偏部署工程，可行性最高但差异化一般。

**故事线**：

```
第0幕  暴露问题：LLaDA-8B 实测 TPS vs Qwen2.5-7B AR，dLLM 更慢 → 卖点不成立
第1幕  定位瓶颈：逐步耗时 profiling，量化"重复前向 prompt"占多少 FLOPs
第2幕  工程解法：block-wise 近似 KV cache（PrefixCache / DualCache）→ ~3x
第3幕  算法解法：固定置信度阈值并行解码 → 再 ~3x，但精度开始掉
第4幕  自己的改进：误差预算式自适应阈值 → 同精度下更快
第5幕  业务结论：扫 batch × 输出长度，给出"什么场景该上 dLLM"决策矩阵
```

**风险**：第 2 幕的双向注意力近似 KV cache 有实现坑（mask 对齐、block 边界）。兜底是复用 Fast-dLLM 官方 cache 实现，精力全押第 4 幕。

---

## 可用模型清单

| 模型 | 规模 | 用途 | 备注 |
|---|---|---|---|
| `diffusionfamily/diffugpt-s` | 127M | 方案 1/2 的小型试验台 | 官方另放了 GSM8K-symbolic LoRA |
| `diffusionfamily/diffugpt-m` | 355M | 同上 | |
| `inclusionAI/LLaDA-MoE-7B-A1B-Instruct` | 7B 总参 / 1.4B 激活 | 方案 2/3/4/6 的主力 | FLOPs 约为 LLaDA-8B 的 1/5，能力对标 Qwen2.5-3B-Instruct |
| `GSAI-ML/LLaDA-8B-Instruct` | 8B | 对照 / 复现基线 | bf16 约 16GB |
| `Dream-org/Dream-v0-Base-7B` | 7B | 第二个 dLLM 对照 | |
| `Qwen2.5-0.5B` | 0.5B | 方案 1 的转换起点 | |
| `Qwen2.5-Coder-7B` | 7B | 方案 4 的 AR FIM 对照 | 有原生 FIM 支持 |

## 成本控制手段

- HF 缓存挂载到 Google Drive，避免每次 session 重下十几 GB 权重
- 评测集抽样（GSM8K 取 200 条），生成长度控制在 256
- 全程固定 seed，实验可复现
- 优先选 MoE 模型（激活参数少）而非 dense 8B
- A100 抢不到时的降级预案：L4（24GB）+ gen_length 128 + 100 条样本

## 关键参考

- Fast-dLLM（NVIDIA, ICLR'26）— KV cache + 置信度并行解码，arXiv 2505.22618
- Fast-dLLM v2 — block diffusion 适配 + 分层缓存，arXiv 2509.26328
- DiffuLLaMA / DiffuGPT（HKUNLP, ICLR'25）— AR→dLLM 转换，arXiv 2410.17891
- d1（UCLA）— masked SFT + diffu-GRPO，arXiv 2504.12216
- LLaDA-MoE（inclusionAI）— MoE 架构 dLLM，arXiv 2509.24389

## 决策记录

- [x] **选定方案 2**（后训练 / RL），并与方案 6 的推理加速交叉：把 Fast-dLLM 接进 diffu-GRPO 的 rollout 环节。详见 `docs/plan-diffu-grpo.md`
- [x] 用 dense `LLaDA-8B-Instruct`，不用 MoE —— MoE 省的是计算不是显存（7B 权重 bf16 仍需约 14GB），而 d1 官方代码按 dense 写，适配成本高于收益
- [x] 放宽时间约束：从「5 小时」改为「12–16 小时墙钟，可跨 Colab session」
- [ ] superpowers skill 是否安装
