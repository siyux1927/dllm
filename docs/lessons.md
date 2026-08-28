# 踩过的坑

`nn.TransformerEncoderLayer` 在 eval 模式下走融合快速路径，**绕过** `linear1` **/** `linear2`
**子模块**，挂在上面的 LoRA 会被静默忽略——前向照常返回结果，训练看着也在跑，但适配器是死的。
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

## 由此衍生出的三条自检

- `resolve_target_modules`：LLaDA 派生自 OLMo，注意力输出叫 `attn_out` 而非 `o_proj`，
FFN 是 `ff_proj` / `up_proj` / `ff_out` 而非 `gate/up/down_proj`。匹配不上就带着模型真实的
模块名单报错；**只匹配上一部分同样报错**——部分命中比全不命中更危险，它会给你一个和预期
不同却照常训练的模型。d1 官方正是照 Llama 命名配置的，实际只挂上了 q/k/v/up 四类。
- `probe_padding_invariance`：已经查出结论——**LLaDA 直接忽略** `attention_mask`
（`forward` 里算完加性 bias 随即 `attention_bias = None`），真实 token 会注意到 padding。
补齐到固定长度使每行 padding 量恒定，可复现性不受影响；d1 官方同样如此，保持一致
才可比，故不修（取舍见 plan 第 2 节）。探针留着的职责已经从「发现问题」变成
「发现变化」：换模型、上游补上掩码、或 codec 改成按批内最大长度补齐，这个数字会动。
- `Policy.trainable_parameters` **只返回** `default` **适配器**。peft 会把 `old` 适配器一并
交出来，眼下靠 `as_old` 里的 `no_grad` 兜底才没出错——但那样「θ_old 会不会被优化」
就取决于一个远处的上下文管理器。`grpo_loss` 里也对 θ_old、θ_ref 显式 `detach`。
