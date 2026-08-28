"""加载 LLaDA-8B-Instruct 并装配成本项目的 Policy。

这里的每个校验都是为了把「静默失效」变成「显式报错」。dLLM 的训练一跑就是几小时 A100，
配错了却照常跑完是最贵的失败模式：

- `resolve_target_modules` 确认 LoRA 的目标模块名真的能匹配上。LLaDA 派生自 OLMo，
  注意力输出叫 attn_out 而非 o_proj，FFN 是 ff_proj / up_proj / ff_out 而非
  gate/up/down_proj。名字配错时 peft 只在「一个都不命中」时报错，部分命中会静默跳过。
- `lm_head_exclusion` 把和 FFN 下投影重名的词表投影（都叫 ff_out）排除在 LoRA 之外。
- `probe_padding_invariance` 确认左侧 padding 不会污染真实位置的输出。LLaDA 官方的
  generate 是按单条或等长 prompt 写的，从没验证过带 padding 的批处理。
- `probe_lora_is_live` 确认适配器真的参与前向（P0 阶段被这个坑过一次）。
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from dllm.config import Config
from dllm.train.policy import DEFAULT_ADAPTER, OLD_ADAPTER, Policy

ALL_LINEAR = "all-linear"

# Llama 命名 → LLaDA 命名。d1 官方按 Llama 那套配置 LoRA，右边这三个于是全部落空。
LLAMA_TO_LLADA = {
    "o_proj": "attn_out",
    "gate_proj": "ff_proj",
    "down_proj": "ff_out",
}


@dataclass
class LLaDABundle:
    policy: Policy
    tokenizer: object
    codec: LLaDAPromptCodec
    encode: Callable[[Sequence[str]], tuple[torch.Tensor, torch.Tensor]]
    decode: Callable[[torch.Tensor], list[str]]
    mask_id: int
    eos_token_id: int | None
    target_modules: object


def linear_module_suffixes(model: nn.Module) -> Counter:
    """统计模型里所有 nn.Linear 的名字末段，用来发现真实的模块命名。"""
    counter: Counter = Counter()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            counter[name.rsplit(".", 1)[-1]] += 1
    return counter


def resolve_target_modules(model: nn.Module, configured: Sequence[str] | str):
    """校验配置里的 target_modules 能匹配上模型，匹配不上就带着候选名单报错。"""
    if isinstance(configured, str):
        if configured == ALL_LINEAR:
            return ALL_LINEAR
        configured = [configured]

    available = linear_module_suffixes(model)
    matched = [name for name in configured if name in available]
    missing = [name for name in configured if name not in available]

    if not matched:
        raise ValueError(
            f"LoRA target_modules {list(configured)} 在模型里一个都匹配不上。\n"
            f"该模型的线性层命名为: {dict(available)}\n"
            f"请改配置，或改用 '{ALL_LINEAR}'。"
        )
    if missing:
        hint = ""
        renames = {
            name: LLAMA_TO_LLADA[name]
            for name in missing
            if LLAMA_TO_LLADA.get(name) in available
        }
        if renames:
            pairs = "、".join(f"{old} → {new}" for old, new in renames.items())
            hint = f"\n看起来是用了 Llama 的命名，LLaDA 对应的是: {pairs}"
        raise ValueError(
            f"LoRA target_modules 只匹配上一部分，命中 {matched}，缺失 {missing}。\n"
            f"该模型的线性层命名为: {dict(available)}{hint}\n"
            f"部分命中比全不命中更危险，会得到一个和预期不同却照常训练的模型。"
        )
    return list(matched)


def output_embedding_name(model: nn.Module) -> str | None:
    """词表投影模块的完整名字，找不到就返回 None。

    LLaDA 里它叫 transformer.ff_out，和每个块里的 FFN 下投影 blocks.N.ff_out 重名。
    PEFT 按名字末段匹配，target_modules 写 "ff_out" 会把这个 [d_model, vocab] 的大矩阵
    一并挂上 LoRA——r=128 时多出约 1700 万可训练参数，而且训练的是输出分布本身，
    与「在每个块的线性层上做低秩适配」是两回事。d1 也没有训它。
    """
    head = getattr(model, "get_output_embeddings", lambda: None)()
    if head is None:
        return None
    for name, module in model.named_modules():
        if module is head:
            return name
    return None


def lm_head_exclusion(model: nn.Module, targets: Sequence[str] | str) -> str | None:
    """target_modules 会连带命中词表投影时，返回一条只匹配它的正则交给 PEFT 排除。

    必须用正则而不是名字列表。PEFT 的两种排除语义差别很大：
    列表按名字末段匹配（`key.endswith("." + item)`），字符串按 `re.fullmatch`。
    走列表就无法表达「只排这一个模块」——填 "ff_out" 会把每个块的 FFN 下投影一起排掉，
    等于 FFN 压根没挂适配器；填完整路径又只在词表投影恰好嵌套得够深时才安全。
    全匹配正则与模块在树里的深浅无关，两种情形都对。
    """
    name = output_embedding_name(model)
    if name is None:
        return None
    suffix = name.rsplit(".", 1)[-1]
    hit = targets == ALL_LINEAR or (not isinstance(targets, str) and suffix in targets)
    return re.escape(name) if hit else None


class LLaDAPromptCodec:
    """把 Countdown 的题面转成模型输入。

    左侧 padding：扩散采样的补全紧接在 prompt 之后，右侧 padding 会把掩码区推到序列中间。
    """

    def __init__(self, tokenizer, max_prompt_length: int, use_chat_template: bool = True) -> None:
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.use_chat_template = use_chat_template and hasattr(tokenizer, "apply_chat_template")
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        if self.pad_token_id is None:
            raise ValueError("tokenizer 既没有 pad_token_id 也没有 eos_token_id，无法左侧补齐")

    def _render(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def encode(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        rows, masks = [], []
        for prompt in prompts:
            ids = self.tokenizer(self._render(prompt), add_special_tokens=False)["input_ids"]
            # 超长时保留末尾：题面的关键信息（数字与目标）在后半段
            ids = ids[-self.max_prompt_length :]
            pad = self.max_prompt_length - len(ids)
            rows.append([self.pad_token_id] * pad + ids)
            masks.append([0] * pad + [1] * len(ids))
        return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> list[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)


@torch.no_grad()
def probe_padding_invariance(
    model: nn.Module,
    codec: LLaDAPromptCodec,
    prompt: str,
    mask_id: int,
    completion_length: int = 16,
) -> float:
    """同一条 prompt 补上不同长度的 padding，比较真实位置上 logits 的最大差异。

    返回值应接近 0。明显大于 0 有两种成因，都会影响结果但严重程度不同：

    1. **padding 泄漏进注意力**。attention_mask 没起作用，真实 token 看到了 padding。
       这是硬伤，必须修。
    2. **位置编码随 padding 平移**。用可学习的绝对位置嵌入时，左侧补齐会把真实 token
       整体后移，位置嵌入随之改变。RoPE 只依赖相对位置，不受影响。

    本项目把所有 prompt 补齐到固定的 max_prompt_length（而非批内最大长度），
    所以即便存在成因 2，同一条 prompt 的 padding 量也始终相同，结果仍可复现；
    代价只是短 prompt 被推到靠后的绝对位置。成因 1 则没有这种回旋余地。
    """
    ids = codec.tokenizer(codec._render(prompt), add_special_tokens=False)["input_ids"]
    device = next(model.parameters()).device

    def run(pad_len: int) -> torch.Tensor:
        row = [codec.pad_token_id] * pad_len + list(ids)
        mask = [0] * pad_len + [1] * len(ids)
        x = torch.tensor([row + [mask_id] * completion_length], device=device)
        attn = torch.tensor([mask + [1] * completion_length], device=device)
        logits = model(x, attention_mask=attn).logits
        return logits[:, -completion_length:, :].float()

    short = run(1)
    long = run(1 + max(8, len(ids) // 2))
    return (short - long).abs().max().item()


@torch.no_grad()
def probe_lora_is_live(policy: Policy, sample_ids: torch.Tensor) -> bool:
    """确认 LoRA 真的参与前向。

    P0 阶段踩过的坑：nn.TransformerEncoderLayer 的融合快速路径会绕过被包装的子模块，
    适配器挂得上、参数也在更新，但对输出毫无影响，训练看着一切正常。
    """
    baseline = policy.model(sample_ids).logits.clone()
    touched = []
    for name, param in policy.model.named_parameters():
        if "lora_B" in name and DEFAULT_ADAPTER in name:
            param.add_(1.0)
            touched.append(param)
    if not touched:
        raise ValueError("模型里没有 default 适配器的 lora_B 参数，LoRA 没有真正装上")
    changed = not torch.allclose(baseline, policy.model(sample_ids).logits)
    for param in touched:
        param.sub_(1.0)
    return changed


def load_llada(
    config: Config,
    device: str | torch.device = "cuda",
    validate: bool = True,
) -> LLaDABundle:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    model_cfg = config.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_path, trust_remote_code=model_cfg.trust_remote_code
    )
    base = AutoModel.from_pretrained(
        model_cfg.model_path,
        trust_remote_code=model_cfg.trust_remote_code,
        torch_dtype=getattr(torch, model_cfg.torch_dtype),
        attn_implementation=model_cfg.attn_implementation,
    ).to(device)

    target_modules = resolve_target_modules(base, model_cfg.lora_target_modules)
    lora = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=target_modules,
        exclude_modules=lm_head_exclusion(base, target_modules),
    )
    model = get_peft_model(base, lora)
    # θ_old 是第二个适配器而非独立模型副本，省掉一份 16GB 权重
    model.add_adapter(OLD_ADAPTER, lora)
    model.set_adapter(DEFAULT_ADAPTER)

    policy = Policy(model, is_peft=True)
    codec = LLaDAPromptCodec(tokenizer, config.sampling.max_prompt_length)

    if validate:
        sample = torch.full((1, 8), model_cfg.mask_token_id, dtype=torch.long, device=device)
        if not probe_lora_is_live(policy, sample):
            raise RuntimeError(
                "LoRA 已挂载但改动适配器权重不影响输出，适配器是死的。"
                "多半是模型前向绕过了被包装的子模块。"
            )

    return LLaDABundle(
        policy=policy,
        tokenizer=tokenizer,
        codec=codec,
        encode=codec.encode,
        decode=codec.decode,
        mask_id=model_cfg.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        target_modules=target_modules,
    )
