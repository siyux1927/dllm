"""掩码扩散语言模型的采样循环，实现对齐 LLaDA 官方 generate.py。

生成按块从左到右推进，块内并行去噪：每一步前向整个序列，按置信度选出若干个位置解掩码，
其余继续保持掩码。块级推进是 LLaDA 的半自回归设定，也是后续接 Fast-dLLM 块级 KV cache
的前提。

这里是单步耗时里最大的一段，也是 P4 要动的地方。具体占比取决于 diffusion_steps 与 μ 的
取值，由 `dllm.experiment.step_compute_budget` 推算、P1 实测核对——不要凭印象引用这个数字。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dllm.config import SamplingConfig


@dataclass
class GenerationOutput:
    sequences: torch.Tensor  # (B, P + G) 含 prompt
    completions: torch.Tensor  # (B, G)
    prompt_length: int
    num_forward_passes: int


def add_gumbel_noise(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Gumbel-max 采样。

    LLaDA 官方写作 exp(logits) / (-log u)^T，取 argmax 后等价于 argmax(logits + T*Gumbel)，
    这里直接用后者，数值上更稳。

    全程原地：logits 在采样时是 (B, block, V)，每个同尺寸 fp32 临时张量都是几百 MB，
    照数学式子直写会同时活着五六个。
    """
    if temperature <= 0:
        return logits
    noise = torch.rand(logits.shape, device=logits.device, generator=generator)
    noise.clamp_min_(torch.finfo(torch.float32).tiny)
    noise.log_().neg_().log_().neg_()  # -log(-log u)
    return noise.mul_(temperature).add_(logits)


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """把每条样本待解码的 token 数尽量均匀地分配到各步上。

    返回 (B, steps)，行和等于该样本的掩码总数。除不尽的余数摊在前几步。
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = base.expand(-1, steps).clone()
    step_index = torch.arange(steps, device=mask_index.device).unsqueeze(0)
    num_transfer += (step_index < remainder).long()
    return num_transfer


def _forward_logits(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    if attention_mask is None:
        return model(input_ids).logits
    return model(input_ids, attention_mask=attention_mask).logits


@torch.no_grad()
def generate(
    model,
    prompt_ids: torch.Tensor,
    sampling: SamplingConfig,
    mask_id: int,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """prompt_ids 需为左侧对齐到同一长度的 (B, P)，配套的 attention_mask 用于屏蔽 padding。"""
    device = prompt_ids.device
    batch_size, prompt_length = prompt_ids.shape
    gen_length = sampling.max_completion_length

    x = torch.full(
        (batch_size, prompt_length + gen_length), mask_id, dtype=torch.long, device=device
    )
    x[:, :prompt_length] = prompt_ids

    full_attention_mask = None
    if attention_mask is not None:
        completion_mask = torch.ones(
            (batch_size, gen_length), dtype=attention_mask.dtype, device=device
        )
        full_attention_mask = torch.cat([attention_mask, completion_mask], dim=1)

    num_forward_passes = 0
    for block in range(sampling.num_blocks):
        lo = prompt_length + block * sampling.block_length
        hi = lo + sampling.block_length
        num_transfer = get_num_transfer_tokens(
            x[:, lo:hi] == mask_id, sampling.steps_per_block
        )

        for step in range(sampling.steps_per_block):
            block_mask = x[:, lo:hi] == mask_id
            if not block_mask.any():
                break

            logits = _forward_logits(model, x, full_attention_mask)
            num_forward_passes += 1

            # 块外位置的 logits 一律用不上（下面的 confidence 只在块内取值），
            # 切块让噪声与 softmax 的 fp32 临时张量小 8 倍，这是采样阶段的显存主项
            block_logits = logits[:, lo:hi, :]
            x0 = add_gumbel_noise(block_logits, sampling.temperature, generator).argmax(dim=-1)

            if sampling.remasking == "low_confidence":
                probs = F.softmax(block_logits.float(), dim=-1)
                confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            else:
                confidence = torch.rand(
                    x0.shape, device=device, generator=generator, dtype=torch.float32
                )
            del logits, block_logits

            # 只允许解当前仍是掩码的位置
            confidence = confidence.masked_fill(~block_mask, float("-inf"))

            for row in range(batch_size):
                k = int(num_transfer[row, step])
                if k <= 0:
                    continue
                available = int(torch.isfinite(confidence[row]).sum())
                k = min(k, available)
                if k <= 0:
                    continue
                selected = torch.topk(confidence[row], k=k).indices
                x[row, lo + selected] = x0[row, selected]

    return GenerationOutput(
        sequences=x,
        completions=x[:, prompt_length:],
        prompt_length=prompt_length,
        num_forward_passes=num_forward_passes,
    )
