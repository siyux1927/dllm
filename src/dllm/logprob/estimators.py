"""掩码扩散语言模型的逐 token log-probability 估计。

自回归模型靠链式法则一次前向就能拿到序列 log-prob，扩散模型没有这个分解，这是把 GRPO
搬到 dLLM 上的核心障碍。这里实现两个估计器：

- `one_step_token_logprobs`：d1 的做法。把补全全部置为掩码、prompt 按概率随机掩码，
  做一次前向，用单步去噪的输出当作 log π(o^k | q)。每次梯度更新只花一次前向。
- `monte_carlo_token_logprobs`：LLaDA 官方的做法（默认 128 个样本）。对补全按随机比例
  反复加噪再前向，取被掩码位置的平均。贵得多，但更接近模型真实的条件分布。

**两者估计的不是同一个量**：单步估计条件在「补全全掩码」这一最极端的上下文上，
蒙特卡洛则在各种部分可见的上下文上取平均。这个差异本身就是 P2 要量化的对象，
也是理解为什么官方要把 clip 范围 ε 从 0.2 放宽到 0.5 的关键。

把 `completion_mask_ratio=1.0`、`p_mask_prompt=0.0`、`num_samples=1` 传给蒙特卡洛版本，
它会退化成与单步估计完全相同的计算——这条性质被用作单测。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_prompt_mask(
    prompt_ids: torch.Tensor,
    p_mask_prompt: float,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """采一个 prompt 掩码模式 q'。

    d1 的关键设计：每次梯度更新换一个模式，等价于对同一批 (prompt, completion) 造出
    多个扰动视角，起到正则与数据增强的作用，从而允许把每批的梯度更新次数 μ 从常规的 2
    提到 12，减少昂贵的在线采样次数。

    padding 位置不参与掩码——它们本来就被 attention_mask 屏蔽，掩码它们只会浪费扰动预算。
    """
    if p_mask_prompt <= 0:
        return torch.zeros_like(prompt_ids, dtype=torch.bool)
    noise = torch.rand(prompt_ids.shape, device=prompt_ids.device, generator=generator)
    mask = noise < p_mask_prompt
    if attention_mask is not None:
        mask &= attention_mask.bool()
    return mask


def build_estimation_sequence(
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_id: int,
    prompt_mask: torch.Tensor | None = None,
    completion_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """拼出送进模型的序列：掩码后的 prompt 接掩码后的补全。"""
    prompt = prompt_ids
    if prompt_mask is not None:
        prompt = torch.where(prompt_mask, torch.full_like(prompt_ids, mask_id), prompt_ids)
    completion = completion_ids
    if completion_mask is not None:
        completion = torch.where(
            completion_mask, torch.full_like(completion_ids, mask_id), completion_ids
        )
    return torch.cat([prompt, completion], dim=1)


def _forward_logits(model, input_ids, attention_mask):
    if attention_mask is None:
        return model(input_ids).logits
    return model(input_ids, attention_mask=attention_mask).logits


def _full_attention_mask(
    attention_mask: torch.Tensor | None, completion_length: int
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    ones = torch.ones(
        (attention_mask.shape[0], completion_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, ones], dim=1)


def one_step_token_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_id: int,
    prompt_mask: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """一次前向估计逐 token log-prob，返回 (B, C)。

    `prompt_mask` 由调用方传入而非内部采样，因为 diffu-GRPO 的目标函数要求
    π_θ 与 π_θold 的 log-prob 在**同一个** q' 上计算（论文式 4 中 q' 在期望内共享）。
    """
    completion_length = completion_ids.shape[1]
    all_masked = torch.ones_like(completion_ids, dtype=torch.bool)
    x = build_estimation_sequence(
        prompt_ids, completion_ids, mask_id, prompt_mask, all_masked
    )
    logits = _forward_logits(model, x, _full_attention_mask(attention_mask, completion_length))
    completion_logits = logits[:, prompt_ids.shape[1] :, :]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)
    return log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def monte_carlo_token_logprobs(
    model,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    mask_id: int,
    num_samples: int = 128,
    completion_mask_ratio: float | None = None,
    p_mask_prompt: float = 0.0,
    attention_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """多次前向的蒙特卡洛估计，返回 (估计值 (B, C), 有效样本数 (B, C))。

    `completion_mask_ratio` 为 None 时每条样本各采一个 t ~ U(0, 1] 作为掩码比例；
    给定具体值时固定使用该比例。没被掩过的 token 计数为 0，其估计值置为 0，
    调用方应按计数过滤。
    """
    batch_size, completion_length = completion_ids.shape
    device = completion_ids.device
    total = torch.zeros((batch_size, completion_length), dtype=torch.float32, device=device)
    counts = torch.zeros((batch_size, completion_length), dtype=torch.float32, device=device)
    full_mask = _full_attention_mask(attention_mask, completion_length)

    for _ in range(num_samples):
        if completion_mask_ratio is None:
            ratio = torch.rand((batch_size, 1), device=device, generator=generator).clamp_min(
                1.0 / completion_length
            )
        else:
            ratio = torch.full((batch_size, 1), float(completion_mask_ratio), device=device)
        noise = torch.rand(
            (batch_size, completion_length), device=device, generator=generator
        )
        completion_mask = noise < ratio
        prompt_mask = sample_prompt_mask(
            prompt_ids, p_mask_prompt, attention_mask, generator=generator
        )
        x = build_estimation_sequence(
            prompt_ids, completion_ids, mask_id, prompt_mask, completion_mask
        )
        logits = _forward_logits(model, x, full_mask)
        log_probs = F.log_softmax(logits[:, prompt_ids.shape[1] :, :].float(), dim=-1)
        token_log_probs = log_probs.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        total += token_log_probs * completion_mask
        counts += completion_mask

    estimate = total / counts.clamp_min(1.0)
    return estimate, counts
