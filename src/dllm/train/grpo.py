"""diffu-GRPO 的优势估计与损失。

与标准 GRPO 的差别只在 log-prob 从哪来：AR 模型靠链式法则，这里靠 `dllm.logprob` 的
单步近似。近似带来的 ratio 噪声正是官方把 clip 范围 ε 放到 0.5 的原因，
所以 `ratio_out_of_range_frac` 这个统计量必须记录——它是 P2 分析的直接证据。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass
class GRPOStats:
    loss: float
    policy_loss: float
    kl: float
    ratio_mean: float
    ratio_std: float
    ratio_out_of_range_frac: float  # ratio 落在 [1-ε, 1+ε] 之外的比例
    clip_active_frac: float  # 真正被 clip 分支接管的比例
    advantage_abs_mean: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_advantages(
    rewards: torch.Tensor,
    num_generations: int,
    scale_rewards: bool = True,
) -> torch.Tensor:
    """组内归一化的优势。

    rewards 形状 (num_prompts * num_generations,)，同一 prompt 的若干条补全必须相邻。
    scale_rewards 对应 TRL GRPOTrainer 的同名参数，默认 True；注意 d1 论文式 2 写的是
    只减均值、不除标准差。
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards 应为一维，收到 {tuple(rewards.shape)}")
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            f"rewards 数量 {rewards.numel()} 不能被 num_generations {num_generations} 整除"
        )

    grouped = rewards.view(-1, num_generations).float()
    advantages = grouped - grouped.mean(dim=1, keepdim=True)
    if scale_rewards:
        # 退化组必须显式置零，不能靠 eps 兜除零：分子是 float32 舍入残差（0.2 附近约 2^-26），
        # 除以 1e-4 会把它放大一万倍，于是 zero_advantage_frac 的 1e-8 判据漏报整步空转
        degenerate = (grouped == grouped[:, :1]).all(dim=1, keepdim=True)
        scaled = advantages / (grouped.std(dim=1, keepdim=True) + 1e-4)
        advantages = torch.where(degenerate, torch.zeros_like(scaled), scaled)
    return advantages.reshape(-1)


def grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    epsilon: float,
    beta: float = 0.0,
    logp_ref: torch.Tensor | None = None,
) -> tuple[torch.Tensor, GRPOStats]:
    """返回标量损失与诊断统计。

    logp_new / logp_old / logp_ref 形状 (B, C)，且必须是在**同一个** prompt 掩码模式 q'
    上算出来的——论文式 4 中 q' 位于期望之内，三者共享。
    """
    if beta > 0 and logp_ref is None:
        raise ValueError("beta > 0 时必须提供 logp_ref")

    # θ_old 与 θ_ref 在目标函数里是常数。调用方目前已在 no_grad 下取它们，这里再断一次，
    # 使得「梯度只流向 θ」这条性质由损失函数自身保证，而不依赖调用方的上下文管理器。
    logp_old = logp_old.detach()
    if logp_ref is not None:
        logp_ref = logp_ref.detach()

    advantages = advantages.unsqueeze(1)
    mask = completion_mask.float()
    token_count = mask.sum(dim=1).clamp_min(1.0)

    log_ratio = logp_new - logp_old
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_token_loss = -torch.min(unclipped, clipped)

    token_loss = policy_token_loss
    kl_value = torch.zeros((), device=logp_new.device)
    if beta > 0:
        # k3 估计量，非负且方差低于朴素的 (ref - new)
        log_ratio_ref = logp_ref - logp_new
        kl_token = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
        token_loss = token_loss + beta * kl_token
        kl_value = (kl_token * mask).sum(dim=1).div(token_count).mean()

    loss = (token_loss * mask).sum(dim=1).div(token_count).mean()

    with torch.no_grad():
        policy_loss = (policy_token_loss * mask).sum(dim=1).div(token_count).mean()
        masked_ratio = ratio[mask.bool()]
        if masked_ratio.numel() == 0:
            ratio_mean = ratio_std = out_of_range = 0.0
        else:
            ratio_mean = masked_ratio.mean().item()
            ratio_std = masked_ratio.std().item() if masked_ratio.numel() > 1 else 0.0
            out_of_range = (
                ((masked_ratio < 1.0 - epsilon) | (masked_ratio > 1.0 + epsilon))
                .float()
                .mean()
                .item()
            )
        clip_active = ((clipped < unclipped).float() * mask).sum() / mask.sum().clamp_min(1.0)
        stats = GRPOStats(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl=kl_value.item(),
            ratio_mean=ratio_mean,
            ratio_std=ratio_std,
            ratio_out_of_range_frac=out_of_range,
            clip_active_frac=clip_active.item(),
            advantage_abs_mean=advantages.abs().mean().item(),
        )

    return loss, stats
