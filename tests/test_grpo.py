import pytest
import torch

from dllm.train.grpo import compute_advantages, grpo_loss


def test_advantages_are_zero_mean_within_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    adv = compute_advantages(rewards, num_generations=3, scale_rewards=False)
    assert torch.allclose(adv.view(-1, 3).mean(dim=1), torch.zeros(2), atol=1e-6)


def test_advantages_without_scaling_are_centered_rewards():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    adv = compute_advantages(rewards, num_generations=4, scale_rewards=False)
    assert torch.allclose(adv, torch.tensor([-1.5, -0.5, 0.5, 1.5]))


def test_advantages_with_scaling_are_normalized():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    adv = compute_advantages(rewards, num_generations=4, scale_rewards=True)
    assert adv.std(unbiased=True).item() == pytest.approx(1.0, abs=0.01)


def test_uniform_group_gives_zero_advantage():
    """组内奖励全同时优势必须是 0，不能因为除以零标准差变成 NaN。"""
    rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])
    adv = compute_advantages(rewards, num_generations=4, scale_rewards=True)
    assert torch.isfinite(adv).all()
    assert torch.allclose(adv, torch.zeros(4), atol=1e-6)


def test_advantages_reject_misaligned_group_size():
    with pytest.raises(ValueError, match="不能被"):
        compute_advantages(torch.zeros(5), num_generations=2)


def _inputs(batch=4, length=6):
    torch.manual_seed(0)
    logp_old = -torch.rand(batch, length)
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])[:batch]
    mask = torch.ones(batch, length)
    return logp_old, advantages, mask


def test_loss_is_zero_when_policy_unchanged_and_advantages_cancel():
    logp_old, _, mask = _inputs()
    advantages = torch.tensor([1.0, -1.0, 2.0, -2.0])
    loss, stats = grpo_loss(
        logp_old.clone(), logp_old, advantages, mask, epsilon=0.5, beta=0.0
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert stats.ratio_mean == pytest.approx(1.0, abs=1e-6)
    assert stats.ratio_out_of_range_frac == 0.0


def test_loss_is_differentiable():
    logp_old, advantages, mask = _inputs()
    logp_new = logp_old.clone().requires_grad_(True)
    loss, _ = grpo_loss(logp_new, logp_old, advantages, mask, epsilon=0.5, beta=0.0)
    loss.backward()
    assert logp_new.grad is not None
    assert torch.isfinite(logp_new.grad).all()


def test_positive_advantage_pushes_logprob_up():
    logp_old = torch.full((1, 3), -1.0)
    logp_new = logp_old.clone().requires_grad_(True)
    loss, _ = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), torch.ones(1, 3), epsilon=0.5, beta=0.0
    )
    loss.backward()
    assert (logp_new.grad < 0).all(), "优势为正时应通过降低损失来提高该 token 的 log-prob"


def test_clipping_activates_for_large_ratio():
    logp_old = torch.zeros(1, 4)
    logp_new = torch.full((1, 4), 2.0)  # ratio = e^2 ≈ 7.39，远超 1+ε
    _, stats = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), torch.ones(1, 4), epsilon=0.5, beta=0.0
    )
    assert stats.ratio_out_of_range_frac == 1.0
    assert stats.clip_active_frac == 1.0


def test_wider_epsilon_clips_less():
    """ε 从 0.2 放宽到 0.5 的直接效果，也是官方那样设置的动机。"""
    logp_old = torch.zeros(1, 64)
    logp_new = torch.randn(1, 64) * 0.3
    _, tight = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), torch.ones(1, 64), epsilon=0.2, beta=0.0
    )
    _, wide = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), torch.ones(1, 64), epsilon=0.5, beta=0.0
    )
    assert wide.ratio_out_of_range_frac < tight.ratio_out_of_range_frac


def test_kl_term_is_non_negative_and_increases_loss():
    logp_old, advantages, mask = _inputs()
    logp_new = logp_old.clone()
    logp_ref = logp_old - 0.5
    base, _ = grpo_loss(logp_new, logp_old, advantages, mask, epsilon=0.5, beta=0.0)
    with_kl, stats = grpo_loss(
        logp_new, logp_old, advantages, mask, epsilon=0.5, beta=0.04, logp_ref=logp_ref
    )
    assert stats.kl > 0
    assert with_kl.item() > base.item()


def test_kl_is_zero_when_policy_equals_reference():
    logp_old, advantages, mask = _inputs()
    _, stats = grpo_loss(
        logp_old.clone(),
        logp_old,
        advantages,
        mask,
        epsilon=0.5,
        beta=0.04,
        logp_ref=logp_old.clone(),
    )
    assert stats.kl == pytest.approx(0.0, abs=1e-7)


def test_beta_without_reference_raises():
    logp_old, advantages, mask = _inputs()
    with pytest.raises(ValueError, match="必须提供 logp_ref"):
        grpo_loss(logp_old.clone(), logp_old, advantages, mask, epsilon=0.5, beta=0.04)


def test_gradient_flows_only_into_the_current_policy():
    """θ_old 与 θ_ref 在目标函数里是常数。

    若梯度能流进它们，损失会额外获得一条「把 logp_old 压低」的下降路径——ratio 照样变大、
    loss 照样下降、训练照常跑完，优化的却不是 GRPO 的目标。
    """
    logp_old, advantages, mask = _inputs()
    logp_old = logp_old.clone().requires_grad_(True)
    logp_ref = torch.randn_like(logp_old).requires_grad_(True)
    logp_new = torch.randn_like(logp_old).requires_grad_(True)

    loss, _ = grpo_loss(
        logp_new, logp_old, advantages, mask, epsilon=0.5, beta=0.04, logp_ref=logp_ref
    )
    loss.backward()

    assert logp_new.grad is not None, "梯度必须流向 θ"
    assert logp_old.grad is None, "梯度不应流向 θ_old"
    assert logp_ref.grad is None, "梯度不应流向 θ_ref"


def test_completion_mask_excludes_padded_positions():
    logp_old = torch.zeros(1, 4)
    logp_new = torch.tensor([[0.0, 0.0, 5.0, 5.0]])
    full = torch.ones(1, 4)
    partial = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    _, masked_stats = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), partial, epsilon=0.5, beta=0.0
    )
    _, full_stats = grpo_loss(
        logp_new, logp_old, torch.tensor([1.0]), full, epsilon=0.5, beta=0.0
    )
    assert masked_stats.ratio_mean == pytest.approx(1.0, abs=1e-6)
    assert full_stats.ratio_mean > 1.0
