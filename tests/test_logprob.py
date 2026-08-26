import torch

from dllm.logprob.estimators import (
    build_estimation_sequence,
    monte_carlo_token_logprobs,
    one_step_token_logprobs,
    sample_prompt_mask,
)


def _completions(batch: int, length: int, mask_id: int) -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randint(0, mask_id, (batch, length))


def test_prompt_mask_is_empty_at_zero_probability(prompt_ids):
    assert not sample_prompt_mask(prompt_ids, 0.0).any()


def test_prompt_mask_rate_is_roughly_p():
    ids = torch.zeros(64, 128, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    rate = sample_prompt_mask(ids, 0.15, generator=g).float().mean().item()
    assert 0.12 < rate < 0.18


def test_prompt_mask_skips_padding(prompt_ids):
    attention_mask = torch.ones_like(prompt_ids)
    attention_mask[:, :2] = 0
    g = torch.Generator().manual_seed(0)
    mask = sample_prompt_mask(prompt_ids, 0.9, attention_mask=attention_mask, generator=g)
    assert not mask[:, :2].any(), "padding 位置不该被掩码"


def test_build_sequence_applies_masks(mask_id):
    prompt = torch.tensor([[1, 2, 3]])
    completion = torch.tensor([[4, 5]])
    prompt_mask = torch.tensor([[False, True, False]])
    completion_mask = torch.tensor([[True, False]])
    x = build_estimation_sequence(prompt, completion, mask_id, prompt_mask, completion_mask)
    assert x.tolist() == [[1, mask_id, 3, mask_id, 5]]


def test_one_step_shape(tiny_model, prompt_ids, mask_id):
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    logps = one_step_token_logprobs(tiny_model, prompt_ids, completion, mask_id)
    assert logps.shape == completion.shape
    assert torch.isfinite(logps).all()
    assert (logps <= 0).all(), "log-prob 必须非正"


def test_one_step_is_differentiable(tiny_model, prompt_ids, mask_id):
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    logps = one_step_token_logprobs(tiny_model, prompt_ids, completion, mask_id)
    logps.sum().backward()
    grads = [p.grad for p in tiny_model.parameters() if p.grad is not None]
    assert grads, "梯度必须能回传到模型参数"
    assert any(g.abs().sum() > 0 for g in grads)


def test_one_step_matches_degenerate_monte_carlo(tiny_model, prompt_ids, mask_id):
    """蒙特卡洛在「补全全掩码、prompt 不掩码、单样本」下应退化成单步估计。

    这条性质把两个独立实现绑在一起，任一侧改坏都会被抓到。
    """
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    with torch.no_grad():
        one_step = one_step_token_logprobs(tiny_model, prompt_ids, completion, mask_id)
    mc, counts = monte_carlo_token_logprobs(
        tiny_model,
        prompt_ids,
        completion,
        mask_id,
        num_samples=1,
        completion_mask_ratio=1.0,
        p_mask_prompt=0.0,
    )
    assert (counts == 1).all()
    assert torch.allclose(one_step, mc, atol=1e-5)


def test_monte_carlo_counts_grow_with_samples(tiny_model, prompt_ids, mask_id):
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    g = torch.Generator().manual_seed(0)
    _, counts = monte_carlo_token_logprobs(
        tiny_model, prompt_ids, completion, mask_id, num_samples=32, generator=g
    )
    assert counts.min() > 0, "32 个样本下每个 token 都该被掩到过"
    assert counts.max() <= 32


def test_monte_carlo_differs_from_one_step_under_partial_context(
    tiny_model, prompt_ids, mask_id
):
    """两个估计器条件在不同上下文上，本就不该相等——这个差异正是 P2 要量化的对象。"""
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    with torch.no_grad():
        one_step = one_step_token_logprobs(tiny_model, prompt_ids, completion, mask_id)
    g = torch.Generator().manual_seed(0)
    mc, _ = monte_carlo_token_logprobs(
        tiny_model, prompt_ids, completion, mask_id, num_samples=16, generator=g
    )
    assert not torch.allclose(one_step, mc, atol=1e-3)


def test_prompt_mask_changes_the_estimate(tiny_model, prompt_ids, mask_id):
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    g = torch.Generator().manual_seed(0)
    prompt_mask = sample_prompt_mask(prompt_ids, 0.5, generator=g)
    with torch.no_grad():
        clean = one_step_token_logprobs(tiny_model, prompt_ids, completion, mask_id)
        perturbed = one_step_token_logprobs(
            tiny_model, prompt_ids, completion, mask_id, prompt_mask=prompt_mask
        )
    assert not torch.allclose(clean, perturbed)


def test_shared_prompt_mask_gives_identical_estimates(tiny_model, prompt_ids, mask_id):
    """diffu-GRPO 要求 π_θ 与 π_θold 在同一个 q' 上取 log-prob，
    因此传入相同 prompt_mask 必须得到完全一致的结果。"""
    completion = _completions(prompt_ids.shape[0], 8, mask_id)
    g = torch.Generator().manual_seed(0)
    prompt_mask = sample_prompt_mask(prompt_ids, 0.3, generator=g)
    with torch.no_grad():
        a = one_step_token_logprobs(
            tiny_model, prompt_ids, completion, mask_id, prompt_mask=prompt_mask
        )
        b = one_step_token_logprobs(
            tiny_model, prompt_ids, completion, mask_id, prompt_mask=prompt_mask
        )
    assert torch.equal(a, b)
