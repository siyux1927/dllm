import pytest
import torch

from dllm.config import SamplingConfig
from dllm.sampling.diffusion import add_gumbel_noise, generate, get_num_transfer_tokens


def test_num_transfer_tokens_sums_to_mask_count():
    mask_index = torch.tensor([[True, True, True, False], [True, False, False, False]])
    num_transfer = get_num_transfer_tokens(mask_index, steps=2)
    assert num_transfer.shape == (2, 2)
    assert num_transfer.sum(dim=1).tolist() == [3, 1]


def test_num_transfer_tokens_puts_remainder_first():
    mask_index = torch.ones(1, 5, dtype=torch.bool)
    num_transfer = get_num_transfer_tokens(mask_index, steps=2)
    assert num_transfer.tolist() == [[3, 2]]


def test_num_transfer_tokens_handles_no_masks():
    mask_index = torch.zeros(2, 4, dtype=torch.bool)
    assert get_num_transfer_tokens(mask_index, steps=3).sum() == 0


def test_gumbel_noise_is_identity_at_zero_temperature():
    logits = torch.randn(2, 3, 5)
    assert torch.equal(add_gumbel_noise(logits, 0.0), logits)


def test_gumbel_noise_perturbs_at_positive_temperature():
    logits = torch.randn(2, 3, 5)
    assert not torch.allclose(add_gumbel_noise(logits, 1.0), logits.float())


def test_generate_shapes(tiny_model, tiny_sampling, prompt_ids, mask_id):
    out = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    batch, prompt_len = prompt_ids.shape
    assert out.prompt_length == prompt_len
    assert out.completions.shape == (batch, tiny_sampling.max_completion_length)
    assert out.sequences.shape == (batch, prompt_len + tiny_sampling.max_completion_length)


def test_generate_leaves_no_mask_tokens(tiny_model, tiny_sampling, prompt_ids, mask_id):
    out = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    assert not (out.completions == mask_id).any(), "所有补全位置都应被解码"


def test_generate_preserves_prompt(tiny_model, tiny_sampling, prompt_ids, mask_id):
    out = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    assert torch.equal(out.sequences[:, : prompt_ids.shape[1]], prompt_ids)


def test_generate_forward_pass_count(tiny_model, tiny_sampling, prompt_ids, mask_id):
    """前向次数等于扩散步数。这是 P1 耗时拆解与 P4 加速比的计数基准。"""
    out = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    assert out.num_forward_passes == tiny_sampling.diffusion_steps


def test_greedy_generation_is_deterministic(tiny_model, tiny_sampling, prompt_ids, mask_id):
    a = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    b = generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    assert torch.equal(a.completions, b.completions)


def test_sampling_is_reproducible_with_seeded_generator(
    tiny_model, prompt_ids, mask_id
):
    sampling = SamplingConfig(
        max_prompt_length=6,
        max_completion_length=8,
        block_length=4,
        diffusion_steps=4,
        temperature=1.0,
    )
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    a = generate(tiny_model, prompt_ids, sampling, mask_id, generator=g1)
    b = generate(tiny_model, prompt_ids, sampling, mask_id, generator=g2)
    assert torch.equal(a.completions, b.completions)


def test_random_remasking_still_fills_everything(tiny_model, prompt_ids, mask_id):
    sampling = SamplingConfig(
        max_prompt_length=6,
        max_completion_length=8,
        block_length=4,
        diffusion_steps=4,
        temperature=0.0,
        remasking="random",
    )
    out = generate(tiny_model, prompt_ids, sampling, mask_id)
    assert not (out.completions == mask_id).any()


def test_generate_accepts_attention_mask(tiny_model, tiny_sampling, prompt_ids, mask_id):
    attention_mask = torch.ones_like(prompt_ids)
    attention_mask[:, :2] = 0  # 模拟左侧 padding
    out = generate(tiny_model, prompt_ids, tiny_sampling, mask_id, attention_mask=attention_mask)
    assert not (out.completions == mask_id).any()


def test_blocks_are_decoded_left_to_right(tiny_model, tiny_sampling, prompt_ids, mask_id):
    """记录每次前向时各位置的掩码状态，确认后一块在前一块解完前始终保持掩码。"""
    observed: list[torch.Tensor] = []
    original_forward = tiny_model.forward

    def spy(input_ids, attention_mask=None):
        observed.append((input_ids == mask_id).clone())
        return original_forward(input_ids, attention_mask=attention_mask)

    tiny_model.forward = spy
    try:
        generate(tiny_model, prompt_ids, tiny_sampling, mask_id)
    finally:
        tiny_model.forward = original_forward

    prompt_len = prompt_ids.shape[1]
    block_len = tiny_sampling.block_length
    steps_per_block = tiny_sampling.steps_per_block
    # 第一块的最后一步：第二块必须仍然全是掩码
    snapshot = observed[steps_per_block - 1]
    second_block = snapshot[:, prompt_len + block_len : prompt_len + 2 * block_len]
    assert second_block.all(), "第一块尚未解完时，第二块不应被解码"


def test_config_rejects_indivisible_lengths():
    with pytest.raises(ValueError, match="必须能被"):
        SamplingConfig(max_completion_length=10, block_length=4, diffusion_steps=4)


def test_config_rejects_indivisible_steps():
    with pytest.raises(ValueError, match="必须能被块数"):
        SamplingConfig(max_completion_length=8, block_length=4, diffusion_steps=3)
