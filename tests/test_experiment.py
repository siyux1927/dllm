"""算力预算与 Amdahl 上限的测试。

这些数字决定 P4 值不值得做，所以推导本身要有测试兜着。
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

from dllm.config import Config
from dllm.experiment import (
    amdahl_speedup,
    build_problems,
    build_tiny_bundle,
    shrink_config_for_cpu,
    step_compute_budget,
)
from dllm.models.llada import probe_lora_is_live
from dllm.models.tiny import TinyTokenizer
from dllm.rewards.countdown import extract_answer, safe_eval


def test_budget_counts_every_forward_in_a_step():
    config = Config()
    config.grpo.num_iterations = 12
    config.grpo.beta = 0.04
    budget = step_compute_budget(config, num_forward_passes=64, backward_multiplier=2.0)

    assert budget.generation == 64
    assert budget.logprob == 24  # μ 次内更新，每次取 θ_old 与 θ_ref
    assert budget.policy == 36  # μ 次「一前向 + 一反向(按 2 折算)」
    assert budget.total == 124


def test_generation_share_is_around_half_at_mu_12():
    """μ=12 时采样只占一半左右，不是直觉上的 70-80%。

    这个数字直接决定 P4 的收益上限，必须在花 A100 之前算清楚。
    """
    config = Config()
    budget = step_compute_budget(config, num_forward_passes=64)
    assert 0.45 < budget.generation_share < 0.60


def test_smaller_mu_makes_generation_dominate():
    """μ 是 diffu-GRPO 的核心旋钮：它同时决定采样占比和采样成本的摊薄程度。"""
    config = Config()
    config.grpo.num_iterations = 2
    small_mu = step_compute_budget(config, num_forward_passes=64)
    config.grpo.num_iterations = 12
    large_mu = step_compute_budget(config, num_forward_passes=64)
    assert small_mu.generation_share > 0.85
    assert small_mu.generation_share > large_mu.generation_share


def test_budget_drops_reference_pass_without_kl():
    config = Config()
    config.grpo.beta = 0.0
    budget = step_compute_budget(config, num_forward_passes=64)
    assert budget.logprob == config.grpo.num_iterations


def test_budget_defaults_to_configured_diffusion_steps():
    config = Config()
    assert step_compute_budget(config).generation == config.sampling.diffusion_steps


def test_amdahl_is_bounded_by_the_untouched_part():
    # 只占一半的部分即使加速到无穷，端到端也不会超过 2 倍
    assert amdahl_speedup(0.5, 1000.0) == pytest.approx(2.0, abs=1e-2)
    assert amdahl_speedup(0.5, 3.0) == pytest.approx(1.5, abs=1e-6)
    assert amdahl_speedup(0.8, 3.0) == pytest.approx(2.14, abs=1e-2)


def test_amdahl_identity_at_unit_speedup():
    assert amdahl_speedup(0.73, 1.0) == pytest.approx(1.0)


def test_amdahl_rejects_nonsense_inputs():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        amdahl_speedup(1.5, 2.0)
    with pytest.raises(ValueError, match="应为正数"):
        amdahl_speedup(0.5, 0.0)


def test_tiny_tokenizer_roundtrips_answer_tags():
    """替身分词器必须能解码出 <answer> 标签和算式。

    否则随机小模型的补全永远拿不到奖励，奖励方差为 0、优势全零、梯度为零——
    冒烟测试会在「什么都没发生」的状态下通过。
    """
    tokenizer = TinyTokenizer(max_number=10)
    vocab = tokenizer._vocab
    ids = [vocab.index("<answer>"), vocab.index("3"), vocab.index("*"),
           vocab.index("4"), vocab.index("</answer>")]
    text = tokenizer.batch_decode([ids])[0]
    assert extract_answer(text) is not None
    assert safe_eval(extract_answer(text))[0] == 12


def test_tiny_tokenizer_skips_special_tokens():
    tokenizer = TinyTokenizer()
    ids = [tokenizer.pad_token_id, tokenizer.mask_token_id, 10, tokenizer.eos_token_id]
    assert tokenizer.batch_decode([ids])[0] == tokenizer._vocab[10]


def test_tiny_tokenizer_encoding_is_stable_across_processes():
    """内置 hash() 对字符串按进程加盐，同一段文本换个进程会得到另一串 id。"""
    script = (
        "import sys; sys.path.insert(0, 'src');"
        "from dllm.models.tiny import TinyTokenizer;"
        "print(TinyTokenizer()('numbers 41 8 2 target 66')['input_ids'])"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(runs) == 1, f"不同 PYTHONHASHSEED 下编码结果不一致: {runs}"


def test_tiny_bundle_mirrors_the_llada_bundle():
    config = shrink_config_for_cpu(Config())
    bundle = build_tiny_bundle(config)

    assert bundle.policy.is_peft
    assert bundle.target_modules == list(config.model.lora_target_modules)
    assert bundle.mask_id == bundle.tokenizer.mask_token_id
    ids, mask = bundle.encode(["Using the numbers [1, 2, 3], make 6."])
    assert ids.shape == (1, config.sampling.max_prompt_length)
    assert mask.shape == ids.shape
    assert bundle.decode(ids)  # 不抛异常即可，内容无意义


def test_tiny_bundle_validates_lora_is_live():
    """CPU 路径与 GPU 路径走同一条自检，不该更宽松。"""
    config = shrink_config_for_cpu(Config())
    bundle = build_tiny_bundle(config, validate=True)
    assert probe_lora_is_live(bundle.policy, torch.zeros((1, 8), dtype=torch.long))


def test_shrunk_config_still_has_a_meaningful_budget():
    """缩配置是为了跑得快，但不能把各阶段的相对结构缩没了。"""
    config = shrink_config_for_cpu(Config())
    budget = step_compute_budget(config)
    assert config.grpo.num_iterations >= 4
    assert 0.15 < budget.generation_share < 0.85


def test_eval_problems_never_overlap_training():
    config = Config()
    config.data.num_train = 200
    config.data.num_eval = 50
    train, evaluation = build_problems(config)

    def key(p):
        return (tuple(sorted(p.numbers)), p.target)

    assert len(train) == 200 and len(evaluation) == 50
    assert not {key(p) for p in train} & {key(p) for p in evaluation}
