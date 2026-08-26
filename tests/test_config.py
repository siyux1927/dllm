"""配置校验。

那几个「不要改」的超参一旦被手滑改掉，训练照样能跑，只是结果悄悄变坏。
这里把它们钉死，改动必须是显式的。
"""

from pathlib import Path

import pytest

from dllm.config import Config, SamplingConfig

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "countdown_base.yaml"


@pytest.fixture(scope="module")
def config() -> Config:
    return Config.from_yaml(CONFIG_PATH)


def test_yaml_loads(config):
    assert config.model.model_path == "GSAI-ML/LLaDA-8B-Instruct"
    assert config.model.mask_token_id == 126336


def test_lora_alpha_stays_below_rank(config):
    """官方就是 alpha(64) < r(128)。按 alpha=2r 的习惯改会让有效学习率变成 4 倍。"""
    assert config.model.lora_alpha == 64
    assert config.model.lora_r == 128
    assert config.model.lora_alpha < config.model.lora_r


def test_epsilon_stays_wide(config):
    """log-prob 是单步近似、ratio 噪声大，收紧到常见的 0.2 会让几乎每个 token 都被截断。"""
    assert config.grpo.epsilon == 0.5


def test_inner_iterations_stay_high(config):
    """μ=12 是 d1 的核心卖点，靠随机 prompt masking 的正则效应支撑。"""
    assert config.grpo.num_iterations == 12
    assert config.grpo.random_masking is True


def test_adam_beta2_is_not_the_default(config):
    assert config.optim.adam_beta2 == 0.99


def test_checkpoint_frequency_survives_colab_disconnects(config):
    assert config.run.save_steps <= 25


def test_rollouts_per_step(config):
    assert config.rollouts_per_step == 24


def test_sampling_geometry_is_consistent(config):
    assert config.sampling.num_blocks == 4
    assert config.sampling.steps_per_block == 16
    assert (
        config.sampling.num_blocks * config.sampling.block_length
        == config.sampling.max_completion_length
    )


def test_cache_is_off_for_baseline(config):
    """P3 基线必须是 vanilla 采样，加速要到 P4 才引入，否则基线数字没有意义。"""
    assert config.sampling.use_cache is False
    assert config.sampling.confidence_threshold is None


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError, match="未知字段"):
        Config.from_dict({"grpo": {"epsilonn": 0.5}})


def test_roundtrip_through_dict(config):
    assert Config.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_sampling_config_validates_divisibility():
    with pytest.raises(ValueError):
        SamplingConfig(max_completion_length=128, block_length=32, diffusion_steps=6)
