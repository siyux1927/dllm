import pytest
import torch

from dllm.config import SamplingConfig
from dllm.models.tiny import TinyMaskedDiffusionLM

VOCAB_SIZE = 64
MASK_ID = 63


@pytest.fixture
def mask_id() -> int:
    return MASK_ID


@pytest.fixture
def tiny_model() -> TinyMaskedDiffusionLM:
    torch.manual_seed(0)
    model = TinyMaskedDiffusionLM(vocab_size=VOCAB_SIZE, hidden_size=32, num_layers=2)
    model.eval()
    return model


@pytest.fixture
def tiny_sampling() -> SamplingConfig:
    return SamplingConfig(
        max_prompt_length=6,
        max_completion_length=8,
        block_length=4,
        diffusion_steps=4,
        temperature=0.0,
    )


@pytest.fixture
def prompt_ids() -> torch.Tensor:
    torch.manual_seed(1)
    # 取值上限避开 MASK_ID，防止 prompt 里混入掩码 token
    return torch.randint(0, MASK_ID, (3, 6))
