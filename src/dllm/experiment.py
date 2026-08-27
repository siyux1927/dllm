"""把配置装配成可运行的实验，以及单步算力预算的推导。

`step_compute_budget` 与 `amdahl_speedup` 在这里而不是写死在脚本里，是因为它们决定
P4 值不值得做：如果采样只占单步的一半，那么把采样加速 3 倍，端到端也只有 1.5 倍。
这个上限应该在花 A100 之前用 P1 的实测数字算出来，而不是做完才发现。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dllm.config import Config
from dllm.data.countdown import CountdownProblem, build_countdown_dataset
from dllm.models.llada import LLaDABundle, LLaDAPromptCodec
from dllm.models.tiny import TinyMaskedDiffusionLM, TinyTokenizer
from dllm.rewards.countdown import batch_rewards
from dllm.train.policy import DEFAULT_ADAPTER, OLD_ADAPTER, Policy


@dataclass
class ComputeBudget:
    """以「一次前向」为单位的单步开销拆解。"""

    generation: float
    logprob: float
    policy: float

    @property
    def total(self) -> float:
        return self.generation + self.logprob + self.policy

    @property
    def generation_share(self) -> float:
        return self.generation / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "budget/generation_fwd": self.generation,
            "budget/logprob_fwd": self.logprob,
            "budget/policy_fwd": self.policy,
            "budget/total_fwd": self.total,
            "budget/generation_share": self.generation_share,
        }


def step_compute_budget(
    config: Config,
    num_forward_passes: int | None = None,
    backward_multiplier: float = 2.0,
) -> ComputeBudget:
    """单个 outer step 的理论开销，单位是「一次全序列前向」。

    - 采样：每个扩散步一次前向，共 diffusion_steps 次。
    - log-prob：每次内更新要取 θ_old 和 θ_ref，各一次无梯度前向。
    - 策略更新：每次内更新一次带梯度前向加一次反向，反向按 backward_multiplier 折算。

    三段的 batch 与序列长度相同，所以可以直接按前向次数相加。
    """
    grpo = config.grpo
    generation = float(
        num_forward_passes if num_forward_passes is not None else config.sampling.diffusion_steps
    )
    refs_per_iter = 2.0 if grpo.beta > 0 else 1.0
    logprob = grpo.num_iterations * refs_per_iter
    policy = grpo.num_iterations * (1.0 + backward_multiplier)
    return ComputeBudget(generation=generation, logprob=logprob, policy=policy)


def amdahl_speedup(accelerated_fraction: float, local_speedup: float) -> float:
    """只加速其中一段时的端到端加速比。

    P4 把 Fast-dLLM 的 KV cache 与并行解码装进采样环节，能拿到的整体收益被这个上限锁死。
    """
    if not 0.0 <= accelerated_fraction <= 1.0:
        raise ValueError(f"被加速部分的占比应在 [0, 1]，收到 {accelerated_fraction}")
    if local_speedup <= 0:
        raise ValueError(f"局部加速比应为正数，收到 {local_speedup}")
    return 1.0 / ((1.0 - accelerated_fraction) + accelerated_fraction / local_speedup)


def build_problems(config: Config) -> tuple[list[CountdownProblem], list[CountdownProblem]]:
    """训练集与不重叠的 held-out 评测集。"""
    data = config.data
    train = build_countdown_dataset(data.num_train, data, seed=config.run.seed)
    evaluation = build_countdown_dataset(
        data.num_eval, data, seed=config.run.seed + 1, exclude=train
    )
    return train, evaluation


def shrink_config_for_cpu(config: Config) -> Config:
    """把配置缩到 CPU 上几十秒能跑完，同时保持各阶段的相对结构不变。

    μ 与 diffusion_steps 都保留 4 以上，否则算力预算那段的占比会失真到看不出问题。
    """
    config.sampling.max_prompt_length = 24
    config.sampling.max_completion_length = 16
    config.sampling.block_length = 8
    config.sampling.diffusion_steps = 8
    config.grpo.num_generations = 4
    config.grpo.num_prompts_per_step = 2
    config.grpo.num_iterations = 4
    config.data.num_train = 64
    config.data.num_eval = 16
    return config


def build_tiny_bundle(config: Config, device: str = "cpu", validate: bool = True) -> LLaDABundle:
    """用 CPU 小模型拼出与 `load_llada` 同构的 bundle。

    存在的意义是让 P1/P2 脚本能在本地把整条路径跑一遍——参数解析、探针、训练器接线、
    指标聚合、JSON 落盘。脚本里的低级错误不该等到 Colab 上加载完 16GB 权重才暴露。
    """
    import torch
    from peft import LoraConfig, get_peft_model

    from dllm.models.llada import probe_lora_is_live, resolve_target_modules

    tokenizer = TinyTokenizer(max_number=config.data.max_operand)
    base = TinyMaskedDiffusionLM(
        vocab_size=tokenizer.vocab_size, hidden_size=32, num_layers=2
    ).to(device)

    target_modules = resolve_target_modules(base, config.model.lora_target_modules)
    lora = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=config.model.lora_dropout, target_modules=target_modules
    )
    model = get_peft_model(base, lora)
    model.add_adapter(OLD_ADAPTER, lora)
    model.set_adapter(DEFAULT_ADAPTER)

    policy = Policy(model, is_peft=True)
    if validate:
        # 与 load_llada 走同一条自检，CPU 路径不该比 GPU 路径宽松
        sample = torch.zeros((1, 8), dtype=torch.long, device=device)
        if not probe_lora_is_live(policy, sample):
            raise RuntimeError("LoRA 已挂载但改动适配器权重不影响输出，适配器是死的")

    codec = LLaDAPromptCodec(tokenizer, config.sampling.max_prompt_length)
    return LLaDABundle(
        policy=policy,
        tokenizer=tokenizer,
        codec=codec,
        encode=codec.encode,
        decode=codec.decode,
        mask_id=tokenizer.mask_token_id,
        eos_token_id=tokenizer.eos_token_id,
        target_modules=target_modules,
    )


def make_reward_fn(config: Config):
    grpo = config.grpo

    def reward_fn(texts: Sequence[str], problems: Sequence[Any]):
        return batch_rewards(
            texts, problems, grpo.reward_format_weight, grpo.reward_correct_weight
        )

    return reward_fn
