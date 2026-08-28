"""训练配置。

默认值对齐 d1 官方 `diffu-grpo/slurm_scripts/train.yaml`，偏离之处在字段注释中说明理由，
完整对照表见 `docs/plan-diffu-grpo.md` 第 2、3 节。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# LLaDA 词表中的 mask token，写死在模型权重里，不能改
LLADA_MASK_TOKEN_ID = 126336


@dataclass
class ModelConfig:
    model_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    torch_dtype: str = "bfloat16"
    # 只能是 eager，别改成 sdpa 或 flash_attention_2，会直接抛 ValueError。
    #
    # LLaDAModelLM 是 trust_remote_code 加载的自定义架构，没有声明 _supports_sdpa，
    # 于是 transformers 的 attn dispatch 一律拒绝。而这个值实际上无关性能：
    # LLaDA 的远端代码（改自 OLMo）在自己的 attention 里直接调
    # F.scaled_dot_product_attention，走的什么核由它自己决定，HF 这个参数管不着。
    # 换句话说填 eager 并不会退化成朴素实现，只是把 HF 的 dispatch 让开而已。
    attn_implementation: str = "eager"
    trust_remote_code: bool = True
    mask_token_id: int = LLADA_MASK_TOKEN_ID

    lora_r: int = 128
    # 官方就是 alpha < r（缩放系数 0.5）。不要按 alpha=2r 的习惯改，
    # 那会让有效学习率变成 4 倍
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    # LLaDA 的命名（派生自 OLMo），不是 Llama 那套。对应关系：
    #   attn_out ↔ o_proj、ff_proj ↔ gate_proj、ff_out ↔ down_proj
    #
    # d1 官方在这里填的是 Llama 的名字，其中 o_proj / gate_proj / down_proj 对 LLaDA
    # 一个都匹配不上。PEFT 只在「全不命中」时才报错，部分命中就静默跳过，
    # 于是官方实际训练的只有 q/k/v/up 四类。差异见 plan-diffu-grpo.md 第 2 节。
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_out",
        "ff_proj",
        "up_proj",
        "ff_out",
    )


@dataclass
class SamplingConfig:
    max_prompt_length: int = 128  # 官方 200，Countdown 的 prompt 只有 60-80 token
    max_completion_length: int = 128  # 官方 256，Countdown 答案是一个算式
    block_length: int = 32
    diffusion_steps: int = 64  # 官方 128
    temperature: float = 1.0
    remasking: str = "low_confidence"  # low_confidence | random

    # Fast-dLLM 加速开关，P4 阶段启用
    use_cache: bool = False
    dual_cache: bool = False
    confidence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.max_completion_length % self.block_length != 0:
            raise ValueError(
                f"max_completion_length({self.max_completion_length}) 必须能被 "
                f"block_length({self.block_length}) 整除"
            )
        num_blocks = self.max_completion_length // self.block_length
        if self.diffusion_steps % num_blocks != 0:
            raise ValueError(
                f"diffusion_steps({self.diffusion_steps}) 必须能被块数({num_blocks})整除"
            )
        if self.remasking not in ("low_confidence", "random"):
            raise ValueError(f"未知的 remasking 策略: {self.remasking}")

    @property
    def num_blocks(self) -> int:
        return self.max_completion_length // self.block_length

    @property
    def steps_per_block(self) -> int:
        return self.diffusion_steps // self.num_blocks


@dataclass
class GRPOConfig:
    num_generations: int = 6  # group 大小 G
    num_prompts_per_step: int = 4  # 官方 8 卡跑 16，单卡降到 4
    num_iterations: int = 12  # μ。d1 的核心：随机 prompt masking 的正则效应让 μ 能从 2 提到 12
    # 官方就是 0.5，远宽于常见的 0.2。因为 log-prob 是单步近似、ratio 噪声大，
    # 收紧到 0.2 会导致几乎每个 token 都被截断
    epsilon: float = 0.5
    beta: float = 0.04  # KL 系数
    p_mask_prompt: float = 0.15
    random_masking: bool = True
    # TRL GRPOTrainer 的默认行为（d1 未覆盖此项）。注意 d1 论文 Eq.2 写的是不除 std
    scale_rewards: bool = True

    reward_format_weight: float = 0.2
    reward_correct_weight: float = 1.0

    # 纯显存开关，梯度与全批逐位相等；A100-40GB 上 24 条会 OOM，激活约 1.7GB/条
    micro_batch_size: int = 4

    @property
    def rollout_size(self) -> int:
        return self.num_prompts_per_step * self.num_generations


@dataclass
class OptimConfig:
    learning_rate: float = 3e-6
    lr_scheduler_type: str = "constant_with_warmup"
    warmup_ratio: float = 0.0001
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99  # 是 0.99 不是 0.999
    weight_decay: float = 0.1
    max_grad_norm: float = 0.2  # 比常见的 1.0 严格得多


@dataclass
class RunConfig:
    seed: int = 42
    max_steps: int = 200
    save_steps: int = 25  # 官方 100；Colab 会断线，加密到 25
    eval_steps: int = 50
    log_every: int = 1
    output_dir: str = "checkpoints/countdown_base"
    # Colab 上指向 Drive。checkpoint 约 4GB 而 Drive 写入约 10-20MB/s，
    # 存一次要几分钟，所以本地高频存、Drive 低频镜像，覆盖两种不同的故障
    mirror_dir: str | None = None
    mirror_every: int = 4  # 每 mirror_every 次本地保存镜像一次
    metrics_path: str = "results/metrics.csv"
    report_to: str = "none"  # none | wandb
    device: str = "auto"


@dataclass
class DataConfig:
    task: str = "countdown"
    num_train: int = 4000
    num_eval: int = 500
    num_operands: int = 3
    min_operand: int = 1
    max_operand: int = 60
    min_target: int = 10
    max_target: int = 500


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @property
    def rollouts_per_step(self) -> int:
        return self.grpo.num_prompts_per_step * self.grpo.num_generations

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        sections = {
            "model": ModelConfig,
            "sampling": SamplingConfig,
            "grpo": GRPOConfig,
            "optim": OptimConfig,
            "run": RunConfig,
            "data": DataConfig,
        }
        kwargs = {}
        for name, klass in sections.items():
            section = raw.get(name) or {}
            unknown = set(section) - set(klass.__dataclass_fields__)
            if unknown:
                raise ValueError(f"配置段 {name} 含未知字段: {sorted(unknown)}")
            kwargs[name] = klass(**section)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
