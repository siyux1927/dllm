"""diffu-GRPO 训练循环。

没有复用 TRL 的 GRPOTrainer，理由有三：它的采样环节是为自回归模型写的，套上扩散采样
需要大幅覆写；本项目的核心产出（分阶段耗时拆解、log-prob 估计诊断、rollout 加速对照）
都要求对循环内部有完全控制权；单卡场景用不上它背后那套分布式设施。

一个 outer step 的结构：

    采样 G×P 条 rollout  ──►  算奖励与优势  ──►  快照 θ_old
                                                    │
                                    ┌───────────────┘
                                    ▼
                       重复 μ 次：换一个 prompt 掩码模式 q'，
                       在同一个 q' 上取 θ_old / θ_ref / θ 的 log-prob，
                       算 GRPO 损失并更新一次

每次内更新都换 q'，是 d1 的核心设计：等价于对同一批数据造出多个扰动视角，起到正则作用，
使 μ 能从常规 GRPO 的 2 提到 12，从而摊薄昂贵的在线采样成本。
论文式 4 要求 π_θ 与 π_θold 的 log-prob 在**同一个** q' 上计算，这里通过把 prompt_mask
显式传给三次估计来保证。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from dllm.config import Config
from dllm.logprob.estimators import one_step_token_logprobs, sample_prompt_mask
from dllm.sampling.diffusion import generate
from dllm.train.checkpoint import (
    checkpoint_bytes,
    load_checkpoint,
    load_rng_state,
    mirror_checkpoint,
    rng_state,
    save_checkpoint,
)
from dllm.train.grpo import compute_advantages, grpo_loss
from dllm.train.policy import Policy
from dllm.utils.metrics import MetricsLogger
from dllm.utils.timing import PhaseTimer

EncodeFn = Callable[[Sequence[str]], "tuple[torch.Tensor, torch.Tensor]"]
DecodeFn = Callable[[torch.Tensor], "list[str]"]
RewardFn = Callable[[Sequence[str], Sequence[Any]], "list[Any]"]


@dataclass
class RolloutBatch:
    problems: list[Any]
    prompt_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    completion_ids: torch.Tensor
    completion_mask: torch.Tensor
    texts: list[str]
    rewards: torch.Tensor
    format_rewards: torch.Tensor
    correct_rewards: torch.Tensor
    advantages: torch.Tensor
    num_forward_passes: int


class PromptSampler:
    """按 epoch 打乱的取样游标。游标与 epoch 会进 checkpoint，续训时不会重复刷同一批题。"""

    def __init__(self, num_items: int, batch_size: int, seed: int) -> None:
        if num_items < batch_size:
            raise ValueError(f"题目数 {num_items} 少于每步所需的 {batch_size}")
        self.num_items = num_items
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._order = self._shuffle(0)

    def _shuffle(self, epoch: int) -> list[int]:
        generator = torch.Generator().manual_seed(self.seed + epoch)
        return torch.randperm(self.num_items, generator=generator).tolist()

    def next_indices(self) -> list[int]:
        if self.cursor + self.batch_size > self.num_items:
            self.epoch += 1
            self.cursor = 0
            self._order = self._shuffle(self.epoch)
        indices = self._order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return indices

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self.cursor = state["cursor"]
        self._order = self._shuffle(self.epoch)


def completion_mask_from_eos(
    completion_ids: torch.Tensor, eos_token_id: int | None
) -> torch.Tensor:
    """EOS 之后的位置不计入损失。

    扩散模型生成的是定长序列，EOS 之后往往是无意义的填充，把它们算进逐 token 平均
    会稀释真正的学习信号。
    """
    if eos_token_id is None:
        return torch.ones_like(completion_ids, dtype=torch.float32)
    is_eos = completion_ids == eos_token_id
    has_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1)
    length = completion_ids.shape[1]
    lengths = torch.where(has_eos, first_eos + 1, torch.full_like(first_eos, length))
    positions = torch.arange(length, device=completion_ids.device).unsqueeze(0)
    return (positions < lengths.unsqueeze(1)).float()


class DiffuGRPOTrainer:
    def __init__(
        self,
        policy: Policy,
        config: Config,
        problems: Sequence[Any],
        encode_fn: EncodeFn,
        decode_fn: DecodeFn,
        reward_fn: RewardFn,
        mask_id: int,
        device: torch.device | str = "cpu",
        eos_token_id: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        timer: PhaseTimer | None = None,
        logger: MetricsLogger | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.problems = list(problems)
        self.encode_fn = encode_fn
        self.decode_fn = decode_fn
        self.reward_fn = reward_fn
        self.mask_id = mask_id
        self.device = torch.device(device)
        self.eos_token_id = eos_token_id
        self.logger = logger
        self.timer = timer or PhaseTimer(
            sync=torch.cuda.synchronize if self.device.type == "cuda" else None
        )

        self.optimizer = optimizer or self._build_optimizer()
        self.scheduler = scheduler
        self.sampler = PromptSampler(
            len(self.problems), config.grpo.num_prompts_per_step, config.run.seed
        )
        self.generator = torch.Generator(device="cpu").manual_seed(config.run.seed)
        self.step_index = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        optim = self.config.optim
        return torch.optim.AdamW(
            self.policy.trainable_parameters(),
            lr=optim.learning_rate,
            betas=(optim.adam_beta1, optim.adam_beta2),
            weight_decay=optim.weight_decay,
        )

    def rollout(self, indices: Sequence[int]) -> RolloutBatch:
        grpo = self.config.grpo
        problems = [self.problems[i] for i in indices]
        prompt_ids, prompt_attention_mask = self.encode_fn([p.prompt for p in problems])
        prompt_ids = prompt_ids.to(self.device)
        prompt_attention_mask = prompt_attention_mask.to(self.device)

        # 同一 prompt 的 G 条补全必须相邻，compute_advantages 按此分组
        prompt_ids = prompt_ids.repeat_interleave(grpo.num_generations, dim=0)
        prompt_attention_mask = prompt_attention_mask.repeat_interleave(
            grpo.num_generations, dim=0
        )
        expanded_problems = [p for p in problems for _ in range(grpo.num_generations)]

        with self.timer.phase("generation"):
            output = generate(
                self.policy.model,
                prompt_ids,
                self.config.sampling,
                self.mask_id,
                attention_mask=prompt_attention_mask,
                generator=self.generator,
            )

        with self.timer.phase("reward"):
            texts = self.decode_fn(output.completions)
            breakdowns = self.reward_fn(texts, expanded_problems)
            rewards = torch.tensor(
                [b.total for b in breakdowns], dtype=torch.float32, device=self.device
            )
            format_rewards = torch.tensor(
                [b.format for b in breakdowns], dtype=torch.float32, device=self.device
            )
            correct_rewards = torch.tensor(
                [b.correct for b in breakdowns], dtype=torch.float32, device=self.device
            )
            advantages = compute_advantages(
                rewards, grpo.num_generations, grpo.scale_rewards
            )

        return RolloutBatch(
            problems=expanded_problems,
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            completion_ids=output.completions,
            completion_mask=completion_mask_from_eos(output.completions, self.eos_token_id),
            texts=texts,
            rewards=rewards,
            format_rewards=format_rewards,
            correct_rewards=correct_rewards,
            advantages=advantages,
            num_forward_passes=output.num_forward_passes,
        )

    def optimize(self, batch: RolloutBatch) -> dict[str, float]:
        grpo = self.config.grpo
        self.policy.sync_old()

        accumulated: list[dict[str, float]] = []
        grad_norms: list[float] = []

        for _ in range(grpo.num_iterations):
            prompt_mask = None
            if grpo.random_masking:
                prompt_mask = sample_prompt_mask(
                    batch.prompt_ids,
                    grpo.p_mask_prompt,
                    batch.prompt_attention_mask,
                    generator=self.generator if batch.prompt_ids.is_cpu else None,
                )

            estimator_kwargs = {
                "prompt_ids": batch.prompt_ids,
                "completion_ids": batch.completion_ids,
                "mask_id": self.mask_id,
                "prompt_mask": prompt_mask,
                "attention_mask": batch.prompt_attention_mask,
            }

            with self.timer.phase("logprob"):
                with self.policy.as_old() as old_model:
                    logp_old = one_step_token_logprobs(old_model, **estimator_kwargs)
                logp_ref = None
                if grpo.beta > 0:
                    with self.policy.as_ref() as ref_model:
                        logp_ref = one_step_token_logprobs(ref_model, **estimator_kwargs)

            with self.timer.phase("forward_backward"):
                logp_new = one_step_token_logprobs(self.policy.model, **estimator_kwargs)
                loss, stats = grpo_loss(
                    logp_new,
                    logp_old,
                    batch.advantages,
                    batch.completion_mask,
                    epsilon=grpo.epsilon,
                    beta=grpo.beta,
                    logp_ref=logp_ref,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.trainable_parameters(), self.config.optim.max_grad_norm
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

            grad_norms.append(float(grad_norm))
            accumulated.append(stats.to_dict())

        averaged = {
            key: sum(entry[key] for entry in accumulated) / len(accumulated)
            for key in accumulated[0]
        }
        averaged["grad_norm"] = sum(grad_norms) / len(grad_norms)
        return averaged

    def step(self) -> dict[str, Any]:
        self.timer.reset()
        indices = self.sampler.next_indices()
        batch = self.rollout(indices)
        optim_stats = self.optimize(batch)
        self.step_index += 1

        # 格式奖励与正确性奖励分开记录：Countdown 的格式分极易被刷，
        # 只看总奖励会把「学会输出标签但不解题」误判成进步
        metrics: dict[str, Any] = {
            "step": self.step_index,
            "epoch": self.sampler.epoch,
            "reward/total_mean": batch.rewards.mean().item(),
            "reward/format_mean": batch.format_rewards.mean().item(),
            "reward/correct_mean": batch.correct_rewards.mean().item(),
            "reward/total_std": batch.rewards.std().item() if batch.rewards.numel() > 1 else 0.0,
            "reward/zero_advantage_frac": (batch.advantages.abs() < 1e-8).float().mean().item(),
            "completion/mean_length": batch.completion_mask.sum(dim=1).mean().item(),
            "completion/eos_hit_frac": (
                batch.completion_mask.sum(dim=1) < batch.completion_mask.shape[1]
            )
            .float()
            .mean()
            .item(),
            "generation/forward_passes": batch.num_forward_passes,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        metrics.update({f"grpo/{k}": v for k, v in optim_stats.items()})
        metrics.update(self.timer.summary())

        if self.logger is not None and self.step_index % self.config.run.log_every == 0:
            self.logger.log(metrics)
        return metrics

    def train(self, max_steps: int | None = None) -> list[dict[str, Any]]:
        run = self.config.run
        target = max_steps or run.max_steps
        history = []
        while self.step_index < target:
            metrics = self.step()
            if self.step_index % run.save_steps == 0:
                metrics.update(self.checkpoint())
                if self.logger is not None:
                    self.logger.log(metrics)
            history.append(metrics)
        return history

    def checkpoint(self) -> dict[str, Any]:
        """本地保存，并按 mirror_every 的节奏镜像到 Drive。

        存盘耗时一并记进指标：4GB 的 checkpoint 写 Drive 要几分钟，如果它悄悄吃掉了
        三成训练时间，只有把它计量出来才看得见。
        """
        run = self.config.run
        started = perf_counter()
        self.save(run.output_dir)
        result: dict[str, Any] = {
            "checkpoint/local_s": perf_counter() - started,
            "checkpoint/size_mb": checkpoint_bytes(run.output_dir) / 1e6,
            "checkpoint/mirrored": 0,
        }

        saves = self.step_index // max(run.save_steps, 1)
        if run.mirror_dir and saves % max(run.mirror_every, 1) == 0:
            started = perf_counter()
            mirror_checkpoint(run.output_dir, run.mirror_dir)
            result["checkpoint/mirror_s"] = perf_counter() - started
            result["checkpoint/mirrored"] = 1
        return result

    def resume(self, directory: str | Path | None = None) -> bool:
        """从 checkpoint 恢复，没有则返回 False。

        优先读本地；本地没有（会话换了机器就是这种情况）再读 Drive 镜像。
        """
        run = self.config.run
        candidates = [directory] if directory else [run.output_dir, run.mirror_dir]
        for candidate in candidates:
            if not candidate:
                continue
            state = load_checkpoint(candidate, map_location=str(self.device))
            if state is not None:
                self.load_state_dict(state)
                return True
        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "model": _policy_state_dict(self.policy),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "sampler": self.sampler.state_dict(),
            "generator": self.generator.get_state(),
            "rng": rng_state(),
            "config": self.config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.step_index = state["step_index"]
        _load_policy_state_dict(self.policy, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state["scheduler"] is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        self.sampler.load_state_dict(state["sampler"])
        self.generator.set_state(state["generator"])
        load_rng_state(state["rng"])

    def save(self, directory: str | Path) -> Path:
        return save_checkpoint(directory, self.state_dict())


def _policy_state_dict(policy: Policy) -> dict[str, torch.Tensor]:
    if policy.is_peft:
        from peft import get_peft_model_state_dict

        return get_peft_model_state_dict(policy.model, adapter_name="default")
    return policy.model.state_dict()


def _load_policy_state_dict(policy: Policy, state: dict[str, torch.Tensor]) -> None:
    if policy.is_peft:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(policy.model, state, adapter_name="default")
        return
    policy.model.load_state_dict(state)
