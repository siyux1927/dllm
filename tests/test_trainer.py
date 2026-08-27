"""训练循环的 CPU 全链路冒烟测试。

跑的是真实的 PEFT 策略（不是 plain 拷贝那条降级路径），所以适配器切换、θ_old 快照、
LoRA 状态存取这些最容易在 Colab 上翻车的环节，在本地就被覆盖了。
"""

from __future__ import annotations

import shutil

import pytest
import torch

from dllm.config import (
    Config,
    DataConfig,
    GRPOConfig,
    ModelConfig,
    OptimConfig,
    RunConfig,
    SamplingConfig,
)
from dllm.data.countdown import build_countdown_dataset
from dllm.models.tiny import TinyMaskedDiffusionLM
from dllm.rewards.countdown import RewardBreakdown, batch_rewards
from dllm.train.checkpoint import load_checkpoint, mirror_checkpoint
from dllm.train.loop import DiffuGRPOTrainer, PromptSampler, completion_mask_from_eos
from dllm.train.policy import Policy

peft = pytest.importorskip("peft")

VOCAB_SIZE = 64
MASK_ID = 63
PAD_ID = 0
EOS_ID = 1
MAX_PROMPT = 16


class ToyTokenizer:
    """字符级玩具分词器，只为让全链路能在 CPU 上跑通，不追求可逆。"""

    def encode(self, prompts) -> tuple[torch.Tensor, torch.Tensor]:
        rows, masks = [], []
        for text in prompts:
            ids = [2 + (ord(c) % (MASK_ID - 2)) for c in text][-MAX_PROMPT:]
            pad = MAX_PROMPT - len(ids)
            rows.append([PAD_ID] * pad + ids)  # 左侧补齐，让补全紧接在 prompt 之后
            masks.append([0] * pad + [1] * len(ids))
        return torch.tensor(rows), torch.tensor(masks)

    def decode(self, ids: torch.Tensor) -> list[str]:
        return ["".join(chr(65 + int(i) % 26) for i in row) for row in ids]


def stub_reward_fn(texts, problems):
    """按补全内容给出可复现的伪随机奖励。

    真实奖励函数在随机初始化的小模型上恒为 0，优势会全零、梯度也全零，
    那样的冒烟测试查不出任何东西。这里制造组内差异，才能真正压到 GRPO 的更新路径。
    """
    out = []
    for text in texts:
        h = sum(ord(c) * (i + 1) for i, c in enumerate(text)) % 97
        correct = 1.0 if h % 3 == 0 else 0.0
        fmt = 1.0 if h % 2 == 0 else 0.0
        out.append(RewardBreakdown(fmt, correct, 0.2 * fmt + correct))
    return out


def make_config(**overrides) -> Config:
    config = Config(
        model=ModelConfig(mask_token_id=MASK_ID),
        sampling=SamplingConfig(
            max_prompt_length=MAX_PROMPT,
            max_completion_length=8,
            block_length=4,
            diffusion_steps=4,
            temperature=0.0,
        ),
        grpo=GRPOConfig(num_generations=3, num_prompts_per_step=2, num_iterations=2, beta=0.04),
        optim=OptimConfig(learning_rate=1e-2),
        run=RunConfig(seed=0, max_steps=2, save_steps=1000),
        data=DataConfig(),
    )
    for key, value in overrides.items():
        setattr(config.run, key, value)
    return config


def make_trainer(reward_fn=stub_reward_fn, decode_fn=None, **overrides) -> DiffuGRPOTrainer:
    torch.manual_seed(0)
    base = TinyMaskedDiffusionLM(vocab_size=VOCAB_SIZE, hidden_size=32, num_layers=2)
    lora = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=list(ModelConfig().lora_target_modules),
    )
    model = peft.get_peft_model(base, lora)
    model.add_adapter("old", lora)
    model.set_adapter("default")
    policy = Policy(model, is_peft=True)

    config = make_config(**overrides)
    tokenizer = ToyTokenizer()
    problems = build_countdown_dataset(12, config.data, seed=0)
    return DiffuGRPOTrainer(
        policy=policy,
        config=config,
        problems=problems,
        encode_fn=tokenizer.encode,
        decode_fn=decode_fn or tokenizer.decode,
        reward_fn=reward_fn,
        mask_id=MASK_ID,
        eos_token_id=EOS_ID,
    )


def test_completion_mask_stops_after_first_eos():
    ids = torch.tensor([[5, 6, EOS_ID, 9], [5, 6, 7, 8]])
    mask = completion_mask_from_eos(ids, EOS_ID)
    assert mask.tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]


def test_completion_mask_is_all_ones_without_eos_id():
    ids = torch.tensor([[5, EOS_ID, 7]])
    assert completion_mask_from_eos(ids, None).tolist() == [[1, 1, 1]]


def test_prompt_sampler_has_no_repeats_within_epoch():
    sampler = PromptSampler(num_items=8, batch_size=4, seed=0)
    first, second = sampler.next_indices(), sampler.next_indices()
    assert len(set(first) | set(second)) == 8
    assert sampler.epoch == 0
    sampler.next_indices()
    assert sampler.epoch == 1


def test_prompt_sampler_state_roundtrip():
    a = PromptSampler(8, 4, seed=3)
    a.next_indices()
    b = PromptSampler(8, 4, seed=999)
    b.load_state_dict(a.state_dict())
    assert a.next_indices() == b.next_indices()


def test_sampler_rejects_too_small_dataset():
    with pytest.raises(ValueError, match="少于每步所需"):
        PromptSampler(num_items=2, batch_size=4, seed=0)


def test_rollout_groups_generations_contiguously():
    trainer = make_trainer()
    batch = trainer.rollout(trainer.sampler.next_indices())
    grpo = trainer.config.grpo
    expected = grpo.num_prompts_per_step * grpo.num_generations
    assert batch.completion_ids.shape == (expected, trainer.config.sampling.max_completion_length)
    # 同一 prompt 的 G 条补全必须相邻，compute_advantages 依赖这个排布
    for i in range(grpo.num_prompts_per_step):
        group = batch.problems[i * grpo.num_generations : (i + 1) * grpo.num_generations]
        assert len(set(group)) == 1


def test_rollout_advantages_are_zero_mean_per_group():
    trainer = make_trainer()
    batch = trainer.rollout(trainer.sampler.next_indices())
    grouped = batch.advantages.view(-1, trainer.config.grpo.num_generations)
    assert torch.allclose(grouped.mean(dim=1), torch.zeros(grouped.shape[0]), atol=1e-5)


def test_rollout_leaves_no_mask_tokens():
    trainer = make_trainer()
    batch = trainer.rollout(trainer.sampler.next_indices())
    assert not (batch.completion_ids == MASK_ID).any()


def test_step_updates_only_lora_weights():
    trainer = make_trainer()
    before = {
        name: p.detach().clone()
        for name, p in trainer.policy.model.named_parameters()
    }
    trainer.step()

    changed = [
        name
        for name, p in trainer.policy.model.named_parameters()
        if not torch.equal(before[name], p.detach())
    ]
    assert changed, "一步之后必须有参数发生变化"
    assert all("lora" in name for name in changed), f"底座权重不该被改动: {changed[:3]}"
    # default 被优化器更新；old 被 sync_old 覆写成快照。两者都该动，底座一律不动。
    assert any("default" in name for name in changed), "训练中的适配器应被更新"
    assert not any(
        "lora" not in name for name in changed
    ), "只有 LoRA 参数可以变动"


def test_step_metrics_split_format_and_correctness():
    trainer = make_trainer()
    metrics = trainer.step()
    for key in ("reward/total_mean", "reward/format_mean", "reward/correct_mean"):
        assert key in metrics
    # 总奖励 = 0.2 * 格式 + 1.0 * 正确性，两者必须能被独立读出
    assert metrics["reward/total_mean"] == pytest.approx(
        0.2 * metrics["reward/format_mean"] + metrics["reward/correct_mean"], abs=1e-6
    )


def test_step_metrics_report_phase_timing():
    """第 1 幕的论据来源：generation / logprob / forward_backward 必须分开计时。"""
    trainer = make_trainer()
    metrics = trainer.step()
    fractions = 0.0
    for phase in ("generation", "reward", "logprob", "forward_backward"):
        assert f"time/{phase}_s" in metrics
        fractions += metrics[f"time/{phase}_frac"]
    assert fractions == pytest.approx(1.0, abs=1e-6)
    assert metrics["time/total_s"] > 0


def test_step_records_ratio_diagnostics():
    """ratio 越界比例是判断 ε 该设多宽的直接证据，必须每步都有。"""
    trainer = make_trainer()
    metrics = trainer.step()
    assert "grpo/ratio_out_of_range_frac" in metrics
    assert "grpo/clip_active_frac" in metrics
    assert "grpo/kl" in metrics
    assert metrics["grpo/grad_norm"] >= 0


def test_first_inner_iteration_has_unit_ratio():
    """内更新第一轮 θ 与 θ_old 相同，ratio 应恰为 1；不为 1 说明 θ_old 快照错了。"""
    trainer = make_trainer(num_iterations=1)
    trainer.config.grpo.num_iterations = 1
    metrics = trainer.step()
    assert metrics["grpo/ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert metrics["grpo/ratio_out_of_range_frac"] == 0.0


def test_metrics_are_written_to_csv(tmp_path):
    from dllm.utils.metrics import MetricsLogger

    path = tmp_path / "metrics.csv"
    trainer = make_trainer()
    trainer.logger = MetricsLogger(path)
    trainer.step()
    trainer.step()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, "一行表头加两行数据"
    assert "reward/correct_mean" in lines[0]


def test_resume_reproduces_uninterrupted_run(tmp_path):
    """把「断线后续训」自动化成断言：中断恢复后的第 3 步必须与不中断时逐位一致。"""
    reference = make_trainer()
    reference.step()
    reference.step()
    expected = reference.step()

    interrupted = make_trainer()
    interrupted.step()
    interrupted.step()
    interrupted.save(tmp_path)

    revived = make_trainer()
    state = load_checkpoint(tmp_path)
    assert state is not None
    revived.load_state_dict(state)
    assert revived.step_index == 2

    actual = revived.step()
    for key in ("reward/total_mean", "grpo/loss", "grpo/ratio_mean", "grpo/kl"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-8), key


def test_checkpoint_stores_only_adapter_weights(tmp_path):
    trainer = make_trainer()
    trainer.step()
    trainer.save(tmp_path)
    state = load_checkpoint(tmp_path)
    assert all("lora" in name for name in state["model"]), "checkpoint 不该含底座权重"
    assert state["sampler"]["cursor"] > 0
    assert "rng" in state and "generator" in state


def test_real_reward_path_end_to_end():
    """把 decode 替换成写死的合法答案，验证真实奖励函数确实接得上训练循环。"""
    captured: list[float] = []

    def decode_fn(ids: torch.Tensor) -> list[str]:
        return ["<answer>1 + 1</answer>"] * ids.shape[0]

    def reward_fn(texts, problems):
        results = batch_rewards(texts, problems)
        captured.extend(r.format for r in results)
        return results

    trainer = make_trainer(reward_fn=reward_fn, decode_fn=decode_fn)
    metrics = trainer.step()
    assert captured and all(f == 1.0 for f in captured), "格式分应被真实奖励函数判为满分"
    assert metrics["reward/format_mean"] == pytest.approx(1.0)
    # 答案与题目无关，正确率必须是 0——否则说明数字校验没起作用
    assert metrics["reward/correct_mean"] == 0.0


def test_train_runs_multiple_steps_and_checkpoints(tmp_path):
    trainer = make_trainer()
    trainer.config.run.save_steps = 2
    trainer.config.run.output_dir = str(tmp_path)
    history = trainer.train(max_steps=2)
    assert len(history) == 2
    assert trainer.step_index == 2
    assert load_checkpoint(tmp_path) is not None


def test_resume_reads_local_checkpoint(tmp_path):
    """resume() 是把 load_checkpoint 真正接上训练循环的那一环。

    此前 checkpoint 存得好好的，却没有任何代码路径会去读它——存了等于没存。
    """
    trainer = make_trainer()
    trainer.config.run.output_dir = str(tmp_path)
    trainer.step()
    trainer.step()
    trainer.save(tmp_path)

    revived = make_trainer()
    revived.config.run.output_dir = str(tmp_path)
    assert revived.resume() is True
    assert revived.step_index == 2


def test_resume_falls_back_to_the_drive_mirror(tmp_path):
    """Colab 重连后换了机器，本地目录是空的，必须能从 Drive 镜像接上。"""
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    trainer = make_trainer()
    trainer.config.run.output_dir = str(local)
    trainer.config.run.mirror_dir = str(drive)
    trainer.step()
    trainer.save(local)
    assert mirror_checkpoint(local, drive) is not None

    shutil.rmtree(local)  # 模拟 /content 被清空

    revived = make_trainer()
    revived.config.run.output_dir = str(local)
    revived.config.run.mirror_dir = str(drive)
    assert revived.resume() is True
    assert revived.step_index == 1


def test_resume_returns_false_when_nothing_saved(tmp_path):
    trainer = make_trainer()
    trainer.config.run.output_dir = str(tmp_path / "missing")
    trainer.config.run.mirror_dir = str(tmp_path / "also-missing")
    assert trainer.resume() is False
    assert trainer.step_index == 0


def test_mirror_happens_on_schedule_not_every_save(tmp_path):
    """Drive 上的 checkpoint 约 4GB、写一次要几分钟，不能每次保存都镜像。"""
    local = tmp_path / "local"
    drive = tmp_path / "drive"
    trainer = make_trainer()
    trainer.config.run.output_dir = str(local)
    trainer.config.run.mirror_dir = str(drive)
    trainer.config.run.save_steps = 1
    trainer.config.run.mirror_every = 3

    mirrored = [trainer.checkpoint()["checkpoint/mirrored"] for _ in _steps(trainer, 6)]
    assert mirrored == [0, 0, 1, 0, 0, 1], f"应每 3 次保存镜像一次，实际 {mirrored}"
    assert load_checkpoint(drive) is not None


def test_checkpoint_reports_size_and_duration(tmp_path):
    """存盘耗时必须计量。4GB 写 Drive 要几分钟，不计量就看不见它吃掉了多少训练时间。"""
    trainer = make_trainer()
    trainer.config.run.output_dir = str(tmp_path)
    trainer.step()
    stats = trainer.checkpoint()
    assert stats["checkpoint/size_mb"] > 0
    assert stats["checkpoint/local_s"] >= 0


def _steps(trainer, count):
    """推进 step_index 而不真的训练，用于测保存节奏。"""
    for _ in range(count):
        trainer.step_index += 1
        yield trainer.step_index
