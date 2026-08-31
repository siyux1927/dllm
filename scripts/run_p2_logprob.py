"""P2：量化单步 log-prob 近似的误差，并给 ε=0.5 找出实证依据。

三个实验：

A. 单步近似 vs 蒙特卡洛真值。前者条件在「补全全掩码」这一最极端的上下文上，
   后者在各种部分可见的上下文上取平均。两者估计的本来就不是同一个量，这里量化差多少。

B. 同一份权重、两个不同的 prompt 掩码模式 q'，算出的 ratio 本该恒等于 1。
   实测的离散程度就是估计器的噪声底噪。把它和 ε 摆在一起看，就知道为什么官方要把
   clip 范围从常见的 0.2 放宽到 0.5——收紧的话，被截断的绝大部分是噪声而不是策略更新。

C. 噪声随 p_mask_prompt 的变化。掩码越多、正则越强，但 ratio 噪声也越大，
   p_mask_prompt=0.15 是这条权衡上的取值。

用法：
    python scripts/run_p2_logprob.py --config configs/countdown_base.yaml --mc-samples 128
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# 必须早于 CUDA 初始化，理由同 run_p1_profile.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from dllm.config import Config  # noqa: E402
from dllm.experiment import (  # noqa: E402
    build_problems,
    build_tiny_bundle,
    make_reward_fn,
    shrink_config_for_cpu,
)
from dllm.logprob.estimators import (  # noqa: E402
    monte_carlo_token_logprobs,
    one_step_token_logprobs,
    sample_prompt_mask,
)
from dllm.models.llada import load_llada  # noqa: E402
from dllm.train.loop import DiffuGRPOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/countdown_base.yaml")
    p.add_argument("--num-prompts", type=int, default=2, help="用几道题的 rollout 做分析")
    p.add_argument("--mc-samples", type=int, default=128, help="蒙特卡洛样本数，LLaDA 官方为 128")
    p.add_argument("--out", default="results/p2_logprob.json")
    p.add_argument("--pairs-csv", default="results/p2_pairs.csv")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--tiny",
        action="store_true",
        help="用 CPU 小模型跑通整条脚本路径，不加载 LLaDA。数字无意义，只验证接线",
    )
    return p.parse_args()


def write_report(report: dict, path: Path) -> None:
    """原子落盘，每完成一个实验就调一次。

    三个实验加起来要跑上百次前向，掉在第三个上不该把前两个的结果一起赔进去。
    先写临时文件再替换，避免写到一半掉线留下截断的 JSON。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).item()) if denom > 0 else 0.0


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    def rank(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort()
        ranks = torch.empty_like(x)
        ranks[order] = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        return ranks

    return pearson(rank(a), rank(b))


def ratio_stats(log_ratio: torch.Tensor, epsilon: float) -> dict[str, float]:
    ratio = log_ratio.exp()
    return {
        "log_ratio_std": float(log_ratio.std().item()),
        "log_ratio_abs_mean": float(log_ratio.abs().mean().item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_p05": float(ratio.quantile(0.05).item()),
        "ratio_p95": float(ratio.quantile(0.95).item()),
        "out_of_range_frac_eps_0.2": float(
            ((ratio < 0.8) | (ratio > 1.2)).float().mean().item()
        ),
        f"out_of_range_frac_eps_{epsilon:g}": float(
            ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float().mean().item()
        ),
    }


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    device = torch.device("cpu" if args.tiny else args.device)
    report: dict = {"config_path": args.config, "tiny": args.tiny}
    out_path = Path(args.out)

    if args.tiny:
        print("CPU 小模型冒烟模式：只验证脚本接线，输出的数字不具备任何意义\n")
        config = shrink_config_for_cpu(config)
        bundle = build_tiny_bundle(config, device="cpu")
    else:
        bundle = load_llada(config, device=device, validate=True)
    config.grpo.num_prompts_per_step = args.num_prompts

    codec = bundle.codec
    train_problems, _ = build_problems(config)

    trainer = DiffuGRPOTrainer(
        policy=bundle.policy,
        config=config,
        problems=train_problems,
        encode_fn=codec.encode,
        decode_fn=codec.decode,
        reward_fn=make_reward_fn(config),
        mask_id=bundle.mask_id,
        device=device,
        eos_token_id=bundle.eos_token_id,
    )

    print("采样一批 rollout 作为分析对象")
    batch = trainer.rollout(trainer.sampler.next_indices())
    valid = batch.completion_mask.bool()
    print(f"  {batch.completion_ids.shape[0]} 条补全，有效 token {int(valid.sum())} 个")

    shared = {
        "prompt_ids": batch.prompt_ids,
        "completion_ids": batch.completion_ids,
        "mask_id": bundle.mask_id,
        "attention_mask": batch.prompt_attention_mask,
    }

    print("\n" + "=" * 70)
    print(f"实验 A：单步近似 vs 蒙特卡洛真值（{args.mc_samples} 样本）")
    print("=" * 70)
    with torch.no_grad():
        one_step = one_step_token_logprobs(bundle.policy.model, prompt_mask=None, **shared)
        mc, counts = monte_carlo_token_logprobs(
            bundle.policy.model,
            batch.prompt_ids,
            batch.completion_ids,
            bundle.mask_id,
            num_samples=args.mc_samples,
            attention_mask=batch.prompt_attention_mask,
        )

    # 没被掩到过的 token 其蒙特卡洛估计无意义，必须按计数过滤
    usable = valid & (counts > 0)
    a = one_step[usable].float().cpu()
    b = mc[usable].float().cpu()
    diff = a - b
    experiment_a = {
        "num_tokens": int(usable.sum()),
        "one_step_mean": float(a.mean()),
        "monte_carlo_mean": float(b.mean()),
        "bias_one_step_minus_mc": float(diff.mean()),
        "mean_abs_diff": float(diff.abs().mean()),
        "median_abs_diff": float(diff.abs().median()),
        "pearson": pearson(a, b),
        "spearman": spearman(a, b),
    }
    report["experiment_a"] = experiment_a
    print(f"  单步均值 {experiment_a['one_step_mean']:.3f}   蒙特卡洛均值 "
          f"{experiment_a['monte_carlo_mean']:.3f}")
    print(f"  偏差(单步-MC) {experiment_a['bias_one_step_minus_mc']:+.3f}   "
          f"平均绝对差 {experiment_a['mean_abs_diff']:.3f}")
    print(f"  Pearson {experiment_a['pearson']:.3f}   Spearman {experiment_a['spearman']:.3f}")
    bias = experiment_a["bias_one_step_minus_mc"]
    if bias < -0.05:
        print("  单步估计系统性偏低，符合预期：全掩码是信息最少的上下文。")
    elif bias > 0.05:
        print("  单步估计反而偏高，值得追一下——全掩码本该是信息最少的上下文。")
    else:
        print("  两者几乎无系统性偏差。")
    print("  GRPO 只用 ratio 的相对关系，所以要紧的是相关性而非绝对值。")

    Path(args.pairs_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.pairs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["one_step", "monte_carlo"])
        writer.writerows(zip(a.tolist(), b.tolist(), strict=True))
    print(f"  逐 token 配对数据已写入 {args.pairs_csv}")
    write_report(report, out_path)

    print("\n" + "=" * 70)
    print("实验 B：同一份权重下的 ratio 噪声底噪")
    print("=" * 70)
    generator = torch.Generator(device="cpu").manual_seed(config.run.seed)
    with torch.no_grad():
        logps = []
        for _ in range(2):
            mask = sample_prompt_mask(
                batch.prompt_ids,
                config.grpo.p_mask_prompt,
                batch.prompt_attention_mask,
                generator=generator if batch.prompt_ids.is_cpu else None,
            )
            logps.append(one_step_token_logprobs(bundle.policy.model, prompt_mask=mask, **shared))
    noise_floor = ratio_stats((logps[0] - logps[1])[valid].float().cpu(), config.grpo.epsilon)
    report["experiment_b"] = noise_floor
    write_report(report, out_path)
    print("  θ 完全没有更新，ratio 本该恒为 1。实测：")
    for key, value in noise_floor.items():
        print(f"    {key:<28}{value:>8.4f}")

    tight = noise_floor["out_of_range_frac_eps_0.2"]
    loose = noise_floor[f"out_of_range_frac_eps_{config.grpo.epsilon:g}"]
    print(f"\n  ε=0.2 时 {tight:.1%} 的 token 会被噪声顶出 clip 区，"
          f"ε={config.grpo.epsilon:g} 时是 {loose:.1%}。")
    # 结论跟着数据走。噪声本来就小的话，这组测量并不能支持「ε 必须放宽」，
    # 硬把它说成支持，就是先有结论再找证据。
    if tight - loose > 0.05:
        print("  ε 放宽有实证支持：收到 0.2 的话，被截断的绝大部分是估计噪声而非策略更新。")
    else:
        print("  注意：这组测量并不足以支持「ε 必须放宽」——策略没动时噪声本就不大。")
        print("  更可能的解释是噪声随 θ 真正偏离 θ_old 而放大，这要到 P3 训练起来后，")
        print("  看 grpo/ratio_out_of_range_frac 这个指标随步数的变化才能验证。")

    print("\n" + "=" * 70)
    print("实验 C：噪声随 p_mask_prompt 的变化")
    print("=" * 70)
    sweep = {}
    for p_mask in (0.0, 0.15, 0.3, 0.5):
        with torch.no_grad():
            pair = []
            for _ in range(2):
                mask = sample_prompt_mask(
                    batch.prompt_ids, p_mask, batch.prompt_attention_mask
                )
                pair.append(
                    one_step_token_logprobs(bundle.policy.model, prompt_mask=mask, **shared)
                )
        sweep[f"{p_mask:g}"] = ratio_stats(
            (pair[0] - pair[1])[valid].float().cpu(), config.grpo.epsilon
        )
        report["experiment_c"] = sweep
        write_report(report, out_path)
        print(
            f"  p_mask={p_mask:<5g} log-ratio 标准差 {sweep[f'{p_mask:g}']['log_ratio_std']:.4f}"
            f"   ε=0.2 外占比 {sweep[f'{p_mask:g}']['out_of_range_frac_eps_0.2']:.1%}"
        )
    print("  p_mask=0 时两次前向输入完全相同，噪声应为 0，可作为这套测量的自检。")

    write_report(report, out_path)
    print(f"\n已写入 {out_path}")


if __name__ == "__main__":
    main()
