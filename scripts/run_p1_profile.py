"""P1：打通 LLaDA 并测出单步耗时拆解。

产出第 1 幕的全部论据：
1. 装配自检——LoRA 真的参与前向，padding 不污染真实位置。
2. 单步各阶段的实测耗时与占比。
3. 实测占比对上理论前向次数预算，两者对不上就说明有别的开销。
4. 由采样占比推出 P4 的端到端收益上限——决定 P4 值不值得做，在花 A100 之前就该知道。

用法（在仓库根目录）：
    python -m scripts.run_p1_profile --config configs/countdown_base.yaml --steps 3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from dllm.config import Config
from dllm.experiment import (
    amdahl_speedup,
    build_problems,
    build_tiny_bundle,
    make_reward_fn,
    shrink_config_for_cpu,
    step_compute_budget,
)
from dllm.models.llada import (
    linear_module_suffixes,
    load_llada,
    probe_padding_invariance,
)
from dllm.sampling.diffusion import generate
from dllm.train.loop import DiffuGRPOTrainer
from dllm.utils.metrics import MetricsLogger

# bf16 下 padding 位置的数值扰动不可能严格为零，这个阈值区分「舍入噪声」与「真的泄漏」
PADDING_LEAK_TOLERANCE = 0.05


PHASES = ["generation", "reward", "logprob", "forward_backward"]


def profile_one_setting(trainer, args, config) -> dict:
    """跑若干步并聚合，返回该设定下的实测耗时、理论预算与 Amdahl 推算。"""
    records = []
    for i in range(args.warmup + args.steps):
        metrics = trainer.step()
        tag = "预热" if i < args.warmup else "统计"
        print(
            f"  [{tag}] step {metrics['step']}  "
            f"总耗时 {metrics['time/total_s']:.1f}s  "
            f"采样 {metrics.get('time/generation_frac', 0):.1%}  "
            f"奖励 {metrics['reward/total_mean']:.3f}"
        )
        if i >= args.warmup:
            records.append(metrics)

    def mean(key: str) -> float:
        return sum(r.get(key, 0.0) for r in records) / len(records)

    timing = {"total_s": mean("time/total_s")}
    for phase in PHASES:
        timing[f"{phase}_s"] = mean(f"time/{phase}_s")
        timing[f"{phase}_frac"] = mean(f"time/{phase}_frac")

    print("\n  各阶段平均耗时")
    print(f"    {'阶段':<20}{'秒':>10}{'占比':>10}")
    for phase in PHASES:
        print(f"    {phase:<20}{timing[f'{phase}_s']:>10.2f}{timing[f'{phase}_frac']:>9.1%}")
    print(f"    {'合计':<20}{timing['total_s']:>10.2f}")

    forward_passes = int(mean("generation/forward_passes"))
    budget = step_compute_budget(config, num_forward_passes=forward_passes)
    print("\n  理论前向次数预算（单位：一次全序列前向）")
    print(f"    采样        {budget.generation:>8.0f}")
    print(
        f"    log-prob    {budget.logprob:>8.0f}"
        f"   (μ={config.grpo.num_iterations} × θ_old/θ_ref)"
    )
    print(f"    策略更新    {budget.policy:>8.0f}   (μ × 前向 + 反向按 2 折算)")
    print(f"    采样占比    {budget.generation_share:>8.1%}   实测 {timing['generation_frac']:.1%}")

    gap = abs(budget.generation_share - timing["generation_frac"])
    if gap > 0.15:
        print(
            f"    注意：理论与实测差 {gap:.1%}。可能来自反向的真实开销倍数不是 2、"
            "采样循环里逐行 topk 的 Python 开销、或显存换页。"
        )

    share = timing["generation_frac"]
    projections = {f"{s:g}x": amdahl_speedup(share, s) for s in args.target_speedups}
    ceiling = 1.0 / (1.0 - share) if share < 1 else float("inf")
    print("\n  P4 收益推算（按实测采样占比）")
    for name, value in projections.items():
        print(f"    采样加速 {name:<6} → 端到端 {value:.2f}x")
    print(f"    采样加速至无穷 → 端到端上限 {ceiling:.2f}x")

    return {
        "timing": timing,
        "budget": budget.to_dict(),
        "generation_forward_passes": forward_passes,
        "amdahl": {
            "generation_frac": share,
            "ceiling": ceiling,
            "projections": projections,
        },
    }


def print_sweep_comparison(sweep: dict, config: Config) -> None:
    """把 diffusion_steps 的取舍摆成一张表。

    这是个真实的权衡而非纯优化：步数越少单步越快，但采样占比随之下降，
    P4 能拿到的端到端收益也跟着缩水——省下的正是后面要攻的那部分。
    """
    print("\n" + "=" * 70)
    print("决策：diffusion_steps 取多少")
    print("=" * 70)
    completion = config.sampling.max_completion_length
    print(
        f"  {'steps':>7}{'每步解码':>10}{'单步秒':>10}"
        f"{'采样占比':>10}{'P4上限':>9}{'200步耗时':>12}"
    )
    for key in sorted(sweep, key=int):
        entry = sweep[key]
        steps = int(key)
        total_s = entry["timing"]["total_s"]
        print(
            f"  {steps:>7}{completion/steps:>10.1f}{total_s:>10.1f}"
            f"{entry['timing']['generation_frac']:>9.1%}"
            f"{entry['amdahl']['ceiling']:>8.2f}x"
            f"{total_s*200/3600:>11.1f}h"
        )
    print("\n  每步解码 1 个 token 才是干净的基线。大于 1 意味着基线本身已经在并行解码，")
    print("  P4 的加速有一部分只是把这份预支的收益兑现一次，而不是新增的。")


def run_sweep(
    trainer,
    args,
    config: Config,
    report: dict,
    out_path: Path,
    profile_fn=None,
) -> dict:
    """依次在每个 diffusion_steps 取值上测一遍，**每测完一档就落盘**。

    不能等全部跑完才写：Colab 掉线是常态，在第二档断掉会把第一档已经测好的数据
    一并赔进去，而那一档可能刚烧了十分钟 A100。
    """
    profile_fn = profile_fn or profile_one_setting
    configured_steps = config.sampling.diffusion_steps
    sweep_values = args.sweep_diffusion_steps or [configured_steps]
    report["sweep"] = {}
    try:
        for steps_value in sweep_values:
            print("\n" + "=" * 70)
            print(
                f"耗时拆解：diffusion_steps={steps_value}"
                f"（预热 {args.warmup} 步 + 统计 {args.steps} 步）"
            )
            print("=" * 70)
            trainer.config = set_diffusion_steps(config, steps_value)
            report["sweep"][str(steps_value)] = profile_fn(trainer, args, config)
            write_report(report, out_path)
            print(f"  已落盘 {out_path}")
    finally:
        # set_diffusion_steps 原地改 config，不还回去的话调用方读到的是最后一档，
        # 于是 report["config"] 记 64、report["timing"] 记 128，两个数字在同一份报告里对不上
        trainer.config = set_diffusion_steps(config, configured_steps)
    return report


def write_report(report: dict, path: Path) -> None:
    """原子落盘。

    直接覆写的话，恰好在写到一半时掉线会留下一个截断的 JSON，
    连上一次成功的结果也一并毁掉。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def set_diffusion_steps(config: Config, steps: int) -> Config:
    """改 diffusion_steps 并重跑校验。

    直接赋值会绕过 SamplingConfig.__post_init__ 的整除性检查，
    留下一个非法但不报错的配置。
    """
    config.sampling = replace(config.sampling, diffusion_steps=steps)
    return config


def set_completion_length(config: Config, length: int) -> Config:
    """改 max_completion_length 并重跑校验，理由同 set_diffusion_steps。"""
    config.sampling = replace(config.sampling, max_completion_length=length)
    return config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/countdown_base.yaml")
    p.add_argument("--steps", type=int, default=3, help="计入统计的步数")
    p.add_argument("--warmup", type=int, default=1, help="丢弃的预热步数，首步含编译与显存分配")
    p.add_argument("--out", default="results/p1_profile.json")
    p.add_argument(
        "--metrics-csv",
        default=None,
        help="逐步指标 CSV 的落点。Colab 上请指向 Drive，否则掉线即丢",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--target-speedups",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0, 5.0],
        help="假设采样环节能加速这些倍数，推算端到端收益"
    )
    p.add_argument(
        "--sweep-diffusion-steps",
        type=int,
        nargs="+",
        default=None,
        help="依次在这些 diffusion_steps 取值上测一遍，用于决定该取多少。留空则只测配置里的值",
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=None,
        help="覆盖配置里的补全长度，用于测截断影响。留空则用配置值",
    )
    p.add_argument(
        "--tiny",
        action="store_true",
        help="用 CPU 小模型跑通整条脚本路径，不加载 LLaDA。数字无意义，只验证接线",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.from_yaml(args.config)
    if args.max_completion_length is not None:
        # 必须在 shrink_config_for_cpu 和建 trainer 之前，否则改的是一份已经被读过的配置
        config = set_completion_length(config, args.max_completion_length)
    device = torch.device("cpu" if args.tiny else args.device)
    report: dict = {"config_path": args.config, "tiny": args.tiny}
    out_path = Path(args.out)

    print("=" * 70)
    if args.tiny:
        print("CPU 小模型冒烟模式：只验证脚本接线，输出的数字不具备任何意义")
        if args.max_completion_length is not None:
            print("  注意：--max-completion-length 在 tiny 模式下被缩表覆盖，实际用 16")
        config = shrink_config_for_cpu(config)
        bundle = build_tiny_bundle(config, device="cpu")
    else:
        print("加载 LLaDA-8B-Instruct")
        bundle = load_llada(config, device=device, validate=True)
    report["config"] = config.to_dict()
    print(f"LoRA 目标模块: {bundle.target_modules}")
    print(f"mask_id={bundle.mask_id}  eos_token_id={bundle.eos_token_id}")
    trainable = sum(p.numel() for p in bundle.policy.trainable_parameters())
    total = sum(p.numel() for p in bundle.policy.model.parameters())
    print(f"可训练参数 {trainable:,} / 总参数 {total:,} ({trainable/total:.3%})")
    print("LoRA 存活自检通过：改动适配器权重确实改变了输出")
    report["model"] = {
        "target_modules": bundle.target_modules,
        "trainable_params": trainable,
        "total_params": total,
        "linear_suffixes": dict(linear_module_suffixes(bundle.policy.model)),
    }

    train_problems, _ = build_problems(config)
    codec = bundle.codec

    print("\n" + "=" * 70)
    print("padding 不变性自检")
    print("=" * 70)
    leak = probe_padding_invariance(
        bundle.policy.model, codec, train_problems[0].prompt, bundle.mask_id
    )
    print(f"不同 padding 长度下补全位置 logits 的最大差异: {leak:.4f}")
    if leak > PADDING_LEAK_TOLERANCE:
        if args.tiny:
            print("  小模型用可学习的绝对位置嵌入，左侧补齐会整体后移真实 token。")
            print("  属预期，与 LLaDA 无关。")
        else:
            print("  未通过——这是 LLaDA 的已知行为，不是本项目的 bug，也不必去修：")
            print("    LLaDAModel.forward 把 attention_mask 算成加性 bias 后随即丢弃")
            print("    （紧跟一行 attention_bias = None），传了等于没传。")
            print("  不破坏可复现性：补齐到固定 max_prompt_length，每行 padding 量只由")
            print("  自身 prompt 长度决定。代价是真实 token 会注意到 padding，补全质量")
            print("  被系统性拉低——但 d1 官方同样如此，保持一致才可比。")
            print("  取舍与备选方案见 docs/plan-diffu-grpo.md 第 2 节。")
    else:
        print("  通过：左侧 padding 不影响真实位置的输出")
    report["padding_leak"] = leak
    write_report(report, out_path)

    print("\n" + "=" * 70)
    print("生成抽查")
    print("=" * 70)
    problem = train_problems[0]
    print(f"题目: numbers={problem.numbers}  target={problem.target}")
    prompt_ids, prompt_mask = codec.encode([problem.prompt])
    output = generate(
        bundle.policy.model,
        prompt_ids.to(device),
        config.sampling,
        bundle.mask_id,
        attention_mask=prompt_mask.to(device),
    )
    sample_text = codec.decode(output.completions)[0]
    print(f"补全（{output.num_forward_passes} 次前向）:\n{sample_text}")
    report["sample_completion"] = sample_text
    report["generation_forward_passes"] = output.num_forward_passes
    write_report(report, out_path)

    metrics_csv = args.metrics_csv or config.run.metrics_path.replace(".csv", "_p1.csv")
    logger = MetricsLogger(metrics_csv)
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
        logger=logger,
    )

    run_sweep(trainer, args, config, report, out_path)

    default_key = str(config.sampling.diffusion_steps)
    # 扫参跳过默认档时会退到第一档，光看 timing/amdahl 认不出是哪一档，引数字必须能对上
    headline_key = default_key if default_key in report["sweep"] else next(iter(report["sweep"]))
    chosen = report["sweep"][headline_key]
    report["headline_diffusion_steps"] = int(headline_key)
    report["timing"] = chosen["timing"]
    report["budget"] = chosen["budget"]
    report["amdahl"] = chosen["amdahl"]

    if len(report["sweep"]) > 1:
        print_sweep_comparison(report["sweep"], config)

    logger.close()
    write_report(report, out_path)
    print(f"\n已写入 {out_path}")
    print(f"逐步指标 {metrics_csv}")


if __name__ == "__main__":
    main()
