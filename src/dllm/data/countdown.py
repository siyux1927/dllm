"""Countdown 任务的合成数据。

给定若干个数与一个目标数，要求用四则运算、每个数恰好用一次，凑出目标。
选它的理由见 `docs/plan-diffu-grpo.md` 第 2 节：d1 论文中 Countdown 的增益是 +26.2%，
而 GSM8K 只有 +3.9%，信噪比差 6 倍；且奖励可由规则判定、无噪声。

题目自己合成而非下载，是为了能精确控制难度分布并保证 held-out 集干净。
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

PROMPT_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be "
    "used once. Show your work in <think> </think> tags. And return the final answer in "
    "<answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)


@dataclass(frozen=True)
class CountdownProblem:
    numbers: tuple[int, ...]
    target: int

    @property
    def prompt(self) -> str:
        return format_prompt(self.numbers, self.target)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["numbers"] = list(self.numbers)
        return d


def format_prompt(numbers: Sequence[int], target: int) -> str:
    return PROMPT_TEMPLATE.format(numbers=list(numbers), target=target)


def reachable_targets(numbers: Sequence[int]) -> set[int]:
    """枚举用给定数字（每个恰好一次）可以凑出的所有整数。

    合成阶段只允许整除，好处是目标数干净。奖励函数那边不受此限制——模型给出
    含非整数中间值的算式（如 (a/b)*c）只要最终等于目标，同样算对。
    """
    results: set[int] = set()

    def recurse(vals: tuple[int, ...]) -> None:
        if len(vals) == 1:
            results.add(vals[0])
            return
        n = len(vals)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                rest = tuple(v for k, v in enumerate(vals) if k != i and k != j)
                a, b = vals[i], vals[j]
                candidates = [a + b, a - b, a * b]
                if b != 0 and a % b == 0:
                    candidates.append(a // b)
                for c in candidates:
                    recurse((*rest, c))

    recurse(tuple(numbers))
    return results


def _sample_problem(rng: random.Random, cfg) -> CountdownProblem | None:
    numbers = tuple(
        rng.randint(cfg.min_operand, cfg.max_operand) for _ in range(cfg.num_operands)
    )
    candidates = [
        t for t in reachable_targets(numbers) if cfg.min_target <= t <= cfg.max_target
    ]
    if not candidates:
        return None
    return CountdownProblem(numbers=numbers, target=rng.choice(sorted(candidates)))


def build_countdown_dataset(
    num_problems: int,
    cfg,
    seed: int,
    exclude: Iterable[CountdownProblem] = (),
) -> list[CountdownProblem]:
    """生成去重后的题目列表。

    `exclude` 用于保证 held-out 评测集与训练集不重叠——同一组数字加同一个目标数
    出现在两边会让评测结果虚高。
    """
    rng = random.Random(seed)
    seen = {(tuple(sorted(p.numbers)), p.target) for p in exclude}
    problems: list[CountdownProblem] = []
    # 采样可能撞上无解组合或重复，给足重试次数后仍不够就报错，避免静默返回短列表
    max_attempts = num_problems * 200
    for _ in range(max_attempts):
        if len(problems) >= num_problems:
            break
        problem = _sample_problem(rng, cfg)
        if problem is None:
            continue
        key = (tuple(sorted(problem.numbers)), problem.target)
        if key in seen:
            continue
        seen.add(key)
        problems.append(problem)
    if len(problems) < num_problems:
        raise RuntimeError(
            f"只生成出 {len(problems)}/{num_problems} 道题，请放宽 target 范围或操作数取值范围"
        )
    return problems


def save_jsonl(problems: Sequence[CountdownProblem], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for p in problems:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[CountdownProblem]:
    problems = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            problems.append(
                CountdownProblem(numbers=tuple(d["numbers"]), target=int(d["target"]))
            )
    return problems
