"""Countdown 的奖励函数。

格式奖励与正确性奖励必须分开记录。Countdown 的格式奖励极易被刷——模型会先学会
输出合法的 <answer> 标签但不解题，此时总奖励上升而正确率不动。只看总奖励会被骗。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from dllm.data.countdown import CountdownProblem

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# 防止畸形输出导致求值爆栈或耗时
_MAX_EXPR_CHARS = 256
_MAX_EXPR_NODES = 64

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


@dataclass
class RewardBreakdown:
    format: float
    correct: float
    total: float
    equation: str | None = None
    reason: str | None = None


def extract_answer(text: str) -> str | None:
    """取最后一个 <answer> 块。取最后一个而非第一个，是因为模型可能在 think 里
    先写出候选答案再修正。"""
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def has_think_block(text: str) -> bool:
    return bool(_THINK_RE.search(text))


def _normalize(expr: str) -> str:
    # 模型常写成 "a + b = c"，只取等号左边
    if "=" in expr:
        expr = expr.split("=", 1)[0]
    expr = expr.strip().rstrip(".").strip()
    # 常见的全角与花括号替换
    for src, dst in (("×", "*"), ("÷", "/"), ("−", "-"), ("[", "("), ("]", ")")):
        expr = expr.replace(src, dst)
    return expr


def safe_eval(expr: str) -> tuple[Fraction | None, list[int], str | None]:
    """求值一个只含整数与四则运算的表达式。

    返回 (值, 用到的整数字面量, 失败原因)。用 Fraction 而非 float 是为了让
    (a / b) * c 这类含非整数中间值的算式也能精确判定，不受浮点误差影响。
    """
    expr = _normalize(expr)
    if not expr:
        return None, [], "空表达式"
    if len(expr) > _MAX_EXPR_CHARS:
        return None, [], "表达式过长"

    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None, [], "语法错误"

    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_EXPR_NODES:
        return None, [], "表达式过于复杂"

    literals: list[int] = []

    def _eval(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise ValueError("只允许整数字面量")
            literals.append(node.value)
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARYOPS):
                raise ValueError("不允许的一元运算符")
            value = _eval(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ValueError("不允许的二元运算符")
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        raise ValueError("不允许的语法结构")

    try:
        value = _eval(tree)
    except ZeroDivisionError:
        return None, literals, "除以零"
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return None, literals, str(exc)
    return value, literals, None


def compute_reward(
    completion: str,
    problem: CountdownProblem,
    format_weight: float = 0.2,
    correct_weight: float = 1.0,
) -> RewardBreakdown:
    answer = extract_answer(completion)
    if answer is None:
        return RewardBreakdown(0.0, 0.0, 0.0, None, "缺少 <answer> 标签")

    fmt = 1.0
    value, literals, reason = safe_eval(answer)
    if value is None:
        return RewardBreakdown(fmt, 0.0, format_weight * fmt, answer, reason)

    # 每个给定数字必须恰好用一次：比较多重集而非集合，否则 "3 * 3" 能骗过只给一个 3 的题
    if sorted(literals) != sorted(problem.numbers):
        return RewardBreakdown(fmt, 0.0, format_weight * fmt, answer, "数字使用不符")

    correct = 1.0 if value == problem.target else 0.0
    reason = None if correct else f"求值为 {value}，目标 {problem.target}"
    total = format_weight * fmt + correct_weight * correct
    return RewardBreakdown(fmt, correct, total, answer, reason)


def batch_rewards(
    completions: Sequence[str],
    problems: Sequence[CountdownProblem],
    format_weight: float = 0.2,
    correct_weight: float = 1.0,
) -> list[RewardBreakdown]:
    if len(completions) != len(problems):
        raise ValueError(f"completions({len(completions)}) 与 problems({len(problems)}) 数量不一致")
    return [
        compute_reward(c, p, format_weight, correct_weight)
        for c, p in zip(completions, problems)
    ]
