from fractions import Fraction

from dllm.data.countdown import CountdownProblem
from dllm.rewards.countdown import (
    batch_rewards,
    compute_reward,
    extract_answer,
    safe_eval,
)

PROBLEM = CountdownProblem(numbers=(3, 5, 7), target=22)


def test_extract_answer_takes_last_block():
    text = "<answer> 1 + 1 </answer> then <answer> 2 + 2 </answer>"
    assert extract_answer(text) == "2 + 2"


def test_extract_answer_missing_returns_none():
    assert extract_answer("no tags here") is None
    assert extract_answer("<answer>   </answer>") is None


def test_safe_eval_basic():
    value, literals, reason = safe_eval("3 * 5 + 7")
    assert value == 22
    assert sorted(literals) == [3, 5, 7]
    assert reason is None


def test_safe_eval_uses_exact_arithmetic():
    # (1 / 3) * 3 用浮点会得到 0.9999...，用 Fraction 才精确等于 1
    value, _, _ = safe_eval("(1 / 3) * 3")
    assert value == Fraction(1)


def test_safe_eval_strips_equals_sign():
    value, literals, _ = safe_eval("3 * 5 + 7 = 22")
    assert value == 22
    assert sorted(literals) == [3, 5, 7]


def test_safe_eval_rejects_code_execution():
    for expr in ("__import__('os').system('ls')", "open('x')", "a + b", "[1,2]"):
        value, _, reason = safe_eval(expr)
        assert value is None, f"{expr!r} 不该被求值"
        assert reason is not None


def test_safe_eval_rejects_division_by_zero():
    value, _, reason = safe_eval("3 / 0")
    assert value is None
    assert reason == "除以零"


def test_safe_eval_rejects_float_literals():
    value, _, reason = safe_eval("3.5 + 1")
    assert value is None
    assert reason is not None


def test_correct_answer_gets_full_reward():
    r = compute_reward("<think>try</think><answer>3 * 5 + 7</answer>", PROBLEM)
    assert r.format == 1.0
    assert r.correct == 1.0
    assert r.total == 0.2 + 1.0


def test_format_only_gets_partial_reward():
    """能刷格式分但答案错，这正是需要监控的奖励 hacking 形态。"""
    r = compute_reward("<answer>3 + 5 + 7</answer>", PROBLEM)
    assert r.format == 1.0
    assert r.correct == 0.0
    assert r.total == 0.2


def test_missing_tags_gets_zero():
    r = compute_reward("the answer is 3 * 5 + 7", PROBLEM)
    assert r.format == 0.0
    assert r.correct == 0.0
    assert r.total == 0.0


def test_reusing_a_number_is_rejected():
    problem = CountdownProblem(numbers=(3, 5, 7), target=9)
    r = compute_reward("<answer>3 * 3</answer>", problem)
    assert r.correct == 0.0
    assert r.reason == "数字使用不符"


def test_omitting_a_number_is_rejected():
    r = compute_reward("<answer>15 + 7</answer>", PROBLEM)
    assert r.correct == 0.0
    assert r.reason == "数字使用不符"


def test_non_integer_intermediate_is_accepted():
    problem = CountdownProblem(numbers=(1, 3, 6), target=2)
    r = compute_reward("<answer>(1 / 3) * 6</answer>", problem)
    assert r.correct == 1.0


def test_batch_rewards_aligns_with_problems():
    problems = [PROBLEM, CountdownProblem((1, 2, 3), 9)]
    completions = ["<answer>3 * 5 + 7</answer>", "<answer>1 + 2 + 3</answer>"]
    results = batch_rewards(completions, problems)
    assert [r.correct for r in results] == [1.0, 0.0]


def test_weights_are_configurable():
    r = compute_reward(
        "<answer>3 * 5 + 7</answer>", PROBLEM, format_weight=0.0, correct_weight=2.0
    )
    assert r.total == 2.0
