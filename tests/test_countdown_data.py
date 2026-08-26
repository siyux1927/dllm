import pytest

from dllm.config import DataConfig
from dllm.data.countdown import (
    CountdownProblem,
    build_countdown_dataset,
    format_prompt,
    load_jsonl,
    reachable_targets,
    save_jsonl,
)


def test_reachable_targets_small_case():
    targets = reachable_targets([2, 3, 4])
    for expected in (9, 24, 10, 2, 14):
        assert expected in targets, f"{expected} 应该可达"


def test_reachable_targets_only_exact_division():
    # 7 / 2 不整除，合成阶段不应产生它
    targets = reachable_targets([7, 2])
    assert 14 in targets and 9 in targets and 5 in targets
    assert 3 not in targets


def test_reachable_targets_single_number():
    assert reachable_targets([5]) == {5}


def test_build_dataset_is_deduplicated():
    cfg = DataConfig(num_train=200)
    problems = build_countdown_dataset(200, cfg, seed=0)
    assert len(problems) == 200
    keys = {(tuple(sorted(p.numbers)), p.target) for p in problems}
    assert len(keys) == 200


def test_build_dataset_is_deterministic():
    cfg = DataConfig()
    a = build_countdown_dataset(50, cfg, seed=7)
    b = build_countdown_dataset(50, cfg, seed=7)
    assert a == b


def test_eval_set_excludes_train_set():
    cfg = DataConfig()
    train = build_countdown_dataset(100, cfg, seed=1)
    evalset = build_countdown_dataset(50, cfg, seed=2, exclude=train)
    train_keys = {(tuple(sorted(p.numbers)), p.target) for p in train}
    for p in evalset:
        assert (tuple(sorted(p.numbers)), p.target) not in train_keys


def test_generated_targets_are_actually_reachable():
    cfg = DataConfig()
    for p in build_countdown_dataset(100, cfg, seed=3):
        assert p.target in reachable_targets(p.numbers)
        assert cfg.min_target <= p.target <= cfg.max_target


def test_prompt_mentions_numbers_and_target():
    prompt = format_prompt([3, 5, 7], 22)
    assert "[3, 5, 7]" in prompt
    assert "22" in prompt
    assert "<answer>" in prompt


def test_jsonl_roundtrip(tmp_path):
    problems = [CountdownProblem((1, 2, 3), 9), CountdownProblem((10, 4, 2), 22)]
    path = tmp_path / "probs.jsonl"
    save_jsonl(problems, path)
    assert load_jsonl(path) == problems


def test_impossible_config_raises():
    cfg = DataConfig(min_target=10_000, max_target=10_001, max_operand=3)
    with pytest.raises(RuntimeError, match="只生成出"):
        build_countdown_dataset(10, cfg, seed=0)
