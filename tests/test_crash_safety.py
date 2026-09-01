"""掉线安全性测试。

Colab 会话随时可能断，这不是异常而是常态。所有「跑了很久才拿到」的东西都必须能
在断线中活下来，而且活下来的必须是可用的——半个 JSON 比没有 JSON 更糟，
因为它会把上一次成功的结果也一起毁掉。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def p1():
    return _load("run_p1_profile")


@pytest.fixture(scope="module")
def p2():
    return _load("run_p2_logprob")


def test_write_report_creates_missing_directories(p1, tmp_path):
    target = tmp_path / "a" / "b" / "report.json"
    p1.write_report({"x": 1}, target)
    assert json.loads(target.read_text(encoding="utf-8")) == {"x": 1}


def test_write_report_leaves_no_temp_file_behind(p1, tmp_path):
    """临时文件必须被 replace 掉。残留的 .tmp 会让人误以为上次写盘没成功。"""
    target = tmp_path / "report.json"
    p1.write_report({"x": 1}, target)
    assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


def test_write_report_preserves_previous_content_on_failure(p1, tmp_path):
    """写新内容失败时，旧结果必须原封不动。

    直接覆写的话，恰好在写到一半掉线会留下截断的 JSON，
    连上一次成功的结果一起毁掉——那才是最坏的情况。
    """
    target = tmp_path / "report.json"
    p1.write_report({"stage": "first"}, target)

    class Unserializable:
        pass

    with pytest.raises(TypeError):
        p1.write_report({"stage": Unserializable()}, target)

    assert json.loads(target.read_text(encoding="utf-8")) == {"stage": "first"}


def test_partial_report_is_still_valid_json(p1, tmp_path):
    """跑到一半的报告要能被绘图单元格读进去，而不是抛解析错误。"""
    target = tmp_path / "report.json"
    report = {"padding_leak": 0.01}
    p1.write_report(report, target)
    report["sweep"] = {"64": {"timing": {"total_s": 1.0}}}
    p1.write_report(report, target)

    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded["padding_leak"] == 0.01
    assert set(loaded["sweep"]) == {"64"}


def test_sweep_persists_each_setting_before_the_next(p1, tmp_path):
    """核心断言：第一档测完就落盘，不等第二档。

    否则在第二档掉线，第一档那十分钟 A100 就白烧了。
    """
    from dllm.config import Config

    out = tmp_path / "p1.json"
    config = Config()
    report: dict = {}
    seen: list[set[str]] = []

    class FakeArgs:
        sweep_diffusion_steps = [64, 128]
        warmup = 0
        steps = 1

    class FakeTrainer:
        config = None

    def fake_profile(trainer, args, cfg):
        # 进入本档时，磁盘上应已有前面各档的结果
        if out.exists():
            seen.append(set(json.loads(out.read_text(encoding="utf-8"))["sweep"]))
        else:
            seen.append(set())
        return {"timing": {"total_s": 1.0, "generation_frac": 0.5}}

    p1.run_sweep(FakeTrainer(), FakeArgs(), config, report, out, profile_fn=fake_profile)

    assert seen == [set(), {"64"}], f"每档结束都应落盘一次，实际观察到 {seen}"
    assert set(json.loads(out.read_text(encoding="utf-8"))["sweep"]) == {"64", "128"}


def test_sweep_restores_the_configured_diffusion_steps(p1, tmp_path):
    """扫参改完 config 必须还回去，否则 headline 悄悄变成最后一档。

    实测报告里 config 记 64、timing 记 128，两个数字在同一份 JSON 里对不上，
    而 report["config"] 是扫参前写的，光看文件看不出是哪一档，绘图和 README 全会引错。
    """
    from dllm.config import Config

    config = Config()
    configured = config.sampling.diffusion_steps

    class FakeArgs:
        sweep_diffusion_steps = [64, 128]
        warmup = 0
        steps = 1

    class FakeTrainer:
        config = None

    p1.run_sweep(
        FakeTrainer(),
        FakeArgs(),
        config,
        {},
        tmp_path / "p1.json",
        profile_fn=lambda *_: {"timing": {"total_s": 1.0}},
    )
    assert config.sampling.diffusion_steps == configured


def test_sweep_revalidates_diffusion_steps(p1):
    """直接给 config.sampling.diffusion_steps 赋值会绕过整除性校验，
    留下一个非法却不报错的配置。"""
    from dllm.config import Config

    config = Config()
    with pytest.raises(ValueError, match="必须能被块数"):
        p1.set_diffusion_steps(config, 63)


def test_p2_uses_the_same_atomic_writer(p2, tmp_path):
    target = tmp_path / "p2.json"
    p2.write_report({"experiment_a": {"pearson": 0.9}}, target)
    assert json.loads(target.read_text(encoding="utf-8"))["experiment_a"]["pearson"] == 0.9
    assert [p.name for p in tmp_path.iterdir()] == ["p2.json"]
