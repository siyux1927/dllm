"""环境自检的测试。

自检本身出错的代价很高：它是上 Colab 后的第一道关，漏判会让人带着半好不坏的环境
一路跑到加载 16GB 权重才炸，而那时的报错指向 transformers 内部，看不出病因。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def check_env():
    spec = importlib.util.spec_from_file_location("check_env", SCRIPTS / "check_env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_env"] = module
    spec.loader.exec_module(module)
    return module


class FakeOps:
    """模拟 torch.ops.torchvision，取 nms 时按需抛错。"""

    def __init__(self, broken: bool) -> None:
        self._broken = broken

    @property
    def nms(self):
        if self._broken:
            raise RuntimeError("operator torchvision::nms does not exist")
        return object()


def fake_torch(version: str, broken_ops: bool = False):
    torch = types.SimpleNamespace()
    torch.__version__ = version
    torch.ops = types.SimpleNamespace(torchvision=FakeOps(broken_ops))
    return torch


@pytest.fixture
def stub_torchvision():
    """让 `import torchvision` 成功，好把探针推进到算子检查那一步。"""
    sys.modules.setdefault("torchvision", types.ModuleType("torchvision"))
    yield
    sys.modules.pop("torchvision", None)


def test_broken_ops_are_reported_with_the_actual_fix(check_env, stub_torchvision, monkeypatch):
    """真实报的错是 'operator torchvision::nms does not exist'，
    追溯栈却指向 transformers。自检必须把它翻译成「版本没配对」并给出修复命令。"""
    monkeypatch.setattr(check_env.md, "version", lambda name: "0.24.0+cu124")
    failures = check_env.check_torch_companions(fake_torch("2.6.0+cu124", broken_ops=True))

    assert len(failures) == 1
    message = failures[0]
    assert "ABI 不匹配" in message
    assert "torchvision==0.21.0" in message, "必须给出与 torch 2.6.0 配对的具体版本"
    assert "torch==2.6.0" in message


def test_working_ops_pass_even_if_torchaudio_mismatches(check_env, stub_torchvision, monkeypatch):
    """torchaudio 本项目从头到尾没有 import，不能因为它挡住整个自检。"""
    versions = {"torchvision": "0.21.0+cu124", "torchaudio": "2.11.0+cu124"}
    monkeypatch.setattr(check_env.md, "version", lambda name: versions[name])
    assert check_torch_ok(check_env, fake_torch("2.6.0+cu124"))


def test_missing_torchvision_is_not_a_failure(check_env, monkeypatch, capsys):
    """torchvision 没装反而无害：transformers 会跳过相关分支，不该判失败。"""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "torchvision":
            raise ModuleNotFoundError("No module named 'torchvision'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "torchvision", raising=False)

    def not_found(name):
        raise check_env.md.PackageNotFoundError(name)

    monkeypatch.setattr(check_env.md, "version", not_found)

    assert check_torch_ok(check_env, fake_torch("2.6.0+cu124"))
    # 确认真的走了「未安装」这条分支，而不是碰巧没报错
    assert "未安装 torchvision，跳过" in capsys.readouterr().out


def test_ops_probe_beats_a_matching_version_number(check_env, stub_torchvision, monkeypatch):
    """版本号对得上也可能坏：判据必须是算子探针，不是版本比对。

    Colab 上真实发生过的情形就是这种——版本装对了，但轮子是按另一个 torch 编译的。
    """
    monkeypatch.setattr(check_env.md, "version", lambda name: "0.21.0+cu124")
    failures = check_env.check_torch_companions(fake_torch("2.6.0+cu124", broken_ops=True))
    assert failures, "版本号一致但算子取不到时，仍必须判失败"


def test_unknown_torch_version_does_not_crash(check_env, stub_torchvision, monkeypatch):
    """torch 换了版本而 COMPANIONS 还没更新时，自检不该自己崩掉。"""
    monkeypatch.setattr(check_env.md, "version", lambda name: "0.24.0")
    assert check_torch_ok(check_env, fake_torch("2.9.0+cu128"))


def check_torch_ok(check_env, torch) -> bool:
    return check_env.check_torch_companions(torch) == []
