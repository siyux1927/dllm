"""统一 θ、θ_old、θ_ref 三份策略的取用方式。

diffu-GRPO 每次内更新都需要三个前向：当前策略（带梯度）、更新前的策略（算 ratio 分母）、
参考策略（算 KL）。三者在真实训练里是同一个底座配不同的 LoRA 适配器，在 CPU 单测里
则是三份独立的小模型；这个类把差异挡住，让训练循环两边通用。

θ_ref 取的是 LoRA 关闭后的原始底座，因此不需要额外显存——这是单卡 40GB 能跑 8B 的关键，
官方那套独立 reference model 会再吃掉 16GB。
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Iterator

import torch
from torch import nn

DEFAULT_ADAPTER = "default"
OLD_ADAPTER = "old"


class Policy:
    def __init__(self, model: nn.Module, is_peft: bool | None = None) -> None:
        self.model = model
        self.is_peft = _looks_like_peft(model) if is_peft is None else is_peft
        self._old_model: nn.Module | None = None
        self._ref_model: nn.Module | None = None

        if not self.is_peft:
            # 没有适配器可切，只能各留一份拷贝。仅用于小模型单测。
            self._old_model = copy.deepcopy(model).eval()
            self._ref_model = copy.deepcopy(model).eval()
            for p in self._old_model.parameters():
                p.requires_grad_(False)
            for p in self._ref_model.parameters():
                p.requires_grad_(False)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def sync_old(self) -> None:
        """把当前策略快照为 θ_old。每个 outer step 开始时调用一次。"""
        if self.is_peft:
            _copy_peft_adapter(self.model, DEFAULT_ADAPTER, OLD_ADAPTER)
        else:
            assert self._old_model is not None
            self._old_model.load_state_dict(self.model.state_dict())

    @contextmanager
    def as_old(self) -> Iterator[nn.Module]:
        if not self.is_peft:
            assert self._old_model is not None
            with torch.no_grad():
                yield self._old_model
            return
        previous = _active_adapter(self.model)
        self.model.set_adapter(OLD_ADAPTER)
        try:
            with torch.no_grad():
                yield self.model
        finally:
            self.model.set_adapter(previous)

    @contextmanager
    def as_ref(self) -> Iterator[nn.Module]:
        if not self.is_peft:
            assert self._ref_model is not None
            with torch.no_grad():
                yield self._ref_model
            return
        with self.model.disable_adapter(), torch.no_grad():
            yield self.model


def _looks_like_peft(model: nn.Module) -> bool:
    return hasattr(model, "set_adapter") and hasattr(model, "disable_adapter")


def _active_adapter(model: nn.Module) -> str:
    active = getattr(model, "active_adapters", None)
    if active:
        return active[0]
    return getattr(model, "active_adapter", DEFAULT_ADAPTER)


def _copy_peft_adapter(model: nn.Module, source: str, target: str) -> None:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    state = get_peft_model_state_dict(model, adapter_name=source)
    set_peft_model_state_dict(model, state, adapter_name=target)
