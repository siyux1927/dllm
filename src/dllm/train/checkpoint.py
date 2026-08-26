"""断点续训。

Colab 一次 session 撑不完 5-8 小时的训练，掉线是常态而非意外。除了权重，
优化器状态、调度器、数据游标和各路随机数生成器状态都必须一起存——
少存任何一样，恢复后的训练轨迹都和没断过时不同，实验就不可复现了。

正式开跑前务必手动 kill 一次验证恢复真的能接上。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

STATE_FILENAME = "state.pt"


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(directory: str | Path, payload: dict[str, Any]) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / STATE_FILENAME
    # 先写临时文件再原子替换，避免恰好在写盘时掉线留下半个损坏的 checkpoint
    tmp = target.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_checkpoint(directory: str | Path, map_location: str = "cpu") -> dict[str, Any] | None:
    target = Path(directory) / STATE_FILENAME
    if not target.exists():
        return None
    return torch.load(target, map_location=map_location, weights_only=False)
