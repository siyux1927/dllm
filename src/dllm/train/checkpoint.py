"""断点续训。

Colab 一次 session 撑不完 5-8 小时的训练，掉线是常态而非意外。除了权重，
优化器状态、调度器、数据游标和各路随机数生成器状态都必须一起存——
少存任何一样，恢复后的训练轨迹都和没断过时不同，实验就不可复现了。

正式开跑前务必手动 kill 一次验证恢复真的能接上。
"""

from __future__ import annotations

import random
import shutil
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


def checkpoint_bytes(directory: str | Path) -> int:
    target = Path(directory) / STATE_FILENAME
    return target.stat().st_size if target.exists() else 0


def mirror_checkpoint(source: str | Path, destination: str | Path) -> Path | None:
    """把 checkpoint 复制到另一处（通常是 Google Drive）。

    为什么要分两级而不是直接存到 Drive：LoRA r=128 挂在 LLaDA-8B 的七个投影上约 3.4 亿参数，
    加上 AdamW 的一阶二阶矩，单个 checkpoint 约 4GB。Drive 是网络挂载，写入约 10-20MB/s，
    存一次要 3-7 分钟——比一个训练步还慢。所以高频存本地（掉进程能救），
    低频镜像到 Drive（掉会话能救），两种故障覆盖的代价不同，频率也就该不同。
    """
    src = Path(source) / STATE_FILENAME
    if not src.exists():
        return None
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / STATE_FILENAME
    tmp = target.with_suffix(".tmp")
    shutil.copyfile(src, tmp)
    tmp.replace(target)
    return target
