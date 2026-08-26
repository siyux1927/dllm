"""分阶段计时。

第 1 幕的全部论据来自这里：要证明 rollout 占单步耗时的 70-80%，就得从第一步开始
把 generation / reward / logprob / forward_backward 四段分开记。事后补不回来。

CUDA 是异步的，不同步就计时会把等待算到下一段头上，所以真实训练必须传 sync 回调。
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter
from typing import Callable, Iterator


class PhaseTimer:
    def __init__(self, sync: Callable[[], None] | None = None) -> None:
        self._sync = sync or (lambda: None)
        self._durations: dict[str, float] = defaultdict(float)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        self._sync()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._durations[name] += perf_counter() - start

    def reset(self) -> None:
        self._durations.clear()

    @property
    def total(self) -> float:
        return sum(self._durations.values())

    def summary(self, prefix: str = "time") -> dict[str, float]:
        """返回各阶段秒数与占比，键形如 time/generation_s、time/generation_frac。"""
        total = self.total
        out: dict[str, float] = {f"{prefix}/total_s": total}
        for name, seconds in sorted(self._durations.items()):
            out[f"{prefix}/{name}_s"] = seconds
            out[f"{prefix}/{name}_frac"] = seconds / total if total > 0 else 0.0
        return out
