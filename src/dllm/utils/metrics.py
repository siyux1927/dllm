"""指标打点。

每步写一行 CSV 并立刻 flush。Colab 会断线，缓冲在内存里的指标会跟着进程一起消失，
所以宁可牺牲一点 IO 也要保证已经跑出来的数据落盘。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class MetricsLogger:
    def __init__(
        self,
        path: str | Path,
        report_to: str = "none",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
        resume: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._wandb = None

        if resume and self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), None)
            if header:
                self._fieldnames = header
        elif self.path.exists():
            self.path.unlink()

        if report_to == "wandb":
            import wandb

            self._wandb = wandb
            wandb.init(project="dllm-grpo", name=run_name, config=config, resume="allow")

    def log(self, row: dict[str, Any]) -> None:
        flat = {k: _to_scalar(v) for k, v in row.items()}
        write_header = False
        if self._fieldnames is None:
            self._fieldnames = list(flat)
            write_header = True

        # 首行之后不再扩列：缺的补空，多的丢弃，保证 CSV 始终是规整表格
        aligned = {name: flat.get(name, "") for name in self._fieldnames}

        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(aligned)

        if self._wandb is not None:
            self._wandb.log(flat)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()


def _to_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            return value
    return value
