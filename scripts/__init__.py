"""入口脚本包，一律用 `python -m scripts.xxx` 从仓库根目录运行。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须早于 CUDA 初始化。采样期申请 GB 级大块、优化期申请微批小块，交替会把分配器的段切碎
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# src layout 下包不在仓库根目录，`-m` 只会把根目录放进 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
