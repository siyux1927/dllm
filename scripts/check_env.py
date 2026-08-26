"""环境自检：确认 dllm-dev 的依赖版本与 d1 官方 env.yml 对齐，且关键接口可用。"""

import importlib.metadata as md
import platform
import sys

# 版本必须与 d1 官方 env.yml 一致，否则本地写的代码到 Colab 上可能行为不同
EXPECTED = {
    "torch": "2.6.0",
    "transformers": "4.49.0",
    "accelerate": "1.4.0",
    "peft": "0.15.1",
    "datasets": "3.3.2",
    "trl": "0.16.0.dev0",
}


def main() -> int:
    print(f"python  {platform.python_version()}  ({sys.executable})")

    failures = []
    for name, expected in EXPECTED.items():
        try:
            actual = md.version(name)
        except md.PackageNotFoundError:
            failures.append(f"{name}: 未安装")
            print(f"  {name:<14} 未安装")
            continue
        # torch 的本地版本号带 +cpu / +cu124 后缀
        base = actual.split("+")[0]
        ok = base == expected
        print(f"  {name:<14} {actual:<16} {'ok' if ok else f'期望 {expected}'}")
        if not ok:
            failures.append(f"{name}: {actual} != {expected}")

    import torch

    print(f"\ntorch.cuda.is_available() = {torch.cuda.is_available()}")

    from trl import GRPOConfig, GRPOTrainer  # noqa: F401

    print("trl GRPOTrainer / GRPOConfig 导入成功")

    from peft import LoraConfig  # noqa: F401
    from transformers import AutoModel, AutoTokenizer  # noqa: F401

    print("peft LoraConfig、transformers Auto* 导入成功")

    # d1 依赖的 GRPOConfig 字段，缺任何一个都说明 trl 版本不对
    required_fields = {"num_iterations", "epsilon", "beta", "num_generations"}
    missing = required_fields - set(GRPOConfig.__dataclass_fields__)
    if missing:
        failures.append(f"GRPOConfig 缺少字段: {sorted(missing)}")
        print(f"GRPOConfig 缺少字段: {sorted(missing)}")
    else:
        print(f"GRPOConfig 关键字段齐全: {sorted(required_fields)}")

    if failures:
        print("\n自检失败:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\n自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
