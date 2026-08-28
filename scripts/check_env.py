"""环境自检：确认 dllm-dev 的依赖版本与 d1 官方 env.yml 对齐，且关键接口可用。

自检的价值在于把「装到一半坏了但要等真正用的时候才炸」变成「装完立刻说清楚哪儿不对」。
torchvision 的 ABI 检查就是这么加进来的：只换 torch 不换 torchvision，
错误会以一条六十行、指向 transformers 内部的 traceback 出现，根本看不出是版本没配对。
"""

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

# torch 与 torchvision / torchaudio 的 C++ 扩展是配对编译的，版本错开就注册不上算子
# https://github.com/pytorch/pytorch/wiki/PyTorch-Versions
COMPANIONS = {"2.6.0": {"torchvision": "0.21.0", "torchaudio": "2.6.0"}}


def check_torch_companions(torch) -> list[str]:
    """确认 torchvision 的 C++ 算子真的注册上了。

    判据是算子探针而非版本号。版本对不上未必坏，版本对得上也未必好——真正要紧的是
    `torchvision::nms` 能不能取到。它取不到时，transformers.image_utils 那句
    无条件的 `from torchvision import io` 会炸，连 AutoModel 都加载不了，
    而报出来的 traceback 指向 transformers 内部，看不出病因是版本没配对。

    只有 torchvision 阻断。torchaudio 本项目从头到尾没有任何地方 import，
    为它挡住整个自检是误报。
    """
    failures = []
    torch_base = torch.__version__.split("+")[0]
    expected = COMPANIONS.get(torch_base, {})
    want_vision = expected.get("torchvision")

    for name, want in expected.items():
        try:
            actual = md.version(name)
        except md.PackageNotFoundError:
            # 没装反而无害：transformers 会跳过相关分支
            print(f"  {name:<14} 未安装（无害，相关分支会被跳过）")
            continue
        matched = actual.split("+")[0] == want
        note = "ok" if matched else f"期望 {want}（配 torch {torch_base}）"
        if not matched and name == "torchaudio":
            note += " —— 仅提示，本项目不用它"
        print(f"  {name:<14} {actual:<16} {note}")

    fix = (
        f"pip install torch=={torch_base} torchvision=={want_vision} "
        "--extra-index-url https://download.pytorch.org/whl/cu124"
    )
    try:
        import torchvision  # noqa: F401

        # 取属性这个动作本身就是探针：算子没注册成功时，这一句会抛 RuntimeError
        nms_op = torch.ops.torchvision.nms
        assert nms_op is not None
    except ModuleNotFoundError:
        print("  torchvision 算子  未安装 torchvision，跳过")
    except Exception as exc:  # noqa: BLE001 - 就是要兜住任意底层异常并翻译成人话
        failures.append(
            f"torchvision 的算子注册失败（{type(exc).__name__}: {exc}）。"
            "\n    这是 torch 与 torchvision 的 ABI 不匹配，不是代码问题："
            "\n    二者的 C++ 扩展配对编译，只换其中一个就会这样。"
            "\n    transformers.image_utils 会 import torchvision，所以它坏了 AutoModel 就用不了。"
            f"\n    修复：{fix}"
        )
    else:
        print("  torchvision 算子  torchvision::nms 可用")
    return failures


def report_version_failures(failures: list[str]) -> int:
    """把版本不符翻译成下一步该做什么。

    全都不符和只有一个不符，病因完全不同：前者几乎一定是「这个会话里还没装依赖」。
    Colab 回收运行时后 pip 装的东西全没，Drive 上的文件却还在，
    于是很容易误以为环境还是上次那个。
    """
    print("\n自检失败:")
    for f in failures:
        print(f"  - {f}")

    if len(failures) >= len(EXPECTED) - 1:
        print(
            "\n几乎所有钉死的包都不对，说明这个会话里还没装依赖——"
            "\n上面那些是 Colab 的预装版本。"
            "\nColab 回收运行时后 pip 装的包全部消失（Drive 上的文件不受影响），"
            "\n所以每开一个新会话都要重装一次："
            "\n"
            "\n    !pip install -r requirements-colab.txt"
            "\n"
            "\n装完必须重启运行时，再从头跑一遍 notebook。"
        )
    else:
        print(
            "\n只有部分包不对，多半是某次 pip 把它们升上去了。"
            "\n重装一次即可：!pip install -r requirements-colab.txt"
        )
    print("\n后续导入检查已跳过：这些检查都以钉死的版本为前提。")
    return 1


def main() -> int:
    print(f"python  {platform.python_version()}  ({sys.executable})")
    if sys.version_info >= (3, 13):
        print("  注意：Python 3.13。numpy 1.26.4 没有 cp313 的 wheel，requirements 已按版本分流")

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

    for name in ("numpy", "torchvision", "torchaudio"):
        try:
            print(f"  {name:<14} {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"  {name:<14} 未安装")

    # 版本不对就到此为止。后面每一项检查都以「装的是钉死的那几个版本」为前提，
    # 前提不成立还硬往下走，只会拿到一条指向第三方库内部的 traceback——
    # 那正是这个脚本存在的意义所在，不该由它自己制造。
    if failures:
        return report_version_failures(failures)

    import torch

    print(f"\ntorch.cuda.is_available() = {torch.cuda.is_available()}")

    # 必须在 import transformers / trl 之前查：它们的导入链会踩到 torchvision，
    # 一旦踩爆，报出来的是一条指向 transformers 内部的 traceback，看不出真正的病因
    print("\ntorch 配套包检查")
    companion_failures = check_torch_companions(torch)
    if companion_failures:
        print("\n自检失败:")
        for f in companion_failures:
            print(f"  - {f}")
        print("\n后续导入检查已跳过：修好上面的问题再跑一次。")
        return 1

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
