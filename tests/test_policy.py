import pytest
import torch

from dllm.config import ModelConfig
from dllm.models.tiny import TinyMaskedDiffusionLM
from dllm.train.policy import Policy

peft = pytest.importorskip("peft")


def _perturb(model: torch.nn.Module, scale: float = 0.1) -> None:
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p) * scale)


def test_plain_model_snapshot_is_frozen_until_synced(tiny_model, prompt_ids):
    policy = Policy(tiny_model, is_peft=False)
    policy.sync_old()

    with policy.as_old() as old:
        before = old(prompt_ids).logits.clone()

    _perturb(tiny_model)

    with policy.as_old() as old:
        after = old(prompt_ids).logits
    assert torch.allclose(before, after), "θ_old 在 sync 之前不应跟着 θ 变"

    policy.sync_old()
    with policy.as_old() as old:
        resynced = old(prompt_ids).logits
    assert not torch.allclose(before, resynced), "sync 之后 θ_old 应追上 θ"


def test_plain_reference_never_moves(tiny_model, prompt_ids):
    policy = Policy(tiny_model, is_peft=False)
    with policy.as_ref() as ref:
        before = ref(prompt_ids).logits.clone()
    _perturb(tiny_model)
    policy.sync_old()
    with policy.as_ref() as ref:
        after = ref(prompt_ids).logits
    assert torch.allclose(before, after), "θ_ref 必须始终是初始策略"


def test_old_context_disables_gradients(tiny_model, prompt_ids):
    policy = Policy(tiny_model, is_peft=False)
    policy.sync_old()
    with policy.as_old() as old:
        assert not old(prompt_ids).logits.requires_grad


def _peft_policy():
    torch.manual_seed(0)
    base = TinyMaskedDiffusionLM(vocab_size=64, hidden_size=32, num_layers=2)
    # 用与真实训练相同的 target_modules，这样「适配器挂不上」会在 CPU 上就暴露
    config = peft.LoraConfig(
        r=4,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=list(ModelConfig().lora_target_modules),
    )
    model = peft.get_peft_model(base, config)
    model.add_adapter("old", config)
    model.set_adapter("default")
    model.eval()
    return Policy(model, is_peft=True)


def test_peft_policy_is_detected_automatically():
    policy = _peft_policy()
    assert Policy(policy.model).is_peft


def test_target_modules_are_actually_wrapped_and_invoked():
    """回归测试。

    之前小模型基于 nn.TransformerEncoderLayer，它在 eval 模式下走融合快速路径、
    绕过线性子模块，于是 LoRA 挂得上却完全不参与前向——训练照常收敛似的跑，
    适配器却是死的。这里同时断言「被包装」和「被调用」。
    """
    policy = _peft_policy()
    wrapped = [
        module
        for name, module in policy.model.named_modules()
        if type(module).__module__.startswith("peft.tuners.lora")
        and hasattr(module, "lora_A")
    ]
    assert wrapped, "没有任何模块被 LoRA 包装"

    calls: list[str] = []
    originals = {}
    for name, module in policy.model.named_modules():
        if hasattr(module, "lora_A"):
            originals[name] = module.forward

            def spy(*args, _name=name, _orig=module.forward, **kwargs):
                calls.append(_name)
                return _orig(*args, **kwargs)

            module.forward = spy
    try:
        with torch.no_grad():
            policy.model(torch.randint(0, 60, (1, 4)))
    finally:
        for name, module in policy.model.named_modules():
            if name in originals:
                module.forward = originals[name]

    assert calls, "LoRA 模块存在但前向没有经过它们"


def test_peft_reference_is_the_adapter_free_backbone(prompt_ids):
    """θ_ref 取的是关掉 LoRA 后的底座，因此不额外占显存。"""
    policy = _peft_policy()
    with torch.no_grad():
        with policy.as_ref() as ref:
            ref_logits = ref(prompt_ids).logits.clone()

    for name, p in policy.model.named_parameters():
        if "lora_B" in name and "default" in name:
            with torch.no_grad():
                p.add_(1.0)  # LoRA 初始化时 B 为零，加偏置才会真正改变输出

    with torch.no_grad():
        active_logits = policy.model(prompt_ids).logits
        with policy.as_ref() as ref:
            ref_again = ref(prompt_ids).logits

    assert not torch.allclose(ref_logits, active_logits), "启用适配器后输出应改变"
    assert torch.allclose(ref_logits, ref_again), "关闭适配器后应回到同一个底座输出"


def test_peft_sync_old_copies_adapter_weights(prompt_ids):
    policy = _peft_policy()

    for name, p in policy.model.named_parameters():
        if "lora_B" in name and "default" in name:
            with torch.no_grad():
                p.add_(0.5)

    policy.sync_old()
    with torch.no_grad():
        current = policy.model(prompt_ids).logits.clone()
        with policy.as_old() as old:
            snapshot = old(prompt_ids).logits.clone()
    assert torch.allclose(current, snapshot), "sync 之后 θ_old 应与 θ 完全一致"

    for name, p in policy.model.named_parameters():
        if "lora_B" in name and "default" in name:
            with torch.no_grad():
                p.add_(0.5)

    with torch.no_grad():
        moved = policy.model(prompt_ids).logits
        with policy.as_old() as old:
            still = old(prompt_ids).logits
    assert not torch.allclose(moved, still), "θ 继续更新时 θ_old 应保持不动"
    assert torch.allclose(snapshot, still)


def test_peft_context_restores_active_adapter(prompt_ids):
    policy = _peft_policy()
    policy.sync_old()
    with policy.as_old():
        pass
    assert policy.model.active_adapter == "default"


def test_peft_trainable_parameters_are_lora_only():
    policy = _peft_policy()
    names = [n for n, p in policy.model.named_parameters() if p.requires_grad]
    assert names, "应有可训练参数"
    assert all("lora" in n for n in names), "只有 LoRA 适配器该被训练"


def test_trainable_parameters_exclude_the_old_adapter():
    """优化器不该持有 θ_old 的参数。

    当前 peft 版本下 add_adapter 只把激活的适配器标为可训练，所以这里手动把 old 的参数
    置为 requires_grad，构造出那个危险状态：只靠 requires_grad 筛选的话，优化器会一并
    收下 θ_old。这条筛选不能依赖 peft 的默认行为。
    """
    policy = _peft_policy()
    old_params = [p for name, p in policy.model.named_parameters() if ".old." in name]
    assert old_params, "测试前提：old 适配器确实存在"
    for p in old_params:
        p.requires_grad_(True)

    trainable = {id(p) for p in policy.trainable_parameters()}
    assert not any(id(p) in trainable for p in old_params)
    expected = {
        id(p)
        for name, p in policy.model.named_parameters()
        if ".default." in name and p.requires_grad
    }
    assert trainable == expected


def test_trainable_parameters_raise_when_default_adapter_missing():
    torch.manual_seed(0)
    base = TinyMaskedDiffusionLM(vocab_size=64, hidden_size=32, num_layers=2)
    config = peft.LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, target_modules=list(ModelConfig().lora_target_modules)
    )
    model = peft.get_peft_model(base, config, adapter_name="policy")
    with pytest.raises(ValueError, match="没找到任何属于"):
        Policy(model, is_peft=True).trainable_parameters()
