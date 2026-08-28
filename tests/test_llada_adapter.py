"""LLaDA 装配层的校验逻辑测试，全部不需要下载权重。

这些校验存在的意义是把「配错了但照常训练」变成「立刻报错」。
"""

from __future__ import annotations

import pytest
import torch

from dllm.config import ModelConfig
from dllm.models.llada import (
    ALL_LINEAR,
    LLaDAPromptCodec,
    linear_module_suffixes,
    lm_head_exclusion,
    probe_lora_is_live,
    resolve_target_modules,
)
from dllm.models.tiny import TinyMaskedDiffusionLM
from dllm.train.policy import Policy

peft = pytest.importorskip("peft")


class FakeTokenizer:
    """够用的最小分词器，字符级映射，便于断言 padding 与截断行为。"""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, with_chat_template: bool = False) -> None:
        if with_chat_template:
            self.apply_chat_template = self._apply_chat_template

    def _apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        return "U:" + messages[0]["content"] + ("|A:" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 50) for c in text]}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["".join(chr(65 + int(i) % 26) for i in row) for row in ids]


def test_linear_suffixes_match_llada_style_naming():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    suffixes = linear_module_suffixes(model)
    for name in ModelConfig().lora_target_modules:
        assert name in suffixes, f"{name} 应存在于小模型中"


def test_resolve_accepts_matching_modules():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    resolved = resolve_target_modules(model, ["q_proj", "attn_out"])
    assert resolved == ["q_proj", "attn_out"]


def test_resolve_passes_through_all_linear():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    assert resolve_target_modules(model, ALL_LINEAR) == ALL_LINEAR


def test_resolve_rejects_completely_wrong_names():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    with pytest.raises(ValueError, match="一个都匹配不上") as exc:
        resolve_target_modules(model, ["att_proj", "wqkv"])
    assert "q_proj" in str(exc.value), "报错里应给出候选名单"


def test_resolve_rejects_partial_match():
    """部分命中比全不命中更危险：会得到一个和预期不同、却照常训练的模型。"""
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    with pytest.raises(ValueError, match="只匹配上一部分"):
        resolve_target_modules(model, ["q_proj", "att_proj"])


def test_llama_names_are_rejected_with_the_llada_translation():
    """d1 官方原样填的就是这份 Llama 命名，对 LLaDA 只命中 q/k/v/up。

    静默少挂三类模块，训练照跑、曲线照有，只是模型和以为的不是同一个。
    报错必须给出改名方案，否则下一个人还得再查一遍 OLMo 的源码。
    """
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    d1_official = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    with pytest.raises(ValueError, match="只匹配上一部分") as exc:
        resolve_target_modules(model, d1_official)

    message = str(exc.value)
    assert "o_proj → attn_out" in message
    assert "gate_proj → ff_proj" in message
    assert "down_proj → ff_out" in message


def test_config_default_targets_all_resolve():
    """配置里的默认目标必须在结构与 LLaDA 对齐的替身上全部命中。"""
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    targets = list(ModelConfig().lora_target_modules)
    assert resolve_target_modules(model, targets) == targets


# --- 词表投影与 FFN 下投影重名 -----------------------------------------------


def test_lm_head_shares_its_name_with_the_ffn_down_projection():
    """替身必须保留 LLaDA 这个重名，否则下面那条排除逻辑在 CPU 上根本测不到。"""
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    assert linear_module_suffixes(model)["ff_out"] == 3, "2 个块内下投影 + 1 个词表投影"


def test_targeting_ff_out_excludes_only_the_vocab_projection():
    """排除项是全匹配正则，只咬住词表投影本身，块内的同名下投影不受影响。"""
    import re

    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    pattern = lm_head_exclusion(model, ["q_proj", "ff_out"])
    assert re.fullmatch(pattern, "ff_out")
    assert not re.fullmatch(pattern, "layers.0.mlp.ff_out")


def test_no_exclusion_when_ff_out_is_not_targeted():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    assert lm_head_exclusion(model, ["q_proj", "attn_out"]) is None


def test_vocab_projection_stays_frozen_while_block_projections_train():
    """这条才是真正要守的东西：排除词表投影，但不能把块内的下投影一起排掉。

    PEFT 的排除规则和匹配规则一样是按名字末段，排除项若写成 "ff_out"，
    会把每个块的 FFN 下投影也一并排掉——等于 FFN 根本没挂上适配器，而且悄无声息。
    """
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    targets = list(ModelConfig().lora_target_modules)
    wrapped = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=targets,
            exclude_modules=lm_head_exclusion(model, targets),
        ),
    )
    adapted = {
        name.rsplit(".lora_A", 1)[0]
        for name, _ in wrapped.named_modules()
        if name.endswith("lora_A")
    }
    assert not any(name.endswith("base_model.model.ff_out") for name in adapted), (
        "词表投影不该被挂上 LoRA"
    )
    block_downs = [name for name in adapted if ".mlp.ff_out" in name]
    assert len(block_downs) == 2, f"两个块的 FFN 下投影都该挂上，实际 {sorted(adapted)}"


def test_codec_left_pads_to_fixed_length():
    codec = LLaDAPromptCodec(FakeTokenizer(), max_prompt_length=12)
    ids, mask = codec.encode(["abc", "abcdefgh"])
    assert ids.shape == (2, 12)
    # 左侧补齐，让补全紧接在 prompt 之后而不是被推到序列中间
    assert mask[0].tolist() == [0] * 9 + [1] * 3
    assert mask[1].tolist() == [0] * 4 + [1] * 8
    assert (ids[0][:9] == 0).all()


def test_codec_truncates_keeping_the_tail():
    """题面的关键信息（数字与目标）在后半段，截断必须保留末尾。"""
    codec = LLaDAPromptCodec(FakeTokenizer(), max_prompt_length=4)
    ids, mask = codec.encode(["abcdefgh"])
    full = FakeTokenizer()("abcdefgh")["input_ids"]
    assert ids[0].tolist() == full[-4:]
    assert mask[0].tolist() == [1, 1, 1, 1]


def test_codec_applies_chat_template_when_available():
    plain = LLaDAPromptCodec(FakeTokenizer(with_chat_template=False), max_prompt_length=32)
    chat = LLaDAPromptCodec(FakeTokenizer(with_chat_template=True), max_prompt_length=32)
    assert chat.use_chat_template and not plain.use_chat_template
    assert not torch.equal(plain.encode(["hi"])[0], chat.encode(["hi"])[0])


def test_codec_falls_back_to_eos_when_pad_missing():
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = None
    codec = LLaDAPromptCodec(tokenizer, max_prompt_length=8)
    assert codec.pad_token_id == tokenizer.eos_token_id


def test_codec_raises_without_any_padding_token():
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="无法左侧补齐"):
        LLaDAPromptCodec(tokenizer, max_prompt_length=8)


def _peft_policy(target_modules):
    torch.manual_seed(0)
    base = TinyMaskedDiffusionLM(vocab_size=64, hidden_size=32, num_layers=2)
    lora = peft.LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, target_modules=list(target_modules)
    )
    model = peft.get_peft_model(base, lora)
    model.add_adapter("old", lora)
    model.set_adapter("default")
    model.eval()
    return Policy(model, is_peft=True)


def test_probe_detects_live_lora():
    policy = _peft_policy(ModelConfig().lora_target_modules)
    assert probe_lora_is_live(policy, torch.randint(0, 60, (1, 6)))


def test_probe_restores_weights_after_checking():
    """探针会临时改权重，必须原样还原，否则它自己就污染了训练。"""
    policy = _peft_policy(ModelConfig().lora_target_modules)
    sample = torch.randint(0, 60, (1, 6))
    with torch.no_grad():
        before = policy.model(sample).logits.clone()
    probe_lora_is_live(policy, sample)
    with torch.no_grad():
        after = policy.model(sample).logits
    assert torch.allclose(before, after)


def test_probe_raises_when_no_adapter_present():
    model = TinyMaskedDiffusionLM(vocab_size=64, hidden_size=32, num_layers=2)
    policy = Policy(model, is_peft=False)
    with pytest.raises(ValueError, match="LoRA 没有真正装上"):
        probe_lora_is_live(policy, torch.randint(0, 60, (1, 6)))
