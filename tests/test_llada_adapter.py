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
    resolved = resolve_target_modules(model, ["q_proj", "o_proj"])
    assert resolved == ["q_proj", "o_proj"]


def test_resolve_passes_through_all_linear():
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    assert resolve_target_modules(model, ALL_LINEAR) == ALL_LINEAR


def test_resolve_rejects_completely_wrong_names():
    """LLaDA 派生自 OLMo，线性层可能叫 att_proj / ff_proj。配错时必须报错并列出真实命名。"""
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    with pytest.raises(ValueError, match="一个都匹配不上") as exc:
        resolve_target_modules(model, ["att_proj", "ff_proj"])
    assert "q_proj" in str(exc.value), "报错里应给出候选名单"


def test_resolve_rejects_partial_match():
    """部分命中比全不命中更危险：会得到一个和预期不同、却照常训练的模型。"""
    model = TinyMaskedDiffusionLM(hidden_size=32, num_layers=2)
    with pytest.raises(ValueError, match="只匹配上一部分"):
        resolve_target_modules(model, ["q_proj", "att_proj"])


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
