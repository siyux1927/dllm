"""供 CPU 单测使用的极小掩码扩散语言模型。

存在的意义是让采样、log-prob 估计、GRPO loss 这些逻辑能在没有 GPU 的机器上跑通全链路。
接口与结构刻意向 LLaDA 看齐：双向注意力、输出不做位移、线性层命名沿用
q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj，
这样单测里的 LoRA target_modules 与真实配置是同一份，能提前暴露适配器挂不上的问题。

注意：这里刻意不用 nn.TransformerEncoderLayer。它在 eval 模式下会走融合快速路径，
绕过 linear1 / linear2 子模块，导致挂在这些模块上的 LoRA 被静默忽略——
前向照常返回结果，但适配器毫无作用。
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TinyModelOutput:
    logits: torch.Tensor


class TinyTokenizer:
    """词表极小的替身分词器，接口对齐 HF tokenizer 中本项目用到的那部分。

    词表里刻意放进 <answer> 标签、数字和四则运算符：这样随机初始化的小模型也能偶尔
    吐出格式合法的补全，奖励于是有非零方差、优势不全为零、梯度真的流动。
    若解码结果全是无意义字符，冒烟测试会在「损失恒为 0」的情况下通过，等于没测。
    """

    pad_token_id = 0
    eos_token_id = 1
    mask_token_id = 2

    def __init__(self, max_number: int = 60) -> None:
        self._vocab = ["<pad>", "<eos>", "<mask>"]
        self._vocab += ["<think>", "</think>", "<answer>", "</answer>"]
        self._vocab += [str(n) for n in range(max_number + 1)]
        self._vocab += ["+", "-", "*", "/", "(", ")", "=", "and", "then", "so"]
        self._first_regular = 3

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def apply_chat_template(
        self, messages, add_generation_prompt: bool = False, tokenize: bool = False
    ) -> str:
        text = " ".join(m["content"] for m in messages)
        return f"user: {text}" + (" assistant:" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        # 用 crc32 而非内置 hash：后者对字符串按进程加盐，同一段文本换个进程就是另一串 id
        span = self.vocab_size - self._first_regular
        ids = [
            self._first_regular + (zlib.crc32(word.encode("utf-8")) % span)
            for word in text.split()
        ]
        return {"input_ids": ids}

    def batch_decode(
        self, sequences: Sequence[Sequence[int]], skip_special_tokens: bool = True
    ) -> list[str]:
        skip = {self.pad_token_id, self.eos_token_id, self.mask_token_id}
        out = []
        for row in sequences:
            tokens = [
                self._vocab[int(i)]
                for i in row
                if not (skip_special_tokens and int(i) in skip)
            ]
            out.append(" ".join(tokens))
        return out


class TinyAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size({hidden_size}) 必须能被 num_heads({num_heads}) 整除")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        batch, seq_len, _ = hidden.shape
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        query = self.q_proj(hidden).view(shape).transpose(1, 2)
        key = self.k_proj(hidden).view(shape).transpose(1, 2)
        value = self.v_proj(hidden).view(shape).transpose(1, 2)
        # 不传 is_causal，扩散语言模型是双向注意力
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_bias)
        context = context.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(context)


class TinyMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = TinyAttention(hidden_size, num_heads)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.mlp = TinyMLP(hidden_size, intermediate_size)

    def forward(self, hidden: torch.Tensor, attn_bias: torch.Tensor | None) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), attn_bias)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class TinyMaskedDiffusionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 64,
        hidden_size: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        intermediate_size: int | None = None,
        max_position: int = 512,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_positions = nn.Embedding(max_position, hidden_size)
        self.layers = nn.ModuleList(
            TinyBlock(hidden_size, num_heads, intermediate_size or hidden_size * 2)
            for _ in range(num_layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> TinyModelOutput:
        _, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)

        attn_bias = None
        if attention_mask is not None:
            # (B, L) 的 0/1 掩码转成 (B, 1, 1, L) 的加性掩码，0 的位置置为 -inf
            keep = attention_mask.bool()[:, None, None, :]
            attn_bias = torch.zeros(keep.shape, dtype=hidden.dtype, device=hidden.device)
            attn_bias = attn_bias.masked_fill(~keep, -math.inf)

        for layer in self.layers:
            hidden = layer(hidden, attn_bias)
        return TinyModelOutput(logits=self.lm_head(self.norm(hidden)))
