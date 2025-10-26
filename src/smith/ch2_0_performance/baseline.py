import dataclasses
import wandb
import random
import torch as t
from datasets import load_dataset
from torch.utils.data import DataLoader
from jaxtyping import Int, Float
import torch.nn as nn
import einops
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import math
import dataclasses
from torch.optim import Adam
import numpy as np
import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity


@dataclasses.dataclass
class TransformerConfig:
  vocab_size: int
  d_model: int
  d_head: int
  n_head: int
  max_seq_len: int
  n_layers: int


class BaselineTransformer(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.config = config
    d_model = config.d_model
    self.d_head = config.d_head
    self.vocab_size = config.vocab_size
    self.max_seq_len = config.max_seq_len
    self.n_layers = config.n_layers

    self.embed = nn.Parameter(t.empty(config.vocab_size, d_model))
    self.embed = nn.init.kaiming_uniform_(self.embed)
    self.positional_embed = nn.Parameter(t.empty(self.max_seq_len, d_model))
    self.positional_embed = nn.init.kaiming_uniform_(self.positional_embed)

    self.blocks = nn.ModuleList([
        nn.ModuleDict({"params": nn.ParameterDict({
            "Q": nn.Parameter(t.empty(config.n_head, d_model, self.d_head)),
            "K": nn.Parameter(t.empty(config.n_head, d_model, self.d_head)),
            "V": nn.Parameter(t.empty(config.n_head, d_model, self.d_head)),
            "O": nn.Parameter(t.empty(config.n_head * self.d_head, d_model)),
            "W_in": nn.Parameter(t.empty(d_model, 4*d_model)),
            "W_out": nn.Parameter(t.empty(4*d_model, d_model)),
            }),
            "norm": nn.LayerNorm(d_model),
        })
        for _ in range(self.n_layers)
    ])
    for pd in self.blocks:
        for p in pd.params.values():
            nn.init.kaiming_uniform_(p)
    self.act_fn = nn.GELU()

    self.register_buffer(
        "attention_mask",
        t.empty(config.max_seq_len, config.max_seq_len),
        persistent=False)
    self.attention_mask = t.ones((config.max_seq_len, config.max_seq_len)) * float("-inf")
    self.attention_mask = t.triu(self.attention_mask, diagonal=1)

    #batch_x_arange = einops.repeat(t.arange(config.max_seq_len, device=x.device), "seq -> batch seq", batch=batch_size).detach()
    self.register_buffer(
        "one_hot",
        F.one_hot(t.arange(config.max_seq_len), num_classes=self.max_seq_len).to(t.float32),
        persistent=False)

  def forward(self, x: Int[t.Tensor, "batch seq b"]) -> Int[t.Tensor, "batch seq d_vocab"]:
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    d_head = t.tensor(self.d_head).to(t.long)

    x = einops.einsum(
        F.one_hot(x, num_classes=self.vocab_size).to(t.float32),
        self.embed,
        "batch seq d_vocab, d_vocab d_model -> batch seq d_model")
    # print(f"{self.one_hot.shape=} {self.one_hot.tile(batch_size, 1, 1).shape=} {self.one_hot.repeat}")
    positional_x = einops.einsum(
        self.one_hot.tile(batch_size, 1, 1),
        self.positional_embed,
        "batch seq max_seq_len, max_seq_len d_model -> batch seq d_model")

    x = x + positional_x[..., :seq_len, :]

    for i in range(self.n_layers):

        xQ = einops.einsum(x, self.blocks[i].params.Q, "batch q_seq d_model, n_head d_model d_head -> batch n_head q_seq d_head")
        Kx = einops.einsum(x, self.blocks[i].params.K, "batch k_seq d_model, n_head d_model d_head -> batch n_head k_seq d_head")
        attention_scores = einops.einsum(xQ, Kx, "batch n_head q_seq d_head, batch n_head k_seq d_head -> batch n_head q_seq k_seq")
        # print(f"{d_head.shape=}")
        attention_scores = attention_scores / math.sqrt(self.d_head)
        # czy na pewno kompatybilne jest attention z attention_mask ?
        # print(f"{i=} {self=}")
        masked_attention_scores = attention_scores + self.attention_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

        masked_attention = F.softmax(masked_attention_scores, dim=3) # Over keys

        # print(1*(masked_attention[0] > 0))


        xV = einops.einsum(x, self.blocks[i].params.V, "batch seq d_model, n_head d_model d_head -> batch n_head seq d_head")

        z = einops.einsum(masked_attention, xV, "batch n_head q_seq k_seq, batch n_head k_seq d_head -> batch q_seq n_head d_head")
        z = einops.rearrange(z, "batch seq n_head d_head -> batch seq (n_head d_head)")

        # attn_out = einops.einsum(z, self.blocks[i].params.O, "batch seq n_head_x_d_head, n_head_x_d_head d_model -> batch seq d_model")
        attn_out = z @ self.blocks[i].params.O
        x = x + attn_out
        x = self.blocks[i].norm(x)

        # mlp
        pre = einops.einsum(x, self.blocks[i].params.W_in, "batch seq d_model, d_model d_mlp -> batch seq d_mlp")
        post = self.act_fn(pre)
        mlp_out = einops.einsum(post, self.blocks[i].params.W_out, "batch seq d_mlp, d_mlp d_model -> batch seq d_model")
        x = x + mlp_out

    #unembed
    x = einops.einsum(x, self.embed, "batch seq d_model, d_vocab d_model -> batch seq d_vocab")

    return x