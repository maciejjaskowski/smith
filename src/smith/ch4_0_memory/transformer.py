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
  learning_rate: float
  batch_size: int
  epochs: int

class Transformer(nn.Module):

  def __init__(self, config: TransformerConfig):
    super().__init__()
    self.config = config
    d_model = config.d_model
    self.d_head = config.d_head
    self.vocab_size = config.vocab_size
    self.max_seq_len = config.max_seq_len
    self.n_layers = config.n_layers

    self.embed = nn.Parameter(t.empty(config.vocab_size, d_model))
    nn.init.kaiming_uniform_(self.embed)
    self.positional_embed = nn.Parameter(t.empty(self.max_seq_len, d_model))
    nn.init.kaiming_uniform_(self.positional_embed)

    self.blocks = nn.ModuleList([
        nn.ModuleDict({"params": nn.ParameterDict({
            "Q": nn.Parameter(t.empty(d_model, config.n_head * self.d_head)),
            "K": nn.Parameter(t.empty(d_model, config.n_head * self.d_head)),
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

    attention_mask = t.ones((config.max_seq_len, config.max_seq_len)) * float("-inf")
    attention_mask = t.triu(attention_mask, diagonal=1)
    self.register_buffer(
        "attention_mask",
        attention_mask,
        persistent=False)
    

    #batch_x_arange = einops.repeat(t.arange(config.max_seq_len, device=x.device), "seq -> batch seq", batch=batch_size).detach()
    self.register_buffer(
        "one_hot",
        F.one_hot(t.arange(config.max_seq_len), num_classes=self.max_seq_len),
        persistent=False)
    
    self.register_buffer(
        "arange",
        t.arange(config.max_seq_len),
        persistent=False,
    )

  def forward(self, x: Int[t.Tensor, "batch seq b"]) -> Int[t.Tensor, "batch seq d_vocab"]:
    batch_size = x.shape[0]
    seq_len = x.shape[1]
    d_head = t.tensor(self.d_head).to(t.long)

    x = F.embedding(x, self.embed) # [batch seq d_model]
    # Mem: B * S * D

    positional_x = F.embedding(
        self.arange[:seq_len],
        self.positional_embed) # [seq d_model]
    # Mem: S * D
    x = x + positional_x
    # Mem: B * S * D

    for i in range(self.n_layers):
        xQ = x @ self.blocks[i].params.Q
        # Mem: B * S * D * N
        xQ = einops.rearrange(xQ, "batch seq (n_head d_head) -> batch n_head seq d_head", n_head=self.config.n_head).contiguous()
        # Mem: B * S * D * N

        Kx = x @ self.blocks[i].params.K                
        # Mem: B * S * D * N
        Kx = einops.rearrange(Kx, "batch seq (n_head d_head) -> batch n_head d_head seq", n_head=self.config.n_head).contiguous()
        # Mem: B * S * D * N
        attention_scores = einops.einsum(
            xQ, Kx,
            "batch n_head q_seq d_head, batch n_head d_head k_seq -> batch n_head q_seq k_seq") 
        # Mem: B * S * S * N

        attention_scores = attention_scores / math.sqrt(self.d_head)
        # Mem: B * S * S * N
        masked_attention_scores = attention_scores + self.attention_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)
        # Mem: B * S * S * N
        masked_attention = F.softmax(masked_attention_scores, dim=-1)
        # Mem: B * S * S * N

        xV = einops.einsum(x, self.blocks[i].params.V, "batch seq d_model, n_head d_model d_head -> batch n_head seq d_head")
        # Mem: B * N * S * H
        z = einops.einsum(masked_attention, xV, "batch n_head q_seq k_seq, batch n_head k_seq d_head -> batch q_seq n_head d_head")
        # Mem: B * S * N * H
        z = einops.rearrange(z, "batch seq n_head d_head -> batch seq (n_head d_head)")


        attn_out = einops.einsum(z, self.blocks[i].params.O, "batch seq n_head_x_d_head, n_head_x_d_head d_model -> batch seq d_model")
        # Mem: B * S * D
        x = x + attn_out
        # Mem: B * S * D
        
        x = self.blocks[i].norm(x)
        # Mem: B * S * D

        # mlp
        pre = einops.einsum(x, self.blocks[i].params.W_in, "batch seq d_model, d_model d_mlp -> batch seq d_mlp")
        # Mem: B * S * 4D
        post = self.act_fn(pre)
        # Mem: B * S * 4D
        mlp_out = einops.einsum(post, self.blocks[i].params.W_out, "batch seq d_mlp, d_mlp d_model -> batch seq d_model")
        # Mem: B * S * D
        x = x + mlp_out
        # Mem: B * S * D

    #unembed
    x = einops.einsum(x, self.embed, "batch seq d_model, d_vocab d_model -> batch seq d_vocab")
    # Mem: B * S * V

    return x

    def loss(self, logits, labels):
        loss = nn.CrossEntropyLoss()
        return loss(logits.view(-1, self.config.vocab_size), labels.view(-1))
        

if __name__ == "__main__":
    # Simple test
    config = TransformerConfig(
        vocab_size=67,
        d_model=512,
        n_head=8,
        d_head=64,
        n_layers=6,
        max_seq_len=128,
        learning_rate=0.00005,
        batch_size=None,
        epochs=None,
    )
    model = Transformer(config)
    model.to("cuda")


    
    model_params = t.cuda.memory_allocated()
    print(f"Model parameters: {model_params/1024/1024} MB")

    optim_params = 0
    # optim = Adam(model.parameters(), lr=config.learning_rate)

    # optim_params = t.cuda.memory_allocated() - model_params    
    # print(f"Model parameters: {optim_params/1024/1024} MB")
    

    batch_size = 4
    seq_len = 32
    x = t.randint(0, config.vocab_size, (batch_size, seq_len)).to("cuda")  
    inp = x[:, :-1]
    pred = model(inp)
    
    activations = t.cuda.memory_allocated() - optim_params - model_params
    print(f"Model activations: {activations/1024/1024} MB")
    
    print(f"{pred.shape=} {x.shape=}")
    loss_val = nn.CrossEntropyLoss()(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                    target=einops.rearrange(x[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos
            
    loss_val.backward()
    gradients = t.cuda.memory_allocated() - activations    
    print(f"Gradients: {gradients/1024/1024} MB")
    
    #out = model(x)
    
    
    
    
    # Teraz chce zmierzyc zuzycie pamiec przy pomocy torch.cuda.memory_allocated() itp.
    # Jak rowniez policzyc na palcach zuzycie przed forward:
    #  - parametry modelu oraz parametry optymalizatora (osobno)
    #  - zuzycie pamieci po zrobioniu forward pass (czyli ile zajmuja aktywacje)
    #  - zuzycie pamieci po zrobioniu backward pass (czyli ile zajmuja gradienty)
    
    # Mem: B * S * D
    # Mem: B * S * D
    # Mem: -- B * S * D -- B * S
    # Mem: B * S * D * N
    # Mem: B * S * D * N
    # Mem: B * S * D * N
    # Mem: B * S * D * N
    # Mem: B * S * S * N
    # Mem: B * S * S * N
    # Mem: B * S * S * N
    # Mem: B * S * S * N
    # Mem: B * N * S * H
    # Mem: B * S * N * H
    # Mem: -- B * S * N * D --
    # Mem: B * S * D
    # Mem: B * S * D
    # Mem: B * S * D
    
    # Mem: B * S * D * 4D
    # Mem: B * S * D * 4D
    # Mem: B * S * D
    # Mem: B * S * D
    # Mem: B * S * V
    
    # B * S -> 1 -> 1 * 4 * 32 / 1024 / 1024 = 0.00012207031 MB
    # B * S * D -> 7. -> 7 * 4 * 32 * 512/1024/1024 = 0.4375 MB
    # B * S * D * N -> 5 -> 5 * 4 * 32 * 512 * 8 /1024 / 1024 = 2.5 MB
    # B * S * S * N -> 4 -> 4 * 4 * 32 * 32 * 8 /1024 /1024 = 0.125 MB
    # B * S * N * H -> 2 -> 2 * 4 * 32 * 8 * 64 /1024 /1024 = 0.125 MB
    # B * S * 4D -> 2 -> 2 * 4 * 32 * 2048 /1024 /1024 = 0.5 MB
    # B * S * V -> 1 -> 1 * 4 * 32 * 67 / 1024 / 1024 = 0.008 MB
    
    # 4 MB paremetrów -> 16 MB pamięci * 4 layers = 64 MB pamięci
    

    # w trakcie activations, odpalaja sie rowniez wagi Adama, ktore robia drugie tyle.
    
    # jak odpalam bez optimizera, to wychodzi 60.78 MB
    
    # 4 * (0.00012207031 + 0.4375 + 2.5 + 0.125 + 0.125 + 0.5 + 0.008) = 59.129 MB