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


class KVCachedTransformer(nn.Module):

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

  def forward(self, x: Int[t.Tensor, "batch seq"]) -> Int[t.Tensor, "batch seq d_vocab"]:
    with record_function("preparation"):
        batch_size = x.shape[0]
        q_seq = x.shape[1]
        d_head = t.tensor(self.d_head).to(t.long)

    with record_function("embeddings"):
        x = F.embedding(x, self.embed) # [batch seq d_model]

        positional_x = F.embedding(
            self.arange[:q_seq],
            self.positional_embed) # [batch seq d_model]
        x = x + positional_x

    for i in range(self.n_layers):
        with record_function("layer"):
            with record_function("QK"):
                xQ = x @ self.blocks[i].params.Q
                xQ = einops.rearrange(xQ, "batch seq (n_head d_head) -> batch n_head seq d_head", n_head=self.config.n_head).contiguous()

                Kx = x @ self.blocks[i].params.K                
                Kx = einops.rearrange(Kx, "batch seq (n_head d_head) -> batch n_head d_head seq", n_head=self.config.n_head).contiguous()
            with record_function("attention_scores"):
                attention_scores = einops.einsum(
                    xQ, Kx,
                    "batch n_head q_seq d_head, batch n_head d_head k_seq -> batch n_head q_seq k_seq") # Czy to moze byc szybsze?
                attention_scores = attention_scores / math.sqrt(self.d_head)
                masked_attention_scores = attention_scores + self.attention_mask[:q_seq, :q_seq].unsqueeze(0).unsqueeze(0)

                masked_attention = F.softmax(masked_attention_scores, dim=-1) # Over keys

            with record_function("attention_output"):
                xV = einops.einsum(x, self.blocks[i].params.V, "batch seq d_model, n_head d_model d_head -> batch n_head seq d_head")

                z = einops.einsum(masked_attention, xV, "batch n_head q_seq k_seq, batch n_head k_seq d_head -> batch q_seq n_head d_head")
                z = einops.rearrange(z, "batch seq n_head d_head -> batch seq (n_head d_head)")

                attn_out = einops.einsum(z, self.blocks[i].params.O, "batch seq n_head_x_d_head, n_head_x_d_head d_model -> batch seq d_model")
                
                x = x + attn_out
            
            
            x = self.blocks[i].norm(x)

            # mlp
            with record_function("mlp"):
                pre = einops.einsum(x, self.blocks[i].params.W_in, "batch seq d_model, d_model d_mlp -> batch seq d_mlp")
                post = self.act_fn(pre)
                mlp_out = einops.einsum(post, self.blocks[i].params.W_out, "batch seq d_mlp, d_mlp d_model -> batch seq d_model")
                x = x + mlp_out

        #unembed
        with record_function("unembedding"):
            x = einops.einsum(x, self.embed, "batch seq d_model, d_vocab d_model -> batch seq d_vocab")
            
        return  x
        
  def kv_cached_forward(self, 
                          x: Int[t.Tensor, "batch seq"],   # tu powinienem dac wszystko czy tylko dopelnienie? raczej to drugie.
                          input_pos: int,
                          cached_Kx: Int[t.Tensor, "n_layer batch n_head d_head p_seq"], 
                          cached_xV: Int[t.Tensor, "n_layer batch n_head p_seq d_head"],
                          ) -> Int[t.Tensor, "batch seq d_vocab"]:
        """
        cached_Kx and cached_xV are changed in place
        """
      
        # Podejrzewam, ze to dodawanie kolejnych elementow zabija performance
        # powinienem trzymac cached_Kx jako liste i p_seq jako pierwszy wymiar? 
        
        
        # Przemyslenie:
        #  swoją drogą w przypadku kv_cache ja zawsze bede sprawdzal tylko jeden element naprzód... czyli mój seq_len będzie 1 ???
        #.    na ile to moze uproscic implementacje? 
        
        with record_function("preparation"):
            q_seq = x.shape[1]

        with record_function("embeddings"):
            x = F.embedding(x, self.embed) # [batch seq d_model]

            positional_x = F.embedding( 
                self.arange[input_pos:input_pos+q_seq],  #DONE: Biore pod uwage, ze to jest dopelnienie
                self.positional_embed) # [batch seq d_model]
            x = x + positional_x

        for i in range(self.n_layers):
            with record_function("layer"): #FIXME: musze zebrac Kx po wszystkich warstwach
                with record_function("QK"):
                    xQ = x @ self.blocks[i].params.Q 
                    xQ = einops.rearrange(xQ, "batch seq (n_head d_head) -> batch n_head seq d_head", n_head=self.config.n_head).contiguous()

                    # DONE: tu musze polaczyc nowe Kx z tym z cache
                    _Kx = x @ self.blocks[i].params.K                
                    _Kx = einops.rearrange(_Kx, "batch seq (n_head d_head) -> batch n_head seq d_head ", n_head=self.config.n_head).contiguous()
                    cached_Kx[i, :, :, input_pos:input_pos+1, :] = _Kx
                                        
                with record_function("attention_scores"):
                    attention_scores = einops.einsum(
                        xQ, cached_Kx[i, :, :, :input_pos+1],  # FIXME? Tu przemnazam za duzo, ale za to mam dobrze zaaranzowany tensor. To nie jest problem, bo attention_mask skasują nadmiarowe napisy
                        "batch n_head q_seq d_head, batch n_head k_seq d_head  -> batch n_head q_seq k_seq")
                    attention_scores = attention_scores / math.sqrt(self.d_head)
                    # DONE: biorą pod uwagę cały zakres, nawet ten scache'owany dla k, ale tylko ten nowy dla q
                    #masked_attention_scores = attention_scores + self.attention_mask[cached_len:cached_len + q_seq, :cached_len + q_seq].unsqueeze(0).unsqueeze(0)
                    #FIXME: Off by one???
                    masked_attention_scores = attention_scores + self.attention_mask[input_pos:input_pos+1, :input_pos+1].unsqueeze(0).unsqueeze(0)

                    masked_attention = F.softmax(masked_attention_scores, dim=-1) # Over keys

                with record_function("attention_output"):
                    # Done: tu musze polaczyc nowe xV z tym z cache po wymiarze p_seq
                    _xV = einops.einsum(x, self.blocks[i].params.V, "batch p_seq d_model, n_head d_model d_head -> batch n_head p_seq d_head")
                    cached_xV[i, :, :, input_pos:input_pos+1] = _xV

                    z = einops.einsum(masked_attention, cached_xV[i, :, :, :input_pos+1, :], "batch n_head q_seq k_seq, batch n_head k_seq d_head -> batch q_seq n_head d_head")
                    z = einops.rearrange(z, "batch q_seq n_head d_head -> batch q_seq (n_head d_head)")

                    attn_out = einops.einsum(z, self.blocks[i].params.O, "batch q_seq n_head_x_d_head, n_head_x_d_head d_model -> batch q_seq d_model")
                    
                    x = x + attn_out
                
                
                x = self.blocks[i].norm(x)

                # mlp
                with record_function("mlp"):
                    pre = einops.einsum(x, self.blocks[i].params.W_in, "batch seq d_model, d_model d_mlp -> batch seq d_mlp")
                    post = self.act_fn(pre)
                    mlp_out = einops.einsum(post, self.blocks[i].params.W_out, "batch seq d_mlp, d_mlp d_model -> batch seq d_model")
                    x = x + mlp_out
                    
        #unembed
        with record_function("unembedding"):
            x = einops.einsum(x, self.embed, "batch seq d_model, d_vocab d_model -> batch seq d_vocab")
            
        return x
            

    

def generate(model_type, config, checkpoint: str, batch_size: int, max_seq_len: int, start_tokens: list[int], device: str):
  model = model_type(config)
#   model.load_state_dict(t.load(checkpoint))
  model = model.to(device)
  with t.inference_mode():
    bos_tensor = t.Tensor(start_tokens).to(t.long).to(device)
    input = einops.repeat(
        bos_tensor, "s -> batch s",
        batch=batch_size
    )
    for i in range(max_seq_len-1):
      proba = F.softmax(model(input).detach() , dim=-1)
      next_proba = proba[:,-1,:]
      next_sampled = t.multinomial(next_proba, num_samples=1) # TODO: dodac temperature
    #   print(i)
    #   print(f"{next_sampled.shape=} {next_proba.shape=}")
    #   print(f"{next_sampled=} {next_proba.max(dim=-1)=} {next_proba[0, next_sampled[0][0]]=}")
    #   print("---")

      input = t.concat([input, next_sampled], dim=-1)

    return input, proba


def generate_kv_cached(model_type, config, checkpoint: str, batch_size: int, max_seq_len: int, start_tokens: list[int], device: str):
  model = model_type(config)
#   model.load_state_dict(t.load(checkpoint))
  model = model.to(device)
  model.eval()
  with t.inference_mode():
    bos_tensor = t.Tensor(start_tokens).to(t.long).to(device)
    next_sampled = einops.repeat(
        bos_tensor, "s -> batch s",
        batch=batch_size
    )
    cached_Kx = t.empty((config.n_layers, batch_size, config.n_head, max_seq_len, config.d_head), dtype=t.float32).to(device) * 0
    cached_xV = t.empty((config.n_layers, batch_size, config.n_head, max_seq_len, config.d_head), dtype=t.float32).to(device) * 0
    generated = torch.empty((batch_size, 0), dtype=t.int64)
    for i in range(max_seq_len-1):
      x = model.kv_cached_forward(next_sampled, i, cached_Kx=cached_Kx, cached_xV=cached_xV)
    #   print(i)
      
    #   print(f"{x.shape=} {cached_Kx.shape=} {cached_xV.shape=}")
      proba = F.softmax(x , dim=-1)
      next_proba = proba[:,-1,:]
      next_sampled = t.multinomial(next_proba, num_samples=1) # TODO: dodac temperature
    #   print(f"{next_sampled.shape=} {next_proba.shape=}")
    #   print(f"{next_sampled=} {next_proba.max(dim=-1)=} {next_proba[0, next_sampled[0][0]]=}")
    #   print("---")

      generated = t.concat([generated, next_sampled], dim=-1)

    return generated, proba


if __name__ == "__main__":
    config = TransformerConfig(
        vocab_size=67,
        d_model=512,
        d_head=64,
        n_head=8,
        max_seq_len=256,
        n_layers=12,
        learning_rate=3e-4,
        batch_size=4,
        epochs=1,
    )
    # FIXME: Tu chce zaladowac model
    # Wydajnosc moge mierzyc bez tego, ale poprawnosci juz tak nie zmierze
    
    device = "cuda" if t.cuda.is_available() else "cpu"
    
    start_tokens = [1]

    start = time.time()
    generated, proba = generate(
        KVCachedTransformer,
        config,
        checkpoint=None,
        batch_size=4,
        max_seq_len=256,
        start_tokens=start_tokens,
        device=device
    )
    print(f"Generated sequences without kv cachein {time.time() - start} seconds:")
    print(generated)
    print(len(generated[0]))
    
    start = time.time()
    generated, proba = generate_kv_cached(
        KVCachedTransformer,
        config,
        checkpoint=None,
        batch_size=4,
        max_seq_len=256,
        start_tokens=start_tokens,
        device=device
    )
    print(f"Generated sequences using kv cache in {time.time() - start} seconds:")
    print(generated)
    print(len(generated[0]))