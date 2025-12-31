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
from smith.ch2_0_performance.baseline import TransformerConfig, BaselineTransformer


def get_dataloader(max_seq_len, batch_size, reproducible=False):
    ds = load_dataset("Trelis/tiny-shakespeare")
    
    chunks = []
    for text in ds['train'].to_pandas()['Text']:
        text = ["_bos_"] + list(text) + ["_eos_"]
        for i in range(0, len(text), max_seq_len):
            chunks.append(text[i:i+max_seq_len])
        
    unique_chars = set()

    for chunk in chunks:
        unique_chars.update(chunk)

    token_to_vocab = list(sorted(unique_chars))
    vocab_to_token = {v:t for t,v in enumerate(token_to_vocab)}

    def tokenize(example, vocab_to_token: dict[str, int]):
        return [vocab_to_token[x] for x in example]

    tokenized = [tokenize(chunk, vocab_to_token) for chunk in chunks]
    
    def collate_fn(examples):
        max_len = max([len(e) for e in examples])

        inputs = []
        for e in examples:
            inputs.append(t.LongTensor(e + [vocab_to_token['_eos_']] * (max_len - len(e))))

        return {"inputs": t.stack(inputs)}

    if reproducible:
        g = torch.Generator()
        g.manual_seed(1337)

        def _seed_worker(worker_id):
            # Each worker gets a different, deterministic seed
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        train_loader = DataLoader(tokenized, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                              worker_init_fn=_seed_worker,
                              generator=g,
                            )
    else:
        train_loader = DataLoader(tokenized, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              collate_fn=collate_fn,
                            )
    
    return train_loader, vocab_to_token, token_to_vocab

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
    with record_function("preparation"):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        d_head = t.tensor(self.d_head).to(t.long)

    with record_function("embeddings"):
        # print(f"{x.shape=} {self.embed.shape=}")
        x = F.embedding(x, self.embed) # [batch seq d_model]

        positional_x = F.embedding(
            self.arange[:seq_len],
            self.positional_embed) # [batch seq d_model]
        # print(f"{self.arange[:seq_len].shape=} {self.positional_embed.shape=} {positional_x.shape=} {x.shape=}")
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
                # print(f"{d_head.shape=}")
                attention_scores = attention_scores / math.sqrt(self.d_head)
                # czy na pewno kompatybilne jest attention z attention_mask ?
                # print(f"{i=} {self=}")
                masked_attention_scores = attention_scores + self.attention_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

                masked_attention = F.softmax(masked_attention_scores, dim=-1) # Over keys

            # print(1*(masked_attention[0] > 0))

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

    return x


def train(model_type, config, name):
# Start a new wandb run to track this script.
    # Start a new wandb run to track this script.
    batch_size = config.batch_size
    epochs = 1
    device = "cuda"
    learning_rate = config.learning_rate
    
    train_loader, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=config.max_seq_len, 
                                  batch_size=batch_size,
                                  reproducible=True)
    config.vocab_size = len(vocab_to_token)
    
    model = model_type(config).to(device)
    
    with wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="",
        # Set the wandb project where this run will be logged.
        project="ch3_1_fsdp",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": learning_rate,
            "architecture": name,
            "dataset": "tiny-shakespear",
            "d_head": config.d_head,
            "d_model": config.d_model,
            "batch_size": batch_size,
            "n_head": config.n_head,
            "n_layers": config.n_layers,
            "epochs": epochs,
        },
    ) as run:
        # run.watch(model)
        
        optim = Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        epochs = config.epochs
        steps = 0
        start = time.time()
        for epoch in range(epochs):
            print("=" * 80)
            print(f"{epoch=}")
            for batch in train_loader:
                
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    inp = batch['inputs'].to(device, non_blocking=True)
                    pred = model(inp[:, :-1]) # 0 -> 1, 1 -> 2, ... n-1 -> eos
                    loss_val = loss_fn(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                                    target=einops.rearrange(inp[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos

                    if steps % 400 == 0:
                        print(f"{loss_val.item()}")
                    loss_val.backward()
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                run.log({"loss": loss_val.item(),
                         "epoch": epoch,
                         "step": steps,
                        })
                steps += 1 

        print(f"Elapsed time: {time.time() - start}")
        os.makedirs(".models", exist_ok=True)
        torch.save(model.state_dict(), f".models/model_{run.id}.pth")


        
if __name__ == "__main__":
    train(Transformer, profile_config3, str(type(Transformer)))
    
