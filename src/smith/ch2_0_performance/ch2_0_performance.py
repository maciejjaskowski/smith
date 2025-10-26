import dataclasses
from datasets import load_dataset
import einops
from jaxtyping import Int, Float
import math
import numpy as np
import wandb
import random
import time
import torch as t
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim import Adam
from torch.utils.data import DataLoader
from smith.ch2_0_performance.baseline import TransformerConfig





def get_dataloader(max_seq_len, batch_size):
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

    train_loader = DataLoader(tokenized, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                            )
    return train_loader, vocab_to_token, token_to_vocab


# model = BaselineTransformer(TransformerConfig(
#     vocab_size=len(vocab_to_token),
#     d_model=128,
#     d_head=16,
#     n_head=4,
#     max_seq_len=max_seq_len,
#     n_layers=2
# ))



# learning_rate = 0.000025
# epochs = 200
# batch_size = 64

# optim = Adam(model.parameters(), lr=learning_rate)

# Start a new wandb run to track this script.
# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="",
#     # Set the wandb project where this run will be logged.
#     project="transformers",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": learning_rate,
#         "architecture": "Transformer 5",
#         "dataset": "tiny-shakespear",
#         "d_head": model.config.d_head,
#         "d_model": model.config.d_model,
#         "batch_size": batch_size,
#         "n_head": model.config.n_head,
#         "n_layers": model.config.n_layers,
#         "epochs": epochs,
#     },
# )

# transformer_config = TransformerConfig(
#         vocab_size=len(vocab_to_token),
#         d_model=512,
#         d_head=64,
#         n_head=8,
#         max_seq_len=max_seq_len,
#         n_layers=4
#     )

def reproducible_step(transformer_type, config: TransformerConfig):
    """
    Returns loss and gradients for a single training step
    """
    
    model = transformer_type(config)
    model = model.to("cuda")
    
    device = "cuda"
    model.to(device)

    learning_rate = 0.000025
    steps=1
    batch_size = 64

    optim = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(tokenized, batch_size=batch_size,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=4,
                          persistent_workers=True,
                          prefetch_factor=16,
                          # pin_memory=True,
                          )
    batch = next(iter(train_loader))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input = batch['inputs'].to(device, non_blocking=True)

        with record_function("train_step"):
            pred = model(input[:, :-1]) # 0 -> 1, 1 -> 2, ... n-1 -> eos
            loss_val = loss_fn(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                        target=einops.rearrange(input[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos
            
            loss_val.backward()
            optim.step()
            for param in model.parameters():
                grads[param.name] = param.grad.cpu().detach()
            optim.zero_grad(set_to_none=True)
    return loss_val, grad

# # Start a new wandb run to track this script.
# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="",
#     # Set the wandb project where this run will be logged.
#     project="transformers",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": learning_rate,
#         "architecture": "Transformer 5",
#         "dataset": "tiny-shakespear",
#         "d_head": model.config.d_head,
#         "d_model": model.config.d_model,
#         "batch_size": batch_size,
#         "n_head": model.config.n_head,
#         "n_layers": model.config.n_layers,
#         "epochs": epochs,
#     },
# )



# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         profile_memory=True, record_shapes=True,
#         schedule=t.profiler.schedule(wait=1, warmup=1, active=2, repeat=1)) as prof:
#   epochs = 1
#   start = time.time()
#   for epoch in range(epochs):
#       print("=" * 80)
#       print(f"{epoch=}")
#       for i_batch, batch in enumerate(train_loader):

#         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#           input = batch['inputs'].to(device, non_blocking=True)



#           with record_function("train_step"):
#               pred = model(input[:, :-1]) # 0 -> 1, 1 -> 2, ... n-1 -> eos
#               loss_val = loss_fn(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
#                             target=einops.rearrange(input[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos

#               if i_batch % 100 == 0:
#                 print(loss_val)
#               loss_val.backward()
#               optim.step()
#               optim.zero_grad(set_to_none=True)

#           prof.step()
#   print(f"Elapsed time: {time.time() - start}")


# run.finish()
# model.eval()

