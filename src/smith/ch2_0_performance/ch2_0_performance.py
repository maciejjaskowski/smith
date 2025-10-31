import dataclasses
import os
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



def generate(model_type, config, checkpoint: str, batch_size: int, max_seq_len: int, start_tokens: list[int], device: str):
  model = model_type(config)
  model.load_state_dict(torch.load(checkpoint))
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
      print(f"{next_sampled.shape=} {next_proba.shape=}")
      print(f"{next_sampled=} {next_proba.max(dim=-1)=} {next_proba[0, next_sampled[0][0]]=}")
      print("---")

      input = t.concat([input, next_sampled], dim=-1)

    return input, proba


def reproducible_step(transformer_type, config: TransformerConfig):
    """
    Returns loss and gradients for a single training step
    """
    learning_rate = config.learning_rate
    steps=1
    batch_size = config.batch_size
    
    train_loader, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=config.max_seq_len, 
                                  batch_size=batch_size,
                                  reproducible=True)
    
    config.vocab_size = len(vocab_to_token)
    model = transformer_type(config)
    model = model.to("cuda")
    
    device = "cuda"
    model.to(device)

    

    optim = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    
    
    batch = next(iter(train_loader))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        input = batch['inputs'].to(device, non_blocking=True)

        with record_function("train_step"):
            pred = model(input[:, :-1]) # 0 -> 1, 1 -> 2, ... n-1 -> eos
            loss_val = loss_fn(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                        target=einops.rearrange(input[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos
            
            loss_val.backward()
            optim.step()
            grads = {}
            for param in model.parameters():
                grads[param.name] = param.grad.cpu().detach()
            optim.zero_grad(set_to_none=True)
    return loss_val, grads

def set_repro(seed: int = 1337):
    # --- Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # --- PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # --- cuDNN / backends
    torch.backends.cudnn.deterministic = True   # use deterministic algorithms where available
    torch.backends.cudnn.benchmark = False      # disable auto-tuner (keeps alg choice fixed)

    # Disable TF32 to avoid tiny numeric diffs on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enforce determinism across PyTorch ops (may error if an op has no deterministic impl)
    torch.use_deterministic_algorithms(True)

    # (optional) keep matmul precision consistent on PyTorch 2.x
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")  # or "medium" consistently


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
        project="ch2_0_performance",
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



def profile_train(model_type, config, name):
    

    # Start a new wandb run to track this script.
    batch_size = 32
    epochs = 2
    device = "cuda"
    learning_rate = 0.00025
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="",
        # Set the wandb project where this run will be logged.
        project="ch2_0_performance_trace",
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
    )
    
    train_loader, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=config.max_seq_len, 
                                  batch_size=batch_size,
                                  reproducible=True)
    config.vocab_size = len(vocab_to_token)
    
    model = model_type(config).to(device).to(torch.bfloat16)
    
    optim = Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True,
            schedule=t.profiler.schedule(wait=1, warmup=3, active=2, repeat=1)) as prof:
        
        start = time.time()
        for epoch in range(epochs):
            print("=" * 80)
            print(f"{epoch=}")
            for i_batch, batch in enumerate(train_loader):

                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    inp = batch['inputs'].to(device, non_blocking=True)

                    with record_function("train_step"):
                        pred = model(inp[:, :-1]) # 0 -> 1, 1 -> 2, ... n-1 -> eos
                        loss_val = loss_fn(input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                                        target=einops.rearrange(inp[:, 1:], "batch seq -> (batch seq)")) # nie przewiduje bos

                        if i_batch % 100 == 0:
                            print(loss_val)
                        loss_val.backward()
                        optim.step()
                        optim.zero_grad(set_to_none=True)

                prof.step()
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total",row_limit=30))
        print(f"Elapsed time: {time.time() - start}")

    prof.export_chrome_trace(f"/tmp/{run.id}_trace.json")
    run.log_artifact(f"/tmp/{run.id}_trace.json", type="trace")
    run.finish()

profile_config1 = TransformerConfig(
        vocab_size=None,
        d_model=128,
        d_head=64,
        n_head=8,
        max_seq_len=129,
        n_layers=4,
        batch_size=32,
        learning_rate=0.00025,
        epochs=200,
    )

profile_config2 = TransformerConfig(
        vocab_size=None,
        d_model=512,
        d_head=32,
        n_head=8,
        max_seq_len=129,
        n_layers=4,
        batch_size=32,
        learning_rate=0.00025,
        epochs=200,
    )   

profile_config3 = TransformerConfig(
        vocab_size=None,
        d_model=1024,
        d_head=512,
        n_head=32,
        max_seq_len=257,
        n_layers=4,
        batch_size=32,
        learning_rate=0.000025,
        epochs=40,
    )

if __name__ == "__main__":
    for batch_size in [64]:
        for max_seq_len in [128]:
            for reproducible in [True]:

                dataloader, _, _ = get_dataloader(max_seq_len=128, batch_size=batch_size, reproducible=reproducible)
                with wandb.init(
                    # Set the wandb entity where your project will be logged (generally your team name).
                    entity="",
                    # Set the wandb project where this run will be logged.
                    project="ch2_0_performance",
                    # Track hyperparameters and run metadata.
                    config={
                        "architecture": "dataloader",
                        "dataset": "tiny-shakespear",
                        "batch_size": batch_size,
                        "max_seq_len": max_seq_len,
                        "reproducible": reproducible,
                    },
                ) as run: 
                    steps = 0
                    for epoch in range(50):
                        for data in dataloader:
                            data['inputs'].to("cuda", non_blocking=True)
                            run.log({
                                "step": steps,
                                "epoch": epoch,
                            })
                            steps += 1