import dataclasses

from datasets import load_dataset
from datasets.dataset_dict import Dataset
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from smith.ch2_0_performance.ch2_0_performance import TransformerConfig, generate, set_repro, reproducible_step, profile_train, profile_config1, profile_config2, profile_config3, train
from smith.ch3_1_fsdp.transformer import Transformer, get_dataloader
import deepspeed
import torch.nn as nn
import einops
import time
import torch
import torch as t
import os
import numpy as np
import random

from smith.ch3_1_fsdp.wandb_utils import ds_wandb

@dataclasses.dataclass
class TransformerConfig:
  d_model: int
  d_head: int
  n_head: int
  max_seq_len: int
  n_layers: int
  learning_rate: float
  train_batch_size: int
  per_device_batch_size: int
  epochs: int
  

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
    
    
    
    return tokenized, collate_fn,vocab_to_token, token_to_vocab

def train(model_type, name):
    config =TransformerConfig(
        d_model=1024,
        d_head=512,
        n_head=32,
        max_seq_len=257,
        n_layers=4,
        train_batch_size=64,
        per_device_batch_size=32,
        learning_rate=0.00005,
        epochs=40,
    )
    per_device_batch_size = config.per_device_batch_size
    learning_rate = config.learning_rate
    


    world_size = int(os.getenv("WORLD_SIZE", "1")) # This is set by deepspeed
    
    assert config.train_batch_size % (per_device_batch_size * world_size) == 0, "train_batch_size must be divisible by batch_size * world_size"
    gradient_accumulation_steps = config.train_batch_size // (per_device_batch_size * world_size)
    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": config.train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": per_device_batch_size,
        "gradient_clipping": 1.0,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.0
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 100
            }
        },
        "zero_optimization": {
            "stage": 1,  # FSDP equivalent
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
        },
        # "fp16": {
        #     "enabled": False,
        #     "auto_cast": False,
        #     "loss_scale": 0,
        #     "initial_scale_power": 16,
        #     "loss_scale_window": 1000,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1
        # },
        # "activation_checkpointing": {
        #     "partition_activations": True,
        #     "cpu_checkpointing": True,
        #     "contiguous_memory_optimization": False,
        #     "number_checkpoints": None,
        #     "synchronize_checkpoint_boundary": False,
        #     "profile": False
        # },
        "wall_clock_breakdown": False
    }

    tokenized, collate_fn, vocab_to_token, token_to_vocab = get_dataloader(max_seq_len=config.max_seq_len, 
                                  batch_size=per_device_batch_size)

    config.vocab_size = len(vocab_to_token)
    
    model = model_type(config)  # Don't move to device yet - DeepSpeed will handle this
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )
    
    
    train_sampler = DistributedSampler(
        tokenized,
        num_replicas=deepspeed.comm.get_world_size(),
        rank=deepspeed.comm.get_rank(),
        shuffle=True,
    )

    train_loader = DataLoader(tokenized, 
                            batch_size=per_device_batch_size, 
                            sampler=train_sampler,

                            collate_fn=collate_fn,
                        )

    
    
    
    
    is_main_process = (not deepspeed.comm.is_initialized()) or (deepspeed.comm.get_rank() == 0)
    print(f"{deepspeed.comm.get_rank()=} {model_engine.local_rank=}")
    with ds_wandb(
        main=is_main_process,
        entity="",
        project="ch3_1_fsdp",
        config={
            "learning_rate": learning_rate,
            "architecture": name,
            "dataset": "tiny-shakespear",
            "d_head": config.d_head,
            "d_model": config.d_model,
            "batch_size": per_device_batch_size,
            "n_head": config.n_head,
            "n_layers": config.n_layers,
            "epochs": config.epochs,
        },
    ) as run:
        model_engine.train()
        loss_fn = nn.CrossEntropyLoss()
        epochs = config.epochs
        steps = 0
        start = time.time()
        
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)  
            print("=" * 80)
            print(f"{epoch=}")
            for batch in train_loader:
                
                # Use DeepSpeed's forward pass
                
                inp = batch['inputs'].to(model_engine.local_rank)
                pred = model_engine(inp[:, :-1])
                
                loss_val = loss_fn(
                    input=einops.rearrange(pred, "batch seq vocab -> (batch seq) vocab"),
                    target=einops.rearrange(inp[:, 1:], "batch seq -> (batch seq)")
                )
                
                
                    
                # Use DeepSpeed's backward and step

                model_engine.backward(loss_val)
                model_engine.step()
                
                
                loss_mean = loss_val.detach().clone()
                if steps % 100 == 0:
                    print(f"{loss_mean.item()=:.4f}")
                deepspeed.comm.all_reduce(loss_mean)
                loss_mean = loss_mean.item() / deepspeed.comm.get_world_size()
                
                
                if is_main_process:
                    grad_norm = model_engine.get_global_grad_norm()
                    run.log({
                        "loss": loss_mean,
                        "epoch": epoch,
                        "step": steps,
                        "grad_norm": grad_norm,
                    })
                
                steps += 1 

        print(f"Elapsed time: {time.time() - start}")
        
        # Save using DeepSpeed's save_checkpoint
        os.makedirs(".models", exist_ok=True)
        model_engine.save_checkpoint(".models", tag=f"model_{run.id}")
        
if __name__ == "__main__":
    train(Transformer, str(type(Transformer)))