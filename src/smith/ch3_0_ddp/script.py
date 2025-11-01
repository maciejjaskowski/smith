# run with `uv run torchrun --nproc_per_node=10 script.py`
import os
import torch.distributed as dist
dist.init_process_group("gloo")  # nccl is CUDA only
print(f"hello from {dist.get_rank()}/{dist.get_world_size()}")
print(f"{[(k,v) for k, v in os.environ.items() if "TORCH" in k or "WORLD" in k or "RANK" in k]}")
dist.destroy_process_group()