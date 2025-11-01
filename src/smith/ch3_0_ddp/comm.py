# run with `uv run torchrun --nproc_per_node=10 script.py`
import os
import torch.distributed as dist
import torch as t

dist.init_process_group("gloo") # Typically this chooses the backend correctly
try:
    # You can alos pass the backend as env variable
    print(f"hello from {dist.get_rank()}/{dist.get_world_size()}")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cpu"

    # Example big batch on src rank
    src = 0
    if rank == src:
        big = t.randn(128, 3, 4, 4, device=device)   # [N, ...]
        assert big.size(0) % world_size == 0, "batch must divide world_size"
        per = big.size(0) // world_size
        # Equal chunks along batch dim
        chunks = list(big.split(per, dim=0))   
        for chunk in chunks:
            print(f"{chunk.size()=} {chunk.sum()=}")
    else:
        per = 128 // world_size                              # or pass from config
        chunks = None

    # Preallocate receive buffer on every rank
    recv = t.empty(per, 3, 4, 4, device=device)

    # Scatter
    print(f"{dist.get_rank()=} {recv.sum()=}")
    dist.scatter(recv, scatter_list=chunks, src=src)
    print(f"{dist.get_rank()=} {recv.sum()=}")
    
    tensor_list = [t.empty(per, 3, 4, 4, device=device) for _ in range(world_size)]
    
    gathered = t.empty(128, 3, 4, 4, device=device)
    dist.all_gather(tensor_list, recv)
    for i, chunk in enumerate(tensor_list):
            print(f"{dist.get_rank()=} {i=} {chunk.size()=} {chunk.sum()=}")
    # if rank == 0:
    #     print(f"{gathered.size()=} {gathered.sum()=}")
    
finally:
    dist.destroy_process_group()