import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, tensor_length, device):
    setup(rank, world_size)
    data = torch.rand((tensor_length,))
    # warmup 
    for i in range(5):
        # print(f"warmup itr {i} =======================")
        # print(f"rank {rank} data (before all-reduce): {data}")
        dist.all_reduce(data, async_op=False)
        # print(f"rank {rank} data (after all-reduce): {data}")

    dist.barrier()
    print("cuda synchronize and start timer ==================")

    total_loop = 10
    if device != 'cpu':
        torch.cuda.synchronize()
    start = timeit.default_timer()
    
    for _ in range(total_loop):
        dist.all_reduce(data, async_op=False)

    if device != 'cpu':
        torch.cuda.synchronize()

    time_elasped = timeit.default_timer() - start
    print("data size: ", tensor_length / 1024 ** 2)
    print(f"time cost: {time_elasped / total_loop} sec")
    
if __name__ == "__main__":
    world_size = 4
    MB_LENGTH = int(1024 * 1024 / 4)
    device="cpu"
    for tensor_length in [MB_LENGTH, MB_LENGTH * 10, MB_LENGTH * 100, MB_LENGTH * 1024]:
        mp.spawn(fn=distributed_demo, args=(world_size, tensor_length, device ), nprocs=world_size, join=True)