from enum import Enum
import torch
import timeit
import statistics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy, clip_gradient
import numpy as np
from dataclasses import dataclass
import gc
import torch
import torch.cuda.nvtx as nvtx

FORWARD = 'forward'
FORWARD_BACKWARD = 'forward_backward'
FULL = 'full'

@dataclass
class BenchmarkConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    batch_size: int
    context_length: int
    warmup_iterations: int
    iterations: int
    vocab_size: int
    max_norm: float


def cehck_gpu_memory():
    print(torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"Free:  {free / 1024**3:.2f} GB")
    print(f"Total: {total / 1024**3:.2f} GB")
    print(f"Used:  {(total - free) / 1024**3:.2f} GB")    


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def run_step(lm, optimizer, train_data, labels, mode):
    if mode != FORWARD:
        optimizer.zero_grad()

    with torch.cuda.nvtx.range("forward"):
        if mode == FORWARD:
            with torch.no_grad():
                logits = lm(train_data)
                loss = cross_entropy(logits, labels)
        else:
            logits = lm(train_data)
            loss = cross_entropy(logits, labels)

    if mode != FORWARD:
        with torch.cuda.nvtx.range("backward"):
            loss.backward()

    if mode == FULL:
        with torch.cuda.nvtx.range("optimizer"):
            optimizer.step()


def benchmark_model(dataset, config: BenchmarkConfig,  mode, device, warmup_flag=True, dtype=torch.float32):
    lm = BasicsTransformerLM(config.vocab_size, config.context_length, config.d_model, config.num_layers, config.num_heads, config.d_ff, rope_theta=None)
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in lm.parameters()
    )

    print(f"Parameters: {param_bytes / 1024**3:.2f} GiB")

    lm = lm.to(device, dtype=dtype)
    optimizer = AdamW(lm.parameters())

    train_data, labels = get_batch(dataset, config.batch_size, config.context_length, device)

    # warmup
    if warmup_flag:
        for _ in range(config.warmup_iterations):
                run_step(lm, optimizer, train_data, labels, mode)
        
        torch.cuda.synchronize()

    # measurement
    times = []

    for _ in range(config.iterations):
        torch.cuda.synchronize()
        start = timeit.default_timer()

        with torch.cuda.nvtx.range("measurement_step"):
            run_step(lm, optimizer, train_data, labels, mode)
            torch.cuda.synchronize()

        times.append(timeit.default_timer() - start)
    return times


if __name__ == "__main__":
    device = "cuda"

    config = BenchmarkConfig(
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
        batch_size=1,
        context_length=256,
        warmup_iterations=5,
        iterations=3,
        vocab_size=10000,
        max_norm=1.0,
    )

    dataset = np.random.randint(
        low=0,
        high=config.vocab_size,
        size=20000,
        dtype=np.int64,
    )

    mode = FULL

    times = benchmark_model(
        dataset,
        config,
        mode,
        device=device,
        dtype=torch.float32,
    )
    print(times)
    print(f"Mean: {statistics.mean(times) * 1000:.2f} ms")