"""FlashAttention-2 benchmark (Problem flash_benchmarking).

Three implementations, all causal, batch size 1, tile size 16:
  naive  : regular attention from cs336-basics (materializes the full N x N scores)
  tiled  : our FlashAttention-2 in eager PyTorch (Python loops over 16x16 tiles)
  triton : our Triton forward kernel (+ the same eager tiled backward)

Built step by step; this is block 1 (implementations + inputs).
"""

import argparse
import csv
import datetime
import math
import os
import warnings

import torch
import triton
from triton.testing import do_bench

from cs336_basics.model import scaled_dot_product_attention
from cs336_basics.nn_utils import softmax
from cs336_systems.FlashAttnAutoGrad import FlashAttnAutoGrad
from cs336_systems.FlashAttnTriton import FlashAttnTriton

# Harmless: first cuBLAS call on autograd's backward thread sets up the CUDA
# context. Same filter as pyproject's pytest config.
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

# --- Sweep (from the handout) ---------------------------------------------
SEQ_LENS = [2 ** i for i in range(7, 17)]   # 128 ... 65536
D_HEADS = [16, 32, 64, 128]
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

# --- Feasibility limits ----------------------------------------------------
# Eager tiled attention launches ~10 kernels per 16x16 tile, (N/16)^2 tiles:
# measured single forward 5.7s at N=2048, 22.9s at N=4096 (RTX 3070). Beyond
# these limits a single config would take minutes, so we record "skipped".
# Crossover study only needs tiled as a "no fusion" reference line, so stop at
# 1024 (1.5s per forward) to keep the sweep to minutes.
TILED_FWD_MAX_N = 1024
BWD_MAX_N = 2048   # tiled and triton share the same eager tiled backward


def build_impls(n, device="cuda"):
    """Return {name: fn(q, k, v) -> o} for sequence length n.

    The naive causal mask is built here, outside any timed region: a real model
    would build it once, and we don't want to time mask construction. Note the
    mask itself is N x N bools (4GB at N=65536), part of why naive runs out of
    memory.
    """
    mask_cache = {}

    def naive(q, k, v):
        # Same structure as cs336-basics scaled_dot_product_attention (full N x N
        # scores), but follows the shared precision policy: its softmax would
        # run in bf16 for bf16 inputs, ours runs in fp32 like tiled/triton.
        # Mask built on first call (the untimed warmup), so if it alone doesn't
        # fit (4GB at N=65536) that OOM is caught by measure() like any other.
        if "keep" not in mask_cache:
            # cs336-basics convention: True = keep. Causal keeps keys <= query -> lower triangle.
            mask_cache["keep"] = torch.ones(n, n, device=device, dtype=torch.bool).tril()
        s = (q @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
        s = torch.where(mask_cache["keep"], s, float("-inf"))
        p = softmax(s, dim=-1)                     # fp32
        return p.to(v.dtype) @ v

    def tiled(q, k, v):
        return FlashAttnAutoGrad.apply(q, k, v, True)

    def triton(q, k, v):
        return FlashAttnTriton.apply(q, k, v, True)

    return {"naive": naive, "tiled": tiled, "triton": triton}


def make_inputs(n, d, dtype, device="cuda", seed=0):
    """Random Q, K, V, dO of shape [batch=1, n, d], created before any timing.

    - Generated once per config so RNG / allocation never shows up in timings.
    - Fixed seed so every implementation sees the exact same numbers (needed
      for the correctness check, and makes reruns reproducible).
    - requires_grad=True so the same tensors work for forward and backward.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q, k, v = (torch.randn(1, n, d, device=device, dtype=dtype, generator=g).requires_grad_()
               for _ in range(3))
    do = torch.randn(1, n, d, device=device, dtype=dtype, generator=g)
    return q, k, v, do


# --- Block 2: correctness ---------------------------------------------------

def reference(q, k, v, do):
    """Ground truth: naive attention in float64 + autograd.

    float64 (~16 significant digits) so the reference's own rounding error is
    negligible next to fp32/bf16. Otherwise we couldn't tell whose error it is.
    """
    q64, k64, v64 = (t.detach().double().requires_grad_() for t in (q, k, v))
    n = q.shape[-2]
    keep = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
    o = scaled_dot_product_attention(q64, k64, v64, mask=keep)
    o.backward(do.double())
    return o.detach(), (q64.grad, k64.grad, v64.grad)


# Max-abs-error tolerance vs the float64 reference (values are O(1), so
# absolute error ~ relative error). Chosen from the dtype's precision, then
# checked against measured worst error over O/dQ/dK/dV (RTX 3070,
# N=128/512/2048, d=16..128):
#   fp32: all three ~1e-6 (fp32 eps ~1e-7, accumulated over many ops) once
#         Triton uses input_precision="ieee" (with default TF32 it was ~5e-3).
#   bf16: eps ~8e-3; just rounding the inputs costs ~1e-2. With the shared
#         precision policy (fp32 softmax/stats/accumulators everywhere) all
#         three land in 0.8-2.7e-2, flat in N. Before: tiled reached 9.5e-2
#         and grew with N (bf16 accumulators, bf16 L). 5e-2 = ~2x margin.
TOLERANCE = {"fp32": 1e-5, "bf16": 5e-2}


def max_errors(q, k, v, do, impls, with_grad):
    """Max |impl - reference| for O (and dQ, dK, dV if with_grad), per impl."""
    ref_o, ref_grads = reference(q, k, v, do)
    out = {}
    for name, fn in impls.items():
        for t in (q, k, v):
            t.grad = None
        o = fn(q, k, v)
        errs = {"O": (o.double() - ref_o).abs().max().item()}
        if with_grad:
            o.backward(do)
            for g_name, t, r in zip(("dQ", "dK", "dV"), (q, k, v), ref_grads):
                errs[g_name] = (t.grad.double() - r).abs().max().item()
        out[name] = errs
    return out


# --- Block 3: timing ---------------------------------------------------------

def bench_ms(fn, grad_to_none=None):
    """Median latency (ms) via triton.testing.do_bench, which handles what is
    easy to get wrong by hand:
      * CUDA events + synchronize: kernels are async, Python timers alone
        would only measure the launch;
      * warmup runs before timing (JIT compile, allocator, caches);
      * clears L2 cache before each run, so small inputs aren't unrealistically
        served from cache;
      * repeats and returns the median, robust to occasional hiccups.

    What it measures is GPU time, NOT wall time. The L2 flush before each run
    keeps the GPU busy while Python queues the next call's kernels, so CPU
    launch overhead is hidden. Checked (N=1024, d=64, fp32 fwd): naive
    do_bench 0.176ms vs 0.27ms when timed with a sync per call; the 0.176 is
    reproduced by putting a 256MB zero_() before each call. warmup is in ms
    (adaptive count); 0, 5 or 500 warmup loops gave the same median once the
    first (JIT) call is done.
    """
    return do_bench(fn, warmup=25, rep=100, grad_to_none=grad_to_none, return_mode="median")


def measure(name, fn, q, k, v, do, with_bwd=True):
    """{'fwd','bwd','e2e'} -> ms (float), 'OOM', or 'skipped'."""
    n = q.shape[-2]
    res = {"fwd": "skipped", "bwd": "skipped", "e2e": "skipped"}
    if name == "tiled" and n > TILED_FWD_MAX_N:
        return res
    do_bwd = with_bwd and (name == "naive" or n <= BWD_MAX_N)
    grads = [q, k, v]  # reset .grad every rep: otherwise grads accumulate (extra work)

    try:
        # One untimed call (Triton JIT compile happens here). Backward only if we
        # time it: eager tiled bwd at large N would take hours for nothing.
        o = fn(q, k, v)
        if do_bwd:
            o.backward(do)
        del o
        # Forward with autograd on (training setting): naive keeps its N x N P for
        # backward, so this is also where its O(N^2) memory shows up.
        res["fwd"] = bench_ms(lambda: fn(q, k, v))
        if do_bwd:
            # Backward alone: build the graph once, replay only .backward().
            o = fn(q, k, v)
            res["bwd"] = bench_ms(lambda: o.backward(do, retain_graph=True), grad_to_none=grads)
            del o
            res["e2e"] = bench_ms(lambda: fn(q, k, v).backward(do), grad_to_none=grads)
    except torch.OutOfMemoryError:
        # Every planned measurement that didn't finish is OOM; intentionally
        # skipped ones stay "skipped" so the table distinguishes the two.
        planned = ["fwd", "bwd", "e2e"] if do_bwd else ["fwd"]
        for key in planned:
            if not isinstance(res[key], float):
                res[key] = "OOM"
    finally:
        for t in grads:
            t.grad = None
        torch.cuda.empty_cache()  # don't let one config's leftovers cause the next OOM
    return res


# --- Block 4: sweep -----------------------------------------------------------

# Correctness gate, not a precision study: confirm each impl computes attention
# before we trust its timing (a wrong kernel is often a fast one). Forward
# only, small N only, pass/fail.
CHECK_MAX_N = 1024
FIELDS = ["seq_len", "d_head", "dtype", "impl", "fwd_ms", "bwd_ms", "e2e_ms", "correct"]


def check_correct(n, d, dtype_name, impls):
    """{impl: True/False} vs float64 reference, or 'unchecked' for large N."""
    if n > CHECK_MAX_N:
        return {name: "unchecked" for name in impls}
    q, k, v, do = make_inputs(n, d, DTYPES[dtype_name])
    feasible = {name: fn for name, fn in impls.items() if not (name == "tiled" and n > TILED_FWD_MAX_N)}
    errs = max_errors(q, k, v, do, feasible, with_grad=False)
    return {name: (errs[name]["O"] <= TOLERANCE[dtype_name]) if name in errs else "unchecked"
            for name in impls}


def run_sweep(seq_lens, d_heads, dtype_name, with_bwd, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        # Metadata first: numbers without their conditions can't be interpreted later.
        f.write(f"# gpu={torch.cuda.get_device_name()} torch={torch.__version__} triton={triton.__version__} "
                f"date={datetime.datetime.now().isoformat(timespec='seconds')} "
                f"batch=1 heads=1 causal=True tile=16 tl.dot=ieee timing=do_bench median (GPU time)\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for d in d_heads:            # outer: one d = one complete set of lines
            for n in seq_lens:       # inner: points along each line
                impls = build_impls(n)
                correct = check_correct(n, d, dtype_name, impls)
                q, k, v, do = make_inputs(n, d, DTYPES[dtype_name])
                for name, fn in impls.items():
                    r = measure(name, fn, q, k, v, do, with_bwd=with_bwd)
                    row = {"seq_len": n, "d_head": d, "dtype": dtype_name, "impl": name,
                           **{f"{key}_ms": (f"{val:.4f}" if isinstance(val, float) else val)
                              for key, val in r.items()},
                           "correct": correct[name]}
                    w.writerow(row)
                    f.flush()  # a crash / Ctrl-C keeps every finished row
                    print(" ".join(f"{key}={val}" for key, val in row.items()), flush=True)
                del q, k, v, do, impls
                torch.cuda.empty_cache()
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=SEQ_LENS)
    ap.add_argument("--dims", type=int, nargs="+", default=[64])
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    ap.add_argument("--bwd", action="store_true", help="also time bwd/e2e (only up to BWD_MAX_N)")
    ap.add_argument("--out", default="reports/flash_fp32_coarse.csv")
    args = ap.parse_args()
    run_sweep(args.seqs, args.dims, args.dtype, args.bwd, args.out)
