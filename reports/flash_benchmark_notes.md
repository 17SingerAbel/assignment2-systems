# FlashAttention benchmark — lab notes

Hardware: RTX 3070 8GB (46 SMs). torch 2.11.0+cu130, triton 3.6.0.
Config unless noted: batch=1, one head, causal, fp32, tile 16x16, Triton `tl.dot(input_precision="ieee")`.

## 2026-11-24 — Prediction vs first measurement

**Prediction (before measuring):** forward speed order tiled << naive < triton
("Triton is fastest").

**Measured (N=1024, d=64, fp32, do_bench median):**

| impl   | fwd (ms) | bwd (ms) | e2e (ms) |
|--------|---------:|---------:|---------:|
| naive  | 0.176    | 0.420    | 0.582    |
| tiled  | 1468.7   | 1622.0   | 3065.8   |
| triton | 0.214    | 1624.0   | 1645.9   |

Sanity checks passed: fwd + bwd ≈ e2e for each impl; tiled bwd ≈ triton bwd
(same code).

**Prediction held for tiled (slowest), failed for triton vs naive:** naive was
faster at N=1024. Why this config differs from "FlashAttention is faster":

1. **No TF32.** We forced real fp32 (`ieee`) so all three use the same
   precision. With default TF32 + tuned tiles an earlier run gave triton fwd
   0.047 ms on the same config — the tensor-core path was most of the speed.
2. **Tile 16 is small:** little data reuse per load from HBM.
3. **N=1024 is short:** the N x N score matrix is 4 MB, so the HBM traffic
   FlashAttention saves is still cheap; its advantage should grow with N
   (and naive eventually OOMs).
4. **batch=1, one head:** the Triton grid has only N/16 = 64 programs for
   46 SMs. A real model runs batch x heads (e.g. 4 x 12 = 48) problems at
   once, so short-N numbers here understate FlashAttention.

To verify with the full sweep: where (which N) triton overtakes naive, and
where naive OOMs.

## 2026-11-24 — How the timing method changes the number

Question: is `do_bench`'s time-based warmup (warmup=25 ms, count chosen from
an estimate) OK, versus "warm up a few loops, then time N iterations"?

| method (fwd, N=1024 d=64 fp32)                  | naive  | triton |
|-------------------------------------------------|-------:|-------:|
| do_bench (warmup 25ms, rep 100ms)               | 0.176  | 0.214  |
| manual, sync every iter, warmup 0 / 5 / 500     | ~0.27  | ~0.27  |
| manual, no per-iter sync                         | 0.267  | 0.206  |
| manual, no sync + 256MB `zero_()` before each   | 0.175  | 0.204  |

Findings:
- **Warmup count doesn't matter here** (0, 5, 500 loops: same median) once the
  first call (Triton JIT) is done outside timing. Warmup matters for first-call
  costs, not for steady state.
- **What matters is whether CPU launch overhead is inside the measurement.**
  Naive launches ~10 small kernels per call; at N=1024 Python takes ~0.27 ms to
  launch them, longer than the GPU needs (~0.18 ms), so the GPU waits on the
  CPU (launch-bound). do_bench's L2-flush kernel keeps the GPU busy while
  Python queues the next call, hiding that overhead.
- So **do_bench reports GPU time** ("CPU is far enough ahead"). That is the
  right metric to compare kernels, but at small N the wall time of a lone call
  can be higher. Report numbers as GPU time.

Not done: a stress test (sustained load, clocks/thermals over minutes) is a
different experiment from this latency benchmark.
