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

## 2026-11-25 — Crossover study: prediction

Question: from what sequence length does Triton beat naive, and how fast
does its advantage grow? Metric: speedup = naive_fwd / triton_fwd (> 1 means
Triton wins); crossover = N where speedup crosses 1.

Plan: coarse sweep (powers of 2, fwd only) -> refine between the two N that
bracket the crossover -> repeat coarse sweep for other d_head.

**User's prediction:** crossover at N ≈ 2048; N=65536 not reachable (memory).
Note: naive needs N x N scores (16 GB fp32 at N=65536 > 8 GB), Triton only
O(N·d) memory, so expected: naive OOMs, Triton still runs (slowly, N^2 compute).

## 2026-11-25 — Crossover study: results (fp32, d=64, fwd)

Data: `reports/flash_fp32_coarse.csv` (before), `reports/flash_fp32_coarse_skip.csv` (after causal tile skip).

| N     | naive  | triton (no skip) | triton (skip) | skip gain | speedup no-skip -> skip |
|------:|-------:|-----------------:|--------------:|----------:|------------------------:|
| 128   | 0.037  | 0.019            | 0.019         | 1.0x      | 1.93 -> 1.98 |
| 256   | 0.041  | 0.033            | 0.033         | 1.0x      | 1.25 -> 1.25 |
| 512   | 0.058  | 0.063            | 0.061         | 1.0x      | 0.93 -> 0.95 |
| 1024  | 0.175  | 0.214            | 0.144         | 1.48x     | 0.82 -> 1.21 |
| 2048  | 0.641  | 0.580            | 0.408         | 1.42x     | 1.11 -> 1.57 |
| 4096  | 2.41   | 2.21             | 1.38          | 1.60x     | 1.09 -> 1.74 |
| 8192  | 9.41   | 8.71             | 4.54          | 1.92x     | 1.08 -> 2.07 |
| 16384 | 38.5   | 33.3             | 16.8          | 1.97x     | 1.16 -> 2.28 |
| 32768 | OOM    | 130.0            | 65.8          | 1.98x     | – |
| 65536 | OOM    | 519.3            | 261.8         | 1.98x     | – |

(ms, do_bench median GPU time. Correctness checked for N <= 1024; tests pass.)

1. **Without the skip, the advantage did not grow** (speedup flat ~1.1 for
   N >= 2048). The Triton forward was causal-correct but looped over all key
   tiles, computing then masking the future half. Both impls did ~full N^2
   work at ~2 TFLOP/s (Triton N=65536: 1.1 TFLOP / 0.519 s; naive N=16384:
   ~1.8 TFLOP/s; 3070 fp32 peak ~20 TFLOP/s per spec), so the ratio stayed constant.
2. **With the skip, the advantage grows with N** (1.21 at 1024 -> 2.28 at
   16384). Dense naive attention can't skip: it must compute the full N x N
   matmul and mask afterwards. Skipping future tiles is FlashAttention's
   natural ~2x for causal.
3. **Crossover moved** from 1024-2048 (user's prediction 2048 matched the
   no-skip version) to 512-1024.
4. **Memory:** naive OOMs from N=32768; Triton runs to 65536 (O(N·d) memory).
5. **Open question (hypothesis, unverified): why the skip only gives ~1.0-1.5x
   for N <= 2048 but ~2x for N >= 8192.** With the skip, program i does i+1
   key tiles, so work is unbalanced; at small N there are too few programs
   (N=1024: 64 programs, 46 SMs) and wall time is set by the longest program
   (last query tile, still a full row). At large N, thousands of programs in
   many waves balance out. Test: raise batch (more programs) and see if the
   small-N skip gain approaches 2x.
6. Small-N oddity: Triton beats naive at N=128/256 (1.98x, 1.25x), loses at
   512. Guess (unverified): naive's ~10 kernels each have a fixed GPU cost
   that dominates at tiny N; Triton is one kernel.

## 2026-11-25 — Peak forward memory (fp32, d=64, causal)

Extra memory allocated during one forward (autograd on), on top of the inputs:

| N     | naive     | triton   |
|------:|----------:|---------:|
| 4096  | 280 MiB   | 1.0 MiB  |
| 8192  | 1088 MiB  | 2.0 MiB  |
| 16384 | 4352 MiB  | 4.1 MiB  |
| 32768 | OOM       | 8.1 MiB  |
| 65536 | OOM       | 16.2 MiB |

- naive grows 4x per doubling of N (O(N^2)): it materializes N x N scores
  and probabilities in HBM, and autograd keeps them for backward. 4352 MiB at
  N=16384 ≈ 4.25 N x N fp32 tensors (one is 1 GiB).
- triton grows 2x per doubling (O(N·d)): it only writes O (N x d) and L (N).
  16.2 MiB at N=65536 = 65536·64·4 B (O, 16 MiB) + 65536·4 B (L, 0.25 MiB):
  exactly the output, nothing N x N. Scores/probabilities live only in
  on-chip SRAM/registers, one tile at a time; backward recomputes P from L.

Not done: a stress test (sustained load, clocks/thermals over minutes) is a
different experiment from this latency benchmark.
