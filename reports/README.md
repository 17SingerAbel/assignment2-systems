# CS336 Assignment 2 — Nsight Systems Profiling (Draft)

> **Status:** Profiling experiments completed; final report and per-configuration comparisons pending. These notes capture preliminary observations, not a finalized benchmark report.

## 1. Benchmarking vs. Profiling Overhead

I compared the forward-pass runtime of my PyTorch Transformer under three measurement configurations:

| Measurement | Approximate forward runtime |
| --- | ---: |
| Python timing with `torch.cuda.synchronize()` | 14 ms |
| Nsight Systems, lightweight tracing (`cuda,nvtx`) | 19 ms |
| Nsight Systems, extensive tracing | 40 ms |

The substantial reduction from 40 ms to 19 ms after disabling additional tracing suggests that **profiling instrumentation contributed significantly to the discrepancy**. Warmup reduces initialization-related effects, but does not remove tracing overhead during measurement.

**Takeaway:** Use synchronized Python timing as a baseline for end-to-end runtime. Use Nsight Systems to investigate CUDA API activity, kernel launches, and GPU execution. Ensure the compared measurements use equivalent scopes and configurations.

## 2. Forward-Pass CUDA Kernel Breakdown

In one profiling report, the `CUDA GPU Kernel Summary` showed:

| CUDA kernel | Total GPU kernel time | Instances | Share of reported GPU kernel time |
| --- | ---: | ---: | ---: |
| `ampere_sgemm_128x64_tn` | 36.76 ms | 296 | 44.8% |
| `ampere_sgemm_128x32_tn` | 17.94 ms | 384 | 21.9% |
| `ampere_sgemm_128x128_nn` | 3.34 ms | 96 | 4.1% |
| `ampere_sgemm_128x128_tn` | 2.26 ms | 96 | 2.7% |
| Largest `vectorized_elementwise_kernel` entry | 4.16 ms | 192 | 5.1% |

`sgemm` denotes single-precision general matrix multiplication (GEMM). The `128x64` and `128x32` portions identify aspects of the cuBLAS kernel's tiling strategy; they are **not** the dimensions of the model's weight matrices.

The two most expensive GEMM kernel types together accounted for **66.7% of the summed GPU kernel execution time in this report**. This percentage is not a share of end-to-end wall-clock runtime. The instance counts above cover the **entire captured report**, not one forward pass, so they should not be reported as per-forward invocation counts.

## 3. Why FLOPs Do Not Directly Predict Runtime

**Core lesson: Fewer FLOPs do not necessarily mean faster execution.**

The time required by a GPU operation depends not only on its floating-point operation count but also on memory traffic, achievable compute throughput, reductions, launch overhead, and opportunities for fusion.

### Self-attention: Softmax vs. matrix multiplication

For scaled dot-product attention:

\[
S = QK^T / \sqrt{d_k}, \qquad P = \operatorname{softmax}(S), \qquad O = PV.
\]

The two matrix multiplications (`QK^T` and `PV`) perform substantially more FLOPs than Softmax, but my **synchronized operation-level measurements** suggested that attention-score computation and Softmax took surprisingly similar amounts of time in the tested configuration.

A numerically stable Softmax includes a row-wise maximum, subtraction, exponentiation, row-wise sum, and normalization. In an unfused PyTorch implementation, these steps may require multiple elementwise and reduction kernels, with additional launches and intermediate-memory traffic. This can make Softmax relatively expensive despite its lower FLOP count.

**Qualification:** The observation concerns the measured operation runtimes. I have not yet isolated and summed the exact GPU kernel durations attributable to each operation; therefore I should not claim their pure GPU kernel times are identical.

### Reduction and elementwise kernels

The report also contained `MaxOps`, `MeanOps`, `exp_kernel_cuda`, `sigmoid_kernel_cuda`, and other elementwise kernels. These names are consistent with operations used throughout the model, including Softmax, RMSNorm, SwiGLU, and the loss. Kernel names alone do not uniquely identify their Python call sites.

A reduction is not the same as an elementwise operation: Softmax needs reductions to compute a maximum and normalization denominator, while RMSNorm computes a mean of squared activations.

## 4. Full Training Step: Forward, Backward, and AdamW

For one synchronized, profiled full training step, the recorded phase durations were approximately:

| Phase | Measured duration |
| --- | ---: |
| Forward | 51 ms |
| Backward | 101 ms |
| AdamW optimizer step | 69 ms |
| **Sum** | **221 ms** |

The separate synchronized Python timing for this training step was approximately **220 ms**, close to the sum above for this configuration. This agreement should not be generalized to other profiling settings.

Backward takes additional work to compute gradients with respect to both activations and parameters. AdamW performs many per-parameter elementwise updates and reads/writes parameter, gradient, and optimizer-state tensors; its runtime can be significant despite a relatively modest FLOP count.

Comparing the forward-only and full-training kernel summaries, I observed that **GEMM's share decreased while elementwise kernels' share increased**. A plausible explanation is the additional elementwise work in backward and AdamW. A smaller *percentage* of GEMM time does **not** mean the absolute GEMM execution time decreased.

**Important distinction:** AdamW momentum updates are predominantly elementwise operations, **not reductions**. Reduction kernels elsewhere in the training step can come from gradient calculations, normalization, loss computation, or gradient clipping (if enabled).

## 5. Implications for Kernel Fusion

These results motivate investigating kernel fusion and memory-efficient attention:

- Fusing elementwise and reduction operations can reduce kernel launches and intermediate tensor reads/writes.
- Fused Softmax can combine row-wise maximum, exponentiation, normalization, and related steps within fewer GPU kernels.
- FlashAttention reduces expensive reads/writes of large intermediate attention matrices through tiling and online Softmax.
- Fusion is **not automatically faster**: register pressure, occupancy, data layout, and kernel size still matter.

**Main takeaway:** *FLOPs alone do not determine runtime. Kernel launches, memory movement, reductions, and hardware utilization matter too.*

## 6. Items to Complete in the Final Report

- Record the exact model configurations, context lengths, dtype, GPU, and tracing commands for each experiment.
- Scope kernel summaries to the relevant forward or training-step interval; report per-forward invocation counts rather than counts from the whole capture.
- Add the requested comparisons across two model sizes and three context lengths, where available.
- Quantify the forward-only versus full-step GEMM/non-GEMM shares from their corresponding scoped GPU kernel summaries.
- Clearly distinguish synchronized wall-clock durations from cumulative GPU kernel execution time.



# Mixed Precision Benchmark — FP32 vs BF16 AMP

## 1. Experiment Setup

The benchmark compares standard FP32 training with BF16 Automatic Mixed Precision (AMP) training.

Both configurations use:
- FP32 model parameters
- FP32 parameter gradients
- FP32 AdamW optimizer states
- The same model architecture and training configuration

The key difference is that BF16 AMP uses `torch.autocast` during forward computation, allowing eligible operations to run in BF16.

Backward operations follow the computation graph and the dtypes selected during forward. The optimizer runs outside the autocast context.

## 2. Benchmark Results

Results obtained from Nsight Systems profiling:

| Stage | FP32 | BF16 AMP |
|---|---:|---:|
| Forward | 26.9 ms | 32.0 ms |
| Backward | 69.8 ms | 73.4 ms |
| Optimizer | 133.8 ms | 40.1 ms |
| Total | 255.7 ms | 166.6 ms |

BF16 AMP reduced the overall measured execution time by approximately 35%.

## 3. Observations

### Forward and Backward

Surprisingly, BF16 AMP was slightly slower than FP32 in both forward and backward execution.

Possible explanations include:

- Additional FP32-to-BF16 casting operations.
- Autocast dispatch overhead.
- Some operations may not benefit from BF16 Tensor Cores.
- Profiling overhead and measurement variability.

Although BF16 can improve GEMM throughput, it does not reduce the mathematical FLOPs required by the model.

### Optimizer

The most unexpected result was the optimizer execution time:

- FP32: 133.8 ms
- BF16 AMP: 40.1 ms

The FP32 optimizer appeared approximately 3.3 times slower.

This is unexpected because both configurations use FP32 parameters, gradients, and optimizer states.

Therefore, the optimizer should theoretically have similar computational and memory-access requirements in both configurations.

## 4. Hypotheses

The optimizer timing difference has not yet been explained.

Potential causes include:

1. **Profiling overhead:** The custom Python AdamW implementation launches many small tensor operations, potentially increasing tracing overhead.

2. **CUDA asynchronous execution:** NVTX ranges around `optimizer.step()` do not necessarily represent the complete GPU execution time without synchronization.

3. **GPU memory pressure:** FP32 training may require more activation memory than BF16 AMP. However, higher memory allocation alone does not necessarily imply slower execution.

4. **Measurement variability:** GPU scheduling, warmup behavior, and a small number of benchmark iterations may affect the results.

These explanations remain hypotheses and have not been verified.

## 5. Next Steps — Memory Profiling

The next experiment will compare:

- Peak CUDA memory allocated
- Peak CUDA memory reserved
- FP32 vs BF16 AMP activation memory usage
- Optimizer timing after sufficient warmup
- CPU-side NVTX duration vs actual CUDA kernel execution time

The main question to investigate is:

**Why does the FP32 optimizer appear significantly slower than the BF16 AMP optimizer, despite both operating on FP32 parameters and gradients?**

## 6. Preliminary Conclusion

BF16 AMP achieved approximately 35% lower total measured runtime in this experiment.

However, the observed improvement was dominated by the optimizer timing difference rather than faster forward or backward computation.

Since the optimizer operates on FP32 data in both configurations, this difference cannot yet be attributed directly to mixed precision.

Further profiling is required before drawing conclusions about the performance benefits of BF16 AMP.
