# Project State
Updated: 2026-11-18

## Resume card
- Current task: Diagnose the user's failing PyTorch backward test using the updated source; no test execution by the assistant.
- Branch: `flash-attn`; HEAD `7d4b9ca` (2026-11-18, backward-pass checkpoint).
- Last discussion: User reports dQ comparison failure (32766/32768 mismatches; maximum absolute difference about 997.7); inspect softmax-gradient factors and query-specific D slicing.
- Completed artifacts: Benchmark harness, profiling draft and captures, PyTorch tiled forward, Triton tiled forward, and adapter wiring.
- Verification boundary: Artifact existence and current source inspected; no tests, notebooks, GPU experiments, or profiler captures rerun during this review.
- Current local work: Compiled helper has separate dK/dV and dQ tile loops; independent output buffers, integer tile counts, and transpose spelling are now present.
- Blockers: Both loops still use a suspect softmax-gradient factor; the dQ loop uses d_tile left from the preceding loop instead of selecting it for the current query tile. Causal backward handling is absent. Triton backward is unimplemented.
- Confirmed conceptual decision: Preserve independent batch samples; do not flatten batch and sequence into one attention sequence.
- Proposed next action: Student rechecks both ds expressions against the softmax derivative and traces d_tile's query range in the dQ loop, then reruns the focused backward test.
- Teaching boundary: Explain and review student work; do not implement core assignment solutions.
- Execution context: `/home/abel/projects/assignment2-systems`; dependency workflow documented as `uv`; runtime/GPU availability not checked in this review.
- Sync status: This handoff was created locally; no commit, push, or cross-device synchronization performed.

## Evidence and remaining questions
- `cs336_systems/benchmark.py`: forward, forward/backward, and full-step modes; warmup, synchronized timing, NVTX ranges, autocast, and memory snapshots.
- `reports/README.md`: historical forward measurements approximately 14/19/40 ms across timing/tracing settings; figures are preliminary and configuration-dependent.
- Mixed-precision draft: total 255.7 ms FP32 vs 166.6 ms BF16 AMP; the large optimizer timing gap is unexplained and does not establish a causal BF16 speedup.
- Memory note: historical peak observations approximately 4.5 GB FP32 and 2.5 GB BF16; allocation/reservation scope and experiment comparability need clarification.
- PyTorch forward contains tiled online softmax, optional causal masking, and saved Q/K/V/O/L. Commit `c764e91` describes the PyTorch version as done, but does not prove the current working tree passes tests.
- Triton forward contains a query-tile/batch grid, block pointers, explicit strides, online softmax, and causal/key-boundary masking. Correctness and performance were not revalidated.
- Prior 2026-11-01 discussion used Q/K/V `(4,128,64)` and identified batch/sequence semantics and a running-max shape mismatch; the historical summary recorded no verified fix.
- `tests/adapters.py`: FlashAttention implementations are wired in; DDP, FSDP, and sharded optimizer interfaces remain unimplemented.
- `tests/test_attention.py` supplies forward/backward reference comparisons for later validation.

## Tiled backward source review (2026-11-18)
- The student's dV contraction and dK contraction have the expected query-reduction shapes; this does not verify numerical correctness.
- `ds = dp * (dp - d_tile)` needs comparison with the softmax derivative; feedback should guide the student to identify the probability factor.
- `dK = dV = torch.zeros_like(K)` aliases the final output buffers, so writing one can overwrite the other. Tile accumulators currently use out-of-place rebinding; distinguish that from final-buffer slice writes.
- `/` produces float tile counts and does not handle partial tiles; `tranpose` is misspelled in three calls. No assignment implementation was edited or executed.
- Subsequent source review: those loop-count, spelling, and output-aliasing issues were addressed. The user supplied a failing dQ assertion; later dK/dV assertions have not been reached.
- In the updated dQ loop, d_tile is not reselected for the current query block. Equal tile shapes can conceal a stale query range; suggest tracing i/q_start and D slice positions on two distinct query blocks.
