# Project State
Updated: 2026-11-20

## Resume card
- Current task: User is working on distributed_hello_world.py and wants a quick route to a GCP VM with two GPUs. PyTorch FlashAttention is complete per user; its benchmark remains deferred.
- Branch: `flash-attn`; HEAD `c9b3851` (2026-11-18, tiled-backward checkpoint), verified on 2026-11-20.
- Last discussion: Reviewed the CPU/Gloo demo and researched GCP single-VM, two-GPU provisioning; proposed N1 + 2 T4 GPUs using a current Deep Learning VM base image.
- Completed artifacts: Benchmark harness, profiling draft/captures, PyTorch tiled forward/backward (completion confirmed by user), Triton tiled forward, and adapter wiring.
- Verification boundary: Current source and Git history inspected on 2026-11-20; no tests, notebooks, GPU experiments, or profiler captures rerun. No fresh test-pass claim is made.
- Source evidence: Both backward loops now use p_tile in the softmax-gradient expression; the dQ loop selects d_tile for its current query range. The two previously recorded blockers are superseded.
- Remaining scope: Causal backward handling is absent in the inspected PyTorch helper. Triton backward was previously recorded as unimplemented; it was not re-inspected in this correction.
- Confirmed conceptual decision: Preserve independent batch samples; do not flatten batch and sequence into one attention sequence.
- Proposed next action: User checks GCP billing/quota and creates the proposed VM if desired, verifies two GPUs, then adapts the demo for GPU communication. No cloud creation command was executed by the agent.
- Teaching boundary: Explain and review student work; do not implement core assignment solutions.
- Execution context: `/home/abel/projects/assignment2-systems`; dependency workflow documented as `uv`; runtime/GPU availability not checked in this review.
- Sync status: This correction is local; no commit, push, or cross-device synchronization performed in this turn.

## Distributed hello world / GCP planning (2026-11-20)
- User-selected task: distributed_hello_world.py; GPU provisioning guidance is separate from the deferred FlashAttention benchmark.
- Source inspection: the untracked script uses Gloo, CPU tensors, device="cpu", world_size=4, and mp.spawn. A GPU VM alone will not change its communication path. Guide the student through backend, tensor placement, per-process GPU ownership, and process count without implementing the assignment.
- Proposed VM: n1-standard-8 with 2 nvidia-tesla-t4 GPUs, 100 GB boot disk, us-central1-a as an initial supported zone; actual capacity is unknown.
- Proposed image: common-cu129-ubuntu-2404-nvidia-580 from deeplearning-platform-release, listed in the current official image catalog. It supplies a CUDA/driver base; use the repository's uv environment for its Python/PyTorch requirements rather than assuming the image supplies the project's exact dependencies.
- Prerequisites: billing, Compute Engine API, regional T4 quota and applicable global GPU quota with at least two GPUs available. Quota does not guarantee zonal stock.
- Verification: read-only source/config inspection and official Google Cloud documentation lookup only; no runtime tests, dependency installation, cloud account inspection, VM creation, or spending performed.
- Sources: https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-general-purpose ; https://docs.cloud.google.com/deep-learning-vm/docs/images ; https://docs.cloud.google.com/compute/resource-usage ; https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones

## TODO: FlashAttention benchmark (deferred, 2026-11-18)
- [ ] Confirm the assignment's required implementations, input configurations, and timing metrics before designing the experiment.
- [ ] Produce a comparison table including forward latency. Proposed metadata: implementation, GPU, input shape, dtype, causal mode, and latency in ms; determine whether backward latency is also required.
- [ ] Reuse the earlier NVTX + Nsight Systems workflow for profiling. Keep latency baseline measurements separate from profiled measurements and record their measurement scopes.
- [ ] Plan warmup, compilation exclusion, synchronized GPU timing, and repeated measurements with a stated summary statistic.
- [ ] Evaluate access to a B200 server as the user's candidate execution environment. No hardware requirement, provider, rental, spending, or provisioning decision has been made.
- [ ] Preserve the experiment configuration, environment, commands, results table, and relevant profiling captures in the project reports.
- Status: Planning only; no benchmark code changes, experiments, server provisioning, or spending authorized by this TODO request. Resolve backward correctness before interpreting backward performance.

## Evidence and remaining questions
- `cs336_systems/benchmark.py`: forward, forward/backward, and full-step modes; warmup, synchronized timing, NVTX ranges, autocast, and memory snapshots.
- `reports/README.md`: historical forward measurements approximately 14/19/40 ms across timing/tracing settings; figures are preliminary and configuration-dependent.
- Mixed-precision draft: total 255.7 ms FP32 vs 166.6 ms BF16 AMP; the large optimizer timing gap is unexplained and does not establish a causal BF16 speedup.
- Memory note: historical peak observations approximately 4.5 GB FP32 and 2.5 GB BF16; allocation/reservation scope and experiment comparability need clarification.
- PyTorch forward contains tiled online softmax, optional causal masking, and saved Q/K/V/O/L. Commit `c764e91` describes the PyTorch version as done; `c9b3851` adds the tiled-backward checkpoint. On 2026-11-20 the user confirmed the PyTorch version is complete; no tests were rerun during this correction.
- Triton forward contains a query-tile/batch grid, block pointers, explicit strides, online softmax, and causal/key-boundary masking. Correctness and performance were not revalidated.
- Prior 2026-11-01 discussion used Q/K/V `(4,128,64)` and identified batch/sequence semantics and a running-max shape mismatch; the historical summary recorded no verified fix.
- `tests/adapters.py`: FlashAttention implementations are wired in; DDP, FSDP, and sharded optimizer interfaces remain unimplemented.
- `tests/test_attention.py` supplies forward/backward reference comparisons for later validation.

## Tiled backward source review (2026-11-18)
- Historical debugging notes below describe the earlier checkpoint. The 2026-11-20 resume card supersedes their open-blocker status; current source contains both softmax-factor and query-slice corrections.
- The student's dV contraction and dK contraction have the expected query-reduction shapes; this does not verify numerical correctness.
- `ds = dp * (dp - d_tile)` needs comparison with the softmax derivative; feedback should guide the student to identify the probability factor.
- `dK = dV = torch.zeros_like(K)` aliases the final output buffers, so writing one can overwrite the other. Tile accumulators currently use out-of-place rebinding; distinguish that from final-buffer slice writes.
- `/` produces float tile counts and does not handle partial tiles; `tranpose` is misspelled in three calls. No assignment implementation was edited or executed.
- Subsequent source review: those loop-count, spelling, and output-aliasing issues were addressed. The user supplied a failing dQ assertion; later dK/dV assertions have not been reached.
- In the updated dQ loop, d_tile is not reselected for the current query block. Equal tile shapes can conceal a stale query range; suggest tracing i/q_start and D slice positions on two distinct query blocks.
