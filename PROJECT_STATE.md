# Project State
Updated: 2026-11-24

## Resume card
- Current task: Step-by-step walkthrough of Section 4 leftovers. Step 1 DONE, committed `d198752` (causal mask in PyTorch `flash_backward`). User then added causal tile-skip (continue/break) themselves; agent wired `FlashAttnTriton.backward` -> `flash_backward` (+ fixed `ctx.is_casual` typo). bf16 fixed: Triton fwd casts O on store; Triton bwd upcasts all to fp32 then casts grads back (simple, no bf16 speedup in bwd; proper mixed-precision bwd = possible later optimization). All 6 tests pass; causal fp32 grads err ~1e-6; bf16 causal O/grads err ~1e-2 vs fp32 ref (2026-11-24). Committed. Next: step 2 discussion (torch.compile unrolls the Python double loop).
- User decision (2026-11-24): teaching first, one concept per turn. Agent edits in one turn were too many; user reverted them with `git checkout .`.
- Branch `flash-attn`, HEAD `41a2edc`. Tracked files = user's code (causal backward absent; `FlashAttnTriton.backward` raises NotImplementedError; Triton fwd bf16 store lacks a cast to O dtype).
- Untracked agent files still present (reference only, user decides keep/delete): `cs336_systems/flash_benchmark.py`, `cs336_systems/allreduce_benchmark.py`, `scripts/gcp/*`, `reports/flash_benchmark.{csv,md}`. The benchmark depended on the reverted edits and won't run as-is; its results came from reverted code.
- Facts measured 2026-11-24 (RTX 3070) worth re-teaching: compiled 16x16 double-loop backward first-call 24s/102s/446s at N=128/256/512 (unrolling); TF32 makes fp32 Triton error ~2e-3; naive attention OOM at N>=32768.
- Planned order: 1 causal bwd -> 2 compile-unrolling problem -> 3 benchmark design -> 4 reading results -> 5 all-reduce metrics -> 6 GCP scripts (after user installs gcloud).
- Open: update CLAUDE.md "speed is the agent's job" line? (asked, no answer yet); xl vs smaller model for DDP; GCP standing permission.
- Test command: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_attention.py` (ROS pollutes PYTHONPATH).
- Sync status: local only; nothing committed or pushed by agent.

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
