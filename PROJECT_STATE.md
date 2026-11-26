# Project State
Updated: 2026-11-25

## Resume card
- Current task: fp32 FlashAttention benchmark WRAPPED UP (2026-11-25). Report section "FlashAttention-2 Benchmark" in `reports/README.md` (setup, fwd table, tile-skip analysis, memory, bwd/e2e, head dim, IO-aware interpretation, limitations) + 4 figures `reports/flash_*.png` from `cs336_systems/flash_benchmark_plot.py`. Key verified results (RTX 3070, fp32 ieee, causal, batch 1, tile 16): speedup naive/triton fwd at N=16384 = 2.28x (d=64), 8.98x (d=16), 1.23x (d=128); tile skip ~2x at large N; naive OOM from N=32768; memory at N=16384 naive 4352 MiB vs triton 4.1 MiB. Open hypothesis: small-N skip gain limited by load imbalance (test with larger batch). Next: Section 5 distributed (allreduce script + GCP), or optional Triton backward. bf16 parked.
- Published 2026-11-25 (user approved): `flash-attn` pushed to origin (up to `1489818`); website essay live at https://yeqisang.com/writing/flashattention-benchmark/ and LLM project page updated (website commit `f4c24d6`). Repo default branch is `master`, which does not yet contain the flash-attn work — merge when the user decides.
- Stale, uncommitted, to delete with user OK: `reports/flash_benchmark.{csv,md,log}` (reverted 2026-11-24 code).
- User decisions: teaching first, one concept per turn; tile=16 everywhere; no torch.compile; Triton `tl.dot(input_precision="ieee")` (no TF32); one shared precision policy (fp32 S/softmax stats/accumulators, input-dtype matmul inputs) for naive/tiled/triton; backward measured only N<=2048; warmup stays do_bench default 25ms; 2-GPU work belongs to Section 5, not this benchmark.
- Branch `flash-attn`; commits today: `d198752` causal bwd, `bb95256` Triton bwd wiring + tile skip, `7f5c468` remove compile, then precision/benchmark commit (see git log). Not pushed.
- Verified (RTX 3070, 2026-11-24): 6/6 attention tests pass; fp32 err ~1e-6 all impls; N=1024 d=64 fp32 do_bench fwd naive 0.176 / tiled 1469 / triton 0.214 ms (naive beat triton — see reports/flash_benchmark_notes.md); do_bench = GPU time (L2 flush hides CPU launch overhead).
- Untracked, not yet reviewed with user: `cs336_systems/allreduce_benchmark.py`, `scripts/gcp/*` (GCP tooling never run; gcloud not installed).
- TODO after A2: compiled per-tile backward w/ larger tiles; Triton backward; tile-size sweep; TF32 as separate setting; realistic batch x heads=48 run; bf16 sweep.
- Open: CLAUDE.md "speed is the agent's job" line (asked, unanswered); xl vs smaller model for DDP; GCP standing permission.
- Run: `env -u PYTHONPATH -u VIRTUAL_ENV uv run python -m cs336_systems.flash_benchmark`; tests: add `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_attention.py` (ROS pollutes PYTHONPATH).

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
