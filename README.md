# MLX_kernel

Agent-designed fused Metal kernels for batch-size-1 LLM decode on [MLX](https://github.com/ml-explore/mlx).

The question: Metal is a low-resource kernel language for LLMs compared to CUDA. Without
any training, what context and loop structure let an agent write fused kernels that beat
stock MLX (including `mx.compile`)? Candidate ablations:

- **few-shot source**: none / CUDA examples / MLX's own Metal kernels retrieved from
  `mlx/backend/metal/kernels/`
- **roles**: single agent / implementer + independent reviewer
  (in the spirit of [humanize](https://github.com/PolyArch/humanize)'s RLCR loop)
- **feedback**: correctness + time / plus achieved bandwidth and roofline

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

## Run

```bash
uv run python -m kbench.run add_rmsnorm candidates/add_rmsnorm_hand.py
```

A candidate is a Python file defining `forward(*inputs)` with the same signature and
outputs as the task's reference. Build kernels with `kbench.harness.metal_kernel`, not
`mx.fast.metal_kernel` directly (see below).

## Layout

```
kbench/harness.py   correctness check, timing, held-out evaluation, report
kbench/tasks/       one file per task: reference, input generator, shapes, bytes moved
kbench/run.py       CLI
candidates/         candidate kernels (hand-written baselines; agent output later)
```

## Measurement notes

- **Build vs exec.** At decode sizes an op takes ~2 us, and Python graph construction
  is the same order of magnitude (built-in op ~0.35 us, `metal_kernel` call ~1.3 us).
  The harness reports both. Exec is the primary metric because mlx-lm overlaps building
  step t+1 with executing step t.
- **Baseline is the best of eager MLX and `mx.compile`.** `mx.compile` already fuses
  elementwise ops, so comparing against eager alone overstates gains.
- **Unique kernel names.** MLX caches kernels by name, and same-name kernels with
  different sources can silently run stale code (ml-explore/mlx#3832). The
  `metal_kernel` wrapper appends a hash of the source.
- **Held-out shapes** (borrowed from Metal-Sci) are evaluated only at the end, including
  a dimension that is not a multiple of 4, to catch kernels overfit to the search shapes.

## Results so far

Hardware: Mac mini, Apple M6, 16 GB, measured ~135 GB/s DRAM bandwidth, macOS 27.0,
MLX 0.32.3.

`add_rmsnorm` (residual add + RMSNorm, two kernels in MLX), hand-written fused kernel:

| split   | rows | dim  | dtype    | exec us (fused / MLX) | exec speedup | eager build+exec speedup |
|---------|-----:|-----:|----------|----------------------:|-------------:|-------------------------:|
| train   | 1    | 2048 | float16  | 1.96 / 2.46           | 1.25x        | 0.80x                    |
| train   | 1    | 2048 | bfloat16 | 2.04 / 2.44           | 1.20x        | 0.80x                    |
| train   | 1    | 4096 | float16  | 2.04 / 2.44           | 1.19x        | 0.84x                    |
| train   | 1    | 4096 | bfloat16 | 2.01 / 2.49           | 1.24x        | 0.85x                    |
| heldout | 1    | 2560 | bfloat16 | 1.99 / 2.42           | 1.22x        | 0.85x                    |
| heldout | 1    | 5120 | float16  | 2.05 / 2.55           | 1.24x        | 0.88x                    |
| heldout | 8    | 4096 | bfloat16 | 2.25 / 2.97           | 1.32x        | 0.81x                    |
| heldout | 1    | 2050 | float16  | 1.96 / 2.46           | 1.25x        | 0.83x                    |

At bs=1 the op moves tens of KB, so time is dominated by dispatch, not bandwidth.
Fusion wins by removing a dispatch; the extra Python cost of `metal_kernel` eats that
back in pure eager mode but not under `mx.compile`.

## Roadmap

- [ ] more decode tasks: RoPE + KV-cache write, SwiGLU (+ 4-bit matvec), MoE top-k gating
- [ ] agent loop (API-driven, reproducible) and the ablations above
- [ ] anti-cheating checks on candidates (Metal-only, no calls back into MLX ops)
- [ ] end-to-end: patch into mlx-lm on a small model, report tok/s

## Related work

- Gimlet Labs, AI-generated Metal kernels for PyTorch on Apple devices (2025), baseline is
  PyTorch MPS rather than MLX
- [Metal-Sci](https://arxiv.org/abs/2605.09708) (2026), evolutionary LLM kernel search on
  Apple Silicon, scientific compute rather than ML ops
- [ZMLX](https://pypi.org/project/zmlx/), hand-written fused MLX kernels for MoE decode
