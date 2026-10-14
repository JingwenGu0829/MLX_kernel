"""Residual add followed by RMSNorm, as in every pre-norm decoder block:

    h = x + r            # residual stream
    y = rms_norm(h) * w  # input to the next attention / MLP

Eager MLX runs this as two kernels and writes h to memory in between.
"""

import mlx.core as mx

from kbench.harness import DTYPES, Task

EPS = 1e-6


def make_inputs(cfg, seed):
    k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
    dtype = DTYPES[cfg["dtype"]]
    shape = (cfg["rows"], cfg["dim"])
    x = mx.random.normal(shape, key=k1).astype(dtype)
    r = mx.random.normal(shape, key=k2).astype(dtype)
    w = (1.0 + 0.1 * mx.random.normal((cfg["dim"],), key=k3)).astype(dtype)
    return [x, r, w]


def reference(x, r, w):
    h = x + r
    return h, mx.fast.rms_norm(h, w, EPS)


def bytes_moved(cfg):
    itemsize = DTYPES[cfg["dtype"]].size
    # read x, r, w; write h, y
    return (4 * cfg["rows"] * cfg["dim"] + cfg["dim"]) * itemsize


TASK = Task(
    name="add_rmsnorm",
    description=__doc__,
    make_inputs=make_inputs,
    reference=reference,
    shapes=[
        {"rows": 1, "dim": d, "dtype": t}
        for d in (2048, 4096)
        for t in ("float16", "bfloat16")
    ],
    heldout=[
        {"rows": 1, "dim": 2560, "dtype": "bfloat16"},
        {"rows": 1, "dim": 5120, "dtype": "float16"},
        {"rows": 8, "dim": 4096, "dtype": "bfloat16"},
        {"rows": 1, "dim": 2050, "dtype": "float16"},  # not a multiple of 4
    ],
    bytes_moved=bytes_moved,
    tol={"float16": (2e-2, 1e-2), "bfloat16": (6e-2, 2e-2), "float32": (1e-4, 1e-4)},
)
