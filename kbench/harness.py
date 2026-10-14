"""Evaluation harness: correctness against a reference, timing, held-out shapes."""

import hashlib
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx

DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


def metal_kernel(name, input_names, output_names, source, header="", **kwargs):
    """mx.fast.metal_kernel with a content-hashed name.

    MLX caches compiled kernels by name, so two candidates that share a name can
    silently run stale code (ml-explore/mlx#3832). Build every kernel through this.
    """
    digest = hashlib.sha1(f"{header}\0{source}".encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"{name}_{digest}",
        input_names=input_names,
        output_names=output_names,
        source=source,
        header=header,
        **kwargs,
    )


@dataclass
class Task:
    name: str
    description: str
    make_inputs: Callable[[dict, int], list]  # (shape cfg, seed) -> inputs
    reference: Callable  # eager MLX, i.e. what mlx-lm runs today
    shapes: list[dict]  # shown to the agent
    heldout: list[dict]  # only evaluated at the end
    bytes_moved: Callable[[dict], int]  # minimum DRAM traffic, for GB/s
    tol: dict  # dtype name -> (atol, rtol)


def _as_tuple(out):
    return tuple(out) if isinstance(out, (list, tuple)) else (out,)


def check(task: Task, fn: Callable, cfg: dict, seeds=(0, 1, 2)):
    """Return (ok, max_abs_err, message) comparing fn to task.reference."""
    atol, rtol = task.tol[cfg["dtype"]]
    worst = 0.0
    for seed in seeds:
        inputs = task.make_inputs(cfg, seed)
        want = _as_tuple(task.reference(*inputs))
        got = _as_tuple(fn(*inputs))
        mx.eval(want, got)
        if len(got) != len(want):
            return False, float("inf"), f"expected {len(want)} outputs, got {len(got)}"
        for i, (g, w) in enumerate(zip(got, want)):
            if g.shape != w.shape or g.dtype != w.dtype:
                return False, float("inf"), (
                    f"output {i}: got {g.shape} {g.dtype}, want {w.shape} {w.dtype}"
                )
            g32, w32 = g.astype(mx.float32), w.astype(mx.float32)
            if mx.any(mx.isnan(g32)).item():
                return False, float("inf"), f"output {i}: NaN"
            worst = max(worst, mx.max(mx.abs(g32 - w32)).item())
            if not mx.allclose(g32, w32, rtol=rtol, atol=atol).item():
                return False, worst, f"output {i}: mismatch (seed {seed})"
    return True, worst, ""


def time_fn(fn: Callable, inputs: Sequence, reps=256, trials=20, warmup=3):
    """Median (build_us, exec_us) per call.

    build: Python-side graph construction (MLX is lazy, nothing runs yet).
    exec:  mx.eval of `reps` independent calls, divided by `reps`, i.e. encode +
           dispatch + GPU time with the host sync amortized.

    At decode sizes these are the same order of magnitude, and custom kernels pay
    more build cost than built-in ops, so a single wall-clock number is misleading.
    mlx-lm overlaps building step t+1 with executing step t, so exec is the primary
    metric and build only matters if it exceeds exec.
    """
    for _ in range(warmup):
        mx.eval([fn(*inputs) for _ in range(reps)])
    mx.synchronize()
    build, execute = [], []
    for _ in range(trials):
        t0 = time.perf_counter()
        outs = [fn(*inputs) for _ in range(reps)]
        t1 = time.perf_counter()
        mx.eval(outs)
        mx.synchronize()
        t2 = time.perf_counter()
        build.append((t1 - t0) / reps)
        execute.append((t2 - t1) / reps)
    return statistics.median(build) * 1e6, statistics.median(execute) * 1e6


def evaluate(task: Task, fn: Callable, splits=("train", "heldout")) -> dict:
    variants = {
        "cand": fn,
        "cand_compiled": mx.compile(fn),
        "mlx": task.reference,
        "mlx_compiled": mx.compile(task.reference),
    }
    rows = []
    for split in splits:
        for cfg in task.shapes if split == "train" else task.heldout:
            row = {"split": split, **cfg}
            try:
                ok, err, msg = check(task, fn, cfg)
            except Exception as e:  # compile errors, bad launch params, ...
                ok, err, msg = False, float("inf"), f"{type(e).__name__}: {e}"
            row.update(correct=ok, max_err=err, message=msg)
            if ok:
                inputs = task.make_inputs(cfg, 0)
                t = {}
                for key, f in variants.items():
                    try:
                        t[key] = time_fn(f, inputs)
                    except Exception:  # e.g. candidate not traceable by mx.compile
                        t[key] = (float("inf"), float("inf"))
                cand_exec = min(t["cand"][1], t["cand_compiled"][1])
                mlx_exec = min(t["mlx"][1], t["mlx_compiled"][1])
                row.update(
                    timings_us={k: {"build": b, "exec": x} for k, (b, x) in t.items()},
                    cand_exec_us=cand_exec,
                    mlx_exec_us=mlx_exec,
                    speedup_exec=mlx_exec / cand_exec,
                    speedup_eager_total=sum(t["mlx"]) / sum(t["cand"]),
                    gbps=task.bytes_moved(cfg) / cand_exec / 1e3,
                )
            rows.append(row)
    return {
        "task": task.name,
        "device": mx.device_info().get("device_name"),
        "mlx": mx.__version__,
        "macos": platform.mac_ver()[0],
        "results": rows,
    }


def format_report(report: dict) -> str:
    head = (
        f"{'split':8} {'rows':>4} {'dim':>5} {'dtype':9} {'ok':3} {'max_err':>8} "
        f"{'cand_b':>7} {'cand_x':>7} {'mlx_b':>7} {'mlx_x':>7} "
        f"{'spd_x':>6} {'spd_tot':>7} {'GB/s':>6}"
    )
    legend = (
        "times in us; _b = Python graph build, _x = exec (best of eager/mx.compile); "
        "spd_x = exec speedup, spd_tot = eager build+exec speedup"
    )
    lines = [f"{report['task']} on {report['device']} (mlx {report['mlx']})", legend, head]
    for r in report["results"]:
        base = (
            f"{r['split']:8} {r['rows']:>4} {r['dim']:>5} {r['dtype']:9} "
            f"{'Y' if r['correct'] else 'N':3} {r['max_err']:>8.2e} "
        )
        if r["correct"]:
            t = r["timings_us"]
            base += (
                f"{t['cand']['build']:>7.2f} {r['cand_exec_us']:>7.2f} "
                f"{t['mlx']['build']:>7.2f} {r['mlx_exec_us']:>7.2f} "
                f"{r['speedup_exec']:>5.2f}x {r['speedup_eager_total']:>6.2f}x "
                f"{r['gbps']:>6.1f}"
            )
        else:
            base += r["message"][:80]
        lines.append(base)
    return "\n".join(lines)
