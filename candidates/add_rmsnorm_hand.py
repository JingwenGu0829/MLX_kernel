"""Hand-written fused residual add + RMSNorm: one threadgroup per row, one kernel.

Reads x and r once to form h, writes h, reduces sum(h^2) in fp32 within the
threadgroup, then re-reads x and r (cache-resident at decode sizes) to write y.
"""

from kbench.harness import metal_kernel

SOURCE = """
    constexpr float eps = 1e-6f;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    const device T* xr = x + row * D;
    const device T* rr = r + row * D;
    device T* hr = h + row * D;
    device T* yr = y + row * D;

    float acc = 0.0f;
    for (uint i = tid; i < D; i += TG) {
        T hv = static_cast<T>(float(xr[i]) + float(rr[i]));
        hr[i] = hv;
        float f = float(hv);
        acc += f * f;
    }

    threadgroup float partial[32];
    acc = simd_sum(acc);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (TG / 32) ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = metal::rsqrt(partial[0] / float(D) + eps);

    for (uint i = tid; i < D; i += TG) {
        T hv = static_cast<T>(float(xr[i]) + float(rr[i]));
        yr[i] = static_cast<T>(float(hv) * inv) * w[i];
    }
"""

kernel = metal_kernel(
    name="add_rmsnorm_hand",
    input_names=["x", "r", "w"],
    output_names=["h", "y"],
    source=SOURCE,
)

TG = 256


def forward(x, r, w):
    rows, dim = x.shape
    return kernel(
        inputs=[x, r, w],
        template=[("T", x.dtype), ("D", dim), ("TG", TG)],
        grid=(rows * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )
