#include "mmv-fv5.cuh"

// Fermion FV5/FV5B fused F32-activation GEMV.
//
// Weight reconstruction is identical to the CPU reference (ggml-quants.c /
// ggml-cpu/quants.c):
//   FV5:  w[j] = (bp[j] - bn[j]) * (br[j] ? s_hi : s_lo)
//   FV5B: w[j] = s * qs[j]
// The dot against the raw f32 activation vector factors into masked f32
// activation sums (sum_lo/sum_hi for FV5), scaled once per row: the per-row
// dual scales are stored as exact f32 copies replicated into every block of
// the row (see block_fv5 in ggml-common.h), so scaling the row totals is
// exact w.r.t. the container's per-row scales. All accumulation is f32.
//
// One CUDA block per output row; each thread walks plane bytes (8 weights per
// byte) with a stride of blockDim.x, then the two partial sums are reduced
// block-wide (warp shuffle + shared memory).

#define MMV_FV5_BLOCK_SIZE 128

static __global__ void mul_mat_vec_fv5_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5 * x = (const block_fv5 *) vx + row*nblocks_per_row;

    float sum_lo = 0.0f;
    float sum_hi = 0.0f;

    const int64_t nbytes = nblocks_per_row * (QK_FV5/8);

    for (int64_t idx = tid; idx < nbytes; idx += MMV_FV5_BLOCK_SIZE) {
        const int64_t ib = idx >> 5;          // block index (32 plane bytes per block)
        const int     j  = (int)(idx & 31);   // plane byte within block

        const uint8_t bp = x[ib].bp[j];
        const uint8_t bn = x[ib].bn[j];
        if (!(bp | bn)) {
            continue;
        }
        const uint8_t br = x[ib].br[j];
        const float * yj = y + ib*QK_FV5 + 8*j;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const uint8_t bit = 1u << b;
            float v = 0.0f;
            if (bp & bit) {
                v = yj[b];
            } else if (bn & bit) {
                v = -yj[b];
            }
            // br is a subset of bp|bn, so for zero weights v == 0 either way
            if (br & bit) {
                sum_hi += v;
            } else {
                sum_lo += v;
            }
        }
    }

    // block-wide reduction of (sum_lo, sum_hi)
    float2 sums = make_float2(sum_lo, sum_hi);
    sums = warp_reduce_sum(sums);

    __shared__ float2 s_sums[MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_sums[warp_id] = sums;
    }
    __syncthreads();
    if (warp_id == 0) {
        sums = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sums[lane_id] : make_float2(0.0f, 0.0f);
        sums = warp_reduce_sum(sums);
        if (lane_id == 0) {
            // per-row scales are replicated into every block: block 0 is exact
            dst[row] = x[0].s_lo*sums.x + x[0].s_hi*sums.y;
        }
    }
}

static __global__ void mul_mat_vec_fv5b_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5b * x = (const block_fv5b *) vx + row*nblocks_per_row;

    float sum = 0.0f;

    const int64_t ngroups = nblocks_per_row * (QK_FV5/8); // groups of 8 int8

    for (int64_t idx = tid; idx < ngroups; idx += MMV_FV5_BLOCK_SIZE) {
        const int64_t ib = idx >> 5;
        const int     j  = (int)(idx & 31);

        const int8_t * q  = x[ib].qs + 8*j;
        const float  * yj = y + ib*QK_FV5 + 8*j;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            sum += (float) q[b] * yj[b];
        }
    }

    sum = warp_reduce_sum(sum);

    __shared__ float s_sum[MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
        s_sum[warp_id] = sum;
    }
    __syncthreads();
    if (warp_id == 0) {
        sum = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sum[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            dst[row] = x[0].s * sum;
        }
    }
}

bool ggml_cuda_can_mul_mat_vec_fv5(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return (src0->type == GGML_TYPE_FV5 || src0->type == GGML_TYPE_FV5B) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1;
}

void ggml_cuda_mul_mat_vec_fv5(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat_vec_fv5(src0, src1, dst));
    GGML_ASSERT(src0->ne[0] % QK_FV5 == 0);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const int64_t nrows           = src0->ne[1];
    const int64_t nblocks_per_row = src0->ne[0] / QK_FV5;

    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(MMV_FV5_BLOCK_SIZE, 1, 1);

    cudaStream_t stream = ctx.stream();

    if (src0->type == GGML_TYPE_FV5) {
        mul_mat_vec_fv5_f32<<<block_nums, block_dims, 0, stream>>>(
            src0->data, (const float *) src1->data, (float *) dst->data, nblocks_per_row);
    } else {
        mul_mat_vec_fv5b_f32<<<block_nums, block_dims, 0, stream>>>(
            src0->data, (const float *) src1->data, (float *) dst->data, nblocks_per_row);
    }
}
