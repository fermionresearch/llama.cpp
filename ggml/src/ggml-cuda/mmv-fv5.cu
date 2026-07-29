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
// byte) with a stride of blockDim.x, then the partial sums are reduced
// block-wide (warp shuffle + shared memory). The kernels are templated on
// ncols_dst (1..MMV_FV5_MAX_BATCH_SIZE activation columns, mirroring
// mmvf.cu/mmvq.cu): the bit-plane bytes are loaded once and the masked sums
// are accumulated for every column, so the weight traffic stays 1x for the
// small batches of speculative verify.

#define MMV_FV5_BLOCK_SIZE 128

template <int ncols_dst>
static __global__ void mul_mat_vec_fv5_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row, const int64_t stride_col_y, const int64_t stride_col_dst) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5 * x = (const block_fv5 *) vx + row*nblocks_per_row;

    float2 sums[ncols_dst];
#pragma unroll
    for (int c = 0; c < ncols_dst; ++c) {
        sums[c] = make_float2(0.0f, 0.0f);
    }

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
            if (!((bp | bn) & bit)) {
                continue;
            }
            const float sgn = (bp & bit) ? 1.0f : -1.0f;
#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                const float v = sgn*yj[c*stride_col_y + b];
                // br is a subset of bp|bn, so zero weights never reach here
                if (br & bit) {
                    sums[c].y += v;
                } else {
                    sums[c].x += v;
                }
            }
        }
    }

    // block-wide reduction of (sum_lo, sum_hi) per column
#pragma unroll
    for (int c = 0; c < ncols_dst; ++c) {
        sums[c] = warp_reduce_sum(sums[c]);
    }

    __shared__ float2 s_sums[ncols_dst][MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            s_sums[c][warp_id] = sums[c];
        }
    }
    __syncthreads();
    if (warp_id == 0) {
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            float2 s = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sums[c][lane_id] : make_float2(0.0f, 0.0f);
            s = warp_reduce_sum(s);
            if (lane_id == 0) {
                // per-row scales are replicated into every block: block 0 is exact
                dst[c*stride_col_dst + row] = x[0].s_lo*s.x + x[0].s_hi*s.y;
            }
        }
    }
}

template <int ncols_dst>
static __global__ void mul_mat_vec_fv5b_f32(
        const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
        const int64_t nblocks_per_row, const int64_t stride_col_y, const int64_t stride_col_dst) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const block_fv5b * x = (const block_fv5b *) vx + row*nblocks_per_row;

    float sums[ncols_dst] = {0.0f};

    const int64_t ngroups = nblocks_per_row * (QK_FV5/8); // groups of 8 int8

    for (int64_t idx = tid; idx < ngroups; idx += MMV_FV5_BLOCK_SIZE) {
        const int64_t ib = idx >> 5;
        const int     j  = (int)(idx & 31);

        const int8_t * q  = x[ib].qs + 8*j;
        const float  * yj = y + ib*QK_FV5 + 8*j;

#pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float w = (float) q[b];
#pragma unroll
            for (int c = 0; c < ncols_dst; ++c) {
                sums[c] += w * yj[c*stride_col_y + b];
            }
        }
    }

#pragma unroll
    for (int c = 0; c < ncols_dst; ++c) {
        sums[c] = warp_reduce_sum(sums[c]);
    }

    __shared__ float s_sums[ncols_dst][MMV_FV5_BLOCK_SIZE/WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0) {
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            s_sums[c][warp_id] = sums[c];
        }
    }
    __syncthreads();
    if (warp_id == 0) {
#pragma unroll
        for (int c = 0; c < ncols_dst; ++c) {
            float s = lane_id < MMV_FV5_BLOCK_SIZE/WARP_SIZE ? s_sums[c][lane_id] : 0.0f;
            s = warp_reduce_sum(s);
            if (lane_id == 0) {
                dst[c*stride_col_dst + row] = x[0].s * s;
            }
        }
    }
}

template <int ncols_dst>
static void launch_mul_mat_vec_fv5_cuda(
        const ggml_type type, const void * vx, const float * y, float * dst,
        const int64_t nrows, const int64_t nblocks_per_row,
        const int64_t stride_col_y, const int64_t stride_col_dst, cudaStream_t stream) {
    const dim3 block_nums(nrows, 1, 1);
    const dim3 block_dims(MMV_FV5_BLOCK_SIZE, 1, 1);

    if (type == GGML_TYPE_FV5) {
        mul_mat_vec_fv5_f32<ncols_dst><<<block_nums, block_dims, 0, stream>>>(
            vx, y, dst, nblocks_per_row, stride_col_y, stride_col_dst);
    } else {
        mul_mat_vec_fv5b_f32<ncols_dst><<<block_nums, block_dims, 0, stream>>>(
            vx, y, dst, nblocks_per_row, stride_col_y, stride_col_dst);
    }
}

bool ggml_cuda_can_mul_mat_vec_fv5(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    return (src0->type == GGML_TYPE_FV5 || src0->type == GGML_TYPE_FV5B) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] <= MMV_FV5_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1;
}

void ggml_cuda_mul_mat_vec_fv5(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_can_mul_mat_vec_fv5(src0, src1, dst));
    GGML_ASSERT(src0->ne[0] % QK_FV5 == 0);
    GGML_ASSERT(src0->ne[0] == src1->ne[0]);

    const int64_t nrows           = src0->ne[1];
    const int64_t nblocks_per_row = src0->ne[0] / QK_FV5;
    const int64_t ncols_dst       = src1->ne[1];
    const int64_t stride_col_y    = src1->nb[1] / sizeof(float);
    const int64_t stride_col_dst  = dst->nb[1]  / sizeof(float);

    const ggml_type type   = src0->type;
    const void *    vx     = src0->data;
    const float *   y      = (const float *) src1->data;
    float *         dst_d  = (float *) dst->data;

    cudaStream_t stream = ctx.stream();

    switch (ncols_dst) {
        case 1:
            launch_mul_mat_vec_fv5_cuda<1>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 2:
            launch_mul_mat_vec_fv5_cuda<2>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 3:
            launch_mul_mat_vec_fv5_cuda<3>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 4:
            launch_mul_mat_vec_fv5_cuda<4>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 5:
            launch_mul_mat_vec_fv5_cuda<5>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 6:
            launch_mul_mat_vec_fv5_cuda<6>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 7:
            launch_mul_mat_vec_fv5_cuda<7>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        case 8:
            launch_mul_mat_vec_fv5_cuda<8>(type, vx, y, dst_d, nrows, nblocks_per_row, stride_col_y, stride_col_dst, stream);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}
