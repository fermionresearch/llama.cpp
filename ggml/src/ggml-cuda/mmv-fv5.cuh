#pragma once

#include "common.cuh"

#define MMV_FV5_MAX_BATCH_SIZE 8 // Max. batch size for which to use MMV_FV5 kernels.

// Fermion FV5/FV5B fused matrix-vector product against RAW F32 activations.
// Mirrors the CPU vec_dot policy (vec_dot_type == GGML_TYPE_F32): the
// activations are never quantized and all arithmetic is f32, so GPU numerics
// stay in the same class as the f32 container expansion — only the summation
// order differs. Used for decode and small batches (ne11 <=
// MMV_FV5_MAX_BATCH_SIZE, the speculative-verify window); larger batches go
// through the dequant + cuBLAS F32 path.

bool ggml_cuda_can_mul_mat_vec_fv5(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);

void ggml_cuda_mul_mat_vec_fv5(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
