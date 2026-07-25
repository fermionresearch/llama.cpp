# Fermion `fermion-fv5` fork notes

Branch `fermion-fv5` on top of upstream `ggml-org/llama.cpp` commit
`d67c0b4107112e4790774c3a8169e2e3eb24643b` (2026-07-25). This fork adds two
ggml weight types so llama.cpp can LOAD AND RUN Fermion Research's TRTC v4
five-value ternary containers (Neutrino-0.6B, Neutrino-8B) converted to GGUF.
CPU backend only in this cut. The graph is untouched: both models are stock
Qwen3 geometry (`qwen3` arch, per-head Q/K RMSNorm, biasless qkv), which
mainline already implements.

GGUF is the *compatibility* door. The Fermion native runtime and its fused
kernels remain the *fast* door; this fork optimizes for correctness parity
with the container's deployed function, not for speed.

## New types

| ggml type | id | block | bytes/block | bpw | used for |
|---|---|---|---|---|---|
| `GGML_TYPE_FV5`  | 43 | 256 | 104 (f32 `s_lo`, f32 `s_hi`, `bp[32]`, `bn[32]`, `br[32]`) | 3.25 | all attention/MLP linears |
| `GGML_TYPE_FV5B` | 44 | 256 | 260 (f32 `s`, `int8 qs[256]`) | 8.125 | token_embd / output (TRTC int8 records) |

Reconstruction semantics (identical to the container spec in
`scripts/expand_trtc_v4_to_hf.py` of the research repo):

- FV5:  `w[j] = (bp[j] - bn[j]) * (br[j] ? s_hi : s_lo)` — five values per row
  `{0, ±s_lo, ±s_hi}`; the per-row dual scales are stored as exact f32 copies
  replicated into each 256-block; bit-planes keep the container's little bit
  order, so conversion is a plane slice, not a re-encode.
- FV5B: `w[j] = s * qs[j]` with the exact f32 per-row scale.

This is NOT BitNet TQ1_0/TQ2_0 (ternary, single fp16 block scale): five
values + dual per-row scales do not map onto those types.

## Numerics policy

`vec_dot_type = GGML_TYPE_F32` for both types: activations are consumed RAW
(no runtime Q8 quantization), and scales are f32, so every weight value seen
by the CPU backend is bit-identical to the f32 expansion of the container.
The only difference vs an f32 reference forward is summation order. The
`llama-fermion-greedy` tool (below) exists to certify exactly that: greedy
token streams vs the reference cache, with teacher-forced margin forensics
for any argmax flip. KV cache is set to F32 in the gate tool.

## Changed files

- `ggml/include/ggml.h` — enum entries FV5=43, FV5B=44 (appended; no renumbering).
- `ggml/src/ggml-common.h` — `block_fv5`, `block_fv5b` + `QK_FV5`.
- `ggml/src/ggml.c` — type traits (`to_float`; no `from_float_ref`: blocks are
  produced offline by the TRTC v4 → GGUF converter, `llama-quantize` is
  intentionally NOT wired).
- `ggml/src/ggml-quants.{c,h}` — reference `dequantize_row_fv5{,b}` +
  `ggml_validate_row_data` cases (validates scale finiteness + the fv5
  invariants `bp&bn==0`, `br⊆bp|bn`).
- `ggml/src/ggml-cpu/quants.{c,h}` — `ggml_vec_dot_fv5_f32`,
  `ggml_vec_dot_fv5b_f32`: portable scalar with an AVX2 fast path inside
  (masked activation sums mirroring the research repo's NEON decode
  structure); no `arch/*/quants.c` or `arch-fallback.h` churn.
- `ggml/src/ggml-cpu/ggml-cpu.c` — CPU traits (`vec_dot_type = F32`).
- `ggml/src/ggml-cpu/ops.cpp` — `get_rows` cases (embedding lookup for FV5B).
- `include/llama.h`, `src/llama-model-loader.cpp` — `LLAMA_FTYPE_MOSTLY_FV5`
  (=42) name/mapping for clean loader printouts.
- `gguf-py/gguf/constants.py`, `gguf-py/gguf/quants.py` — python-side type
  registration + numpy dequantize (used by the converter's bitwise
  cross-check); float→FV5 quantize deliberately raises.
- `tools/fermion-greedy/` — the correctness-gate tool (exact token-id
  prompts, free + teacher-forced greedy, per-flip logit margins, F32 KV).

## Deliberate non-goals of this cut

- No `llama-quantize` path (conversion is offline from TRTC v4 containers).
- No CUDA/Metal/Vulkan kernels: those backends report the types unsupported
  and fall back to CPU. (Native GPU serving lives in the Fermion runtime.)
- No repack/IMatrix/LoRA integration; training-path ops (`add`, `acc`,
  `out_prod`) on FV5 tensors abort as unsupported — inference never hits them.

## Build

    cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
    cmake --build build -j --target llama-cli llama-bench llama-fermion-greedy

## Gate usage

    llama-fermion-greedy -m neutrino-0p6b.fv5.gguf \
        --prompts prompts_ids.txt --ref ref_ids.txt --steps 128 -t 16 \
        --out gate_receipt.json

`forced_mismatch_total == 0` and `all_free_match == true` is the shipping
gate; any nonzero flip must be margin-classified (near-tie class) in the
release receipts.
