# llama.cpp — Fermion Research fork (`fermion-fv5`)

This is [Fermion Research](https://fermionresearch.com)'s fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). It adds two GGML
weight types so llama.cpp can **load and run Neutrino GGUF packs** — Fermion's
ternary-family models — on the **CPU, CUDA and Metal (Apple GPU)** backends:

| ggml type | id | block | bytes/block | bpw | used for |
|---|---|---|---|---|---|
| `FV5`  | 43 | 256 | 104 (f32 `s_lo`, f32 `s_hi`, 3×32 B bit-planes) | 3.25 | all attention / MLP linears |
| `FV5B` | 44 | 256 | 260 (f32 `s`, 256×int8) | 8.125 | token_embd / output |

Everything else is stock llama.cpp at the pinned base commit — the model graph
is untouched (Neutrino models are stock Qwen3 geometry, which mainline already
implements). The upstream README is preserved as
[README_LLAMACPP.md](README_LLAMACPP.md).

> GGUF is the *compatibility* door. Fermion's native runtime and its fused
> kernels remain the *fast* door; this fork optimizes for correctness parity
> with the deployed container function, not for peak speed.

## Quickstart

You do **not** convert anything. Download a ready-made Neutrino GGUF pack
(`neutrino-8b-fv5.gguf`, ~4.1 GB, or `neutrino-0.6b-base-fv5.gguf`),
then build this fork and point it at the file.

```sh
# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-completion llama-bench llama-fermion-greedy

# CPU + CUDA (full offload supported)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
      -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DCMAKE_CUDA_ARCHITECTURES=75\;89
cmake --build build -j --target llama-completion llama-bench llama-fermion-greedy

# macOS / Apple silicon (Metal is ON by default; full offload supported)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-completion llama-bench llama-fermion-greedy llama-server
```

On Apple silicon you no longer need `-DGGML_METAL=OFF`: the fork has native
Metal kernels for FV5/FV5B, so the default Metal build offloads every layer
with `-ngl 99`.

`-DGGML_CUDA_NO_VMM=ON` is **required** if you build on a machine with the
CUDA toolkit but no driver present (a GPU-less CI/builder box): without it the
link step fails on `cuGetErrorString` from the driver API. On a machine with a
real driver you may drop it. If you are building a portable binary to run on a
different host, also pass `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON
-DGGML_F16C=ON -DGGML_BMI2=ON` — a `-march=native` build has been observed to
SIGILL on an older target CPU.

A quick greedy completion / benchmark:

```sh
./build/bin/llama-completion -m neutrino-8b-fv5.gguf -ngl 99 -p "The capital of France is" -n 64 --temp 0 -no-cnv
./build/bin/llama-bench -m neutrino-8b-fv5.gguf -ngl 99 -p 512 -n 128
```

> **`llama-server` status, stated exactly.** Verified on macOS/Metal with the
> 0.6B pack: one OpenAI-compatible chat round-trip, and prompt caching
> confirmed engaging across two requests with a shared prefix (`cache_n`
> reported by the server, prompt work dropped accordingly). The server has
> not been exercised on the CUDA or CPU-only builds — nothing in the FV5
> path is server-specific, but we only claim what we ran.

**Backends:** CPU, CUDA and Metal. There is **no Vulkan FV5 kernel** yet.
On Apple silicon the Metal path is the supported configuration — the CPU
FV5 kernel is scalar on arm64 (the vectorized path is AVX2-only), so a
CPU-only macOS build is functional but slow by comparison; it remains the
reference for correctness gates.

Stock upstream llama.cpp will **reject** these packs — at the pinned base and
on current upstream master, `GGML_TYPE_COUNT` is 43, so *both* FV5 (43) and
FV5B (44) are out of range there. You need this fork until the types are
upstreamed.

## Correctness

The fork ships a gate tool, `tools/fermion-greedy` (`llama-fermion-greedy`),
that certifies greedy **token identity** (free-running and teacher-forced,
with per-flip logit-margin forensics, F32 KV cache).

Results, stated exactly as measured. Receipts are in the lane's
`receipts/` directory; the raw JSON verdicts are the authority, not this
table.

**CPU backend**, vs the fp32 expansion of the source container:

| model | free-running | teacher-forced |
|---|---|---|
| Neutrino-0.6B-base | 8/8 prompts × 128 steps identical | 0/1024 mismatches |
| Neutrino-8B | 8/8 prompts × 128 steps identical | not run in this lane |

The 8B's 0/1024 teacher-forced CPU figure comes from an earlier lane's
**bring-up** cut, not the shipped product cut. For the shipped cut we have
free-running continuity only, and we are not going to present the older
number as if it covered it.

**CUDA backend (`-ngl 99`)**, vs this fork's certified CPU path, 8 prompts ×
256 greedy steps per model, `NVIDIA_TF32_OVERRIDE=0`:

| model | free-running | teacher-forced | verdict |
|---|---|---|---|
| Neutrino-0.6B-base | **8/8 identical** | 0/2048 | PASS |
| Neutrino-8B | **7/8 identical** | 1/2048 | PASS with one documented near-tie |

**The 8B is not a clean sweep, and here is the one flip in full.** On one of
the eight prompts, at step 0, CUDA picks token 323 and CPU picks token 498.
The logits are `15.631217` vs `15.630566` — a margin of **6.51e-4**, i.e. a
relative gap of ~4e-5. That is an fp32 summation-order tie, below our
preregistered 1e-3 near-tie bar, and because greedy decoding is
path-dependent that single step 0 disagreement makes the whole free-running
trajectory for that prompt diverge. The teacher-forced pass localizes it:
**2047 of 2048 forced steps agree.** Zero unexplained flips, on either model.

The receipt JSON for that run carries a mechanical `"verdict": "FAIL"`
(`all_free_match: false`). We are not hiding that: the script only stamped
PASS when *all* free-runs matched, and this run has one near-tie, so under
the house policy — flips permitted only as documented near-ties within 1e-3,
zero unexplained — the classification is PASS_WITH_NEAR_TIES. No bar was
moved after the fact.

**Calibration, so the near-tie is read at the right size.** This flip class
is not specific to our kernels. On the reference side of the house, greedy
`torch` **disagrees with itself across dtypes at the same rate it disagrees
with our runtime** — 1 prompt in 6 either way, on near-ties of the same
order. A single 6.5e-4 argmax tie is the noise floor of fp32 reduction
ordering, not evidence of a defect. What would be evidence of a defect is an
unexplained flip at a wide margin, and there are none.

**Metal backend (`-ngl 99`, Apple M5, 16 GB)**, vs this fork's certified CPU
path — same binary and prompts as the CUDA gate, 128 greedy steps:

| model | free-running | teacher-forced | verdict |
|---|---|---|---|
| Neutrino-0.6B-base | **8/8 identical** | 0/1024 | PASS |
| Neutrino-0.6B-base, long prompts (136/256 tok, exercises the mat-mat prefill path) | **2/2 identical** | 0/256 | PASS |
| Neutrino-8B (vs the banked fork-CPU streams of the shipping cut, sha `1c13a343…`) | **8/8 identical** | 0/1024 | PASS |

The Metal 0.6B streams are also token-identical to the **banked x86 CPU
streams from the CUDA lane's gate** (8/8 × 128) — two machines, three
backends, one trajectory. Unit-level: `test-backend-ops` passes
MUL_MAT / MUL_MAT_ID / GET_ROWS for `fv5`/`fv5b` against the CPU backend at
llama.cpp's standard tolerance (nmse ≤ 5e-4), including every GEMV shape in
the 0.6B and 8B packs (reduction lengths 1024/2048/3072/4096/12288, batch
1 through 512, and the full 151936-row lm_head).

Measured on the M5 (llama-bench, `-p 512 -n 128`):

| model | backend | pp512 t/s | tg128 t/s |
|---|---|---:|---:|
| Neutrino-0.6B-base | CPU (arm64 scalar) | 14.3 | 5.6 |
| Neutrino-0.6B-base | **Metal** | **6979** (×487) | **157** (×28) |
| Neutrino-8B | CPU (arm64 scalar) | 0.56 | 0.39 |
| Neutrino-8B | **Metal** | **471.8** (×842) | **16.6** (×42) |

Prefill is the headline: the arm64 CPU FV5 kernel is scalar, so prompt
processing was the pain point in every llama.cpp-shaped tool on a Mac.

Numerics policy: activations are consumed **raw in f32** on all backends
(`vec_dot_type = F32` on CPU; fused f32 GEMV + true-f32 cuBLAS with TF32
disabled on CUDA; f32 GEMV + f32 accumulation on Metal), and all scales are
exact f32 copies from the container. Decode (single token) is f32 end to
end on every backend. Batched prefill on Metal goes through llama.cpp's
standard mat-mat kernels, which stage tiles in f16 with f32 accumulation —
the same numerics class every other llama.cpp quant type gets on Metal, and
the greedy gates above measure its end-to-end effect (none observed).
See [FORK_NOTES.md](FORK_NOTES.md) for the full design.

## Branch / base

- Branch: `fermion-fv5`
- Upstream base: `d67c0b4107112e4790774c3a8169e2e3eb24643b` (master, 2026-07-25)
- Patches on top: CPU types + gate tool, then CUDA support, then Metal support.

## Upstream rebase / renumber plan

`FV5 = 43` / `FV5B = 44` were appended **at** `GGML_TYPE_COUNT` of the pinned
base, taking the next two free slots and moving `GGML_TYPE_COUNT` to 45.

**Nothing collides today.** At the pinned base `d67c0b41`, and on upstream
`origin/master` as of 2026-07-27, the highest allocated type is still
`GGML_TYPE_Q2_0 = 42` with `GGML_TYPE_COUNT = 43`. Ids 43 and 44 are unused
upstream.

**The hazard is what happens next, and it is silent.** FV5 and FV5B sit in
exactly the two slots upstream will allocate from. The moment upstream adds
one new type it takes id 43 — and since a GGUF pack stores the numeric type
id in every tensor header, every existing FV5 pack would then be reinterpreted
as that new upstream type by any build that has both. There is no magic, no
version field and no checksum in the tensor header to catch it: the pack
loads, the tensors are decoded with the wrong layout, and the output is
garbage rather than an error. **This is why packs must be re-emitted, not
just re-tagged, on any rebase that crosses a new upstream type allocation.**

Plan of record:

1. Stay on the pinned base `d67c0b41` (this branch) for current releases —
   packs and binaries from this fork are self-consistent, and the ids are
   unambiguous as long as you use this fork's own builds.
2. On rebase: check whether upstream has allocated 43/44. If it has, renumber
   FV5/FV5B to the then-current `GGML_TYPE_COUNT` and bump the ids in
   `ggml/include/ggml.h` (the enum), the `ggml-common.h` block-size asserts,
   `gguf-py/gguf/constants.py` (`GGMLQuantizationType` ids at ~4598-4599, the
   block-shape map at ~4783-4784, and `LlamaFileType.MOSTLY_FV5` at ~4656),
   and `LLAMA_FTYPE_MOSTLY_FV5`; then **re-emit every GGUF pack** and re-run
   both correctness gates before shipping anything.
3. For an upstream PR, propose the types at the head of master's type space.
   We would submit without a float→FV5 reference quantizer, because FV5 blocks
   are produced offline from Fermion containers and a naive float→FV5 rounding
   path would not reproduce them. Note this differs from how TQ1_0/TQ2_0
   landed — both of those *did* ship reference quantizers — so it is a point
   to negotiate with maintainers, not a precedent to lean on. CPU and CUDA
   paths in separate commits for review.

## License

MIT, same as upstream — see [LICENSE](LICENSE). This fork retains upstream's
copyright and attribution: llama.cpp is by The ggml authors; the FV5/FV5B
additions are Copyright (c) 2026 Fermion Research, released under the same
MIT terms.
