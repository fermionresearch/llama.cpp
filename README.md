# llama.cpp — Fermion Research fork (`fermion-fv5`)

This is [Fermion Research](https://fermionresearch.com)'s fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). It adds two GGML
weight types so llama.cpp can **load and run Neutrino GGUF packs** — Fermion's
ternary-family models — on the **CPU and CUDA** backends:

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
(e.g. `neutrino-8b-fv5.gguf`, ~4.1 GB, or the 0.6B cut), verify its
`SHA256SUMS`, then build this fork and point it at the file.

```sh
# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server llama-bench llama-completion

# CPU + CUDA (full offload supported)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CUDA=ON
cmake --build build -j --target llama-server llama-bench llama-completion
```

Run an OpenAI-compatible server, fully offloaded:

```sh
./build/bin/llama-server -m neutrino-8b-fv5.gguf -ngl 99 -c 4096 --host 127.0.0.1 --port 8080
```

Or a quick greedy completion / benchmark:

```sh
./build/bin/llama-completion -m neutrino-8b-fv5.gguf -ngl 99 -p "The capital of France is" -n 64 --temp 0 -no-cnv
./build/bin/llama-bench -m neutrino-8b-fv5.gguf -ngl 99 -p 512 -n 128
```

Stock upstream llama.cpp will **reject** these packs (`FV5B` = type id 44 is
out of range there) — you need this fork until the types are upstreamed.

## Correctness

The fork ships a gate tool, `tools/fermion-greedy` (`llama-fermion-greedy`),
that certifies greedy **token identity** (free-running and teacher-forced,
with per-flip logit-margin forensics, F32 KV cache):

- **CPU backend:** token-identical to the fp32 expansion of the source
  container — 8 prompts × 128 greedy steps, free-run 8/8 identical and 0/1024
  teacher-forced mismatches, for both Neutrino-0.6B-base and Neutrino-8B.
- **CUDA backend (`-ngl 99`):** token-identical to this fork's certified CPU
  path — 8 prompts × 256 greedy steps per model (free-run identical + 0
  teacher-forced mismatches; gate run with `NVIDIA_TF32_OVERRIDE=0`).

Numerics policy: activations are consumed **raw in f32** on both backends
(`vec_dot_type = F32` on CPU; fused f32 GEMV + true-f32 cuBLAS with TF32
disabled on CUDA), and all scales are exact f32 copies from the container —
the only difference vs an f32 reference forward is summation order.
See [FORK_NOTES.md](FORK_NOTES.md) for the full design.

## Branch / base

- Branch: `fermion-fv5`
- Upstream base: `d67c0b4107112e4790774c3a8169e2e3eb24643b` (master, 2026-07-25)
- Patches on top: CPU types + gate tool, then CUDA support.

## Upstream rebase / renumber plan

`FV5 = 43` / `FV5B = 44` were appended past `GGML_TYPE_COUNT` of the pinned
base. Upstream's type space has **since grown to id 43**, so these ids now
collide with newer upstream types. Plan of record:

1. Stay on the pinned base `d67c0b41` (this branch) for current releases —
   packs and binaries from this fork are self-consistent.
2. On rebase: renumber FV5/FV5B to the then-current `GGML_TYPE_COUNT` and
   bump ids in `ggml.h`, `ggml-common.h` asserts, `gguf-py/constants.py`, and
   `LLAMA_FTYPE_MOSTLY_FV5`; re-emit GGUF packs (the pack encodes the type id
   in every tensor header) and re-run both correctness gates before shipping.
3. For an upstream PR, propose the types at the head of master's type space
   with a float→FV5 reference quantizer deliberately omitted (blocks are
   produced offline from Fermion containers), mirroring how TQ1_0/TQ2_0
   landed; CPU + CUDA paths in separate commits for review.

## License

MIT, same as upstream — see [LICENSE](LICENSE). This fork retains upstream's
copyright and attribution: llama.cpp is by The ggml authors; the FV5/FV5B
additions are Copyright (c) 2026 Fermion Research, released under the same
MIT terms.
