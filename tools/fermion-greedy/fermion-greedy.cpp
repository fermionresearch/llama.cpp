// fermion-greedy — deterministic greedy runner for the Fermion FV5 correctness gate.
//
// Reads prompts as EXACT token ids (no tokenizer in the loop), runs pure
// argmax greedy generation, and — when a reference stream is given — also a
// teacher-forced pass that feeds the reference token at every step while
// recording where our argmax disagrees and by what logit margin. Root-cause
// flips (teacher-forced mismatches) are the gate currency: free-run streams
// diverge wholesale after one flip, forced mode localizes every disagreement.
//
// The KV cache is kept in F32 so the only numerical difference vs the f32
// expansion reference is operation order, not storage precision.
//
// usage:
//   llama-fermion-greedy -m model.gguf --prompts prompts.txt [--ref ref.txt]
//                        [--steps 128] [-t nthreads] [--out receipt.json]
//
// prompts.txt: one prompt per line, comma-separated token ids
// ref.txt:     one line per prompt, comma-separated reference continuation ids
// output:      JSON receipt on stdout (or --out file)

#include "llama.h"

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static double now_s() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static std::vector<std::vector<llama_token>> read_id_lines(const std::string & path) {
    std::vector<std::vector<llama_token>> out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", path.c_str());
        exit(1);
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<llama_token> ids;
        std::stringstream ss(line);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) {
                ids.push_back((llama_token) strtol(tok.c_str(), nullptr, 10));
            }
        }
        if (!ids.empty()) {
            out.push_back(std::move(ids));
        }
    }
    return out;
}

struct step_info {
    llama_token top1;
    float       logit_top1;
    llama_token top2;
    float       logit_top2;
};

// argmax + runner-up over the last decoded token's logits
static step_info argmax2(const float * logits, int n_vocab) {
    step_info si = { 0, -1e30f, 0, -1e30f };
    for (int i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (v > si.logit_top1) {
            si.top2 = si.top1; si.logit_top2 = si.logit_top1;
            si.top1 = i;       si.logit_top1 = v;
        } else if (v > si.logit_top2) {
            si.top2 = i;       si.logit_top2 = v;
        }
    }
    return si;
}

int main(int argc, char ** argv) {
    std::string model_path;
    std::string prompts_path;
    std::string ref_path;
    std::string out_path;
    int steps     = 128;
    int n_threads = 8;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char * flag) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "error: %s needs a value\n", flag); exit(1); }
            return argv[++i];
        };
        if      (!strcmp(argv[i], "-m"))        { model_path   = need("-m"); }
        else if (!strcmp(argv[i], "--prompts")) { prompts_path = need("--prompts"); }
        else if (!strcmp(argv[i], "--ref"))     { ref_path     = need("--ref"); }
        else if (!strcmp(argv[i], "--out"))     { out_path     = need("--out"); }
        else if (!strcmp(argv[i], "--steps"))   { steps        = atoi(need("--steps")); }
        else if (!strcmp(argv[i], "-t"))        { n_threads    = atoi(need("-t")); }
        else { fprintf(stderr, "error: unknown arg %s\n", argv[i]); return 1; }
    }
    if (model_path.empty() || prompts_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf --prompts ids.txt [--ref ids.txt] [--steps N] [-t N] [--out f.json]\n", argv[0]);
        return 1;
    }

    const auto prompts = read_id_lines(prompts_path);
    std::vector<std::vector<llama_token>> refs;
    if (!ref_path.empty()) {
        refs = read_id_lines(ref_path);
        if (refs.size() != prompts.size()) {
            fprintf(stderr, "error: %zu prompts but %zu reference lines\n", prompts.size(), refs.size());
            return 1;
        }
    }

    size_t max_prompt = 0;
    for (const auto & p : prompts) {
        max_prompt = p.size() > max_prompt ? p.size() : max_prompt;
    }

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    const double t_load0 = now_s();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "error: failed to load %s\n", model_path.c_str());
        return 1;
    }
    const double t_load = now_s() - t_load0;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = (uint32_t) (max_prompt + steps + 8);
    cparams.n_batch         = (uint32_t) (max_prompt + 8);
    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.type_k          = GGML_TYPE_F32;   // keep KV in f32: gate compares against an f32 reference
    cparams.type_v          = GGML_TYPE_F32;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED; // conservative numerics for the gate
    cparams.no_perf         = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    FILE * out = stdout;
    if (!out_path.empty()) {
        out = fopen(out_path.c_str(), "w");
        if (!out) {
            fprintf(stderr, "error: cannot write %s\n", out_path.c_str());
            return 1;
        }
    }

    fprintf(out, "{\n  \"model\": \"%s\",\n  \"steps\": %d,\n  \"n_threads\": %d,\n  \"load_seconds\": %.2f,\n  \"kv_type\": \"f32\",\n  \"prompts\": [\n",
            model_path.c_str(), steps, n_threads, t_load);

    double total_gen_s    = 0.0;
    double total_prompt_s = 0.0;
    int64_t total_gen_tok    = 0;
    int64_t total_prompt_tok = 0;
    int     total_forced_mismatch = 0;
    bool    all_free_match = true;

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        std::vector<llama_token> prompt = prompts[pi];

        // ---------------- free-running greedy pass ----------------
        llama_memory_clear(llama_get_memory(ctx), true);

        double t0 = now_s();
        llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "error: prompt decode failed (prompt %zu)\n", pi);
            return 1;
        }
        const double prompt_s = now_s() - t0;

        std::vector<llama_token> free_stream;
        float min_margin_free = 1e30f;
        t0 = now_s();
        for (int t = 0; t < steps; ++t) {
            const float * logits = llama_get_logits_ith(ctx, -1);
            step_info si = argmax2(logits, n_vocab);
            const float margin = si.logit_top1 - si.logit_top2;
            if (margin < min_margin_free) {
                min_margin_free = margin;
            }
            free_stream.push_back(si.top1);
            if (t + 1 < steps) {
                llama_token next = si.top1;
                batch = llama_batch_get_one(&next, 1);
                if (llama_decode(ctx, batch) != 0) {
                    fprintf(stderr, "error: decode failed (prompt %zu step %d)\n", pi, t);
                    return 1;
                }
            }
        }
        const double gen_s = now_s() - t0;

        total_prompt_s   += prompt_s;
        total_gen_s      += gen_s;
        total_prompt_tok += (int64_t) prompt.size();
        total_gen_tok    += steps;

        fprintf(out, "    {\n      \"prompt_ids\": [");
        for (size_t i = 0; i < prompt.size(); ++i) {
            fprintf(out, "%s%d", i ? "," : "", prompt[i]);
        }
        fprintf(out, "],\n      \"free_ids\": [");
        for (size_t i = 0; i < free_stream.size(); ++i) {
            fprintf(out, "%s%d", i ? "," : "", free_stream[i]);
        }
        fprintf(out, "],\n      \"free_min_top2_margin\": %.6f,\n", min_margin_free);
        fprintf(out, "      \"prompt_seconds\": %.3f,\n      \"gen_seconds\": %.3f", prompt_s, gen_s);

        if (!refs.empty()) {
            const std::vector<llama_token> & ref = refs[pi];
            const int n_ref = (int) ref.size() < steps ? (int) ref.size() : steps;

            // free-run vs reference (first divergence)
            int first_div = -1;
            for (int t = 0; t < n_ref; ++t) {
                if (free_stream[t] != ref[t]) { first_div = t; break; }
            }
            const bool free_match = first_div < 0;
            all_free_match = all_free_match && free_match;

            // ---------------- teacher-forced pass ----------------
            llama_memory_clear(llama_get_memory(ctx), true);
            batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "error: forced prompt decode failed (prompt %zu)\n", pi);
                return 1;
            }

            int n_mismatch = 0;
            std::string mm_json;
            char buf[256];
            for (int t = 0; t < n_ref; ++t) {
                const float * logits = llama_get_logits_ith(ctx, -1);
                step_info si = argmax2(logits, n_vocab);
                if (si.top1 != ref[t]) {
                    const float logit_ref = logits[ref[t]];
                    snprintf(buf, sizeof(buf),
                             "%s\n        {\"step\": %d, \"got\": %d, \"want\": %d, \"logit_got\": %.6f, \"logit_want\": %.6f, \"margin\": %.6f}",
                             n_mismatch ? "," : "", t, si.top1, ref[t], si.logit_top1, logit_ref, si.logit_top1 - logit_ref);
                    mm_json += buf;
                    n_mismatch++;
                }
                llama_token next = ref[t];   // teacher-force the reference stream
                if (t + 1 < n_ref) {
                    batch = llama_batch_get_one(&next, 1);
                    if (llama_decode(ctx, batch) != 0) {
                        fprintf(stderr, "error: forced decode failed (prompt %zu step %d)\n", pi, t);
                        return 1;
                    }
                }
            }
            total_forced_mismatch += n_mismatch;

            fprintf(out, ",\n      \"free_match\": %s,\n      \"first_divergence\": %d,\n      \"forced_mismatch_count\": %d,\n      \"forced_mismatches\": [%s%s]",
                    free_match ? "true" : "false", first_div, n_mismatch,
                    mm_json.c_str(), n_mismatch ? "\n      " : "");
        }

        fprintf(out, "\n    }%s\n", pi + 1 < prompts.size() ? "," : "");
        fflush(out);
        fprintf(stderr, "prompt %zu/%zu done (%.1f tok/s gen)\n", pi + 1, prompts.size(), steps / gen_s);
    }

    fprintf(out, "  ],\n  \"totals\": {\n");
    fprintf(out, "    \"prompt_tokens\": %" PRId64 ",\n    \"gen_tokens\": %" PRId64 ",\n", total_prompt_tok, total_gen_tok);
    fprintf(out, "    \"prompt_tok_per_s\": %.2f,\n    \"gen_tok_per_s\": %.2f,\n",
            total_prompt_tok / (total_prompt_s > 0 ? total_prompt_s : 1e-9),
            total_gen_tok    / (total_gen_s    > 0 ? total_gen_s    : 1e-9));
    fprintf(out, "    \"forced_mismatch_total\": %d,\n    \"all_free_match\": %s\n  }\n}\n",
            total_forced_mismatch, (refs.empty() || !all_free_match) ? (refs.empty() ? "null" : "false") : "true");

    if (out != stdout) {
        fclose(out);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
