#include "../../src/integration.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct error_ptr {
    char *value = nullptr;
    ~error_ptr() { libia_string_free(value); }
};

static void print_usage(const char *argv0) {
    std::cout
        << "Usage: " << argv0 << " [--model FILE] [--prompt TEXT]\n"
        << "\n"
        << "Defaults:\n"
        << "  --model  /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf\n"
        << "  --prompt \"O que é Rust language?\"\n";
}

static void stream_stdout(libia_generation_kind, const char *text, int32_t, int32_t, void *) {
    if (!text) {
        return;
    }
    std::fputs(text, stdout);
    std::fflush(stdout);
}

} // namespace

int main(int argc, char **argv) {
    std::freopen("/dev/null", "w", stderr);

    std::string model_path = "/home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf";
    std::string prompt = "O que é Rust language?";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i] ? argv[i] : "";
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
            continue;
        }
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
            continue;
        }
    }

    libia_params *params = libia_params_create();
    libia_params_set_model_path(params, model_path.c_str(), nullptr);
    libia_params_set_prompt(params, prompt.c_str(), nullptr);
    libia_params_set_system_prompt(params, "Voce e um assistente tecnico. Pense internamente, mas responda apenas com a resposta final em portugues, clara, objetiva e completa. Nao mostre raciocinio, tags internas ou texto de pensamento.", nullptr);
    libia_params_set_bool(params, "--log-disable", true, nullptr);
    libia_params_set_n_predict(params, 384, nullptr);
    libia_params_set_n_ctx(params, 4096, nullptr);
    libia_params_set_n_batch(params, 512, nullptr);
    libia_params_set_n_ubatch(params, 128, nullptr);
    libia_params_set_temp(params, 0.2, nullptr);
    libia_params_set_top_p(params, 0.95, nullptr);
    libia_params_set_top_k(params, 40, nullptr);

    libia_runtime *runtime = libia_runtime_create(params);
    error_ptr err;

    if (!libia_runtime_load(runtime, &err.value)) {
        std::cerr << "load failed: " << (err.value ? err.value : "unknown error") << "\n";
        libia_runtime_free(runtime);
        libia_params_free(params);
        return 1;
    }

    const libia_chat_message messages[] = {
        { "user", prompt.c_str(), nullptr },
    };

    char *output = libia_runtime_generate_chat_stream(runtime, messages, 1, 512, stream_stdout, nullptr, &err.value);
    if (!output) {
        std::cerr << "generation failed: " << (err.value ? err.value : "unknown error") << "\n";
        libia_runtime_free(runtime);
        libia_params_free(params);
        return 1;
    }

    std::cout << std::endl;
    libia_string_free(output);

    libia_runtime_free(runtime);
    libia_params_free(params);
    return 0;
}
