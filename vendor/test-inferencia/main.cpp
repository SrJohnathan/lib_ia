#include "../../src/integration.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

namespace {
    struct error_ptr {
        char *value = nullptr;
        ~error_ptr() { libia_string_free(value); }
    };

    static void print_usage(const char *argv0) {
        std::cout
                << "Usage: " << argv0 << " [--model FILE] [--prompt TEXT] [--ngl N]\n"
                << "\n"
                << "Defaults:\n"
                << "  --model  /run/media/johnathan/Novo volume/Qwen3.8-27B-MTP-Q4_K_M.gguf\n"
                << "  --prompt \"O que e Rust language?\"\n"
                << "  --ngl    0 (CPU only)\n";
    }

    static void stream_stdout(libia_generation_kind, const char *text, int32_t, int32_t, void *) {
        if (!text || !*text) return;
        std::fputs(text, stdout);
        std::fflush(stdout);
    }
} // namespace

int main(int argc, char **argv) {
    // NÃO redirecionar stderr — precisamos ver os logs do llama.cpp
    // std::freopen("/dev/null", "w", stderr);

    std::string model_path = "/run/media/johnathan/Novo volume/Qwen3.8-27B-MTP-Q4_K_M.gguf";
    std::string prompt = "O que e Rust language?";
    int64_t ngl = 20; // começa seguro (CPU)

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
        if (arg == "--ngl" && i + 1 < argc) {
            ngl = std::stoll(argv[++i]);
            continue;
        }
    }

    std::cout << "Model : " << model_path << "\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "ngl   : " << ngl << "\n\n";

    libia_params *params = libia_params_create();
    if (!params) {
        std::cerr << "failed to create params\n";
        return 1;
    }

    error_ptr err;

    // Modelo e prompt
    if (!libia_params_set_model_path(params, model_path.c_str(), &err.value)) {
        std::cerr << "set_model_path failed: " << (err.value ? err.value : "?") << "\n";
        libia_params_free(params);
        return 1;
    }
    libia_params_set_prompt(params, prompt.c_str(), nullptr);
    libia_params_set_system_prompt(params,
                                   "Voce e um assistente tecnico. Responda em portugues, claro e objetivo.",
                                   nullptr);

    libia_params_set_int(params, "n-gpu-layers", ngl, nullptr);

    // --- DRAFT NA GPU (senão fica na CPU e mata o MTP) ---
    if (libia_params_has_option(params, "spec-draft-ngl")) {
        libia_params_set_int(params, "spec-draft-ngl", ngl, nullptr);
    } else if (libia_params_has_option(params, "n-gpu-layers-draft")) {
        libia_params_set_int(params, "n-gpu-layers-draft", ngl, nullptr);
    } else {
        // fallback pra versão antiga da libia
        libia_params_set_option(params, "--spec-draft-ngl", std::to_string(ngl).c_str(), nullptr);
    }

    if (libia_params_has_option(params, "spec-draft-device")) {
        libia_params_set_string(params, "spec-draft-device", "Vulkan0", nullptr);
    }
    // backend-sampling só suporta 1 output por sequência por batch; com spec-draft-n-max=2
    // (MTP gerando mais de 1 token de draft por vez) isso quebra o llama_decode. Deixa desligado.
    std::cerr << "[debug] has_option(spec-draft-backend-sampling) = "
              << libia_params_has_option(params, "spec-draft-backend-sampling") << "\n";
    if (libia_params_has_option(params, "spec-draft-backend-sampling")) {
        error_ptr bs_err;
        bool ok_bs = libia_params_set_bool(params, "spec-draft-backend-sampling", false, &bs_err.value);
        std::cerr << "[debug] set_bool(spec-draft-backend-sampling,false) ok=" << ok_bs
                  << " err=" << (bs_err.value ? bs_err.value : "-") << "\n";
        bool readback_bs = true;
        bool got_bs = libia_params_get_bool(params, "spec-draft-backend-sampling", &readback_bs);
        std::cerr << "[debug] readback backend-sampling: got=" << got_bs << " value=" << readback_bs << "\n";
    } else {
        std::cerr << "[debug] opcao spec-draft-backend-sampling nao existe segundo has_option\n";
    }

    // Tentativa alternativa via apply_argv (mesmo parser de CLI), caso set_bool nao
    // esteja de fato escrevendo no common_params (mesmo padrao do bug do spec-type).
    {
        const char *bs_argv[] = { "prog", "--no-spec-draft-backend-sampling" };
        error_ptr apply_bs_err;
        bool ok_apply_bs = libia_params_apply_argv(params, 2, bs_argv, &apply_bs_err.value);
        std::cerr << "[debug] apply_argv(--no-spec-draft-backend-sampling) ok=" << ok_apply_bs
                  << " err=" << (apply_bs_err.value ? apply_bs_err.value : "-") << "\n";
    }

    // --- perf igual ao server ---
    libia_params_set_bool(params, "flash-attn", true, nullptr);
    libia_params_set_bool(params, "log-disable", false, nullptr);
    libia_params_set_string(params, "cache-type-k", "q4_0", nullptr);
    libia_params_set_string(params, "cache-type-v", "q4_0", nullptr);

    // --- DEBUG: listar todas as opções que a lib marca como is_spec ---
    std::cerr << "[debug] opcoes is_spec registradas na libia:\n";
    {
        int n = libia_params_option_count();
        bool any_spec = false;
        for (int i = 0; i < n; ++i) {
            libia_option_info info{};
            if (libia_params_option_info(i, &info) && info.is_spec) {
                any_spec = true;
                std::cerr << "  - " << (info.name ? info.name : "?")
                          << " kind=" << (int)info.kind
                          << " hint=" << (info.value_hint ? info.value_hint : "")
                          << "\n";
            }
        }
        if (!any_spec) {
            std::cerr << "  (nenhuma opcao is_spec encontrada!)\n";
        }
    }

    // --- MTP ---
    std::cerr << "[debug] has_option(spec-type)        = " << libia_params_has_option(params, "spec-type") << "\n";
    std::cerr << "[debug] has_option(spec-draft-n-max) = " << libia_params_has_option(params, "spec-draft-n-max") << "\n";

    error_ptr spec_err, draft_err;
    bool ok1 = libia_params_set_string(params, "spec-type", "draft-mtp", &spec_err.value);
    bool ok2 = libia_params_set_int(params, "spec-draft-n-max", 2, &draft_err.value);
    std::cerr << "spec-type: " << ok1 << " (" << (spec_err.value ? spec_err.value : "-") << ")\n";
    std::cerr << "spec-draft-n-max: " << ok2 << " (" << (draft_err.value ? draft_err.value : "-") << ")\n";

    // --- DEBUG: reler o valor de volta pra confirmar que gravou de fato ---
    {
        const char *readback = nullptr;
        bool got = libia_params_get_string(params, "spec-type", &readback);
        std::cerr << "[debug] readback spec-type: got=" << got
                  << " value=" << (readback ? readback : "(null)") << "\n";
    }

    libia_params_set_n_ctx(params, 8192, nullptr);
    libia_params_set_n_batch(params, 512, nullptr);
    libia_params_set_n_ubatch(params, 256, nullptr); // se voltar o erro X<Y, troca pra 512

    libia_params_set_n_predict(params, 1536, nullptr);

    libia_params_set_temp(params, 0.8, nullptr);
    libia_params_set_top_p(params, 0.95, nullptr);
    libia_params_set_top_k(params, 20, nullptr);

    libia_params_set_int(params, "threads", 12, nullptr);
    libia_params_set_int(params, "threads-batch", 12, nullptr);

    // Runtime
    libia_runtime *runtime = libia_runtime_create(params);
    if (!runtime) {
        std::cerr << "failed to create runtime\n";
        libia_params_free(params);
        return 1;
    }

    std::cout << "Loading model...\n";
    if (!libia_runtime_load(runtime, &err.value)) {
        std::cerr << "LOAD FAILED: " << (err.value ? err.value : "unknown error") << "\n";
        libia_runtime_free(runtime);
        libia_params_free(params);
        return 1;
    }
    std::cout << "Model loaded OK\n\n";

    // Geração
    const libia_chat_message messages[] = {
        {"user", prompt.c_str(), nullptr},
    };

    std::cout << "Generating...\n";
    char *output = libia_runtime_generate_chat_stream(
        runtime, messages, 1, 1536, stream_stdout, nullptr, &err.value);

    if (!output) {
        std::cerr << "\nGENERATION FAILED: " << (err.value ? err.value : "unknown error") << "\n";
        libia_runtime_free(runtime);
        libia_params_free(params);
        return 1;
    }

    std::cout << "\n\n--- done ---\n";
    std::cout << "tokens generated : " << libia_runtime_last_generated_tokens(runtime) << "\n";
    std::cout << "tokens/s         : " << libia_runtime_last_tokens_per_second(runtime) << "\n";

    libia_string_free(output);
    libia_runtime_free(runtime);
    libia_params_free(params);
    return 0;
}