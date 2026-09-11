#include "integration.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

namespace {

void free_error(char *&error) {
    if (error) {
        libia_string_free(error);
        error = nullptr;
    }
}

void print_usage(const char *bin) {
    std::cerr
        << "Uso:\n"
        << "  " << bin << " [diffusion.gguf] [vae.sft|vae.safetensors] [llm.gguf] [prompt] [output.png]\n\n"
        << "Sem argumentos, usa os caminhos locais de teste ja configurados no arquivo.\n\n"
        << "Exemplo Z-Image:\n"
        << "  " << bin
        << " /models/z-image-turbo-Q6_K.gguf /models/ae.sft /models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        << " \"a tiny robot reading a book\" /tmp/z-image-test.png\n";
}

bool wait_loaded(libia_image *image, bool expected, const char *label) {
    for (int attempt = 1; attempt <= 12; ++attempt) {
        std::cout << label << " tentativa " << attempt << "/12" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(5));
        if (libia_image_is_loaded(image) == expected) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char **argv) {
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return 0;
    }

    const std::string diffusion_model = argc > 1
        ? argv[1]
        : "/run/media/johnathan/Novo volume/z-image-turbo-Q4_K_M.gguf";
    const std::string vae = argc > 2
        ? argv[2]
        : "/run/media/johnathan/Novo volume/ae.sft";
    const std::string llm = argc > 3
        ? argv[3]
        : "/run/media/johnathan/Novo volume/Qwen3-4B-Instruct-2507-Q4_K_M.gguf";
    const std::string prompt = argc > 4
        ? argv[4]
        : "a small friendly robot holding a sign that says GGUF++";
    const std::string output = argc > 5
        ? argv[5]
        : "/tmp/libia-z-image-test.png";

    if (!std::filesystem::exists(diffusion_model)) {
        std::cerr << "diffusion model nao existe: " << diffusion_model << std::endl;
        return 2;
    }
    if (!std::filesystem::exists(vae)) {
        std::cerr << "vae nao existe: " << vae << std::endl;
        return 2;
    }
    if (!std::filesystem::exists(llm)) {
        std::cerr << "llm/text encoder nao existe: " << llm << std::endl;
        return 2;
    }

    libia_image_params params{};
    libia_image_params_init(&params);
    params.model_path = nullptr;
    params.diffusion_model_path = diffusion_model.c_str();
    params.vae_path = vae.c_str();
    params.llm_path = llm.c_str();
    params.backend = "Vulkan";
    params.params_backend = nullptr;
    params.max_vram = nullptr;
    params.use_mmap = true;
    params.flash_attn = true;
    params.diffusion_flash_attn = true;

    libia_image *image = libia_image_create(&params);
    if (!image) {
        std::cerr << "falha ao criar handle libia_image" << std::endl;
        return 1;
    }

    char *error = nullptr;

    std::cout << "[load] diffusion: " << diffusion_model << std::endl;
    std::cout << "[load] vae:       " << vae << std::endl;
    std::cout << "[load] llm:       " << llm << std::endl;
    if (!libia_image_load(image, &error)) {
        std::cerr << "load failed: " << (error ? error : "erro desconhecido") << std::endl;
        free_error(error);
        libia_image_free(image);
        return 1;
    }

    if (!wait_loaded(image, true, "[load] aguardando confirmar carregamento")) {
        std::cerr << "timeout esperando modelo ficar carregado" << std::endl;
        libia_image_unload(image);
        libia_image_free(image);
        return 1;
    }

    libia_image_generate_params gen{};
    libia_image_generate_params_init(&gen);
    gen.prompt = prompt.c_str();
    gen.negative_prompt = "";
    gen.output_path = output.c_str();
    gen.width = 512;
    gen.height = 512;
    gen.sample_steps = 8;
    gen.seed = -1;
    gen.cfg_scale = 1.0f;
    gen.guidance = 3.5f;
    gen.sample_method = LIBIA_IMAGE_SAMPLE_EULER;
    gen.scheduler = LIBIA_IMAGE_SCHEDULER_DISCRETE;

    libia_image_result result{};
    std::cout << "[generate] prompt: " << prompt << std::endl;
    if (!libia_image_generate(image, &gen, &result, &error)) {
        std::cerr << "generate failed: " << (error ? error : "erro desconhecido") << std::endl;
        free_error(error);
        libia_image_unload(image);
        libia_image_free(image);
        return 1;
    }

    std::cout << "[ok] imagem salva em: " << (result.path ? result.path : output.c_str()) << std::endl;
    std::cout << "[ok] size: " << result.width << "x" << result.height
              << " channels=" << result.channels
              << " seed=" << result.seed << std::endl;
    libia_image_result_free(&result);

    std::cout << "[unload] descarregando modelo de imagem" << std::endl;
    libia_image_unload(image);
    if (!wait_loaded(image, false, "[unload] aguardando confirmar descarregamento")) {
        std::cerr << "timeout esperando modelo descarregar" << std::endl;
        libia_image_free(image);
        return 1;
    }

    libia_image_free(image);
    std::cout << "[done]" << std::endl;
    return 0;
}
