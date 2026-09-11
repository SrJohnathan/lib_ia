#include "integration.h"

#include <stable-diffusion.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace {

struct libia_image_impl {
    libia_image_params params{};
    std::string model_path;
    std::string diffusion_model_path;
    std::string vae_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string llm_path;
    std::string backend;
    std::string params_backend;
    std::string max_vram;
    std::string last_error;
    sd_ctx_t *ctx = nullptr;
};

static std::mutex g_sd_log_mutex;
static std::string g_sd_last_error;

static libia_image_impl *to_impl(libia_image *image) {
    return reinterpret_cast<libia_image_impl *>(image);
}

static const libia_image_impl *to_impl(const libia_image *image) {
    return reinterpret_cast<const libia_image_impl *>(image);
}

static char *dup_string(const std::string &text) {
    char *out = static_cast<char *>(std::malloc(text.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

static const char *nullable_c_str(const std::string &text) {
    return text.empty() ? nullptr : text.c_str();
}

static std::string safe_string(const char *text) {
    return text ? std::string(text) : std::string();
}

static void set_error(libia_image_impl *impl, char **error_out, const std::string &message) {
    if (impl) {
        impl->last_error = message;
    }
    if (!error_out) {
        return;
    }
    libia_string_free(*error_out);
    *error_out = dup_string(message);
}

static const char *sd_log_level_name(sd_log_level_t level) {
    switch (level) {
        case SD_LOG_DEBUG: return "debug";
        case SD_LOG_INFO: return "info";
        case SD_LOG_WARN: return "warn";
        case SD_LOG_ERROR: return "error";
        default: return "log";
    }
}

static void libia_sd_log_callback(sd_log_level_t level, const char *text, void *) {
    const char *message = text ? text : "";
    {
        std::lock_guard<std::mutex> lock(g_sd_log_mutex);
        if (level == SD_LOG_ERROR || level == SD_LOG_WARN) {
            g_sd_last_error = message;
        }
    }
    std::fprintf(stderr, "[stable-diffusion:%s] %s\n", sd_log_level_name(level), message);
}

static void copy_params(libia_image_impl *impl, const libia_image_params *params) {
    impl->model_path = safe_string(params ? params->model_path : nullptr);
    impl->diffusion_model_path = safe_string(params ? params->diffusion_model_path : nullptr);
    impl->vae_path = safe_string(params ? params->vae_path : nullptr);
    impl->clip_l_path = safe_string(params ? params->clip_l_path : nullptr);
    impl->clip_g_path = safe_string(params ? params->clip_g_path : nullptr);
    impl->t5xxl_path = safe_string(params ? params->t5xxl_path : nullptr);
    impl->llm_path = safe_string(params ? params->llm_path : nullptr);
    impl->backend = safe_string(params ? params->backend : nullptr);
    impl->params_backend = safe_string(params ? params->params_backend : nullptr);
    impl->max_vram = safe_string(params ? params->max_vram : nullptr);

    libia_image_params_init(&impl->params);
    if (params) {
        impl->params = *params;
    }
    impl->params.model_path = nullable_c_str(impl->model_path);
    impl->params.diffusion_model_path = nullable_c_str(impl->diffusion_model_path);
    impl->params.vae_path = nullable_c_str(impl->vae_path);
    impl->params.clip_l_path = nullable_c_str(impl->clip_l_path);
    impl->params.clip_g_path = nullable_c_str(impl->clip_g_path);
    impl->params.t5xxl_path = nullable_c_str(impl->t5xxl_path);
    impl->params.llm_path = nullable_c_str(impl->llm_path);
    impl->params.backend = nullable_c_str(impl->backend);
    impl->params.params_backend = nullable_c_str(impl->params_backend);
    impl->params.max_vram = nullable_c_str(impl->max_vram);
}

static sample_method_t to_sd_sample_method(libia_image_sample_method method) {
    switch (method) {
        case LIBIA_IMAGE_SAMPLE_EULER_A: return EULER_A_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_HEUN: return HEUN_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_DPM2: return DPM2_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_DPMPP2S_A: return DPMPP2S_A_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_DPMPP2M: return DPMPP2M_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_LCM: return LCM_SAMPLE_METHOD;
        case LIBIA_IMAGE_SAMPLE_EULER:
        default: return EULER_SAMPLE_METHOD;
    }
}

static scheduler_t to_sd_scheduler(libia_image_scheduler scheduler) {
    switch (scheduler) {
        case LIBIA_IMAGE_SCHEDULER_KARRAS: return KARRAS_SCHEDULER;
        case LIBIA_IMAGE_SCHEDULER_EXPONENTIAL: return EXPONENTIAL_SCHEDULER;
        case LIBIA_IMAGE_SCHEDULER_SIMPLE: return SIMPLE_SCHEDULER;
        case LIBIA_IMAGE_SCHEDULER_DISCRETE:
        default: return DISCRETE_SCHEDULER;
    }
}

static sd_ctx_params_t build_sd_ctx_params(const libia_image_impl *impl) {
    sd_ctx_params_t params{};
    sd_ctx_params_init(&params);
    params.model_path = nullable_c_str(impl->model_path);
    params.diffusion_model_path = nullable_c_str(impl->diffusion_model_path);
    params.vae_path = nullable_c_str(impl->vae_path);
    params.clip_l_path = nullable_c_str(impl->clip_l_path);
    params.clip_g_path = nullable_c_str(impl->clip_g_path);
    params.t5xxl_path = nullable_c_str(impl->t5xxl_path);
    params.llm_path = nullable_c_str(impl->llm_path);
    params.n_threads = impl->params.n_threads > 0
        ? impl->params.n_threads
        : std::max(1u, std::thread::hardware_concurrency());
    params.enable_mmap = impl->params.use_mmap;
    params.flash_attn = impl->params.flash_attn;
    params.diffusion_flash_attn = impl->params.diffusion_flash_attn;
    params.backend = nullable_c_str(impl->backend);
    params.params_backend = nullable_c_str(impl->params_backend);
    params.max_vram = nullable_c_str(impl->max_vram);
    return params;
}

static bool ensure_loaded(libia_image_impl *impl, char **error_out) {
    if (!impl) {
        return false;
    }
    if (impl->ctx) {
        return true;
    }
    if (impl->model_path.empty() && impl->diffusion_model_path.empty()) {
        set_error(impl, error_out, "stable-diffusion model_path/diffusion_model_path is empty");
        return false;
    }

    static std::once_flag sd_once;
    std::call_once(sd_once, []() {
        // The SD sources are linked against llama.cpp's GGML. Let GGML discover
        // the CPU/Vulkan backend set configured by the parent project.
        sd_set_log_callback(libia_sd_log_callback, nullptr);
    });

    sd_ctx_params_t params = build_sd_ctx_params(impl);
    {
        std::lock_guard<std::mutex> lock(g_sd_log_mutex);
        g_sd_last_error.clear();
    }
    impl->ctx = new_sd_ctx(&params);
    if (!impl->ctx) {
        std::string detail;
        {
            std::lock_guard<std::mutex> lock(g_sd_log_mutex);
            detail = g_sd_last_error;
        }
        set_error(
            impl,
            error_out,
            detail.empty()
                ? "failed to create stable-diffusion context"
                : "failed to create stable-diffusion context: " + detail);
        return false;
    }
    if (!sd_ctx_supports_image_generation(impl->ctx)) {
        free_sd_ctx(impl->ctx);
        impl->ctx = nullptr;
        set_error(impl, error_out, "model does not support image generation");
        return false;
    }
    return true;
}

static bool write_png(const char *path, const sd_image_t &image, std::string *error) {
    if (!path || !*path) {
        if (error) {
            *error = "output_path is empty";
        }
        return false;
    }
    if (!image.data || image.width == 0 || image.height == 0 || image.channel == 0) {
        if (error) {
            *error = "stable-diffusion returned an empty image";
        }
        return false;
    }

    const int stride = static_cast<int>(image.width * image.channel);
    const int ok = stbi_write_png(
        path,
        static_cast<int>(image.width),
        static_cast<int>(image.height),
        static_cast<int>(image.channel),
        image.data,
        stride);
    if (!ok && error) {
        *error = "failed to write PNG image";
    }
    return ok != 0;
}

} // namespace

extern "C" {

LIBIA_API void libia_image_params_init(libia_image_params *params) {
    if (!params) {
        return;
    }
    *params = {};
    params->n_threads = static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    params->use_mmap = true;
    params->flash_attn = true;
    params->diffusion_flash_attn = true;
    params->backend = "Vulkan";
    params->params_backend = nullptr;
    params->max_vram = nullptr;
}

LIBIA_API void libia_image_generate_params_init(libia_image_generate_params *params) {
    if (!params) {
        return;
    }
    *params = {};
    params->width = 512;
    params->height = 512;
    params->sample_steps = 20;
    params->seed = -1;
    params->batch_count = 1;
    params->cfg_scale = 7.0f;
    params->guidance = 3.5f;
    params->sample_method = LIBIA_IMAGE_SAMPLE_EULER;
    params->scheduler = LIBIA_IMAGE_SCHEDULER_DISCRETE;
}

LIBIA_API void libia_image_result_free(libia_image_result *result) {
    if (!result) {
        return;
    }
    libia_string_free(result->path);
    *result = {};
}

LIBIA_API libia_image *libia_image_create(const libia_image_params *params) {
    auto *impl = new (std::nothrow) libia_image_impl();
    if (!impl) {
        return nullptr;
    }
    copy_params(impl, params);
    return reinterpret_cast<libia_image *>(impl);
}

LIBIA_API void libia_image_free(libia_image *image) {
    auto *impl = to_impl(image);
    if (!impl) {
        return;
    }
    if (impl->ctx) {
        free_sd_ctx(impl->ctx);
        impl->ctx = nullptr;
    }
    delete impl;
}

LIBIA_API bool libia_image_load(libia_image *image, char **error_out) {
    return ensure_loaded(to_impl(image), error_out);
}

LIBIA_API void libia_image_unload(libia_image *image) {
    auto *impl = to_impl(image);
    if (!impl || !impl->ctx) {
        return;
    }
    free_sd_ctx(impl->ctx);
    impl->ctx = nullptr;
}

LIBIA_API bool libia_image_is_loaded(const libia_image *image) {
    const auto *impl = to_impl(image);
    return impl && impl->ctx;
}

LIBIA_API bool libia_image_generate(
    libia_image *image,
    const libia_image_generate_params *params,
    libia_image_result *out_result,
    char **error_out) {
    auto *impl = to_impl(image);
    if (!impl) {
        return false;
    }
    if (!params) {
        set_error(impl, error_out, "image generation params are null");
        return false;
    }
    if (!params->prompt || !*params->prompt) {
        set_error(impl, error_out, "prompt is empty");
        return false;
    }
    if (!params->output_path || !*params->output_path) {
        set_error(impl, error_out, "output_path is empty");
        return false;
    }
    if (!ensure_loaded(impl, error_out)) {
        return false;
    }

    sd_img_gen_params_t gen{};
    sd_img_gen_params_init(&gen);
    gen.prompt = params->prompt;
    gen.negative_prompt = params->negative_prompt;
    gen.width = params->width > 0 ? params->width : 512;
    gen.height = params->height > 0 ? params->height : 512;
    gen.seed = params->seed;
    gen.batch_count = std::max(1, params->batch_count);
    gen.sample_params.sample_steps = params->sample_steps > 0 ? params->sample_steps : 20;
    gen.sample_params.guidance.txt_cfg = params->cfg_scale > 0.0f ? params->cfg_scale : 7.0f;
    gen.sample_params.guidance.distilled_guidance = params->guidance;
    gen.sample_params.sample_method = to_sd_sample_method(params->sample_method);
    gen.sample_params.scheduler = to_sd_scheduler(params->scheduler);

    sd_image_t *images = generate_image(impl->ctx, &gen);
    if (!images) {
        set_error(impl, error_out, "stable-diffusion image generation failed");
        return false;
    }

    std::string write_error;
    if (!write_png(params->output_path, images[0], &write_error)) {
        free_sd_images(images, gen.batch_count);
        set_error(impl, error_out, write_error);
        return false;
    }

    if (out_result) {
        libia_image_result_free(out_result);
        out_result->path = dup_string(params->output_path);
        out_result->width = static_cast<int32_t>(images[0].width);
        out_result->height = static_cast<int32_t>(images[0].height);
        out_result->channels = static_cast<int32_t>(images[0].channel);
        out_result->seed = gen.seed;
    }
    free_sd_images(images, gen.batch_count);
    return true;
}

LIBIA_API const char *libia_image_last_error(const libia_image *image) {
    const auto *impl = to_impl(image);
    if (!impl) {
        return "image runtime is null";
    }
    return impl->last_error.c_str();
}

} // extern "C"
