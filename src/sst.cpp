#include "integration.h"

#include <whisper.h>
#include <ggml-backend.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <thread>

namespace {

static constexpr int kTargetSampleRate = WHISPER_SAMPLE_RATE;

static void quiet_whisper_log(ggml_log_level level, const char * text, void *) {
    if (level <= GGML_LOG_LEVEL_WARN && text) {
        std::fputs(text, stderr);
    }
}

struct libia_sst_impl {
    std::string model_path;
    std::string vad_model_path;
    std::string language = "auto";
    std::string initial_prompt;

    bool use_gpu = true;
    bool flash_attn = true;
    bool translate = false;
    bool no_context = true;
    bool no_timestamps = false;
    bool print_special = false;
    bool suppress_nst = false;
    bool vad_enabled = false;

    int64_t n_threads = std::min<int64_t>(4, (int64_t) std::thread::hardware_concurrency());
    int64_t n_processors = 1;
    int64_t step_ms = 3000;
    int64_t length_ms = 10000;
    int64_t keep_ms = 200;

    double vad_threshold = 0.5;
    int64_t vad_min_speech_duration_ms = 250;
    int64_t vad_min_silence_duration_ms = 100;
    double vad_max_speech_duration_s = 0.0;
    int64_t vad_speech_pad_ms = 30;
    double vad_samples_overlap = 0.1;

    std::unique_ptr<whisper_context, void (*)(whisper_context *)> ctx{nullptr, whisper_free};

    std::vector<float> pending_audio;
    std::vector<float> stream_history_audio;
    bool loaded = false;
    std::string last_emitted_text;
    std::string last_error;

    libia_sst_impl() = default;
};

static libia_sst_impl *to_impl(libia_sst *sst) {
    return reinterpret_cast<libia_sst_impl *>(sst);
}

static const libia_sst_impl *to_impl(const libia_sst *sst) {
    return reinterpret_cast<const libia_sst_impl *>(sst);
}

static char *dup_string(const std::string &text) {
    char *out = static_cast<char *>(std::malloc(text.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

static void set_error(libia_sst_impl *impl, char **error_out, const std::string &message) {
    impl->last_error = message;
    if (!error_out) {
        return;
    }
    libia_string_free(*error_out);
    *error_out = dup_string(message);
}

static std::string trim_copy(std::string text) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    while (!text.empty() && is_space(static_cast<unsigned char>(text.front()))) {
        text.erase(text.begin());
    }
    while (!text.empty() && is_space(static_cast<unsigned char>(text.back()))) {
        text.pop_back();
    }
    return text;
}

static std::vector<float> collapse_channels(const float *samples, size_t n_frames, int64_t channels) {
    std::vector<float> mono;
    if (!samples || n_frames == 0) {
        return mono;
    }

    channels = std::max<int64_t>(1, channels);
    mono.reserve(n_frames);
    for (size_t i = 0; i < n_frames; ++i) {
        double acc = 0.0;
        for (int64_t c = 0; c < channels; ++c) {
            acc += samples[i * static_cast<size_t>(channels) + static_cast<size_t>(c)];
        }
        mono.push_back(static_cast<float>(acc / static_cast<double>(channels)));
    }
    return mono;
}

static std::vector<float> resample_to_target(const std::vector<float> &input, int64_t sample_rate) {
    if (input.empty() || sample_rate <= 0) {
        return {};
    }
    if (sample_rate == kTargetSampleRate) {
        return input;
    }

    const double ratio = static_cast<double>(kTargetSampleRate) / static_cast<double>(sample_rate);
    const size_t out_size = static_cast<size_t>(std::ceil(static_cast<double>(input.size()) * ratio));
    std::vector<float> output(out_size, 0.0f);
    for (size_t i = 0; i < out_size; ++i) {
        const double src_pos = static_cast<double>(i) / ratio;
        const size_t idx = static_cast<size_t>(src_pos);
        const double frac = src_pos - static_cast<double>(idx);
        const float s0 = input[std::min(idx, input.size() - 1)];
        const float s1 = input[std::min(idx + 1, input.size() - 1)];
        output[i] = static_cast<float>(s0 + (s1 - s0) * frac);
    }
    return output;
}

static std::vector<float> normalize_audio(const float *samples, size_t n_samples, int64_t sample_rate, int64_t channels) {
    if (!samples || n_samples == 0) {
        return {};
    }

    const size_t frames = static_cast<size_t>(n_samples / static_cast<size_t>(std::max<int64_t>(1, channels)));
    auto mono = collapse_channels(samples, frames, channels);
    return resample_to_target(mono, sample_rate);
}

static float chunk_energy(const std::vector<float> &samples) {
    if (samples.empty()) {
        return 0.0f;
    }

    float energy = 0.0f;
    for (float sample : samples) {
        energy += std::fabs(sample);
    }

    return energy / static_cast<float>(samples.size());
}

static bool ensure_loaded(libia_sst_impl *impl, std::string *error) {
    if (impl->loaded && impl->ctx) {
        return true;
    }
    try {
        static std::once_flag backend_once;
        std::call_once(backend_once, []() {
            ggml_backend_load_all();
            whisper_log_set(quiet_whisper_log, nullptr);
        });

        if (impl->model_path.empty()) {
            if (error) {
                *error = "model path is empty";
            }
            return false;
        }

        whisper_context_params cparams = whisper_context_default_params();
        cparams.use_gpu = impl->use_gpu;
        cparams.flash_attn = impl->flash_attn;

        impl->ctx.reset(whisper_init_from_file_with_params(impl->model_path.c_str(), cparams));
        if (!impl->ctx) {
            if (error) {
                *error = "failed to initialize whisper context";
            }
            return false;
        }

        impl->loaded = true;
        return true;
    } catch (const std::exception &ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
}

static whisper_full_params build_full_params(const libia_sst_impl *impl) {
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.strategy = WHISPER_SAMPLING_GREEDY;
    params.n_threads = static_cast<int>(std::max<int64_t>(1, impl->n_threads));
    params.no_context = impl->no_context;
    params.no_timestamps = impl->no_timestamps;
    params.print_special = impl->print_special;
    params.print_progress = false;
    params.print_realtime = false;
    params.print_timestamps = !impl->no_timestamps;
    params.translate = impl->translate;
    params.language = impl->language.empty() ? "auto" : impl->language.c_str();
    params.detect_language = impl->language.empty() || impl->language == "auto";
    params.initial_prompt = impl->initial_prompt.empty() ? nullptr : impl->initial_prompt.c_str();
    params.carry_initial_prompt = false;
    params.suppress_nst = impl->suppress_nst;
    params.single_segment = true;
    params.max_tokens = 0;
    params.token_timestamps = !impl->no_timestamps;
    params.vad = false;
    params.vad_model_path = nullptr;
    return params;
}

static std::string collect_segments(whisper_context *ctx) {
    std::string out;
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (!text || !*text) {
            continue;
        }
        if (!out.empty() && !std::isspace(static_cast<unsigned char>(out.back()))) {
            out.push_back(' ');
        }
        out += text;
    }
    return trim_copy(std::move(out));
}

static void emit_text_segments(libia_sst_stream_callback cb, void *user_data, whisper_context *ctx) {
    if (!cb || !ctx) {
        return;
    }
    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char *text = whisper_full_get_segment_text(ctx, i);
        if (text && *text) {
            cb(LIBIA_SST_GENERATION_KIND_TEXT, text, user_data);
        }
    }
}

static bool transcribe_buffer(
    libia_sst_impl *impl,
    const std::vector<float> &audio,
    std::string &result,
    libia_sst_stream_callback cb,
    void *user_data,
    std::string *error) {
    if (audio.empty()) {
        result.clear();
        return true;
    }
    if (!ensure_loaded(impl, error)) {
        return false;
    }
    if (!impl->ctx) {
        if (error) {
            *error = "whisper context is not ready";
        }
        return false;
    }

    auto params = build_full_params(impl);
    const int processors = static_cast<int>(std::max<int64_t>(1, impl->n_processors));
    const int rc = whisper_full_parallel(impl->ctx.get(), params, audio.data(), static_cast<int>(audio.size()), processors);
    if (rc != 0) {
        if (error) {
            *error = "whisper transcription failed";
        }
        return false;
    }

    result = collect_segments(impl->ctx.get());

    return true;
}

static void clear_buffers(libia_sst_impl *impl) {
    impl->pending_audio.clear();
    impl->stream_history_audio.clear();
}

static bool process_chunk(
    libia_sst_impl *impl,
    const std::vector<float> &chunk,
    bool finalize,
    std::string &result,
    libia_sst_stream_callback cb,
    void *user_data,
    std::string *error) {
    if (chunk.empty() && !finalize) {
        result.clear();
        return true;
    }

    if (!chunk.empty()) {
        impl->pending_audio.insert(impl->pending_audio.end(), chunk.begin(), chunk.end());
    }

    if (!finalize) {
        result.clear();
        return true;
    }

    if (!transcribe_buffer(impl, impl->pending_audio, result, cb, user_data, error)) {
        return false;
    }
    impl->pending_audio.clear();
    impl->stream_history_audio.clear();

    return true;
}

static bool set_bool_field(bool &field, bool value) {
    field = value;
    return true;
}

static bool set_int_field(int64_t &field, int64_t value, char **error_out, const char *name, libia_sst_impl *impl) {
    if (value < 0 && std::string(name) != "--keep-ms" && std::string(name) != "--step-ms") {
        set_error(impl, error_out, std::string("invalid negative value for ") + name);
        return false;
    }
    field = value;
    return true;
}

} // namespace

extern "C" {

libia_sst *libia_sst_create(const char *model_path) {
    auto *impl = new libia_sst_impl{};
    if (model_path) {
        impl->model_path = model_path;
    }
    return reinterpret_cast<libia_sst *>(impl);
}

void libia_sst_free(libia_sst *sst) {
    delete to_impl(sst);
}

bool libia_sst_load(libia_sst *sst, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!sst) {
        return false;
    }
    std::string error;
    if (!ensure_loaded(to_impl(sst), &error)) {
        set_error(to_impl(sst), error_out, error);
        return false;
    }
    return true;
}

void libia_sst_reset(libia_sst *sst) {
    if (!sst) {
        return;
    }
    clear_buffers(to_impl(sst));
}

void libia_sst_clear_audio(libia_sst *sst) {
    if (!sst) {
        return;
    }
    auto *impl = to_impl(sst);
    impl->pending_audio.clear();
}

#define DEFINE_STRING_SETTER(Name, Field) \
    bool libia_sst_set_##Name(libia_sst *sst, const char *value, char **error_out) { \
        if (error_out) { *error_out = nullptr; } \
        if (!sst) { return false; } \
        auto *impl = to_impl(sst); \
        impl->Field = value ? value : ""; \
        return true; \
    }

DEFINE_STRING_SETTER(model_path, model_path)
DEFINE_STRING_SETTER(vad_model_path, vad_model_path)
DEFINE_STRING_SETTER(language, language)
DEFINE_STRING_SETTER(initial_prompt, initial_prompt)

#undef DEFINE_STRING_SETTER

#define DEFINE_BOOL_SETTER(Name, Field) \
    bool libia_sst_set_##Name(libia_sst *sst, bool value, char **error_out) { \
        if (error_out) { *error_out = nullptr; } \
        if (!sst) { return false; } \
        auto *impl = to_impl(sst); \
        impl->Field = value; \
        if constexpr (std::is_same_v<decltype(impl->Field), bool>) { \
            if (!value && std::string(#Name) == std::string("vad_enabled")) { \
                clear_buffers(impl); \
            } \
        } \
        return true; \
    }

DEFINE_BOOL_SETTER(use_gpu, use_gpu)
DEFINE_BOOL_SETTER(flash_attn, flash_attn)
DEFINE_BOOL_SETTER(translate, translate)
DEFINE_BOOL_SETTER(no_context, no_context)
DEFINE_BOOL_SETTER(no_timestamps, no_timestamps)
DEFINE_BOOL_SETTER(print_special, print_special)
DEFINE_BOOL_SETTER(suppress_nst, suppress_nst)
DEFINE_BOOL_SETTER(vad_enabled, vad_enabled)

#undef DEFINE_BOOL_SETTER

#define DEFINE_INT_SETTER(Name, Field) \
    bool libia_sst_set_##Name(libia_sst *sst, int64_t value, char **error_out) { \
        if (error_out) { *error_out = nullptr; } \
        if (!sst) { return false; } \
        auto *impl = to_impl(sst); \
        impl->Field = value; \
        return true; \
    }

DEFINE_INT_SETTER(n_threads, n_threads)
DEFINE_INT_SETTER(n_processors, n_processors)
DEFINE_INT_SETTER(step_ms, step_ms)
DEFINE_INT_SETTER(length_ms, length_ms)
DEFINE_INT_SETTER(keep_ms, keep_ms)
DEFINE_INT_SETTER(vad_min_speech_duration_ms, vad_min_speech_duration_ms)
DEFINE_INT_SETTER(vad_min_silence_duration_ms, vad_min_silence_duration_ms)
DEFINE_INT_SETTER(vad_speech_pad_ms, vad_speech_pad_ms)

#undef DEFINE_INT_SETTER

bool libia_sst_set_vad_threshold(libia_sst *sst, double value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!sst) {
        return false;
    }
    to_impl(sst)->vad_threshold = value;
    return true;
}

bool libia_sst_set_vad_max_speech_duration_s(libia_sst *sst, double value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!sst) {
        return false;
    }
    to_impl(sst)->vad_max_speech_duration_s = value;
    return true;
}

bool libia_sst_set_vad_samples_overlap(libia_sst *sst, double value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!sst) {
        return false;
    }
    to_impl(sst)->vad_samples_overlap = value;
    return true;
}

char *libia_sst_push_audio(libia_sst *sst, const float *samples, size_t n_samples, int64_t sample_rate, int64_t channels, bool finalize, libia_sst_stream_callback on_text, void *user_data, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!sst) {
        set_error(to_impl(sst), error_out, "invalid sst handle");
        return nullptr;
    }

    auto *impl = to_impl(sst);
    std::string result;
    std::string error;
    const auto chunk = normalize_audio(samples, n_samples, sample_rate, channels);
    if (!process_chunk(impl, chunk, finalize, result, on_text, user_data, &error)) {
        set_error(impl, error_out, error);
        return nullptr;
    }
    return dup_string(result);
}

char *libia_sst_flush(libia_sst *sst, libia_sst_stream_callback on_text, void *user_data, char **error_out) {
    return libia_sst_push_audio(sst, nullptr, 0, kTargetSampleRate, 1, true, on_text, user_data, error_out);
}

const char *libia_sst_last_error(const libia_sst *sst) {
    if (!sst) {
        return nullptr;
    }
    return to_impl(sst)->last_error.c_str();
}

} // extern "C"
