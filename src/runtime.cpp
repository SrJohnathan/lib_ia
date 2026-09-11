#include "integration.h"

#include <common/arg.h>
#include <common/chat.h>
#include <common/common.h>
#include <common/sampling.h>
#include <tools/mtmd/mtmd-helper.h>
#include <tools/mtmd/mtmd.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct libia_params_impl {
    common_params params;
    std::map<std::string, std::string> values;
    std::vector<common_chat_tool> tools;
};

struct libia_runtime_impl {
    libia_params_impl params;
    std::unique_ptr<common_init_result> init;
    common_sampler_ptr sampler;
    common_chat_templates_ptr templates;
    mtmd::context_ptr vision;
    mtmd::bitmaps media;
    std::vector<mtmd_helper::video_ptr> videos;
    std::string last_error;
    int64_t last_context_tokens = 0;
    int64_t last_generated_tokens = 0;
    int32_t last_tokens_per_second = 0;
    std::atomic_bool cancel_requested{false};
    bool loaded = false;
};

static std::atomic_bool g_cancel_all_generation{false};

struct stream_split_state {
    std::string buffer;
    bool in_reasoning = false;
};

struct parsed_stream_state {
    std::string generated_text;
    common_chat_msg parsed_msg;
    bool has_parsed_msg = false;
};

static std::string normalize_name(std::string name) {
    if (name.empty()) {
        return name;
    }
    if (name.front() != '-') {
        name = "--" + name;
    }
    std::replace(name.begin(), name.end(), '_', '-');
    return name;
}

static char *dup_string(const std::string &text) {
    char *out = static_cast<char *>(std::malloc(text.size() + 1));
    if (!out) {
        return nullptr;
    }
    std::memcpy(out, text.c_str(), text.size() + 1);
    return out;
}

static void set_error(char **error_out, const std::string &message) {
    if (!error_out) {
        return;
    }
    libia_string_free(*error_out);
    *error_out = dup_string(message);
}

static libia_params_impl *to_impl(libia_params *params) {
    return reinterpret_cast<libia_params_impl *>(params);
}

static const libia_params_impl *to_impl(const libia_params *params) {
    return reinterpret_cast<const libia_params_impl *>(params);
}

static libia_runtime_impl *to_impl(libia_runtime *runtime) {
    return reinterpret_cast<libia_runtime_impl *>(runtime);
}

static const libia_runtime_impl *to_impl(const libia_runtime *runtime) {
    return reinterpret_cast<const libia_runtime_impl *>(runtime);
}

static bool parse_bool_text(const std::string &value) {
    return common_arg_utils::is_truthy(value) || (!common_arg_utils::is_falsey(value) && value.empty());
}

static void finalize_tensor_overrides(common_params &params) {
    auto &overrides = params.tensor_buft_overrides;
    overrides.erase(
        std::remove_if(overrides.begin(), overrides.end(), [](const llama_model_tensor_buft_override &item) {
            return item.pattern == nullptr && item.buft == nullptr;
        }),
        overrides.end());

    if (!overrides.empty()) {
        const size_t max_overrides = llama_max_tensor_buft_overrides();
        if (max_overrides > 0 && overrides.size() >= max_overrides) {
            overrides.resize(max_overrides - 1);
        }
        overrides.push_back({nullptr, nullptr});
        while (overrides.size() < max_overrides) {
            overrides.push_back({nullptr, nullptr});
        }
    }
}

static bool init_runtime(libia_runtime_impl *rt, std::string *error) {
    if (rt->loaded && rt->init && rt->sampler) {
        return true;
    }
    try {
        static std::once_flag common_init_once;
        std::call_once(common_init_once, []() {
            common_init();
        });

        if (!rt->params.params.no_mmproj && !rt->params.params.mmproj.path.empty()) {
            rt->params.params.n_ubatch = std::max<int32_t>(
                rt->params.params.n_ubatch,
                std::max<int32_t>(rt->params.params.n_batch, 512));
        }
        finalize_tensor_overrides(rt->params.params);
        rt->init = common_init_from_params(rt->params.params);
        if (!rt->init) {
            if (error) *error = "failed to initialize model";
            return false;
        }
        rt->sampler.reset(common_sampler_init(rt->init->model(), rt->params.params.sampling));
        if (!rt->sampler) {
            if (error) *error = "failed to initialize sampler";
            return false;
        }
        rt->templates = common_chat_templates_init(rt->init->model(), rt->params.params.chat_template);
        rt->loaded = true;
        return true;
    } catch (const std::exception &ex) {
        if (error) *error = ex.what();
        return false;
    }
}

static void unload_runtime(libia_runtime_impl *rt) {
    if (!rt) {
        return;
    }

    rt->media.entries.clear();
    rt->videos.clear();
    rt->vision.reset();
    rt->templates.reset();
    rt->sampler.reset();
    rt->init.reset();
    rt->last_context_tokens = 0;
    rt->last_generated_tokens = 0;
    rt->last_tokens_per_second = 0;
    rt->loaded = false;
}

static bool decode_tokens_in_batches(
    llama_context *ctx,
    const std::vector<llama_token> &tokens,
    int64_t &n_past,
    int n_batch
) {
    if (tokens.empty()) {
        return true;
    }

    const int batch_size = std::max(1, n_batch);
    int offset = 0;
    while (offset < static_cast<int>(tokens.size())) {
        const int chunk = std::min(batch_size, static_cast<int>(tokens.size()) - offset);
        auto *chunk_data = const_cast<llama_token *>(tokens.data() + offset);
        if (llama_decode(ctx, llama_batch_get_one(chunk_data, chunk)) != 0) {
            return false;
        }
        offset += chunk;
        n_past += chunk;
    }

    return true;
}

static std::string tool_call_to_json(const common_chat_tool_call &tool) {
    nlohmann::ordered_json json;
    json["name"] = tool.name;
    json["call_id"] = tool.id;
    json["command"] = tool.name;
    try {
        json["args"] = nlohmann::ordered_json::parse(tool.arguments.empty() ? "{}" : tool.arguments);
    } catch (const std::exception &) {
        json["args"] = tool.arguments;
    }
    return json.dump();
}

static bool init_vision(libia_runtime_impl *rt, std::string *error) {
    if (rt->params.params.no_mmproj || rt->params.params.mmproj.path.empty()) {
        return true;
    }
    if (rt->vision) {
        return true;
    }
    try {
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = rt->params.params.mmproj_use_gpu;
        mparams.print_timings = true;
        mparams.n_threads = rt->params.params.cpuparams.n_threads;
        mparams.flash_attn_type = rt->params.params.flash_attn_type;
        mparams.warmup = rt->params.params.warmup;
        mparams.image_min_tokens = rt->params.params.image_min_tokens;
        mparams.image_max_tokens = rt->params.params.image_max_tokens;
        mparams.batch_max_tokens = rt->params.params.mtmd_batch_max_tokens;
        rt->vision.reset(mtmd_init_from_file(rt->params.params.mmproj.path.c_str(), rt->init->model(), mparams));
        if (!rt->vision) {
            if (error) *error = "failed to initialize multimodal context";
            return false;
        }
    } catch (const std::exception &ex) {
        if (error) *error = ex.what();
        return false;
    }
    return true;
}

static std::string ensure_media_markers(std::string prompt, const std::string &marker, size_t required) {
    size_t count = 0;
    size_t pos = 0;
    while ((pos = prompt.find(marker, pos)) != std::string::npos) {
        ++count;
        pos += marker.size();
    }
    if (count >= required) {
        return prompt;
    }
    if (!prompt.empty() && prompt.back() != '\n') {
        prompt.push_back('\n');
    }
    for (size_t i = count; i < required; ++i) {
        prompt += marker;
        if (i + 1 < required) {
            prompt.push_back('\n');
        }
    }
    return prompt;
}

static void ensure_media_markers_in_chat_messages(
        std::vector<common_chat_msg> & messages,
        const std::string & marker,
        size_t required) {
    if (required == 0 || marker.empty()) {
        return;
    }

    size_t count = 0;
    for (const auto & msg : messages) {
        size_t pos = 0;
        while ((pos = msg.content.find(marker, pos)) != std::string::npos) {
            ++count;
            pos += marker.size();
        }
    }
    if (count >= required) {
        return;
    }

    auto target = messages.end();
    for (auto it = messages.end(); it != messages.begin();) {
        --it;
        if (it->role == "user") {
            target = it;
            break;
        }
    }
    if (target == messages.end()) {
        common_chat_msg msg;
        msg.role = "user";
        messages.push_back(std::move(msg));
        target = std::prev(messages.end());
    }

    if (!target->content.empty() && target->content.back() != '\n') {
        target->content.push_back('\n');
    }
    for (size_t i = count; i < required; ++i) {
        target->content += marker;
        if (i + 1 < required) {
            target->content.push_back('\n');
        }
    }
}

static bool decode_prompt(libia_runtime_impl *rt, const std::string &prompt, int64_t &n_past, std::string *error) {
    if (!rt->media.entries.empty()) {
        if (!init_vision(rt, error)) {
            return false;
        }
        if (!rt->vision) {
            if (error) *error = "media was queued but no mmproj is configured";
            return false;
        }

        const std::string marker = mtmd_get_marker(rt->vision.get());
        std::string prompt_with_markers = ensure_media_markers(prompt, marker, rt->media.entries.size());
        mtmd_input_text text;
        text.text = prompt_with_markers.c_str();
        text.add_special = false;
        text.parse_special = true;

        mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
        const auto media_ptrs = rt->media.c_ptr();
        const mtmd_bitmap **bitmaps = const_cast<const mtmd_bitmap **>(media_ptrs.data());
        if (mtmd_tokenize(rt->vision.get(), chunks.get(), &text, bitmaps, media_ptrs.size()) != 0) {
            if (error) *error = "failed to tokenize multimodal prompt";
            return false;
        }

        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(rt->vision.get(), rt->init->context(), chunks.get(), n_past, 0, rt->params.params.n_batch, true, &new_n_past) != 0) {
            if (error) *error = "failed to evaluate multimodal prompt";
            return false;
        }
        n_past = new_n_past;
        rt->media.entries.clear();
        rt->videos.clear();
        return true;
    }

    auto tokens = common_tokenize(rt->init->context(), prompt, true, true);
    if (!decode_tokens_in_batches(rt->init->context(), tokens, n_past, rt->params.params.n_batch)) {
        if (error) *error = "failed to evaluate prompt";
        return false;
    }
    return true;
}

static bool reset_generation_state(libia_runtime_impl *rt, std::string *error) {
    if (!rt->init) {
        if (error) *error = "runtime is not initialized";
        return false;
    }

    llama_memory_clear(llama_get_memory(rt->init->context()), true);
    rt->init->reset_samplers();
    rt->sampler.reset(common_sampler_init(rt->init->model(), rt->params.params.sampling));
    if (!rt->sampler) {
        if (error) *error = "failed to reset sampler";
        return false;
    }

    return true;
}

static bool generate_text(
        libia_runtime_impl *rt,
        const std::string &prompt,
        int64_t n_predict_override,
        std::string &out,
        std::string *error,
        libia_stream_callback stream_cb = nullptr,
        void *stream_user = nullptr,
        const common_chat_parser_params *chat_parser_params = nullptr) {
    if (!init_runtime(rt, error)) {
        return false;
    }

    rt->cancel_requested.store(false);
    g_cancel_all_generation.store(false);

    if (!reset_generation_state(rt, error)) {
        return false;
    }

    int64_t n_past = 0;
    if (!decode_prompt(rt, prompt, n_past, error)) {
        return false;
    }
    rt->last_context_tokens = n_past;
    rt->last_generated_tokens = 0;
    rt->last_tokens_per_second = 0;

    const int64_t n_predict = n_predict_override >= 0 ? n_predict_override : rt->params.params.n_predict;
    if (n_predict == 0) {
        out.clear();
        return true;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    auto cleanup = [&]() { llama_batch_free(batch); };

    auto emit_chunk = [&](libia_generation_kind kind, const std::string &text, int32_t mode = 0, int32_t tps = 0) {
        if (text.empty() && mode == 0) {
            return;
        }
        if (stream_cb) {
            stream_cb(kind, text.c_str(), mode, tps, stream_user);
        }
    };

    const std::vector<std::string> reasoning_starts = {
        "<think>",
        "<|think|>",
        "<|START_THINKING|>",
        "[THINK]",
        "<|channel>thought",
        "<|channel>analysis",
        "<|channel>commentary",
    };
    const std::vector<std::string> reasoning_ends = {
        "</think>",
        "<|END_THINKING|>",
        "[/THINK]",
        "[BEGIN FINAL RESPONSE]",
        "<channel|>",
    };

    auto match_marker = [](const std::string & buffer, const std::vector<std::string> & markers, size_t & matched_len) -> bool {
        for (const auto & marker : markers) {
            if (buffer.size() >= marker.size() && buffer.compare(0, marker.size(), marker) == 0) {
                matched_len = marker.size();
                return true;
            }
        }
        return false;
    };

    auto flush_stream_buffer = [&](stream_split_state &state, bool final_flush) {
        const size_t max_marker_len = std::max(
            reasoning_starts.empty() ? size_t(0) : std::max_element(reasoning_starts.begin(), reasoning_starts.end(), [](const auto &a, const auto &b) { return a.size() < b.size(); })->size(),
            reasoning_ends.empty() ? size_t(0) : std::max_element(reasoning_ends.begin(), reasoning_ends.end(), [](const auto &a, const auto &b) { return a.size() < b.size(); })->size()
        );
        const size_t keep = max_marker_len > 1 ? max_marker_len - 1 : 0;

        while (true) {
            size_t matched_len = 0;
            bool matched = false;
            if (state.in_reasoning) {
                matched = match_marker(state.buffer, reasoning_ends, matched_len);
            } else {
                matched = match_marker(state.buffer, reasoning_starts, matched_len);
            }
            if (matched) {
                emit_chunk(state.in_reasoning ? LIBIA_GENERATION_KIND_REASONING : LIBIA_GENERATION_KIND_CONTENT, state.buffer.substr(0, 0));
                state.buffer.erase(0, matched_len);
                state.in_reasoning = !state.in_reasoning;
                continue;
            }

            if (final_flush) {
                emit_chunk(state.in_reasoning ? LIBIA_GENERATION_KIND_REASONING : LIBIA_GENERATION_KIND_CONTENT, state.buffer);
                state.buffer.clear();
            } else if (state.buffer.size() > keep) {
                const size_t flush_len = state.buffer.size() - keep;
                emit_chunk(state.in_reasoning ? LIBIA_GENERATION_KIND_REASONING : LIBIA_GENERATION_KIND_CONTENT, state.buffer.substr(0, flush_len));
                state.buffer.erase(0, flush_len);
            }
            break;
        }
    };

    stream_split_state stream_state;
    parsed_stream_state parsed_state;

    const auto generation_started_at = std::chrono::steady_clock::now();
    int64_t generated_tokens = 0;
    auto current_tps = [&]() -> int32_t {
        const auto elapsed = std::chrono::steady_clock::now() - generation_started_at;
        const double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count();
        if (seconds <= 0.0) {
            return 0;
        }
        return static_cast<int32_t>(std::llround(static_cast<double>(generated_tokens) / seconds));
    };

    auto emit_parsed_snapshot = [&](bool is_partial) {
        if (!stream_cb || !chat_parser_params) {
            return;
        }

        try {
            const common_chat_msg parsed = common_chat_parse(parsed_state.generated_text, is_partial, *chat_parser_params);
            auto emit_snapshot = [&](int32_t mode) {
                const int32_t tps = current_tps();
                emit_chunk(LIBIA_GENERATION_KIND_REASONING, parsed.reasoning_content, mode, tps);
                emit_chunk(LIBIA_GENERATION_KIND_CONTENT, parsed.content.empty() ? parsed.render_content() : parsed.content, mode, tps);
            };

            if (is_partial && parsed_state.has_parsed_msg) {
                try {
                    const auto diffs = common_chat_msg_diff::compute_diffs(parsed_state.parsed_msg, parsed);
                    for (const auto & diff : diffs) {
                        if (!diff.reasoning_content_delta.empty()) {
                            emit_chunk(LIBIA_GENERATION_KIND_REASONING, diff.reasoning_content_delta, 0, current_tps());
                        }
                        if (!diff.content_delta.empty()) {
                            emit_chunk(LIBIA_GENERATION_KIND_CONTENT, diff.content_delta, 0, current_tps());
                        }
                    }
                } catch (const std::exception &) {
                    emit_snapshot(1);
                }
            } else {
                emit_snapshot(is_partial ? 0 : 1);
            }

            parsed_state.parsed_msg = parsed;
            parsed_state.has_parsed_msg = true;
        } catch (const std::exception &ex) {
            if (!is_partial) {
                emit_chunk(LIBIA_GENERATION_KIND_CONTENT, parsed_state.generated_text, 1, current_tps());
            }
        }
    };

    const bool use_chat_parser = stream_cb && chat_parser_params;
    try {
        while (n_predict < 0 || generated_tokens < n_predict) {
            if (rt->cancel_requested.load() || g_cancel_all_generation.load()) {
                break;
            }
            llama_token token = common_sampler_sample(rt->sampler.get(), rt->init->context(), -1, false);
            common_sampler_accept(rt->sampler.get(), token, true);
            if (llama_vocab_is_eog(llama_model_get_vocab(rt->init->model()), token)) {
                break;
            }
            const std::string piece = common_token_to_piece(rt->init->context(), token, true);
            out += piece;
            if (use_chat_parser) {
                parsed_state.generated_text += piece;
                emit_parsed_snapshot(true);
            } else {
                stream_state.buffer += piece;
                flush_stream_buffer(stream_state, false);
            }
            ++generated_tokens;
            common_batch_clear(batch);
            common_batch_add(batch, token, static_cast<llama_pos>(n_past++), {0}, true);
            if (rt->cancel_requested.load() || g_cancel_all_generation.load()) {
                break;
            }
            if (llama_decode(rt->init->context(), batch) != 0) {
                if (error) *error = "failed to decode generated token";
                cleanup();
                return false;
            }
        }
        rt->last_generated_tokens = generated_tokens;
        rt->last_tokens_per_second = current_tps();
        if (use_chat_parser) {
            emit_parsed_snapshot(false);
            if (stream_cb && chat_parser_params) {
                try {
                    const common_chat_msg parsed = common_chat_parse(parsed_state.generated_text, false, *chat_parser_params);
                    for (const auto &tool : parsed.tool_calls) {
                        if (!tool.name.empty()) {
                            emit_chunk(LIBIA_GENERATION_KIND_TOOL_CALL, tool_call_to_json(tool), 1, current_tps());
                        }
                    }
                } catch (const std::exception &) {
                }
            }
        } else {
            flush_stream_buffer(stream_state, true);
        }
        cleanup();
        return true;
    } catch (const std::exception &ex) {
        if (!use_chat_parser) {
            flush_stream_buffer(stream_state, true);
        }
        cleanup();
        if (error) *error = ex.what();
        return false;
    }
}

extern "C" int64_t libia_runtime_last_context_tokens(const libia_runtime *runtime) {
    if (!runtime) {
        return 0;
    }
    return to_impl(runtime)->last_context_tokens;
}

extern "C" int64_t libia_runtime_last_generated_tokens(const libia_runtime *runtime) {
    if (!runtime) {
        return 0;
    }
    return to_impl(runtime)->last_generated_tokens;
}

extern "C" int32_t libia_runtime_last_tokens_per_second(const libia_runtime *runtime) {
    if (!runtime) {
        return 0;
    }
    return to_impl(runtime)->last_tokens_per_second;
}

} // namespace

extern "C" {

libia_runtime *libia_runtime_create(const libia_params *params) {
    auto *rt = new libia_runtime_impl{};
    if (params) {
        rt->params = *to_impl(params);
    }
    return reinterpret_cast<libia_runtime *>(rt);
}

void libia_runtime_free(libia_runtime *runtime) {
    if (!runtime) {
        return;
    }
    unload_runtime(to_impl(runtime));
    delete to_impl(runtime);
}

bool libia_runtime_is_loaded(const libia_runtime *runtime) {
    return runtime && to_impl(runtime)->loaded;
}

bool libia_runtime_load(libia_runtime *runtime, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime) {
        set_error(error_out, "invalid runtime");
        return false;
    }
    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return false;
    }
    return true;
}

void libia_runtime_unload(libia_runtime *runtime) {
    if (!runtime) {
        return;
    }
    to_impl(runtime)->cancel_requested.store(true);
    unload_runtime(to_impl(runtime));
}

void libia_runtime_cancel(libia_runtime *runtime) {
    if (!runtime) {
        return;
    }
    to_impl(runtime)->cancel_requested.store(true);
}

void libia_runtime_cancel_all(void) {
    g_cancel_all_generation.store(true);
}

void libia_runtime_clear_media(libia_runtime *runtime) {
    if (!runtime) {
        return;
    }
    auto *impl = to_impl(runtime);
    impl->media.entries.clear();
    impl->videos.clear();
}

bool libia_runtime_add_media_file(libia_runtime *runtime, const char *path, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime || !path) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return false;
    }
    if (!init_vision(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return false;
    }
    if (!impl->vision) {
        set_error(error_out, "mmproj context is not available");
        return false;
    }

    auto media = mtmd_helper_bitmap_init_from_file(impl->vision.get(), path, false);
    if (!media.bitmap) {
        set_error(error_out, "failed to load media file");
        return false;
    }
    impl->media.entries.emplace_back(media.bitmap);
    if (media.video_ctx) {
        impl->videos.emplace_back(media.video_ctx);
    }
    return true;
}

char *libia_runtime_generate_prompt(libia_runtime *runtime, const char *prompt, int64_t n_predict_override, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime) {
        set_error(error_out, "invalid runtime");
        return nullptr;
    }
    auto *impl = to_impl(runtime);
    std::string error;
    std::string output;
    const std::string prompt_text = prompt ? prompt : impl->params.params.prompt;
    if (!generate_text(impl, prompt_text, n_predict_override, output, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }
    return dup_string(output);
}

char *libia_runtime_generate_chat(libia_runtime *runtime, const libia_chat_message *messages, size_t n_messages, int64_t n_predict_override, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime) {
        set_error(error_out, "invalid runtime");
        return nullptr;
    }

    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }

    std::vector<common_chat_msg> chat_messages;
    if (!impl->params.params.system_prompt.empty()) {
        common_chat_msg sys;
        sys.role = "system";
        sys.content = impl->params.params.system_prompt;
        chat_messages.push_back(std::move(sys));
    }
    for (size_t i = 0; i < n_messages; ++i) {
        common_chat_msg msg;
        msg.role = messages && messages[i].role ? messages[i].role : "user";
        msg.content = messages && messages[i].content ? messages[i].content : "";
        msg.reasoning_content = messages && messages[i].reasoning_content ? messages[i].reasoning_content : "";
        chat_messages.push_back(std::move(msg));
    }
    if (!impl->media.entries.empty()) {
        std::string media_error;
        if (!init_vision(impl, &media_error)) {
            set_error(error_out, media_error);
            impl->last_error = media_error;
            return nullptr;
        }
        if (!impl->vision) {
            set_error(error_out, "media was queued but no mmproj is configured");
            impl->last_error = "media was queued but no mmproj is configured";
            return nullptr;
        }
        ensure_media_markers_in_chat_messages(
                chat_messages,
                mtmd_get_marker(impl->vision.get()),
                impl->media.entries.size());
    }

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_messages);
    inputs.use_jinja = impl->params.params.use_jinja;
    inputs.reasoning_format = impl->params.params.reasoning_format;
    inputs.force_pure_content = impl->params.params.force_pure_content_parser;
    inputs.enable_thinking = impl->params.params.enable_reasoning != 0;
    inputs.parallel_tool_calls = true;
    inputs.tools = impl->params.tools;

    auto chat_params = common_chat_templates_apply(impl->templates.get(), inputs);
    std::string formatted = chat_params.prompt;
    std::string output;
    if (!generate_text(impl, formatted, n_predict_override, output, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }
    return dup_string(output);
}

char *libia_runtime_generate_chat_stream(libia_runtime *runtime, const libia_chat_message *messages, size_t n_messages, int64_t n_predict_override, libia_stream_callback on_text, void *user_data, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime) {
        set_error(error_out, "invalid runtime");
        return nullptr;
    }

    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }

    std::vector<common_chat_msg> chat_messages;
    if (!impl->params.params.system_prompt.empty()) {
        common_chat_msg sys;
        sys.role = "system";
        sys.content = impl->params.params.system_prompt;
        chat_messages.push_back(std::move(sys));
    }
    for (size_t i = 0; i < n_messages; ++i) {
        common_chat_msg msg;
        msg.role = messages && messages[i].role ? messages[i].role : "user";
        msg.content = messages && messages[i].content ? messages[i].content : "";
        msg.reasoning_content = messages && messages[i].reasoning_content ? messages[i].reasoning_content : "";
        chat_messages.push_back(std::move(msg));
    }
    if (!impl->media.entries.empty()) {
        std::string media_error;
        if (!init_vision(impl, &media_error)) {
            set_error(error_out, media_error);
            impl->last_error = media_error;
            return nullptr;
        }
        if (!impl->vision) {
            set_error(error_out, "media was queued but no mmproj is configured");
            impl->last_error = "media was queued but no mmproj is configured";
            return nullptr;
        }
        ensure_media_markers_in_chat_messages(
                chat_messages,
                mtmd_get_marker(impl->vision.get()),
                impl->media.entries.size());
    }

    common_chat_templates_inputs inputs;
    inputs.messages = std::move(chat_messages);
    inputs.use_jinja = impl->params.params.use_jinja;
    inputs.reasoning_format = impl->params.params.reasoning_format;
    inputs.force_pure_content = impl->params.params.force_pure_content_parser;
    inputs.enable_thinking = impl->params.params.enable_reasoning != 0;
    inputs.parallel_tool_calls = true;
    inputs.tools = impl->params.tools;

    auto chat_params = common_chat_templates_apply(impl->templates.get(), inputs);
    std::string formatted = chat_params.prompt;

    common_chat_parser_params parser_params;
    parser_params.format = chat_params.format;
    parser_params.reasoning_format = inputs.reasoning_format;
    parser_params.reasoning_in_content = false;
    parser_params.generation_prompt = chat_params.generation_prompt;
    parser_params.parse_tool_calls = true;
    if (!chat_params.parser.empty()) {
        parser_params.parser.load(chat_params.parser);
    }

    std::string output;
    if (!generate_text(impl, formatted, n_predict_override, output, &error, on_text, user_data, &parser_params)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }
    return dup_string(output);
}

char *libia_runtime_tokenize_text(libia_runtime *runtime, const char *text, bool add_special, bool parse_special, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime || !text) {
        set_error(error_out, "invalid arguments");
        return nullptr;
    }
    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }
    try {
        auto tokens = common_tokenize(impl->init->context(), text, add_special, parse_special);
        std::ostringstream ss;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i != 0) {
                ss << ',';
            }
            ss << tokens[i];
        }
        return dup_string(ss.str());
    } catch (const std::exception &ex) {
        set_error(error_out, ex.what());
        impl->last_error = ex.what();
        return nullptr;
    }
}

char *libia_runtime_detokenize_csv(libia_runtime *runtime, const char *tokens_csv, bool special, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!runtime || !tokens_csv) {
        set_error(error_out, "invalid arguments");
        return nullptr;
    }
    auto *impl = to_impl(runtime);
    std::string error;
    if (!init_runtime(impl, &error)) {
        set_error(error_out, error);
        impl->last_error = error;
        return nullptr;
    }
    try {
        std::vector<llama_token> tokens;
        std::stringstream ss(tokens_csv);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty()) {
                tokens.push_back(static_cast<llama_token>(std::stoll(item)));
            }
        }
        return dup_string(common_detokenize(impl->init->context(), tokens, special));
    } catch (const std::exception &ex) {
        set_error(error_out, ex.what());
        impl->last_error = ex.what();
        return nullptr;
    }
}

char *libia_runtime_system_info(const libia_runtime *runtime) {
    if (!runtime) {
        return nullptr;
    }
    const auto *impl = to_impl(runtime);
    return dup_string(common_params_get_system_info(impl->params.params));
}

const char *libia_runtime_last_error(const libia_runtime *runtime) {
    if (!runtime) {
        return nullptr;
    }
    return to_impl(runtime)->last_error.c_str();
}

bool libia_runtime_supports_image(const libia_runtime *runtime) {
    if (!runtime) {
        return false;
    }
    const auto *impl = to_impl(runtime);
    if (!impl->vision) {
        return false;
    }
    return mtmd_support_vision(impl->vision.get());
}

bool libia_runtime_supports_thinking(const libia_runtime *runtime) {
    if (!runtime) {
        return false;
    }
    const auto *impl = to_impl(runtime);
    if (!impl->templates) {
        return false;
    }
    return common_chat_templates_support_enable_thinking(impl->templates.get());
}

bool libia_runtime_supports_audio(const libia_runtime *runtime) {
    if (!runtime) {
        return false;
    }
    const auto *impl = to_impl(runtime);
    if (!impl->vision) {
        return false;
    }
    return mtmd_support_audio(impl->vision.get());
}

bool libia_runtime_supports_video(const libia_runtime *runtime) {
    if (!runtime) {
        return false;
    }
    const auto *impl = to_impl(runtime);
    if (!impl->vision) {
        return false;
    }
    return mtmd_helper_support_video(impl->vision.get());
}

char *libia_runtime_media_caps_json(const libia_runtime *runtime) {
    std::ostringstream out;
    out << "{";
    out << "\"image\":" << (libia_runtime_supports_image(runtime) ? "true" : "false") << ",";
    out << "\"audio\":" << (libia_runtime_supports_audio(runtime) ? "true" : "false") << ",";
    out << "\"video\":" << (libia_runtime_supports_video(runtime) ? "true" : "false") << ",";
    out << "\"thinking\":" << (libia_runtime_supports_thinking(runtime) ? "true" : "false");
    out << "}";
    return dup_string(out.str());
}

char *libia_runtime_chat_template_caps_json(const libia_runtime *runtime) {
    if (!runtime) {
        return dup_string("{}");
    }
    const auto *impl = to_impl(runtime);
    if (!impl->templates) {
        return dup_string("{}");
    }

    const auto caps = common_chat_templates_get_caps(impl->templates.get());
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto &entry : caps) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << "\"" << entry.first << "\":" << (entry.second ? "true" : "false");
    }
    out << "}";
    return dup_string(out.str());
}

} // extern "C"
