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
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct option_entry {
    common_arg arg;
    std::string primary_name;
    std::vector<std::string> aliases;
    libia_option_kind kind = LIBIA_OPTION_KIND_VOID;
};

struct option_catalog {
    std::vector<option_entry> entries;
    std::map<std::string, size_t> by_name;
};

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
    std::atomic_bool cancel_requested{false};
    bool loaded = false;
};

static std::atomic_bool g_cancel_all_generation{false};

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

static bool parse_bool_text(const std::string &value) {
    return common_arg_utils::is_truthy(value) || (!common_arg_utils::is_falsey(value) && value.empty());
}

static libia_option_kind infer_kind(const common_arg &arg) {
    if (arg.handler_str_str) return LIBIA_OPTION_KIND_STRING_PAIR;
    if (arg.handler_string) return LIBIA_OPTION_KIND_STRING;
    if (arg.handler_int) return LIBIA_OPTION_KIND_INT;
    if (arg.handler_bool) return LIBIA_OPTION_KIND_BOOL;
    return LIBIA_OPTION_KIND_VOID;
}

static const std::vector<enum llama_example> &catalog_examples() {
    static const std::vector<enum llama_example> examples = {
        LLAMA_EXAMPLE_BATCHED,
        LLAMA_EXAMPLE_DEBUG,
        LLAMA_EXAMPLE_COMMON,
        LLAMA_EXAMPLE_SPECULATIVE,
        LLAMA_EXAMPLE_COMPLETION,
        LLAMA_EXAMPLE_CLI,
        LLAMA_EXAMPLE_EMBEDDING,
        LLAMA_EXAMPLE_PERPLEXITY,
        LLAMA_EXAMPLE_RETRIEVAL,
        LLAMA_EXAMPLE_PASSKEY,
        LLAMA_EXAMPLE_IMATRIX,
        LLAMA_EXAMPLE_BENCH,
        LLAMA_EXAMPLE_SERVER,
        LLAMA_EXAMPLE_CVECTOR_GENERATOR,
        LLAMA_EXAMPLE_EXPORT_LORA,
        LLAMA_EXAMPLE_MTMD,
        LLAMA_EXAMPLE_LOOKUP,
        LLAMA_EXAMPLE_PARALLEL,
        LLAMA_EXAMPLE_TTS,
        LLAMA_EXAMPLE_DIFFUSION,
        LLAMA_EXAMPLE_FINETUNE,
        LLAMA_EXAMPLE_FIT_PARAMS,
        LLAMA_EXAMPLE_RESULTS,
        LLAMA_EXAMPLE_EXPORT_GRAPH_OPS,
    };
    return examples;
}

static option_catalog build_catalog() {
    option_catalog out;
    std::set<std::string> seen;

    for (auto ex : catalog_examples()) {
        common_params params;
        auto ctx = common_params_parser_init(params, ex, nullptr);
        for (const auto &arg : ctx.options) {
            auto aliases = arg.get_args();
            if (aliases.empty()) {
                continue;
            }
            const std::string primary = normalize_name(aliases.front());
            if (!seen.insert(primary).second) {
                continue;
            }

            option_entry entry;
            entry.arg = arg;
            entry.primary_name = primary;
            entry.kind = infer_kind(arg);
            entry.aliases.reserve(aliases.size());
            for (const auto &alias : aliases) {
                entry.aliases.push_back(normalize_name(alias));
            }

            const size_t index = out.entries.size();
            out.entries.push_back(std::move(entry));
            for (const auto &alias : out.entries.back().aliases) {
                out.by_name.emplace(alias, index);
            }
        }
    }
    return out;
}

static const option_catalog &catalog() {
    static const option_catalog built = build_catalog();
    return built;
}

static const option_entry *find_entry(const std::string &name) {
    const auto &cat = catalog();
    auto it = cat.by_name.find(normalize_name(name));
    if (it == cat.by_name.end()) {
        return nullptr;
    }
    return &cat.entries[it->second];
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

static void apply_postprocess(common_params &params) {
    postprocess_cpu_params(params.cpuparams, nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams, &params.cpuparams);
    postprocess_cpu_params(params.speculative.draft.cpuparams_batch, &params.cpuparams_batch);

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto &s : params.antiprompt) {
            string_process_escapes(s);
        }
        for (auto &s : params.sampling.dry_sequence_breakers) {
            string_process_escapes(s);
        }
    }
}

static bool store_value(libia_params_impl *impl, const std::string &name, const std::string &value) {
    impl->values[normalize_name(name)] = value;
    const auto *entry = find_entry(name);
    if (entry) {
        impl->values[entry->primary_name] = value;
    }
    return true;
}

static bool apply_entry(common_params &params, const option_entry &entry, const std::string &requested_name, const std::vector<std::string> &values, std::string *error) {
    try {
        switch (entry.kind) {
            case LIBIA_OPTION_KIND_VOID:
                if (entry.arg.handler_void) {
                    entry.arg.handler_void(params);
                }
                return true;
            case LIBIA_OPTION_KIND_BOOL: {
                bool value = true;
                if (!values.empty()) {
                    value = parse_bool_text(values.front());
                } else if (std::find(entry.arg.args_neg.begin(), entry.arg.args_neg.end(), requested_name) != entry.arg.args_neg.end()) {
                    value = false;
                }
                if (entry.arg.handler_bool) {
                    entry.arg.handler_bool(params, value);
                }
                return true;
            }
            case LIBIA_OPTION_KIND_INT:
                if (values.empty()) {
                    throw std::invalid_argument("missing integer value");
                }
                if (entry.arg.handler_int) {
                    entry.arg.handler_int(params, std::stoi(values.front()));
                }
                return true;
            case LIBIA_OPTION_KIND_STRING:
                if (values.empty()) {
                    throw std::invalid_argument("missing string value");
                }
                if (entry.arg.handler_string) {
                    entry.arg.handler_string(params, values.front());
                }
                return true;
            case LIBIA_OPTION_KIND_STRING_PAIR:
                if (values.size() < 2) {
                    throw std::invalid_argument("missing pair values");
                }
                if (entry.arg.handler_str_str) {
                    entry.arg.handler_str_str(params, values[0], values[1]);
                }
                return true;
        }
    } catch (const std::exception &ex) {
        if (error) {
            *error = ex.what();
        }
        return false;
    }
    return false;
}

static bool apply_option_impl(libia_params_impl *impl, const std::string &name, const std::vector<std::string> &values, std::string *error) {
    const auto *entry = find_entry(name);
    if (!entry) {
        if (error) {
            *error = "unknown option: " + name;
        }
        return false;
    }
    if (!apply_entry(impl->params, *entry, normalize_name(name), values, error)) {
        return false;
    }
    if (values.empty()) {
        store_value(impl, name, entry->kind == LIBIA_OPTION_KIND_BOOL ? "true" : "");
    } else if (values.size() == 1) {
        store_value(impl, name, values.front());
    } else {
        store_value(impl, name, values[0] + "," + values[1]);
    }
    return true;
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

static bool generate_text(libia_runtime_impl *rt, const std::string &prompt, int64_t n_predict_override, std::string &out, std::string *error) {
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

    const int64_t n_predict = n_predict_override >= 0 ? n_predict_override : rt->params.params.n_predict;
    if (n_predict == 0) {
        out.clear();
        return true;
    }

    llama_batch batch = llama_batch_init(1, 0, 1);
    auto cleanup = [&]() { llama_batch_free(batch); };

    try {
        while (n_predict < 0 || static_cast<int64_t>(out.size()) < n_predict) {
            if (rt->cancel_requested.load() || g_cancel_all_generation.load()) {
                break;
            }
            llama_token token = common_sampler_sample(rt->sampler.get(), rt->init->context(), -1, false);
            common_sampler_accept(rt->sampler.get(), token, true);
            if (llama_vocab_is_eog(llama_model_get_vocab(rt->init->model()), token)) {
                break;
            }
            out += common_token_to_piece(rt->init->context(), token, true);
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
        cleanup();
        return true;
    } catch (const std::exception &ex) {
        cleanup();
        if (error) *error = ex.what();
        return false;
    }
}

static bool get_numeric(const libia_params_impl *impl, const char *name, double *out) {
    auto it = impl->values.find(normalize_name(name ? name : ""));
    if (it == impl->values.end()) {
        return false;
    }
    try {
        *out = std::stod(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

static bool get_integer(const libia_params_impl *impl, const char *name, int64_t *out) {
    auto it = impl->values.find(normalize_name(name ? name : ""));
    if (it == impl->values.end()) {
        return false;
    }
    try {
        *out = std::stoll(it->second);
        return true;
    } catch (...) {
        return false;
    }
}

static bool get_boolean(const libia_params_impl *impl, const char *name, bool *out) {
    auto it = impl->values.find(normalize_name(name ? name : ""));
    if (it == impl->values.end()) {
        return false;
    }
    *out = parse_bool_text(it->second);
    return true;
}

static bool set_from_scalar(libia_params_impl *impl, const char *name, const std::string &value, std::string *error) {
    return apply_option_impl(impl, name ? name : "", {value}, error);
}

static bool set_from_bool(libia_params_impl *impl, const char *name, bool value, std::string *error) {
    const auto *entry = find_entry(name ? name : "");
    if (!entry) {
        if (error) *error = std::string("unknown option: ") + (name ? name : "");
        return false;
    }
    if (entry->kind != LIBIA_OPTION_KIND_BOOL && entry->kind != LIBIA_OPTION_KIND_VOID) {
        if (error) *error = std::string("option is not boolean: ") + (name ? name : "");
        return false;
    }
    if (entry->arg.handler_bool) {
        entry->arg.handler_bool(impl->params, value);
    } else if (entry->arg.handler_void && value) {
        entry->arg.handler_void(impl->params);
    }
    store_value(impl, name ? name : entry->primary_name, value ? "true" : "false");
    return true;
}

} // namespace

extern "C" {

char *libia_string_dup(const char *text) {
    return dup_string(text ? text : "");
}

void libia_string_free(char *text) {
    std::free(text);
}

libia_params *libia_params_create(void) {
    return reinterpret_cast<libia_params *>(new libia_params_impl{});
}

libia_params *libia_params_clone(const libia_params *params) {
    if (!params) {
        return nullptr;
    }
    return reinterpret_cast<libia_params *>(new libia_params_impl(*to_impl(params)));
}

void libia_params_free(libia_params *params) {
    delete to_impl(params);
}

void libia_params_reset(libia_params *params) {
    if (!params) {
        return;
    }
    auto *impl = to_impl(params);
    impl->params = common_params{};
    impl->values.clear();
}

int libia_params_option_count(void) {
    return static_cast<int>(catalog().entries.size());
}

bool libia_params_option_info(int index, libia_option_info *out_info) {
    if (!out_info || index < 0 || index >= static_cast<int>(catalog().entries.size())) {
        return false;
    }
    const auto &entry = catalog().entries[static_cast<size_t>(index)];
    out_info->name = entry.primary_name.c_str();
    out_info->value_hint = entry.arg.value_hint;
    out_info->value_hint_2 = entry.arg.value_hint_2;
    out_info->help = entry.arg.help.c_str();
    out_info->env = entry.arg.env;
    out_info->kind = entry.kind;
    out_info->is_sampling = entry.arg.is_sampling;
    out_info->is_spec = entry.arg.is_spec;
    out_info->is_preset_only = entry.arg.is_preset_only;
    out_info->is_negated = !entry.arg.args_neg.empty();
    return true;
}

bool libia_params_apply_argv(libia_params *params, int argc, const char *const *argv, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || argc < 1 || !argv) {
        set_error(error_out, "invalid arguments");
        return false;
    }

    auto *impl = to_impl(params);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "";
        if (arg.rfind("--", 0) == 0) {
            arg = normalize_name(arg);
        }
        const auto *entry = find_entry(arg);
        if (!entry) {
            set_error(error_out, "unknown option: " + arg);
            return false;
        }

        std::vector<std::string> values;
        if (entry->kind == LIBIA_OPTION_KIND_BOOL) {
            if (i + 1 < argc) {
                const std::string next = argv[i + 1] ? argv[i + 1] : "";
                if (common_arg_utils::is_truthy(next) || common_arg_utils::is_falsey(next)) {
                    values.emplace_back(next);
                    ++i;
                }
            }
        } else if (entry->kind == LIBIA_OPTION_KIND_STRING_PAIR) {
            if (i + 2 >= argc) {
                set_error(error_out, "missing values for " + arg);
                return false;
            }
            values.emplace_back(argv[++i] ? argv[i] : "");
            values.emplace_back(argv[++i] ? argv[i] : "");
        } else if (entry->kind != LIBIA_OPTION_KIND_VOID) {
            if (i + 1 >= argc) {
                set_error(error_out, "missing value for " + arg);
                return false;
            }
            values.emplace_back(argv[++i] ? argv[i] : "");
        }

        std::string error;
        if (!apply_option_impl(impl, arg, values, &error)) {
            set_error(error_out, error);
            return false;
        }
    }

    apply_postprocess(impl->params);
    return true;
}

bool libia_params_set_option(libia_params *params, const char *name, const char *value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!set_from_scalar(to_impl(params), name, value ? value : "", &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

bool libia_params_set_option_pair(libia_params *params, const char *name, const char *value1, const char *value2, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name || !value1 || !value2) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!apply_option_impl(to_impl(params), name, {value1, value2}, &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

const char *libia_params_get_option(const libia_params *params, const char *name) {
    if (!params || !name) {
        return nullptr;
    }
    const auto *impl = to_impl(params);
    auto it = impl->values.find(normalize_name(name));
    if (it == impl->values.end()) {
        return nullptr;
    }
    return it->second.c_str();
}

bool libia_params_has_option(const libia_params *params, const char *name) {
    return libia_params_get_option(params, name) != nullptr;
}

bool libia_params_set_bool(libia_params *params, const char *name, bool value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!set_from_bool(to_impl(params), name, value, &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

bool libia_params_get_bool(const libia_params *params, const char *name, bool *out_value) {
    if (!params || !name || !out_value) {
        return false;
    }
    return get_boolean(to_impl(params), name, out_value);
}

bool libia_params_set_int(libia_params *params, const char *name, int64_t value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!set_from_scalar(to_impl(params), name, std::to_string(value), &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

bool libia_params_get_int(const libia_params *params, const char *name, int64_t *out_value) {
    if (!params || !name || !out_value) {
        return false;
    }
    return get_integer(to_impl(params), name, out_value);
}

bool libia_params_set_float(libia_params *params, const char *name, double value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!set_from_scalar(to_impl(params), name, std::to_string(value), &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

bool libia_params_get_float(const libia_params *params, const char *name, double *out_value) {
    if (!params || !name || !out_value) {
        return false;
    }
    return get_numeric(to_impl(params), name, out_value);
}

bool libia_params_set_string(libia_params *params, const char *name, const char *value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !name) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    std::string error;
    if (!set_from_scalar(to_impl(params), name, value ? value : "", &error)) {
        set_error(error_out, error);
        return false;
    }
    return true;
}

bool libia_params_get_string(const libia_params *params, const char *name, const char **out_value) {
    if (!params || !name || !out_value) {
        return false;
    }
    const auto *impl = to_impl(params);
    auto it = impl->values.find(normalize_name(name));
    if (it == impl->values.end()) {
        return false;
    }
    *out_value = it->second.c_str();
    return true;
}

bool libia_params_set_tools_json(libia_params *params, const char *tools_json, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params) {
        set_error(error_out, "invalid params");
        return false;
    }

    auto *impl = to_impl(params);
    const std::string raw = tools_json ? tools_json : "";
    if (raw.empty()) {
        impl->tools.clear();
        impl->values.erase("__tools_json");
        return true;
    }

    try {
        auto parsed = nlohmann::ordered_json::parse(raw);
        impl->tools = common_chat_tools_parse_oaicompat(parsed);
        impl->values["__tools_json"] = raw;
        return true;
    } catch (const std::exception &ex) {
        set_error(error_out, std::string("failed to parse tools json: ") + ex.what());
        return false;
    }
}

bool libia_params_set_use_jinja(libia_params *params, bool value, char **error_out) {
    return libia_params_set_bool(params, "--jinja", value, error_out);
}

bool libia_params_get_use_jinja(const libia_params *params, bool *out_value) {
    return libia_params_get_bool(params, "--jinja", out_value);
}

bool libia_params_set_reasoning_format(libia_params *params, const char *value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params || !value) {
        set_error(error_out, "invalid arguments");
        return false;
    }

    const std::string value_str = value;
    const auto format = common_reasoning_format_from_name(value_str);
    if (format == COMMON_REASONING_FORMAT_NONE && value_str != "none") {
        set_error(error_out, "invalid reasoning format");
        return false;
    }

    auto *impl = to_impl(params);
    impl->params.reasoning_format = format;
    impl->values[normalize_name("--reasoning-format")] = value_str;
    return true;
}

bool libia_params_get_reasoning_format(const libia_params *params, const char **out_value) {
    if (!params || !out_value) {
        return false;
    }
    *out_value = common_reasoning_format_name(to_impl(params)->params.reasoning_format);
    return true;
}

bool libia_params_set_enable_reasoning(libia_params *params, int64_t value, char **error_out) {
    if (error_out) {
        *error_out = nullptr;
    }
    if (!params) {
        set_error(error_out, "invalid arguments");
        return false;
    }
    if (value < -1 || value > 1) {
        set_error(error_out, "enable reasoning must be -1, 0, or 1");
        return false;
    }

    auto *impl = to_impl(params);
    impl->params.enable_reasoning = static_cast<int>(value);
    impl->values[normalize_name("--reasoning")] = value < 0 ? "auto" : (value > 0 ? "on" : "off");
    return true;
}

bool libia_params_get_enable_reasoning(const libia_params *params, int64_t *out_value) {
    if (!params || !out_value) {
        return false;
    }
    *out_value = to_impl(params)->params.enable_reasoning;
    return true;
}

bool libia_params_set_enable_thinking(libia_params *params, bool value, char **error_out) {
    return libia_params_set_enable_reasoning(params, value ? 1 : 0, error_out);
}

bool libia_params_get_enable_thinking(const libia_params *params, bool *out_value) {
    if (!params || !out_value) {
        return false;
    }
    int64_t value = 0;
    if (!libia_params_get_enable_reasoning(params, &value)) {
        return false;
    }
    *out_value = value != 0;
    return true;
}

#define DEFINE_STRING_WRAPPERS(Stem, FlagName) \
    bool libia_params_set_##Stem(libia_params *params, const char *value, char **error_out) { \
        return libia_params_set_string(params, FlagName, value, error_out); \
    } \
    bool libia_params_get_##Stem(const libia_params *params, const char **out_value) { \
        return libia_params_get_string(params, FlagName, out_value); \
    }

DEFINE_STRING_WRAPPERS(model_path, "--model")
DEFINE_STRING_WRAPPERS(mmproj_path, "--mmproj")
DEFINE_STRING_WRAPPERS(prompt, "--prompt")
DEFINE_STRING_WRAPPERS(system_prompt, "--system-prompt")
DEFINE_STRING_WRAPPERS(chat_template, "--chat-template")
DEFINE_STRING_WRAPPERS(input_prefix, "--input-prefix")
DEFINE_STRING_WRAPPERS(input_suffix, "--input-suffix")

#undef DEFINE_STRING_WRAPPERS

#define DEFINE_INT_WRAPPERS(Stem, FlagName) \
    bool libia_params_set_##Stem(libia_params *params, int64_t value, char **error_out) { \
        return libia_params_set_int(params, FlagName, value, error_out); \
    } \
    bool libia_params_get_##Stem(const libia_params *params, int64_t *out_value) { \
        return libia_params_get_int(params, FlagName, out_value); \
    }

DEFINE_INT_WRAPPERS(n_predict, "--n-predict")
DEFINE_INT_WRAPPERS(n_ctx, "--ctx-size")
DEFINE_INT_WRAPPERS(n_batch, "--batch-size")
DEFINE_INT_WRAPPERS(n_ubatch, "--ubatch-size")
DEFINE_INT_WRAPPERS(n_keep, "--keep")
DEFINE_INT_WRAPPERS(seed, "--seed")
DEFINE_INT_WRAPPERS(top_k, "--top-k")
DEFINE_INT_WRAPPERS(image_min_tokens, "--image-min-tokens")
DEFINE_INT_WRAPPERS(image_max_tokens, "--image-max-tokens")

#undef DEFINE_INT_WRAPPERS

#define DEFINE_FLOAT_WRAPPERS(Stem, FlagName) \
    bool libia_params_set_##Stem(libia_params *params, double value, char **error_out) { \
        return libia_params_set_float(params, FlagName, value, error_out); \
    } \
    bool libia_params_get_##Stem(const libia_params *params, double *out_value) { \
        return libia_params_get_float(params, FlagName, out_value); \
    }

DEFINE_FLOAT_WRAPPERS(temp, "--temp")
DEFINE_FLOAT_WRAPPERS(top_p, "--top-p")
DEFINE_FLOAT_WRAPPERS(min_p, "--min-p")

#undef DEFINE_FLOAT_WRAPPERS

} // extern "C"
