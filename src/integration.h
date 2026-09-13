#pragma once
#include <iostream>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef LIBIA_BUILD
#    define LIBIA_API __declspec(dllexport)
#  else
#    define LIBIA_API __declspec(dllimport)
#  endif
#else
#  define LIBIA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct libia_params libia_params;
typedef struct libia_runtime libia_runtime;
typedef struct libia_sst libia_sst;
typedef struct libia_image libia_image;

typedef enum libia_option_kind {
    LIBIA_OPTION_KIND_VOID = 0,
    LIBIA_OPTION_KIND_BOOL = 1,
    LIBIA_OPTION_KIND_INT = 2,
    LIBIA_OPTION_KIND_STRING = 3,
    LIBIA_OPTION_KIND_STRING_PAIR = 4,
} libia_option_kind;

typedef struct libia_option_info {
    const char *name;
    const char *value_hint;
    const char *value_hint_2;
    const char *help;
    const char *env;
    libia_option_kind kind;
    bool is_sampling;
    bool is_spec;
    bool is_preset_only;
    bool is_negated;
} libia_option_info;

typedef struct libia_chat_message {
    const char *role;
    const char *content;
    const char *reasoning_content;
} libia_chat_message;

typedef enum libia_generation_kind {
    LIBIA_GENERATION_KIND_CONTENT = 0,
    LIBIA_GENERATION_KIND_REASONING = 1,
    LIBIA_GENERATION_KIND_TOOL_CALL = 2,
} libia_generation_kind;

typedef void (*libia_stream_callback)(libia_generation_kind kind, const char *text, int32_t mode, int32_t tps, void *user_data);

typedef enum libia_sst_generation_kind {
    LIBIA_SST_GENERATION_KIND_TEXT = 0,
    LIBIA_SST_GENERATION_KIND_SPEECH_START = 1,
    LIBIA_SST_GENERATION_KIND_SPEECH_END = 2,
} libia_sst_generation_kind;

typedef void (*libia_sst_stream_callback)(libia_sst_generation_kind kind, const char *text, void *user_data);

LIBIA_API char *libia_string_dup(const char *text);
LIBIA_API void  libia_string_free(char *text);

LIBIA_API libia_params *libia_params_create(void);
LIBIA_API libia_params *libia_params_clone(const libia_params *params);
LIBIA_API void          libia_params_free(libia_params *params);
LIBIA_API void          libia_params_reset(libia_params *params);

LIBIA_API int  libia_params_option_count(void);
LIBIA_API bool libia_params_option_info(int index, libia_option_info *out_info);

LIBIA_API bool libia_params_apply_argv(libia_params *params, int argc, const char *const *argv, char **error_out);
LIBIA_API bool libia_params_set_option(libia_params *params, const char *name, const char *value, char **error_out);
LIBIA_API bool libia_params_set_option_pair(libia_params *params, const char *name, const char *value1, const char *value2, char **error_out);
LIBIA_API const char *libia_params_get_option(const libia_params *params, const char *name);
LIBIA_API bool libia_params_has_option(const libia_params *params, const char *name);

LIBIA_API bool libia_params_set_bool(libia_params *params, const char *name, bool value, char **error_out);
LIBIA_API bool libia_params_get_bool(const libia_params *params, const char *name, bool *out_value);
LIBIA_API bool libia_params_set_int(libia_params *params, const char *name, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_int(const libia_params *params, const char *name, int64_t *out_value);
LIBIA_API bool libia_params_set_float(libia_params *params, const char *name, double value, char **error_out);
LIBIA_API bool libia_params_get_float(const libia_params *params, const char *name, double *out_value);
LIBIA_API bool libia_params_set_string(libia_params *params, const char *name, const char *value, char **error_out);
LIBIA_API bool libia_params_get_string(const libia_params *params, const char *name, const char **out_value);
LIBIA_API bool libia_params_set_tools_json(libia_params *params, const char *tools_json, char **error_out);

LIBIA_API bool libia_params_set_model_path(libia_params *params, const char *path, char **error_out);
LIBIA_API bool libia_params_get_model_path(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_mmproj_path(libia_params *params, const char *path, char **error_out);
LIBIA_API bool libia_params_get_mmproj_path(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_prompt(libia_params *params, const char *prompt, char **error_out);
LIBIA_API bool libia_params_get_prompt(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_system_prompt(libia_params *params, const char *prompt, char **error_out);
LIBIA_API bool libia_params_get_system_prompt(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_chat_template(libia_params *params, const char *value, char **error_out);
LIBIA_API bool libia_params_get_chat_template(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_use_jinja(libia_params *params, bool value, char **error_out);
LIBIA_API bool libia_params_get_use_jinja(const libia_params *params, bool *out_value);
LIBIA_API bool libia_params_set_reasoning_format(libia_params *params, const char *value, char **error_out);
LIBIA_API bool libia_params_get_reasoning_format(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_enable_reasoning(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_enable_reasoning(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_enable_thinking(libia_params *params, bool value, char **error_out);
LIBIA_API bool libia_params_get_enable_thinking(const libia_params *params, bool *out_value);
LIBIA_API bool libia_params_set_input_prefix(libia_params *params, const char *value, char **error_out);
LIBIA_API bool libia_params_get_input_prefix(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_input_suffix(libia_params *params, const char *value, char **error_out);
LIBIA_API bool libia_params_get_input_suffix(const libia_params *params, const char **out_value);
LIBIA_API bool libia_params_set_n_predict(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_n_predict(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_n_ctx(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_n_ctx(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_n_batch(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_n_batch(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_n_ubatch(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_n_ubatch(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_n_keep(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_n_keep(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_seed(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_seed(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_temp(libia_params *params, double value, char **error_out);
LIBIA_API bool libia_params_get_temp(const libia_params *params, double *out_value);
LIBIA_API bool libia_params_set_top_k(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_top_k(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_top_p(libia_params *params, double value, char **error_out);
LIBIA_API bool libia_params_get_top_p(const libia_params *params, double *out_value);
LIBIA_API bool libia_params_set_min_p(libia_params *params, double value, char **error_out);
LIBIA_API bool libia_params_get_min_p(const libia_params *params, double *out_value);
LIBIA_API bool libia_params_set_image_min_tokens(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_image_min_tokens(const libia_params *params, int64_t *out_value);
LIBIA_API bool libia_params_set_image_max_tokens(libia_params *params, int64_t value, char **error_out);
LIBIA_API bool libia_params_get_image_max_tokens(const libia_params *params, int64_t *out_value);

LIBIA_API libia_runtime *libia_runtime_create(const libia_params *params);
LIBIA_API void           libia_runtime_free(libia_runtime *runtime);
LIBIA_API bool           libia_runtime_is_loaded(const libia_runtime *runtime);
LIBIA_API bool           libia_runtime_load(libia_runtime *runtime, char **error_out);
LIBIA_API void           libia_runtime_unload(libia_runtime *runtime);
LIBIA_API void           libia_runtime_cancel(libia_runtime *runtime);
LIBIA_API void           libia_runtime_cancel_all(void);
LIBIA_API void           libia_runtime_clear_media(libia_runtime *runtime);
LIBIA_API bool           libia_runtime_add_media_file(libia_runtime *runtime, const char *path, char **error_out);
LIBIA_API bool           libia_runtime_supports_image(const libia_runtime *runtime);
LIBIA_API bool           libia_runtime_supports_audio(const libia_runtime *runtime);
LIBIA_API bool           libia_runtime_supports_video(const libia_runtime *runtime);
LIBIA_API bool           libia_runtime_supports_thinking(const libia_runtime *runtime);
LIBIA_API char          *libia_runtime_media_caps_json(const libia_runtime *runtime);
LIBIA_API char          *libia_runtime_chat_template_caps_json(const libia_runtime *runtime);

LIBIA_API char *libia_runtime_generate_prompt(libia_runtime *runtime, const char *prompt, int64_t n_predict_override, char **error_out);
LIBIA_API char *libia_runtime_generate_chat(libia_runtime *runtime, const libia_chat_message *messages, size_t n_messages, int64_t n_predict_override, char **error_out);
LIBIA_API char *libia_runtime_generate_chat_stream(libia_runtime *runtime, const libia_chat_message *messages, size_t n_messages, int64_t n_predict_override, libia_stream_callback on_text, void *user_data, char **error_out);
LIBIA_API int64_t        libia_runtime_last_context_tokens(const libia_runtime *runtime);
LIBIA_API int64_t        libia_runtime_last_generated_tokens(const libia_runtime *runtime);
LIBIA_API int32_t        libia_runtime_last_tokens_per_second(const libia_runtime *runtime);

LIBIA_API char *libia_runtime_tokenize_text(libia_runtime *runtime, const char *text, bool add_special, bool parse_special, char **error_out);
LIBIA_API char *libia_runtime_detokenize_csv(libia_runtime *runtime, const char *tokens_csv, bool special, char **error_out);

LIBIA_API char *libia_runtime_system_info(const libia_runtime *runtime);
LIBIA_API const char *libia_runtime_last_error(const libia_runtime *runtime);

LIBIA_API libia_sst *libia_sst_create(const char *model_path);
LIBIA_API void       libia_sst_free(libia_sst *sst);
LIBIA_API bool       libia_sst_load(libia_sst *sst, char **error_out);
LIBIA_API void       libia_sst_reset(libia_sst *sst);
LIBIA_API void       libia_sst_clear_audio(libia_sst *sst);

LIBIA_API bool libia_sst_set_model_path(libia_sst *sst, const char *path, char **error_out);
LIBIA_API bool libia_sst_set_vad_model_path(libia_sst *sst, const char *path, char **error_out);
LIBIA_API bool libia_sst_set_language(libia_sst *sst, const char *value, char **error_out);
LIBIA_API bool libia_sst_set_initial_prompt(libia_sst *sst, const char *value, char **error_out);
LIBIA_API bool libia_sst_set_use_gpu(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_flash_attn(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_translate(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_no_context(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_no_timestamps(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_print_special(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_suppress_nst(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_n_threads(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_n_processors(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_step_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_length_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_keep_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_vad_enabled(libia_sst *sst, bool value, char **error_out);
LIBIA_API bool libia_sst_set_vad_threshold(libia_sst *sst, double value, char **error_out);
LIBIA_API bool libia_sst_set_vad_min_speech_duration_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_vad_min_silence_duration_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_vad_max_speech_duration_s(libia_sst *sst, double value, char **error_out);
LIBIA_API bool libia_sst_set_vad_speech_pad_ms(libia_sst *sst, int64_t value, char **error_out);
LIBIA_API bool libia_sst_set_vad_samples_overlap(libia_sst *sst, double value, char **error_out);

LIBIA_API char *libia_sst_push_audio(libia_sst *sst, const float *samples, size_t n_samples, int64_t sample_rate, int64_t channels, bool finalize, libia_sst_stream_callback on_text, void *user_data, char **error_out);
LIBIA_API char *libia_sst_flush(libia_sst *sst, libia_sst_stream_callback on_text, void *user_data, char **error_out);
LIBIA_API const char *libia_sst_last_error(const libia_sst *sst);

LIBIA_API void libia_runtime_set_keep_context(libia_runtime *runtime, bool keep);
LIBIA_API bool libia_runtime_get_keep_context(const libia_runtime *runtime);
LIBIA_API void libia_runtime_clear_context(libia_runtime *runtime);   // força limpar

typedef enum libia_image_sample_method {
    LIBIA_IMAGE_SAMPLE_EULER = 0,
    LIBIA_IMAGE_SAMPLE_EULER_A = 1,
    LIBIA_IMAGE_SAMPLE_HEUN = 2,
    LIBIA_IMAGE_SAMPLE_DPM2 = 3,
    LIBIA_IMAGE_SAMPLE_DPMPP2S_A = 4,
    LIBIA_IMAGE_SAMPLE_DPMPP2M = 5,
    LIBIA_IMAGE_SAMPLE_LCM = 9,
} libia_image_sample_method;

typedef enum libia_image_scheduler {
    LIBIA_IMAGE_SCHEDULER_DISCRETE = 0,
    LIBIA_IMAGE_SCHEDULER_KARRAS = 1,
    LIBIA_IMAGE_SCHEDULER_EXPONENTIAL = 2,
    LIBIA_IMAGE_SCHEDULER_SIMPLE = 6,
} libia_image_scheduler;

typedef struct libia_image_params {
    const char *model_path;
    const char *diffusion_model_path;
    const char *vae_path;
    const char *clip_l_path;
    const char *clip_g_path;
    const char *t5xxl_path;
    const char *llm_path;
    const char *backend;
    const char *params_backend;
    const char *max_vram;
    int32_t n_threads;
    bool use_mmap;
    bool flash_attn;
    bool diffusion_flash_attn;
} libia_image_params;

typedef struct libia_image_generate_params {
    const char *prompt;
    const char *negative_prompt;
    const char *output_path;
    int32_t width;
    int32_t height;
    int32_t sample_steps;
    int64_t seed;
    int32_t batch_count;
    float cfg_scale;
    float guidance;
    libia_image_sample_method sample_method;
    libia_image_scheduler scheduler;
} libia_image_generate_params;

typedef struct libia_image_result {
    char *path;
    int32_t width;
    int32_t height;
    int32_t channels;
    int64_t seed;
} libia_image_result;

LIBIA_API void libia_image_params_init(libia_image_params *params);
LIBIA_API void libia_image_generate_params_init(libia_image_generate_params *params);
LIBIA_API void libia_image_result_free(libia_image_result *result);

LIBIA_API libia_image *libia_image_create(const libia_image_params *params);
LIBIA_API void         libia_image_free(libia_image *image);
LIBIA_API bool         libia_image_load(libia_image *image, char **error_out);
LIBIA_API void         libia_image_unload(libia_image *image);
LIBIA_API bool         libia_image_is_loaded(const libia_image *image);
LIBIA_API bool         libia_image_generate(libia_image *image, const libia_image_generate_params *params, libia_image_result *out_result, char **error_out);
LIBIA_API const char  *libia_image_last_error(const libia_image *image);

#ifdef __cplusplus
}
#endif
