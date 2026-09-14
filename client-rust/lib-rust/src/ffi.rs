use libc::{c_char, c_double, c_float, c_int, c_void};
use std::os::raw::c_longlong;

#[repr(C)]
pub struct libia_params {
    _private: [u8; 0],
}

#[repr(C)]
pub struct libia_runtime {
    _private: [u8; 0],
}

#[repr(C)]
pub struct libia_sst {
    _private: [u8; 0],
}

#[repr(C)]
pub struct libia_image {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum libia_option_kind {
    LIBIA_OPTION_KIND_VOID = 0,
    LIBIA_OPTION_KIND_BOOL = 1,
    LIBIA_OPTION_KIND_INT = 2,
    LIBIA_OPTION_KIND_STRING = 3,
    LIBIA_OPTION_KIND_STRING_PAIR = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct libia_option_info {
    pub name: *const c_char,
    pub value_hint: *const c_char,
    pub value_hint_2: *const c_char,
    pub help: *const c_char,
    pub env: *const c_char,
    pub kind: libia_option_kind,
    pub is_sampling: bool,
    pub is_spec: bool,
    pub is_preset_only: bool,
    pub is_negated: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct libia_chat_message {
    pub role: *const c_char,
    pub content: *const c_char,
    pub reasoning_content: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum libia_generation_kind {
    LIBIA_GENERATION_KIND_CONTENT = 0,
    LIBIA_GENERATION_KIND_REASONING = 1,
    LIBIA_GENERATION_KIND_TOOL_CALL = 2,
}

pub type libia_stream_callback = Option<
    unsafe extern "C" fn(
        kind: libia_generation_kind,
        text: *const c_char,
        mode: c_int,
        tps: c_int,
        user_data: *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum libia_sst_generation_kind {
    LIBIA_SST_GENERATION_KIND_TEXT = 0,
    LIBIA_SST_GENERATION_KIND_SPEECH_START = 1,
    LIBIA_SST_GENERATION_KIND_SPEECH_END = 2,
}

pub type libia_sst_stream_callback = Option<
    unsafe extern "C" fn(
        kind: libia_sst_generation_kind,
        text: *const c_char,
        user_data: *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum libia_image_sample_method {
    LIBIA_IMAGE_SAMPLE_EULER = 0,
    LIBIA_IMAGE_SAMPLE_EULER_A = 1,
    LIBIA_IMAGE_SAMPLE_HEUN = 2,
    LIBIA_IMAGE_SAMPLE_DPM2 = 3,
    LIBIA_IMAGE_SAMPLE_DPMPP2S_A = 4,
    LIBIA_IMAGE_SAMPLE_DPMPP2M = 5,
    LIBIA_IMAGE_SAMPLE_LCM = 9,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum libia_image_scheduler {
    LIBIA_IMAGE_SCHEDULER_DISCRETE = 0,
    LIBIA_IMAGE_SCHEDULER_KARRAS = 1,
    LIBIA_IMAGE_SCHEDULER_EXPONENTIAL = 2,
    LIBIA_IMAGE_SCHEDULER_SIMPLE = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct libia_image_params {
    pub model_path: *const c_char,
    pub diffusion_model_path: *const c_char,
    pub vae_path: *const c_char,
    pub clip_l_path: *const c_char,
    pub clip_g_path: *const c_char,
    pub t5xxl_path: *const c_char,
    pub llm_path: *const c_char,
    pub backend: *const c_char,
    pub params_backend: *const c_char,
    pub max_vram: *const c_char,
    pub n_threads: c_int,
    pub use_mmap: bool,
    pub flash_attn: bool,
    pub diffusion_flash_attn: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct libia_image_generate_params {
    pub prompt: *const c_char,
    pub negative_prompt: *const c_char,
    pub output_path: *const c_char,
    pub width: c_int,
    pub height: c_int,
    pub sample_steps: c_int,
    pub seed: c_longlong,
    pub batch_count: c_int,
    pub cfg_scale: c_float,
    pub guidance: c_float,
    pub sample_method: libia_image_sample_method,
    pub scheduler: libia_image_scheduler,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct libia_image_result {
    pub path: *mut c_char,
    pub width: c_int,
    pub height: c_int,
    pub channels: c_int,
    pub seed: c_longlong,
}

unsafe extern "C" {
    pub fn libia_string_dup(text: *const c_char) -> *mut c_char;
    pub fn libia_string_free(text: *mut c_char);

    pub fn libia_params_create() -> *mut libia_params;
    pub fn libia_params_clone(params: *const libia_params) -> *mut libia_params;
    pub fn libia_params_free(params: *mut libia_params);
    pub fn libia_params_reset(params: *mut libia_params);

    pub fn libia_params_option_count() -> c_int;
    pub fn libia_params_option_info(index: c_int, out_info: *mut libia_option_info) -> bool;

    pub fn libia_params_apply_argv(
        params: *mut libia_params,
        argc: c_int,
        argv: *const *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_set_option(
        params: *mut libia_params,
        name: *const c_char,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_set_option_pair(
        params: *mut libia_params,
        name: *const c_char,
        value1: *const c_char,
        value2: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_option(
        params: *const libia_params,
        name: *const c_char,
    ) -> *const c_char;
    pub fn libia_params_has_option(params: *const libia_params, name: *const c_char) -> bool;

    pub fn libia_params_set_bool(
        params: *mut libia_params,
        name: *const c_char,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_bool(
        params: *const libia_params,
        name: *const c_char,
        out_value: *mut bool,
    ) -> bool;
    pub fn libia_params_set_int(
        params: *mut libia_params,
        name: *const c_char,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_int(
        params: *const libia_params,
        name: *const c_char,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_float(
        params: *mut libia_params,
        name: *const c_char,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_float(
        params: *const libia_params,
        name: *const c_char,
        out_value: *mut c_double,
    ) -> bool;
    pub fn libia_params_set_string(
        params: *mut libia_params,
        name: *const c_char,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_string(
        params: *const libia_params,
        name: *const c_char,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_tools_json(
        params: *mut libia_params,
        tools_json: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;

    pub fn libia_params_set_model_path(
        params: *mut libia_params,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_model_path(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_mmproj_path(
        params: *mut libia_params,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_mmproj_path(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_prompt(
        params: *mut libia_params,
        prompt: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_prompt(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_system_prompt(
        params: *mut libia_params,
        prompt: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_system_prompt(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_chat_template(
        params: *mut libia_params,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_chat_template(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_use_jinja(
        params: *mut libia_params,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_use_jinja(params: *const libia_params, out_value: *mut bool) -> bool;
    pub fn libia_params_set_reasoning_format(
        params: *mut libia_params,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_reasoning_format(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_enable_reasoning(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_enable_reasoning(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_enable_thinking(
        params: *mut libia_params,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_enable_thinking(
        params: *const libia_params,
        out_value: *mut bool,
    ) -> bool;
    pub fn libia_params_set_input_prefix(
        params: *mut libia_params,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_input_prefix(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_input_suffix(
        params: *mut libia_params,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_input_suffix(
        params: *const libia_params,
        out_value: *mut *const c_char,
    ) -> bool;
    pub fn libia_params_set_n_predict(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_n_predict(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_n_ctx(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_n_ctx(params: *const libia_params, out_value: *mut c_longlong) -> bool;
    pub fn libia_params_set_n_batch(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_n_batch(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_n_ubatch(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_n_ubatch(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_n_keep(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_n_keep(params: *const libia_params, out_value: *mut c_longlong)
    -> bool;
    pub fn libia_params_set_seed(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_seed(params: *const libia_params, out_value: *mut c_longlong) -> bool;
    pub fn libia_params_set_temp(
        params: *mut libia_params,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_temp(params: *const libia_params, out_value: *mut c_double) -> bool;
    pub fn libia_params_set_top_k(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_top_k(params: *const libia_params, out_value: *mut c_longlong) -> bool;
    pub fn libia_params_set_top_p(
        params: *mut libia_params,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_top_p(params: *const libia_params, out_value: *mut c_double) -> bool;
    pub fn libia_params_set_min_p(
        params: *mut libia_params,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_min_p(params: *const libia_params, out_value: *mut c_double) -> bool;
    pub fn libia_params_set_image_min_tokens(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_image_min_tokens(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;
    pub fn libia_params_set_image_max_tokens(
        params: *mut libia_params,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_params_get_image_max_tokens(
        params: *const libia_params,
        out_value: *mut c_longlong,
    ) -> bool;

    pub fn libia_runtime_create(params: *const libia_params) -> *mut libia_runtime;
    pub fn libia_runtime_free(runtime: *mut libia_runtime);
    pub fn libia_runtime_is_loaded(runtime: *const libia_runtime) -> bool;
    pub fn libia_runtime_load(runtime: *mut libia_runtime, error_out: *mut *mut c_char) -> bool;
    pub fn libia_runtime_unload(runtime: *mut libia_runtime);
    pub fn libia_runtime_cancel(runtime: *mut libia_runtime);
    pub fn libia_runtime_cancel_all();
    pub fn libia_runtime_clear_media(runtime: *mut libia_runtime);
    pub fn libia_runtime_add_media_file(
        runtime: *mut libia_runtime,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_runtime_supports_image(runtime: *const libia_runtime) -> bool;
    pub fn libia_runtime_supports_audio(runtime: *const libia_runtime) -> bool;
    pub fn libia_runtime_supports_video(runtime: *const libia_runtime) -> bool;
    pub fn libia_runtime_media_caps_json(runtime: *const libia_runtime) -> *mut c_char;

    pub fn libia_runtime_generate_prompt(
        runtime: *mut libia_runtime,
        prompt: *const c_char,
        n_predict_override: c_longlong,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_runtime_generate_chat(
        runtime: *mut libia_runtime,
        messages: *const libia_chat_message,
        n_messages: usize,
        n_predict_override: c_longlong,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_runtime_generate_chat_stream(
        runtime: *mut libia_runtime,
        messages: *const libia_chat_message,
        n_messages: usize,
        n_predict_override: c_longlong,
        on_text: libia_stream_callback,
        user_data: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_runtime_last_context_tokens(runtime: *const libia_runtime) -> c_longlong;
    pub fn libia_runtime_last_generated_tokens(runtime: *const libia_runtime) -> c_longlong;
    pub fn libia_runtime_last_tokens_per_second(runtime: *const libia_runtime) -> c_double;
    pub fn libia_runtime_supports_thinking(runtime: *const libia_runtime) -> bool;
    pub fn libia_runtime_chat_template_caps_json(runtime: *const libia_runtime) -> *mut c_char;

    pub fn libia_runtime_tokenize_text(
        runtime: *mut libia_runtime,
        text: *const c_char,
        add_special: bool,
        parse_special: bool,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_runtime_detokenize_csv(
        runtime: *mut libia_runtime,
        tokens_csv: *const c_char,
        special: bool,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;

    pub fn libia_runtime_system_info(runtime: *const libia_runtime) -> *mut c_char;
    pub fn libia_runtime_last_error(runtime: *const libia_runtime) -> *const c_char;

    pub fn libia_sst_create(model_path: *const c_char) -> *mut libia_sst;
    pub fn libia_sst_free(sst: *mut libia_sst);
    pub fn libia_sst_load(sst: *mut libia_sst, error_out: *mut *mut c_char) -> bool;
    pub fn libia_sst_reset(sst: *mut libia_sst);
    pub fn libia_sst_clear_audio(sst: *mut libia_sst);
    pub fn libia_sst_set_model_path(
        sst: *mut libia_sst,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_model_path(
        sst: *mut libia_sst,
        path: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_language(
        sst: *mut libia_sst,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_initial_prompt(
        sst: *mut libia_sst,
        value: *const c_char,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_use_gpu(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_flash_attn(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_translate(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_no_context(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_no_timestamps(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_print_special(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_suppress_nst(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_n_threads(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_n_processors(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_step_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_length_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_keep_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_enabled(
        sst: *mut libia_sst,
        value: bool,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_threshold(
        sst: *mut libia_sst,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_min_speech_duration_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_min_silence_duration_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_max_speech_duration_s(
        sst: *mut libia_sst,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_speech_pad_ms(
        sst: *mut libia_sst,
        value: c_longlong,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_set_vad_samples_overlap(
        sst: *mut libia_sst,
        value: c_double,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_sst_push_audio(
        sst: *mut libia_sst,
        samples: *const f32,
        n_samples: usize,
        sample_rate: c_longlong,
        channels: c_longlong,
        finalize: bool,
        on_text: libia_sst_stream_callback,
        user_data: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_sst_flush(
        sst: *mut libia_sst,
        on_text: libia_sst_stream_callback,
        user_data: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_char;
    pub fn libia_sst_last_error(sst: *const libia_sst) -> *const c_char;

    pub fn libia_image_params_init(params: *mut libia_image_params);
    pub fn libia_image_generate_params_init(params: *mut libia_image_generate_params);
    pub fn libia_image_result_free(result: *mut libia_image_result);
    pub fn libia_image_create(params: *const libia_image_params) -> *mut libia_image;
    pub fn libia_image_free(image: *mut libia_image);
    pub fn libia_image_load(image: *mut libia_image, error_out: *mut *mut c_char) -> bool;
    pub fn libia_image_unload(image: *mut libia_image);
    pub fn libia_image_is_loaded(image: *const libia_image) -> bool;
    pub fn libia_image_generate(
        image: *mut libia_image,
        params: *const libia_image_generate_params,
        out_result: *mut libia_image_result,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub fn libia_image_last_error(image: *const libia_image) -> *const c_char;
}
