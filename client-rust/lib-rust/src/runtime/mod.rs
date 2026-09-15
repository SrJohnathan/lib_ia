use crate::error::{Error, Result};
use crate::ffi;
use crate::helpes::{normalize_prop_name, read_c_string, take_owned_c_string, to_cstring};
use crate::traits::{ChatMessage, GenerationType, InferenceEngine, ToolCall};
use base64::Engine;
use libc::c_void;
use std::collections::HashMap;
use std::ffi::CStr;
use std::io::Write;
use std::ptr::NonNull;

struct ParamsHandle(NonNull<ffi::libia_params>);

impl ParamsHandle {
    fn new() -> Result<Self> {
        let ptr = unsafe { ffi::libia_params_create() };
        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer("libia_params_create"))?;
        Ok(Self(ptr))
    }

    fn as_ptr(&self) -> *mut ffi::libia_params {
        self.0.as_ptr()
    }
}

impl Drop for ParamsHandle {
    fn drop(&mut self) {
        unsafe { ffi::libia_params_free(self.as_ptr()) };
    }
}

struct RuntimeHandle(NonNull<ffi::libia_runtime>);

impl RuntimeHandle {
    fn new(params: &ParamsHandle) -> Result<Self> {
        let ptr = unsafe { ffi::libia_runtime_create(params.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer("libia_runtime_create"))?;
        Ok(Self(ptr))
    }

    fn as_ptr(&self) -> *mut ffi::libia_runtime {
        self.0.as_ptr()
    }
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::libia_runtime_unload(self.as_ptr());
            ffi::libia_runtime_free(self.as_ptr());
        };
    }
}

pub struct RuntimeLlama {
    params: ParamsHandle,
    runtime: Option<RuntimeHandle>,
    dirty: bool,
}

unsafe impl Send for RuntimeLlama {}

impl RuntimeLlama {
    pub fn try_new(
        model_path: String,
        ctx: usize,
        n_predict: usize,
        n_batch: usize,
        n_ubatch: usize,
        temp: f32,
        top_p: f32,
        top_k: usize,
        mmproj_path: Option<String>,
        _loras_path: Option<Vec<String>>,
        ngl: Option<usize>,
        system_prompt: Option<String>,
    ) -> Result<Self> {
        let params = ParamsHandle::new()?;

        unsafe {
            let model = to_cstring(&model_path)?;
            if !ffi::libia_params_set_model_path(
                params.as_ptr(),
                model.as_ptr(),
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set model path".to_string()));
            }
            if !ffi::libia_params_set_n_ctx(params.as_ptr(), ctx as i64, std::ptr::null_mut()) {
                return Err(Error::Api("failed to set context size".to_string()));
            }
            if !ffi::libia_params_set_n_predict(
                params.as_ptr(),
                n_predict as i64,
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set n_predict".to_string()));
            }
            if !ffi::libia_params_set_n_batch(params.as_ptr(), n_batch as i64, std::ptr::null_mut())
            {
                return Err(Error::Api("failed to set n_batch".to_string()));
            }
            if !ffi::libia_params_set_n_ubatch(
                params.as_ptr(),
                n_ubatch as i64,
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set n_ubatch".to_string()));
            }
            if !ffi::libia_params_set_temp(params.as_ptr(), temp as f64, std::ptr::null_mut()) {
                return Err(Error::Api("failed to set temp".to_string()));
            }
            if !ffi::libia_params_set_top_p(params.as_ptr(), top_p as f64, std::ptr::null_mut()) {
                return Err(Error::Api("failed to set top_p".to_string()));
            }
            if !ffi::libia_params_set_top_k(params.as_ptr(), top_k as i64, std::ptr::null_mut()) {
                return Err(Error::Api("failed to set top_k".to_string()));
            }

            if let Some(ngl) = ngl {
                let prop = to_cstring("--gpu-layers")?;
                let value = to_cstring(&ngl.to_string())?;
                if !ffi::libia_params_set_option(
                    params.as_ptr(),
                    prop.as_ptr(),
                    value.as_ptr(),
                    std::ptr::null_mut(),
                ) {
                    return Err(Error::Api("failed to set gpu layers".to_string()));
                }
            }

            if let Some(mmproj_path) = mmproj_path {
                let mmproj = to_cstring(&mmproj_path)?;
                if !ffi::libia_params_set_mmproj_path(
                    params.as_ptr(),
                    mmproj.as_ptr(),
                    std::ptr::null_mut(),
                ) {
                    return Err(Error::Api("failed to set mmproj path".to_string()));
                }
            }

            if let Some(system_prompt) = system_prompt {
                let prompt = to_cstring(&system_prompt)?;
                if !ffi::libia_params_set_system_prompt(
                    params.as_ptr(),
                    prompt.as_ptr(),
                    std::ptr::null_mut(),
                ) {
                    return Err(Error::Api("failed to set system prompt".to_string()));
                }
            }
        }

        Ok(Self {
            params,
            runtime: None,
            dirty: true,
        })
    }

    fn ensure_runtime(&mut self) -> Result<&mut RuntimeHandle> {
        if self.dirty {
            self.runtime = None;
            self.dirty = false;
        }

        if self.runtime.is_none() {
            self.runtime = Some(RuntimeHandle::new(&self.params)?);
        }

        let runtime = self.runtime.as_mut().unwrap();
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let loaded = unsafe { ffi::libia_runtime_load(runtime.as_ptr(), &mut error) };
        if !loaded {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "failed to load runtime".to_string())
            };
            return Err(Error::Api(message));
        }
        Ok(runtime)
    }

    pub fn is_loaded(&self) -> bool {
        self.runtime.is_some()
    }

    pub fn generate_stream_loaded_with<F>(&mut self, prompt: &str, on_text: F) -> Result<String>
    where
        F: FnMut(GenerationType, i32, i32),
    {
        self.generate_chat_messages_stream_with(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                reasoning_content: None,
            }],
            on_text,
        )
    }

    pub fn unload(&mut self) {
        if let Some(runtime) = self.runtime.as_mut() {
            unsafe { ffi::libia_runtime_unload(runtime.as_ptr()) };
        }
        self.runtime = None;
    }

    pub fn cancel_generation(&mut self) {
        if let Some(runtime) = self.runtime.as_mut() {
            unsafe { ffi::libia_runtime_cancel(runtime.as_ptr()) };
        }
    }

    pub fn cancel_all_generations() {
        unsafe { ffi::libia_runtime_cancel_all() };
    }

    pub fn set_prompt(&mut self, prompt: &str) -> Result<()> {
        let prompt = to_cstring(prompt)?;
        unsafe {
            if !ffi::libia_params_set_prompt(
                self.params.as_ptr(),
                prompt.as_ptr(),
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set prompt".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_system_prompt(&mut self, prompt: &str) -> Result<()> {
        let prompt = to_cstring(prompt)?;
        unsafe {
            if !ffi::libia_params_set_system_prompt(
                self.params.as_ptr(),
                prompt.as_ptr(),
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set system prompt".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn clear_media(&mut self) -> Result<()> {
        let runtime = self.ensure_runtime()?;
        unsafe { ffi::libia_runtime_clear_media(runtime.as_ptr()) };
        Ok(())
    }

    pub fn add_media_file(&mut self, path: &str) -> Result<()> {
        let runtime = self.ensure_runtime()?;
        let path = to_cstring(path)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let ok = unsafe {
            ffi::libia_runtime_add_media_file(runtime.as_ptr(), path.as_ptr(), &mut error)
        };
        if !ok {
            let message = unsafe {
                take_owned_c_string(error)
                    .unwrap_or_else(|_| "failed to add media file".to_string())
            };
            return Err(Error::Api(message));
        }
        Ok(())
    }

    pub fn add_media_files(&mut self, paths: &[String]) -> Result<()> {
        for path in paths {
            self.add_media_file(path)?;
        }
        Ok(())
    }

    pub fn add_media_base64(&mut self, data: &str) -> Result<()> {
        let payload = if let Some((_, encoded)) = data.split_once("base64,") {
            encoded
        } else {
            data
        };

        let bytes = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .map_err(|_| Error::Api("failed to decode base64 image".to_string()))?;

        let mut tmp = tempfile::NamedTempFile::new()
            .map_err(|err| Error::Api(format!("failed to create temp file for image: {}", err)))?;
        tmp.write_all(&bytes)
            .map_err(|err| Error::Api(format!("failed to write temp image file: {}", err)))?;
        tmp.flush()
            .map_err(|err| Error::Api(format!("failed to flush temp image file: {}", err)))?;

        let path = tmp.path().to_string_lossy().to_string();
        let result = self.add_media_file(&path);
        drop(tmp);
        result
    }

    pub fn add_media_base64s(&mut self, items: &[String]) -> Result<()> {
        for item in items {
            self.add_media_base64(item)?;
        }
        Ok(())
    }

    pub fn add_audio_file(&mut self, path: &str) -> Result<()> {
        self.add_media_file(path)
    }

    pub fn add_audio_files(&mut self, paths: &[String]) -> Result<()> {
        self.add_media_files(paths)
    }

    pub fn add_audio_base64(&mut self, data: &str) -> Result<()> {
        self.add_media_base64(data)
    }

    pub fn add_audio_base64s(&mut self, items: &[String]) -> Result<()> {
        self.add_media_base64s(items)
    }

    pub fn add_video_file(&mut self, path: &str) -> Result<()> {
        self.add_media_file(path)
    }

    pub fn add_video_files(&mut self, paths: &[String]) -> Result<()> {
        self.add_media_files(paths)
    }

    pub fn add_video_base64(&mut self, data: &str) -> Result<()> {
        self.add_media_base64(data)
    }

    pub fn add_video_base64s(&mut self, items: &[String]) -> Result<()> {
        self.add_media_base64s(items)
    }

    pub fn set_use_jinja(&mut self, value: bool) -> Result<()> {
        unsafe {
            if !ffi::libia_params_set_use_jinja(self.params.as_ptr(), value, std::ptr::null_mut()) {
                return Err(Error::Api("failed to set use_jinja".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn use_jinja(&self) -> Option<bool> {
        let mut value = false;
        let ok = unsafe { ffi::libia_params_get_use_jinja(self.params.as_ptr(), &mut value) };
        if ok { Some(value) } else { None }
    }

    pub fn set_reasoning_format(&mut self, value: &str) -> Result<()> {
        let value = to_cstring(value)?;
        unsafe {
            if !ffi::libia_params_set_reasoning_format(
                self.params.as_ptr(),
                value.as_ptr(),
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set reasoning_format".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn reasoning_format(&self) -> Option<String> {
        let mut value: *const libc::c_char = std::ptr::null();
        let ok =
            unsafe { ffi::libia_params_get_reasoning_format(self.params.as_ptr(), &mut value) };
        if !ok {
            return None;
        }
        unsafe { read_c_string(value) }
    }

    pub fn set_enable_reasoning(&mut self, value: i64) -> Result<()> {
        unsafe {
            if !ffi::libia_params_set_enable_reasoning(
                self.params.as_ptr(),
                value as libc::c_longlong,
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set enable_reasoning".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn enable_reasoning(&self) -> Option<i64> {
        let mut value: libc::c_longlong = 0;
        let ok =
            unsafe { ffi::libia_params_get_enable_reasoning(self.params.as_ptr(), &mut value) };
        if ok { Some(value as i64) } else { None }
    }

    pub fn set_enable_thinking(&mut self, value: bool) -> Result<()> {
        unsafe {
            if !ffi::libia_params_set_enable_thinking(
                self.params.as_ptr(),
                value,
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api("failed to set enable_thinking".to_string()));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn enable_thinking(&self) -> Option<bool> {
        let mut value = false;
        let ok = unsafe { ffi::libia_params_get_enable_thinking(self.params.as_ptr(), &mut value) };
        if ok { Some(value) } else { None }
    }

    pub fn set_speculative_type(&mut self, value: &str) -> Result<()> {
        self.set_prop("--spec-type", value)
    }

    pub fn speculative_type(&self) -> Option<String> {
        self.get_prop("--spec-type").ok().flatten()
    }

    pub fn set_spec_draft_model(&mut self, value: &str) -> Result<()> {
        self.set_prop("--spec-draft-model", value)
    }

    pub fn spec_draft_model(&self) -> Option<String> {
        self.get_prop("--spec-draft-model").ok().flatten()
    }

    pub fn set_spec_draft_n_max(&mut self, value: i64) -> Result<()> {
        self.set_prop("--spec-draft-n-max", &value.to_string())
    }

    pub fn set_spec_draft_n_min(&mut self, value: i64) -> Result<()> {
        self.set_prop("--spec-draft-n-min", &value.to_string())
    }

    pub fn set_spec_draft_p_split(&mut self, value: f32) -> Result<()> {
        self.set_prop("--spec-draft-p-split", &value.to_string())
    }

    pub fn set_spec_draft_p_min(&mut self, value: f32) -> Result<()> {
        self.set_prop("--spec-draft-p-min", &value.to_string())
    }

    pub fn set_spec_draft_backend_sampling(&mut self, value: bool) -> Result<()> {
        self.set_prop(
            "--spec-draft-backend-sampling",
            if value { "on" } else { "off" },
        )
    }

    pub fn set_spec_draft_ngl(&mut self, value: i64) -> Result<()> {
        self.set_prop("--spec-draft-ngl", &value.to_string())
    }

    pub fn set_prop(&mut self, prop: &str, value: &str) -> Result<()> {
        let prop_name = normalize_prop_name(prop);
        let prop = to_cstring(&prop_name)?;
        let value = to_cstring(value)?;
        unsafe {
            if !ffi::libia_params_set_option(
                self.params.as_ptr(),
                prop.as_ptr(),
                value.as_ptr(),
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api(format!("failed to set option {}", prop_name)));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_int_prop(&mut self, prop: &str, value: i64) -> Result<()> {
        let prop_name = to_cstring(prop)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        unsafe {
            if !ffi::libia_params_set_int(
                self.params.as_ptr(),
                prop_name.as_ptr(),
                value,
                &mut error,
            ) {
                let message = unsafe {
                    take_owned_c_string(error).unwrap_or_else(|_| {
                        format!("failed to set int option {}", prop)
                    })
                };
                return Err(Error::Api(message));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_string_prop(&mut self, prop: &str, value: &str) -> Result<()> {
        let prop_name = to_cstring(prop)?;
        let prop_value = to_cstring(value)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        unsafe {
            if !ffi::libia_params_set_string(
                self.params.as_ptr(),
                prop_name.as_ptr(),
                prop_value.as_ptr(),
                &mut error,
            ) {
                let message = unsafe {
                    take_owned_c_string(error).unwrap_or_else(|_| {
                        format!("failed to set string option {}", prop)
                    })
                };
                return Err(Error::Api(message));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_bool_prop_raw(&mut self, prop: &str, value: bool) -> Result<()> {
        let prop_name = to_cstring(prop)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        unsafe {
            if !ffi::libia_params_set_bool(
                self.params.as_ptr(),
                prop_name.as_ptr(),
                value,
                &mut error,
            ) {
                let message = unsafe {
                    take_owned_c_string(error).unwrap_or_else(|_| {
                        format!("failed to set bool option {}", prop)
                    })
                };
                return Err(Error::Api(message));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_bool_prop(&mut self, prop: &str, value: bool) -> Result<()> {
        let prop_name = normalize_prop_name(prop);
        let prop = to_cstring(&prop_name)?;
        unsafe {
            if !ffi::libia_params_set_bool(
                self.params.as_ptr(),
                prop.as_ptr(),
                value,
                std::ptr::null_mut(),
            ) {
                return Err(Error::Api(format!("failed to set option {}", prop_name)));
            }
        }
        self.dirty = true;
        Ok(())
    }

    pub fn set_tools_json(&mut self, tools_json: &str) -> Result<()> {
        let tools = to_cstring(tools_json)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let ok = unsafe {
            ffi::libia_params_set_tools_json(self.params.as_ptr(), tools.as_ptr(), &mut error)
        };
        if !ok {
            let message = unsafe {
                take_owned_c_string(error)
                    .unwrap_or_else(|_| "failed to set tools json".to_string())
            };
            return Err(Error::Api(message));
        }
        self.dirty = true;
        Ok(())
    }

    pub fn get_prop(&self, prop: &str) -> Result<Option<String>> {
        let prop_name = normalize_prop_name(prop);
        let prop = to_cstring(&prop_name)?;
        let value = unsafe { ffi::libia_params_get_option(self.params.as_ptr(), prop.as_ptr()) };
        Ok(unsafe { read_c_string(value) })
    }

    pub fn list_props(&self) -> Vec<String> {
        let count = unsafe { ffi::libia_params_option_count() };
        let mut out = Vec::new();
        for index in 0..count {
            let mut info = ffi::libia_option_info {
                name: std::ptr::null(),
                value_hint: std::ptr::null(),
                value_hint_2: std::ptr::null(),
                help: std::ptr::null(),
                env: std::ptr::null(),
                kind: ffi::libia_option_kind::LIBIA_OPTION_KIND_VOID,
                is_sampling: false,
                is_spec: false,
                is_preset_only: false,
                is_negated: false,
            };
            if unsafe { ffi::libia_params_option_info(index, &mut info) } {
                if let Some(name) = unsafe { read_c_string(info.name) } {
                    out.push(name);
                }
            }
        }
        out
    }

    pub fn generate_prompt_text(&mut self, prompt: &str) -> Result<String> {
        let runtime = self.ensure_runtime()?;
        let prompt = to_cstring(prompt)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let output = unsafe {
            ffi::libia_runtime_generate_prompt(runtime.as_ptr(), prompt.as_ptr(), -1, &mut error)
        };
        if output.is_null() {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "generation failed".to_string())
            };
            return Err(Error::Api(message));
        }
        unsafe { take_owned_c_string(output) }
    }

    pub fn generate_chat_text(&mut self, prompt: &str) -> Result<String> {
        self.generate_chat_messages_text(&[ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
            reasoning_content: None,
        }])
    }

    pub fn generate_stream_with<F>(&mut self, prompt: &str, on_text: F) -> Result<String>
    where
        F: FnMut(GenerationType, i32, i32),
    {
        self.generate_chat_messages_stream_with(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt.to_string(),
                reasoning_content: None,
            }],
            on_text,
        )
    }

    pub fn generate_chat_messages_text(&mut self, messages: &[ChatMessage]) -> Result<String> {
        let runtime = self.ensure_runtime()?;
        let owned = build_chat_messages(messages)?;
        let ffi_messages = owned.iter().map(|m| m.message).collect::<Vec<_>>();
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let output = unsafe {
            ffi::libia_runtime_generate_chat(
                runtime.as_ptr(),
                ffi_messages.as_ptr(),
                ffi_messages.len(),
                -1,
                &mut error,
            )
        };
        if output.is_null() {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "generation failed".to_string())
            };
            return Err(Error::Api(message));
        }
        unsafe { take_owned_c_string(output) }
    }

    pub fn generate_chat_messages_stream_with<F>(
        &mut self,
        messages: &[ChatMessage],
        on_text: F,
    ) -> Result<String>
    where
        F: FnMut(GenerationType, i32, i32),
    {
        let runtime = self.ensure_runtime()?;
        let owned = build_chat_messages(messages)?;
        let ffi_messages = owned.iter().map(|m| m.message).collect::<Vec<_>>();

        struct StreamState<'a, F: FnMut(GenerationType, i32, i32)> {
            callback: &'a mut F,
            panicked: bool,
        }

        unsafe extern "C" fn trampoline<F: FnMut(GenerationType, i32, i32)>(
            kind: ffi::libia_generation_kind,
            text: *const libc::c_char,
            mode: libc::c_int,
            tps: libc::c_int,
            user_data: *mut c_void,
        ) {
            if text.is_null() || user_data.is_null() {
                return;
            }
            let state = unsafe { &mut *(user_data as *mut StreamState<'_, F>) };
            if state.panicked {
                return;
            }
            if let Ok(chunk) = unsafe { CStr::from_ptr(text).to_str() } {
                let kind = match kind {
                    ffi::libia_generation_kind::LIBIA_GENERATION_KIND_REASONING => {
                        GenerationType::Reasoning(chunk.to_string())
                    }
                    ffi::libia_generation_kind::LIBIA_GENERATION_KIND_TOOL_CALL => {
                        let tool = serde_json::from_str::<serde_json::Value>(chunk)
                            .ok()
                            .map(tool_call_from_value)
                            .unwrap_or_else(|| ToolCall {
                                name: "tool".to_string(),
                                call_id: String::new(),
                                command: "tool".to_string(),
                                args: serde_json::json!({ "raw": chunk }),
                            });
                        GenerationType::CallTool(tool)
                    }
                    _ => GenerationType::Content(chunk.to_string()),
                };
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (state.callback)(kind, mode as i32, tps as i32);
                }))
                .is_err()
                {
                    state.panicked = true;
                }
            }
        }

        let mut on_text = on_text;
        let mut state = StreamState {
            callback: &mut on_text,
            panicked: false,
        };
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let output = unsafe {
            ffi::libia_runtime_generate_chat_stream(
                runtime.as_ptr(),
                ffi_messages.as_ptr(),
                ffi_messages.len(),
                -1,
                Some(trampoline::<F>),
                &mut state as *mut _ as *mut c_void,
                &mut error,
            )
        };

        if state.panicked {
            if !output.is_null() {
                unsafe { ffi::libia_string_free(output) };
            }
            return Err(Error::Api("stream callback panicked".to_string()));
        }

        if output.is_null() {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "generation failed".to_string())
            };
            return Err(Error::Api(message));
        }
        unsafe { take_owned_c_string(output) }
    }

    pub fn last_context_tokens(&mut self) -> Option<i64> {
        let runtime = self.runtime.as_ref()?;
        let value = unsafe { ffi::libia_runtime_last_context_tokens(runtime.as_ptr()) };
        Some(value as i64)
    }

    pub fn last_generated_tokens(&mut self) -> Option<i64> {
        let runtime = self.runtime.as_ref()?;
        let value = unsafe { ffi::libia_runtime_last_generated_tokens(runtime.as_ptr()) };
        Some(value as i64)
    }

    pub fn last_tokens_per_second(&mut self) -> Option<f64> {
        let runtime = self.runtime.as_ref()?;
        let value = unsafe { ffi::libia_runtime_last_tokens_per_second(runtime.as_ptr()) };
        Some(value)
    }

    pub fn tokenize_text(&mut self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        let runtime = self.ensure_runtime()?;
        let text_c = to_cstring(text)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let raw = unsafe {
            ffi::libia_runtime_tokenize_text(
                runtime.as_ptr(),
                text_c.as_ptr(),
                add_special,
                parse_special,
                &mut error,
            )
        };
        if raw.is_null() {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "tokenize failed".to_string())
            };
            return Err(Error::Api(message));
        }
        let csv = unsafe { take_owned_c_string(raw)? };
        Ok(csv
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.trim().parse::<i32>().ok())
            .collect())
    }

    pub fn system_info(&self) -> Option<String> {
        let runtime = self.runtime.as_ref()?;
        unsafe { read_c_string(ffi::libia_runtime_system_info(runtime.as_ptr())) }
    }

    pub fn supports_thinking(&mut self) -> Option<bool> {
        let runtime = self.ensure_runtime().ok()?;
        Some(unsafe { ffi::libia_runtime_supports_thinking(runtime.as_ptr()) })
    }

    pub fn supports_audio(&mut self) -> Option<bool> {
        let runtime = self.ensure_runtime().ok()?;
        Some(unsafe { ffi::libia_runtime_supports_audio(runtime.as_ptr()) })
    }

    pub fn supports_image(&mut self) -> Option<bool> {
        let runtime = self.ensure_runtime().ok()?;
        Some(unsafe { ffi::libia_runtime_supports_image(runtime.as_ptr()) })
    }

    pub fn supports_video(&mut self) -> Option<bool> {
        let runtime = self.ensure_runtime().ok()?;
        Some(unsafe { ffi::libia_runtime_supports_video(runtime.as_ptr()) })
    }

    pub fn media_caps(&mut self) -> Option<HashMap<String, bool>> {
        let runtime = self.ensure_runtime().ok()?;
        let raw = unsafe { ffi::libia_runtime_media_caps_json(runtime.as_ptr()) };
        let json = unsafe { take_owned_c_string(raw).ok()? };
        let parsed: serde_json::Value = serde_json::from_str(&json).ok()?;
        let mut out = HashMap::new();
        let obj = parsed.as_object()?;
        for (key, value) in obj {
            out.insert(key.clone(), value.as_bool().unwrap_or(false));
        }
        Some(out)
    }

    pub fn chat_template_caps(&mut self) -> Option<HashMap<String, bool>> {
        let runtime = self.ensure_runtime().ok()?;
        let raw = unsafe { ffi::libia_runtime_chat_template_caps_json(runtime.as_ptr()) };
        let json = unsafe { take_owned_c_string(raw).ok()? };
        let parsed: serde_json::Value = serde_json::from_str(&json).ok()?;
        let mut out = HashMap::new();
        let obj = parsed.as_object()?;
        for (key, value) in obj {
            out.insert(key.clone(), value.as_bool().unwrap_or(false));
        }
        Some(out)
    }
}

fn tool_call_from_value(value: serde_json::Value) -> ToolCall {
    let name = value
        .get("name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("tool")
        .to_string();
    let raw_args = value
        .get("args")
        .or_else(|| value.get("arguments"))
        .cloned()
        .unwrap_or_else(|| serde_json::Value::Object(Default::default()));
    ToolCall {
        command: value
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(&name)
            .to_string(),
        call_id: value
            .get("call_id")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        args: normalize_args(raw_args),
        name,
    }
}

/// Torna `args` sempre um objeto, custe o que custar. O lado C do lib_ia
/// despeja `tool.arguments` em `args` como um OBJETO quando o JSON parseia,
/// mas como STRING crua quando o modelo escreve um JSON quebrado/truncado
/// (ex.: prompt com aspas nao escapadas). "missing name/prompt" nos
/// executores acontece quando `args` chega como string e o executor faz
/// `args.get("name")` num `Value::String`.
fn normalize_args(args: serde_json::Value) -> serde_json::Value {
    if let Some(text) = args.as_str() {
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
            return parsed;
        }
        // O modelo pode ter embrulhado o JSON com texto/rails soltos.
        if let Some(open) = text.find('{') {
            if let Some(close) = text.rfind('}') {
                if let Ok(parsed) =
                    serde_json::from_str::<serde_json::Value>(&text[open..=close])
                {
                    return parsed;
                }
            }
        }
        return args;
    }
    args
}

impl InferenceEngine for RuntimeLlama {
    fn new(
        model_path: String,
        ctx: usize,
        n_predict: usize,
        n_batch: usize,
        n_ubatch: usize,
        temp: f32,
        top_p: f32,
        top_k: usize,
        mmproj_path: Option<String>,
        loras_path: Option<Vec<String>>,
        ngl: Option<usize>,
        system_prompt: Option<String>,
    ) -> Self {
        Self::try_new(
            model_path,
            ctx,
            n_predict,
            n_batch,
            n_ubatch,
            temp,
            top_p,
            top_k,
            mmproj_path,
            loras_path,
            ngl,
            system_prompt,
        )
        .expect("failed to create RuntimeLlama")
    }

    fn load(&mut self) -> std::result::Result<(), String> {
        self.ensure_runtime()
            .map(|_| ())
            .map_err(|err| err.to_string())
    }

    fn unload(&mut self) {
        RuntimeLlama::unload(self);
    }

    fn generate(&mut self, prompt: String) -> std::result::Result<String, String> {
        self.generate_chat_text(&prompt)
            .map_err(|err| err.to_string())
    }

    fn generate_stream(
        &mut self,
        call: fn(GenerationType, i32, i32),
    ) -> std::result::Result<String, String> {
        let prompt = self
            .get_prop("prompt")
            .ok()
            .and_then(|value| value)
            .unwrap_or_default();
        self.generate_stream_with(&prompt, |chunk, mode, tps| call(chunk, mode, tps))
            .map_err(|err| err.to_string())
    }

    fn charge_system_prompt(&mut self, prompt: String) {
        let _ = self.set_system_prompt(&prompt);
    }

    fn charge_prop(&mut self, prop: String, value: String) {
        let _ = self.set_prop(&prop, &value);
    }

    fn get_all_props(&self) -> Vec<String> {
        self.list_props()
    }
}

impl Drop for RuntimeLlama {
    fn drop(&mut self) {
        self.unload();
    }
}

struct OwnedChatMessage {
    _role: std::ffi::CString,
    _content: std::ffi::CString,
    _reasoning_content: Option<std::ffi::CString>,
    message: ffi::libia_chat_message,
}

fn build_chat_messages(messages: &[ChatMessage]) -> Result<Vec<OwnedChatMessage>> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = to_cstring(&msg.role)?;
        let content = to_cstring(&msg.content)?;
        let reasoning_content = match &msg.reasoning_content {
            Some(value) if !value.is_empty() => Some(to_cstring(value)?),
            _ => None,
        };
        out.push(OwnedChatMessage {
            message: ffi::libia_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
                reasoning_content: reasoning_content
                    .as_ref()
                    .map(|value| value.as_ptr())
                    .unwrap_or(std::ptr::null()),
            },
            _role: role,
            _content: content,
            _reasoning_content: reasoning_content,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn normalize_args_from_string() {
        // C despeja `args` como string quando `tool.arguments` nao parseia.
        let raw = json!({"name":"ask_agent","call_id":"x","command":"ask_agent",
                         "args":"{\"name\": \"lucy\", \"prompt\": \"ok?\"}"});
        let tool = tool_call_from_value(raw);
        assert_eq!(tool.name, "ask_agent");
        assert_eq!(tool.args["name"], "lucy");
        assert_eq!(tool.args["prompt"], "ok?");

        // String quebrada/embrulhada com texto solto: extrai o JSON no meio.
        let raw = json!({"name":"ask_agent","args":"ANTES {\"name\":\"lucas\",\"prompt\":\"x\"} DEPOIS"});
        let tool = tool_call_from_value(raw);
        assert_eq!(tool.args["name"], "lucas");
    }

    #[test]
    fn tool_call_wrapped_arguments() {
        // Alguns templates embrulham os argumentos num objeto `arguments` interno.
        let raw = json!({"name":"ask_agent","args":{"arguments":{"name":"lucy","prompt":"p"}}});
        let tool = tool_call_from_value(raw);
        assert_eq!(tool.args["arguments"]["name"], "lucy");
    }
}
