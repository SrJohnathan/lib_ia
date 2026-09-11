use crate::error::{Error, Result};
use crate::ffi;
use crate::helpes::{normalize_prop_name, take_owned_c_string, to_cstring};
use crate::traits::{SSTEngine, SSTGenerationType};
use libc::c_void;
use std::ffi::CStr;
use std::ptr::NonNull;

struct SstHandle(NonNull<ffi::libia_sst>);

impl SstHandle {
    fn new(model_path: &str) -> Result<Self> {
        let model = to_cstring(model_path)?;
        let ptr = unsafe { ffi::libia_sst_create(model.as_ptr()) };
        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer("libia_sst_create"))?;
        Ok(Self(ptr))
    }

    fn as_ptr(&self) -> *mut ffi::libia_sst {
        self.0.as_ptr()
    }
}

impl Drop for SstHandle {
    fn drop(&mut self) {
        unsafe { ffi::libia_sst_free(self.as_ptr()) };
    }
}

pub struct WhisperSst {
    model_path: String,
    vad_model_path: Option<String>,
    use_gpu: bool,
    flash_attn: bool,
    language: Option<String>,
    initial_prompt: Option<String>,
    translate: bool,
    no_context: bool,
    no_timestamps: bool,
    print_special: bool,
    suppress_nst: bool,
    vad_enabled: bool,
    n_threads: i64,
    n_processors: i64,
    step_ms: i64,
    length_ms: i64,
    keep_ms: i64,
    vad_threshold: f64,
    vad_min_speech_duration_ms: i64,
    vad_min_silence_duration_ms: i64,
    vad_max_speech_duration_s: f64,
    vad_speech_pad_ms: i64,
    vad_samples_overlap: f64,
    runtime: Option<SstHandle>,
    loaded: bool,
}

impl WhisperSst {
    fn default_n_threads() -> i64 {
        std::cmp::min(
            4_i64,
            std::thread::available_parallelism()
                .map(|n| n.get() as i64)
                .unwrap_or(4),
        )
    }

    fn ensure_runtime(&mut self) -> Result<&mut SstHandle> {
        if self.runtime.is_none() {
            self.runtime = Some(SstHandle::new(&self.model_path)?);
            self.loaded = false;
        }
        Ok(self.runtime.as_mut().unwrap())
    }

    fn apply_settings(&mut self) -> Result<()> {
        let runtime = self
            .runtime
            .as_mut()
            .ok_or(Error::Api("sst runtime is not loaded".to_string()))?;

        let set_bool =
            |f: unsafe extern "C" fn(*mut ffi::libia_sst, bool, *mut *mut libc::c_char) -> bool,
             value: bool,
             name: &'static str,
             runtime: &mut SstHandle|
             -> Result<()> {
                let mut error: *mut libc::c_char = std::ptr::null_mut();
                let ok = unsafe { f(runtime.as_ptr(), value, &mut error) };
                if !ok {
                    let message = unsafe {
                        take_owned_c_string(error)
                            .unwrap_or_else(|_| format!("failed to set {}", name))
                    };
                    return Err(Error::Api(message));
                }
                Ok(())
            };

        let set_int =
            |f: unsafe extern "C" fn(*mut ffi::libia_sst, i64, *mut *mut libc::c_char) -> bool,
             value: i64,
             name: &'static str,
             runtime: &mut SstHandle|
             -> Result<()> {
                let mut error: *mut libc::c_char = std::ptr::null_mut();
                let ok = unsafe { f(runtime.as_ptr(), value, &mut error) };
                if !ok {
                    let message = unsafe {
                        take_owned_c_string(error)
                            .unwrap_or_else(|_| format!("failed to set {}", name))
                    };
                    return Err(Error::Api(message));
                }
                Ok(())
            };

        let set_float =
            |f: unsafe extern "C" fn(*mut ffi::libia_sst, f64, *mut *mut libc::c_char) -> bool,
             value: f64,
             name: &'static str,
             runtime: &mut SstHandle|
             -> Result<()> {
                let mut error: *mut libc::c_char = std::ptr::null_mut();
                let ok = unsafe { f(runtime.as_ptr(), value, &mut error) };
                if !ok {
                    let message = unsafe {
                        take_owned_c_string(error)
                            .unwrap_or_else(|_| format!("failed to set {}", name))
                    };
                    return Err(Error::Api(message));
                }
                Ok(())
            };

        set_bool(ffi::libia_sst_set_use_gpu, self.use_gpu, "use_gpu", runtime)?;
        set_bool(
            ffi::libia_sst_set_flash_attn,
            self.flash_attn,
            "flash_attn",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_translate,
            self.translate,
            "translate",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_no_context,
            self.no_context,
            "no_context",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_no_timestamps,
            self.no_timestamps,
            "no_timestamps",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_print_special,
            self.print_special,
            "print_special",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_suppress_nst,
            self.suppress_nst,
            "suppress_nst",
            runtime,
        )?;
        set_bool(
            ffi::libia_sst_set_vad_enabled,
            self.vad_enabled,
            "vad_enabled",
            runtime,
        )?;
        set_int(
            ffi::libia_sst_set_n_threads,
            self.n_threads,
            "n_threads",
            runtime,
        )?;
        set_int(
            ffi::libia_sst_set_n_processors,
            self.n_processors,
            "n_processors",
            runtime,
        )?;
        set_int(ffi::libia_sst_set_step_ms, self.step_ms, "step_ms", runtime)?;
        set_int(
            ffi::libia_sst_set_length_ms,
            self.length_ms,
            "length_ms",
            runtime,
        )?;
        set_int(ffi::libia_sst_set_keep_ms, self.keep_ms, "keep_ms", runtime)?;
        set_float(
            ffi::libia_sst_set_vad_threshold,
            self.vad_threshold,
            "vad_threshold",
            runtime,
        )?;
        set_int(
            ffi::libia_sst_set_vad_min_speech_duration_ms,
            self.vad_min_speech_duration_ms,
            "vad_min_speech_duration_ms",
            runtime,
        )?;
        set_int(
            ffi::libia_sst_set_vad_min_silence_duration_ms,
            self.vad_min_silence_duration_ms,
            "vad_min_silence_duration_ms",
            runtime,
        )?;
        set_float(
            ffi::libia_sst_set_vad_max_speech_duration_s,
            self.vad_max_speech_duration_s,
            "vad_max_speech_duration_s",
            runtime,
        )?;
        set_int(
            ffi::libia_sst_set_vad_speech_pad_ms,
            self.vad_speech_pad_ms,
            "vad_speech_pad_ms",
            runtime,
        )?;
        set_float(
            ffi::libia_sst_set_vad_samples_overlap,
            self.vad_samples_overlap,
            "vad_samples_overlap",
            runtime,
        )?;

        if let Some(language) = &self.language {
            let value = to_cstring(language)?;
            let mut error: *mut libc::c_char = std::ptr::null_mut();
            let ok = unsafe {
                ffi::libia_sst_set_language(runtime.as_ptr(), value.as_ptr(), &mut error)
            };
            if !ok {
                let message = unsafe {
                    take_owned_c_string(error)
                        .unwrap_or_else(|_| "failed to set language".to_string())
                };
                return Err(Error::Api(message));
            }
        }

        if let Some(prompt) = &self.initial_prompt {
            let value = to_cstring(prompt)?;
            let mut error: *mut libc::c_char = std::ptr::null_mut();
            let ok = unsafe {
                ffi::libia_sst_set_initial_prompt(runtime.as_ptr(), value.as_ptr(), &mut error)
            };
            if !ok {
                let message = unsafe {
                    take_owned_c_string(error)
                        .unwrap_or_else(|_| "failed to set initial_prompt".to_string())
                };
                return Err(Error::Api(message));
            }
        }

        if let Some(vad_model) = &self.vad_model_path {
            let value = to_cstring(vad_model)?;
            let mut error: *mut libc::c_char = std::ptr::null_mut();
            let ok = unsafe {
                ffi::libia_sst_set_vad_model_path(runtime.as_ptr(), value.as_ptr(), &mut error)
            };
            if !ok {
                let message = unsafe {
                    take_owned_c_string(error)
                        .unwrap_or_else(|_| "failed to set vad_model_path".to_string())
                };
                return Err(Error::Api(message));
            }
        }

        let model = to_cstring(&self.model_path)?;
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let ok =
            unsafe { ffi::libia_sst_set_model_path(runtime.as_ptr(), model.as_ptr(), &mut error) };
        if !ok {
            let message = unsafe {
                take_owned_c_string(error)
                    .unwrap_or_else(|_| "failed to set model_path".to_string())
            };
            return Err(Error::Api(message));
        }

        Ok(())
    }

    fn stream_audio<F>(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
        finalize: bool,
        mut on_event: F,
    ) -> Result<String>
    where
        F: FnMut(SSTGenerationType),
    {
        struct StreamState<'a, F: FnMut(SSTGenerationType)> {
            callback: &'a mut F,
        }

        unsafe extern "C" fn trampoline<F: FnMut(SSTGenerationType)>(
            kind: ffi::libia_sst_generation_kind,
            text: *const libc::c_char,
            user_data: *mut c_void,
        ) {
            if user_data.is_null() {
                return;
            }
            let state = unsafe { &mut *(user_data as *mut StreamState<'_, F>) };
            let event = match kind {
                ffi::libia_sst_generation_kind::LIBIA_SST_GENERATION_KIND_SPEECH_START => {
                    SSTGenerationType::SpeechStart
                }
                ffi::libia_sst_generation_kind::LIBIA_SST_GENERATION_KIND_SPEECH_END => {
                    SSTGenerationType::SpeechEnd
                }
                _ => {
                    let text = if text.is_null() {
                        String::new()
                    } else {
                        unsafe { CStr::from_ptr(text) }
                            .to_str()
                            .unwrap_or("")
                            .to_string()
                    };
                    SSTGenerationType::Text(text)
                }
            };
            (state.callback)(event);
        }

        let runtime = self.ensure_runtime()?;
        let mut state = StreamState {
            callback: &mut on_event,
        };
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let output = unsafe {
            ffi::libia_sst_push_audio(
                runtime.as_ptr(),
                samples.as_ptr(),
                samples.len(),
                sample_rate as i64,
                channels as i64,
                finalize,
                Some(trampoline::<F>),
                &mut state as *mut _ as *mut c_void,
                &mut error,
            )
        };

        if output.is_null() {
            let message = unsafe {
                take_owned_c_string(error).unwrap_or_else(|_| "sst push audio failed".to_string())
            };
            return Err(Error::Api(message));
        }

        unsafe { take_owned_c_string(output) }
    }
}

impl SSTEngine for WhisperSst {
    fn new(
        model_path: String,
        vad_model_path: Option<String>,
        use_gpu: bool,
        flash_attn: bool,
        language: Option<String>,
    ) -> Self {
        Self {
            model_path,
            vad_model_path,
            use_gpu,
            flash_attn,
            language,
            initial_prompt: None,
            translate: false,
            no_context: true,
            no_timestamps: false,
            print_special: false,
            suppress_nst: false,
            vad_enabled: false,
            n_threads: Self::default_n_threads(),
            n_processors: 1,
            step_ms: 3000,
            length_ms: 10000,
            keep_ms: 200,
            vad_threshold: 0.5,
            vad_min_speech_duration_ms: 250,
            vad_min_silence_duration_ms: 100,
            vad_max_speech_duration_s: f64::MAX,
            vad_speech_pad_ms: 30,
            vad_samples_overlap: 0.1,
            runtime: None,
            loaded: false,
        }
    }

    fn load(&mut self) -> std::result::Result<(), String> {
        if self.runtime.is_none() {
            self.runtime = Some(SstHandle::new(&self.model_path).map_err(|e| e.to_string())?);
        }
        if self.loaded {
            return Ok(());
        }
        self.apply_settings().map_err(|e| e.to_string())?;

        let runtime = self.runtime.as_mut().unwrap();
        let mut error: *mut libc::c_char = std::ptr::null_mut();
        let ok = unsafe { ffi::libia_sst_load(runtime.as_ptr(), &mut error) };
        if !ok {
            let message = unsafe {
                take_owned_c_string(error)
                    .unwrap_or_else(|_| "failed to load sst runtime".to_string())
            };
            return Err(message);
        }
        self.loaded = true;
        Ok(())
    }

    fn unload(&mut self) {
        self.runtime = None;
        self.loaded = false;
    }

    fn reset(&mut self) {
        if let Some(runtime) = self.runtime.as_ref() {
            unsafe { ffi::libia_sst_reset(runtime.as_ptr()) };
        }
    }

    fn clear_audio(&mut self) {
        if let Some(runtime) = self.runtime.as_ref() {
            unsafe { ffi::libia_sst_clear_audio(runtime.as_ptr()) };
        }
    }

    fn set_vad_enabled(&mut self, value: bool) {
        self.vad_enabled = value;
        self.loaded = false;
    }

    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
        finalize: bool,
        call: fn(SSTGenerationType, i32),
    ) -> std::result::Result<String, String> {
        if !self.loaded {
            self.load()?;
        }
        self.stream_audio(samples, sample_rate, channels, finalize, |event| {
            call(event, 0)
        })
        .map_err(|e| e.to_string())
    }

    fn flush(&mut self, call: fn(SSTGenerationType, i32)) -> std::result::Result<String, String> {
        self.push_audio(&[], 16_000, 1, true, call)
    }

    fn charge_prop(&mut self, prop: String, value: String) {
        let name = normalize_prop_name(&prop);
        match name.as_str() {
            "--model" => self.model_path = value,
            "--vad-model" | "--vad-model-path" => self.vad_model_path = Some(value),
            "--language" => self.language = Some(value),
            "--initial-prompt" | "--prompt" => self.initial_prompt = Some(value),
            "--use-gpu" => self.use_gpu = matches!(value.as_str(), "1" | "true" | "on" | "yes"),
            "--flash-attn" => {
                self.flash_attn = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--translate" => self.translate = matches!(value.as_str(), "1" | "true" | "on" | "yes"),
            "--no-context" => {
                self.no_context = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--no-timestamps" => {
                self.no_timestamps = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--print-special" => {
                self.print_special = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--suppress-nst" => {
                self.suppress_nst = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--vad" | "--vad-enabled" => {
                self.vad_enabled = matches!(value.as_str(), "1" | "true" | "on" | "yes")
            }
            "--n-threads" => self.n_threads = value.parse::<i64>().unwrap_or(self.n_threads),
            "--n-processors" => {
                self.n_processors = value.parse::<i64>().unwrap_or(self.n_processors)
            }
            "--step-ms" => self.step_ms = value.parse::<i64>().unwrap_or(self.step_ms),
            "--length-ms" => self.length_ms = value.parse::<i64>().unwrap_or(self.length_ms),
            "--keep-ms" => self.keep_ms = value.parse::<i64>().unwrap_or(self.keep_ms),
            "--vad-threshold" => {
                self.vad_threshold = value.parse::<f64>().unwrap_or(self.vad_threshold)
            }
            "--vad-min-speech-duration-ms" => {
                self.vad_min_speech_duration_ms = value
                    .parse::<i64>()
                    .unwrap_or(self.vad_min_speech_duration_ms)
            }
            "--vad-min-silence-duration-ms" => {
                self.vad_min_silence_duration_ms = value
                    .parse::<i64>()
                    .unwrap_or(self.vad_min_silence_duration_ms)
            }
            "--vad-max-speech-duration-s" => {
                self.vad_max_speech_duration_s = value
                    .parse::<f64>()
                    .unwrap_or(self.vad_max_speech_duration_s)
            }
            "--vad-speech-pad-ms" => {
                self.vad_speech_pad_ms = value.parse::<i64>().unwrap_or(self.vad_speech_pad_ms)
            }
            "--vad-samples-overlap" => {
                self.vad_samples_overlap = value.parse::<f64>().unwrap_or(self.vad_samples_overlap)
            }
            _ => {}
        }
        self.loaded = false;
    }

    fn get_all_props(&self) -> Vec<String> {
        vec![
            "--model".to_string(),
            "--vad-model-path".to_string(),
            "--language".to_string(),
            "--initial-prompt".to_string(),
            "--use-gpu".to_string(),
            "--flash-attn".to_string(),
            "--translate".to_string(),
            "--no-context".to_string(),
            "--no-timestamps".to_string(),
            "--print-special".to_string(),
            "--suppress-nst".to_string(),
            "--n-threads".to_string(),
            "--n-processors".to_string(),
            "--step-ms".to_string(),
            "--length-ms".to_string(),
            "--keep-ms".to_string(),
            "--vad-enabled".to_string(),
            "--vad-threshold".to_string(),
            "--vad-min-speech-duration-ms".to_string(),
            "--vad-min-silence-duration-ms".to_string(),
            "--vad-max-speech-duration-s".to_string(),
            "--vad-speech-pad-ms".to_string(),
            "--vad-samples-overlap".to_string(),
        ]
    }
}

impl Drop for WhisperSst {
    fn drop(&mut self) {
        self.unload();
    }
}
