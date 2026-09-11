use crate::error::{Error, Result};
use crate::ffi;
use crate::helpes::{read_c_string, take_owned_c_string, to_cstring};
use crate::traits::{
    ImageEngine, ImageGenerateConfig, ImageModelConfig, ImageResult, ImageSampleMethod,
    ImageScheduler,
};
use libc::c_char;
use std::ffi::CString;
use std::ptr::NonNull;

struct ImageHandle(NonNull<ffi::libia_image>);

impl ImageHandle {
    fn new(params: &ffi::libia_image_params) -> Result<Self> {
        let ptr = unsafe { ffi::libia_image_create(params as *const _) };
        let ptr = NonNull::new(ptr).ok_or(Error::NullPointer("libia_image_create"))?;
        Ok(Self(ptr))
    }

    fn as_ptr(&self) -> *mut ffi::libia_image {
        self.0.as_ptr()
    }
}

impl Drop for ImageHandle {
    fn drop(&mut self) {
        unsafe { ffi::libia_image_free(self.as_ptr()) };
    }
}

#[derive(Debug)]
struct ImageParamStrings {
    model_path: CString,
    diffusion_model_path: Option<CString>,
    vae_path: Option<CString>,
    clip_l_path: Option<CString>,
    clip_g_path: Option<CString>,
    t5xxl_path: Option<CString>,
    llm_path: Option<CString>,
    backend: Option<CString>,
    params_backend: Option<CString>,
    max_vram: Option<CString>,
}

impl ImageParamStrings {
    fn new(config: &ImageModelConfig) -> Result<Self> {
        Ok(Self {
            model_path: to_cstring(&config.model_path)?,
            diffusion_model_path: optional_cstring(config.diffusion_model_path.as_deref())?,
            vae_path: optional_cstring(config.vae_path.as_deref())?,
            clip_l_path: optional_cstring(config.clip_l_path.as_deref())?,
            clip_g_path: optional_cstring(config.clip_g_path.as_deref())?,
            t5xxl_path: optional_cstring(config.t5xxl_path.as_deref())?,
            llm_path: optional_cstring(config.llm_path.as_deref())?,
            backend: optional_cstring(config.backend.as_deref())?,
            params_backend: optional_cstring(config.params_backend.as_deref())?,
            max_vram: optional_cstring(config.max_vram.as_deref())?,
        })
    }

    fn to_ffi(&self, config: &ImageModelConfig) -> ffi::libia_image_params {
        let mut params = unsafe {
            let mut raw = std::mem::MaybeUninit::<ffi::libia_image_params>::zeroed();
            ffi::libia_image_params_init(raw.as_mut_ptr());
            raw.assume_init()
        };
        params.model_path = self.model_path.as_ptr();
        params.diffusion_model_path = optional_ptr(&self.diffusion_model_path);
        params.vae_path = optional_ptr(&self.vae_path);
        params.clip_l_path = optional_ptr(&self.clip_l_path);
        params.clip_g_path = optional_ptr(&self.clip_g_path);
        params.t5xxl_path = optional_ptr(&self.t5xxl_path);
        params.llm_path = optional_ptr(&self.llm_path);
        params.backend = optional_ptr(&self.backend);
        params.params_backend = optional_ptr(&self.params_backend);
        params.max_vram = optional_ptr(&self.max_vram);
        params.n_threads = config.n_threads;
        params.use_mmap = config.use_mmap;
        params.flash_attn = config.flash_attn;
        params.diffusion_flash_attn = config.diffusion_flash_attn;
        params
    }
}

pub struct StableDiffusionImage {
    config: ImageModelConfig,
    strings: ImageParamStrings,
    handle: Option<ImageHandle>,
}

unsafe impl Send for StableDiffusionImage {}

impl StableDiffusionImage {
    pub fn try_new(config: ImageModelConfig) -> Result<Self> {
        let strings = ImageParamStrings::new(&config)?;
        Ok(Self {
            config,
            strings,
            handle: None,
        })
    }

    fn ensure_handle(&mut self) -> Result<&mut ImageHandle> {
        if self.handle.is_none() {
            let params = self.strings.to_ffi(&self.config);
            self.handle = Some(ImageHandle::new(&params)?);
        }
        Ok(self.handle.as_mut().expect("image handle is initialized"))
    }

    pub fn last_error(&self) -> Option<String> {
        let handle = self.handle.as_ref()?;
        unsafe { read_c_string(ffi::libia_image_last_error(handle.as_ptr())) }
    }
}

impl ImageEngine for StableDiffusionImage {
    fn new(config: ImageModelConfig) -> Self {
        Self::try_new(config).expect("failed to create StableDiffusionImage")
    }

    fn load(&mut self) -> std::result::Result<(), String> {
        let handle = self.ensure_handle().map_err(|err| err.to_string())?;
        let mut error: *mut c_char = std::ptr::null_mut();
        let ok = unsafe { ffi::libia_image_load(handle.as_ptr(), &mut error) };
        if ok {
            return Ok(());
        }
        Err(unsafe {
            take_owned_c_string(error)
                .unwrap_or_else(|_| "failed to load image runtime".to_string())
        })
    }

    fn unload(&mut self) {
        if let Some(handle) = self.handle.as_ref() {
            unsafe { ffi::libia_image_unload(handle.as_ptr()) };
        }
    }

    fn is_loaded(&self) -> bool {
        self.handle
            .as_ref()
            .is_some_and(|handle| unsafe { ffi::libia_image_is_loaded(handle.as_ptr()) })
    }

    fn generate(
        &mut self,
        config: ImageGenerateConfig,
    ) -> std::result::Result<ImageResult, String> {
        self.load()?;
        let prompt = to_cstring(&config.prompt).map_err(|err| err.to_string())?;
        let negative_prompt =
            optional_cstring(config.negative_prompt.as_deref()).map_err(|err| err.to_string())?;
        let output_path = to_cstring(&config.output_path).map_err(|err| err.to_string())?;

        let mut params = unsafe {
            let mut raw = std::mem::MaybeUninit::<ffi::libia_image_generate_params>::zeroed();
            ffi::libia_image_generate_params_init(raw.as_mut_ptr());
            raw.assume_init()
        };
        params.prompt = prompt.as_ptr();
        params.negative_prompt = optional_ptr(&negative_prompt);
        params.output_path = output_path.as_ptr();
        params.width = config.width;
        params.height = config.height;
        params.sample_steps = config.sample_steps;
        params.seed = config.seed;
        params.batch_count = config.batch_count;
        params.cfg_scale = config.cfg_scale;
        params.guidance = config.guidance;
        params.sample_method = to_ffi_sample_method(config.sample_method);
        params.scheduler = to_ffi_scheduler(config.scheduler);

        let handle = self.ensure_handle().map_err(|err| err.to_string())?;
        let mut result = ffi::libia_image_result {
            path: std::ptr::null_mut(),
            width: 0,
            height: 0,
            channels: 0,
            seed: -1,
        };
        let mut error: *mut c_char = std::ptr::null_mut();
        let ok =
            unsafe { ffi::libia_image_generate(handle.as_ptr(), &params, &mut result, &mut error) };
        if !ok {
            return Err(unsafe {
                take_owned_c_string(error)
                    .unwrap_or_else(|_| "failed to generate image".to_string())
            });
        }

        let path = unsafe { take_owned_c_string(result.path).map_err(|err| err.to_string())? };
        result.path = std::ptr::null_mut();
        let out = ImageResult {
            path,
            width: result.width,
            height: result.height,
            channels: result.channels,
            seed: result.seed,
        };
        unsafe { ffi::libia_image_result_free(&mut result) };
        Ok(out)
    }
}

fn optional_cstring(value: Option<&str>) -> Result<Option<CString>> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(to_cstring)
        .transpose()
}

fn optional_ptr(value: &Option<CString>) -> *const c_char {
    value
        .as_ref()
        .map(|value| value.as_ptr())
        .unwrap_or(std::ptr::null())
}

fn to_ffi_sample_method(method: ImageSampleMethod) -> ffi::libia_image_sample_method {
    match method {
        ImageSampleMethod::Euler => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_EULER,
        ImageSampleMethod::EulerA => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_EULER_A,
        ImageSampleMethod::Heun => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_HEUN,
        ImageSampleMethod::Dpm2 => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_DPM2,
        ImageSampleMethod::Dpmpp2sA => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_DPMPP2S_A,
        ImageSampleMethod::Dpmpp2m => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_DPMPP2M,
        ImageSampleMethod::Lcm => ffi::libia_image_sample_method::LIBIA_IMAGE_SAMPLE_LCM,
    }
}

fn to_ffi_scheduler(scheduler: ImageScheduler) -> ffi::libia_image_scheduler {
    match scheduler {
        ImageScheduler::Discrete => ffi::libia_image_scheduler::LIBIA_IMAGE_SCHEDULER_DISCRETE,
        ImageScheduler::Karras => ffi::libia_image_scheduler::LIBIA_IMAGE_SCHEDULER_KARRAS,
        ImageScheduler::Exponential => {
            ffi::libia_image_scheduler::LIBIA_IMAGE_SCHEDULER_EXPONENTIAL
        }
        ImageScheduler::Simple => ffi::libia_image_scheduler::LIBIA_IMAGE_SCHEDULER_SIMPLE,
    }
}
