use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    pub reasoning_content: Option<String>,
}

pub trait InferenceEngine {
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
    ) -> Self;

    fn load(&mut self) -> Result<(), String>;
    fn unload(&mut self);
    fn generate(&mut self, prompt: String) -> Result<String, String>;
    fn generate_stream(&mut self, call: fn(GenerationType, i32, i32)) -> Result<String, String>;
    fn charge_system_prompt(&mut self, prompt: String);
    fn charge_prop(&mut self, prop: String, value: String);
    fn get_all_props(&self) -> Vec<String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum GenerationType {
    Content(String),
    Reasoning(String),
    ToolPreview(ToolPreview),
    CallTool(ToolCall),
}

pub enum SSTGenerationType {
    Text(String),
    SpeechStart,
    SpeechEnd,
}

pub trait SSTEngine {
    fn new(
        model_path: String,
        vad_model_path: Option<String>,
        use_gpu: bool,
        flash_attn: bool,
        language: Option<String>,
    ) -> Self;

    fn start(&mut self) -> Result<(), String> {
        self.reset();
        self.load()
    }

    fn end(&mut self, call: fn(SSTGenerationType, i32)) -> Result<String, String> {
        self.flush(call)
    }

    fn load(&mut self) -> Result<(), String>;
    fn unload(&mut self);
    fn reset(&mut self);
    fn clear_audio(&mut self);
    fn set_vad_enabled(&mut self, value: bool);
    fn push_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: usize,
        finalize: bool,
        call: fn(SSTGenerationType, i32),
    ) -> Result<String, String>;
    fn flush(&mut self, call: fn(SSTGenerationType, i32)) -> Result<String, String>;
    fn charge_prop(&mut self, prop: String, value: String);
    fn get_all_props(&self) -> Vec<String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageSampleMethod {
    Euler,
    EulerA,
    Heun,
    Dpm2,
    Dpmpp2sA,
    Dpmpp2m,
    Lcm,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImageScheduler {
    Discrete,
    Karras,
    Exponential,
    Simple,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageModelConfig {
    pub model_path: String,
    pub diffusion_model_path: Option<String>,
    pub vae_path: Option<String>,
    pub clip_l_path: Option<String>,
    pub clip_g_path: Option<String>,
    pub t5xxl_path: Option<String>,
    pub llm_path: Option<String>,
    pub backend: Option<String>,
    pub params_backend: Option<String>,
    pub max_vram: Option<String>,
    pub n_threads: i32,
    pub use_mmap: bool,
    pub flash_attn: bool,
    pub diffusion_flash_attn: bool,
}

impl ImageModelConfig {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            diffusion_model_path: None,
            vae_path: None,
            clip_l_path: None,
            clip_g_path: None,
            t5xxl_path: None,
            llm_path: None,
            backend: Some("Vulkan".to_string()),
            params_backend: None,
            max_vram: None,
            n_threads: 0,
            use_mmap: true,
            flash_attn: true,
            diffusion_flash_attn: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageGenerateConfig {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub output_path: String,
    pub width: i32,
    pub height: i32,
    pub sample_steps: i32,
    pub seed: i64,
    pub batch_count: i32,
    pub cfg_scale: f32,
    pub guidance: f32,
    pub sample_method: ImageSampleMethod,
    pub scheduler: ImageScheduler,
}

impl ImageGenerateConfig {
    pub fn new(prompt: String, output_path: String) -> Self {
        Self {
            prompt,
            negative_prompt: None,
            output_path,
            width: 512,
            height: 512,
            sample_steps: 20,
            seed: -1,
            batch_count: 1,
            cfg_scale: 7.0,
            guidance: 3.5,
            sample_method: ImageSampleMethod::Euler,
            scheduler: ImageScheduler::Discrete,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageResult {
    pub path: String,
    pub width: i32,
    pub height: i32,
    pub channels: i32,
    pub seed: i64,
}

pub trait ImageEngine {
    fn new(config: ImageModelConfig) -> Self;
    fn load(&mut self) -> Result<(), String>;
    fn unload(&mut self);
    fn is_loaded(&self) -> bool;
    fn generate(&mut self, config: ImageGenerateConfig) -> Result<ImageResult, String>;
}

pub struct Prop {
    pub key: String,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub call_id: String,
    pub command: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolPreview {
    pub path: String,
    pub action: String,
    pub content_chunk: String,
    pub reset: bool,
    pub done: bool,
    pub truncated: bool,
}
