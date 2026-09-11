use lib_rust::{GenerationType, InferenceEngine, RuntimeLlama};
use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const ANSI_RESET: &str = "\x1b[0m";
const ANSI_THINKING: &str = "\x1b[36m";

#[derive(Copy, Clone, Eq, PartialEq)]
enum StreamPhase {
    Idle,
    Thinking,
    Answer,
}

struct StreamState {
    phase: StreamPhase,
}

impl StreamState {
    fn new() -> Self {
        Self {
            phase: StreamPhase::Idle,
        }
    }
}

static STREAM_STATE: OnceLock<Mutex<StreamState>> = OnceLock::new();

fn stream_state() -> &'static Mutex<StreamState> {
    STREAM_STATE.get_or_init(|| Mutex::new(StreamState::new()))
}

fn parse_arg(flag: &str, default: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            if let Some(value) = args.next() {
                return value;
            }
        }
    }
    default.to_string()
}

fn parse_csv_arg(flag: &str) -> Vec<String> {
    let value = parse_arg(flag, "");
    if value.trim().is_empty() {
        Vec::new()
    } else {
        value
            .split(',')
            .map(|item| item.trim().to_string())
            .filter(|item| !item.is_empty())
            .collect()
    }
}

fn guess_mtp_model(model_path: &str) -> Option<String> {
    let path = Path::new(model_path);
    let dir = path.parent()?;
    let mut candidates: Vec<PathBuf> = fs::read_dir(dir)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with("mtp-") && name.ends_with(".gguf"))
                .unwrap_or(false)
        })
        .collect();
    candidates.sort();
    candidates
        .first()
        .map(|path| path.to_string_lossy().to_string())
}

fn stream_stdout(chunk: GenerationType, _tokens_seconds: i32) {
    let mut state = stream_state().lock().unwrap();
    match chunk {
        GenerationType::Content(text) => {
            if state.phase == StreamPhase::Thinking {
                print!("\n\n");
            }
            state.phase = StreamPhase::Answer;
            print!("{}", text);
            let _ = io::stdout().flush();
        }
        GenerationType::Reasoning(text) => {
            state.phase = StreamPhase::Thinking;
            print!("{}{}{}", ANSI_THINKING, text, ANSI_RESET);
            let _ = io::stdout().flush();
        }
        GenerationType::CallTool(_) => {}
    }
}

fn reset_stream_state() {
    if let Ok(mut state) = stream_state().lock() {
        state.phase = StreamPhase::Idle;
    }
}

fn print_runtime_caps(engine: &mut RuntimeLlama) {
    let media_caps = engine.media_caps().unwrap_or_default();
    let supports_image = engine.supports_image().unwrap_or(false);
    let supports_audio = engine.supports_audio().unwrap_or(false);
    let supports_video = engine.supports_video().unwrap_or(false);
    let use_jinja = engine.use_jinja().unwrap_or(false);
    let reasoning_format = engine
        .reasoning_format()
        .unwrap_or_else(|| "unknown".to_string());
    let enable_reasoning = engine.enable_reasoning().unwrap_or(-999);
    let enable_thinking = engine.enable_thinking().unwrap_or(false);
    let supports_thinking = engine.supports_thinking().unwrap_or(false);
    let caps = engine.chat_template_caps().unwrap_or_default();

    eprintln!("runtime.supports_image = {}", supports_image);
    eprintln!("runtime.supports_audio = {}", supports_audio);
    eprintln!("runtime.supports_video = {}", supports_video);
    if media_caps.is_empty() {
        eprintln!("runtime.media_caps = {{}}");
    } else {
        let mut items: Vec<_> = media_caps.into_iter().collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        let rendered = items
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("runtime.media_caps = {{{}}}", rendered);
    }
    eprintln!("runtime.use_jinja = {}", use_jinja);
    eprintln!("runtime.reasoning_format = {}", reasoning_format);
    eprintln!("runtime.enable_reasoning = {}", enable_reasoning);
    eprintln!("runtime.enable_thinking = {}", enable_thinking);
    eprintln!("runtime.supports_thinking = {}", supports_thinking);

    if caps.is_empty() {
        eprintln!("runtime.chat_template_caps = {{}}");
    } else {
        let mut items: Vec<_> = caps.into_iter().collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        let rendered = items
            .into_iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(", ");
        eprintln!("runtime.chat_template_caps = {{{}}}", rendered);
    }
}

fn main() {
    let model = parse_arg("--model", "/home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf");
    let prompt = parse_arg("--prompt", "O que é Rust language?");
    let reasoning_format = parse_arg("--reasoning-format", "auto");
    let reasoning_mode = parse_arg("--reasoning", "on");
    let n_predict = parse_arg("--n-predict", "-1");
    let spec_type = parse_arg("--spec-type", "draft-mtp");
    let spec_draft_model = parse_arg("--spec-draft-model", "");
    let spec_draft_n_max = parse_arg("--spec-draft-n-max", "4");
    let spec_draft_n_min = parse_arg("--spec-draft-n-min", "0");
    let spec_draft_p_split = parse_arg("--spec-draft-p-split", "0.10");
    let spec_draft_p_min = parse_arg("--spec-draft-p-min", "0.00");
    let spec_draft_backend_sampling = parse_arg("--spec-draft-backend-sampling", "on");
    let spec_draft_ngl = parse_arg("--spec-draft-ngl", "-1");
    let media_files = parse_csv_arg("--image");
    let audio_files = parse_csv_arg("--audio");
    let video_files = parse_csv_arg("--video");
    let media_b64 = parse_csv_arg("--image-b64");
    let audio_b64 = parse_csv_arg("--audio-b64");
    let video_b64 = parse_csv_arg("--video-b64");

    let system_prompt = "Voce e um assistente tecnico. Responda em portugues, com clareza, objetividade e completude.".to_string();

    let guessed_mtp_model = if spec_type == "draft-mtp" && spec_draft_model.trim().is_empty() {
        guess_mtp_model(&model)
    } else {
        None
    };

    let mut engine = RuntimeLlama::new(
        model,
        4096,
        4096,
        512,
        128,
        1.0,
        0.95,
        64,
        None,
        None,
        None,
        Some(system_prompt),
    );

    let _ = engine.set_reasoning_format(&reasoning_format);
    let _ = engine.set_use_jinja(true);
    engine.charge_prop("--fit".to_string(), "off".to_string());
    engine.charge_prop("--n-predict".to_string(), n_predict);
    let _ = engine.set_speculative_type(&spec_type);
    if !spec_draft_model.trim().is_empty() {
        let _ = engine.set_spec_draft_model(&spec_draft_model);
    } else if let Some(guessed) = guessed_mtp_model {
        let _ = engine.set_spec_draft_model(&guessed);
        eprintln!("runtime.spec_draft_model (guessed) = {}", guessed);
    }
    let _ = engine.set_spec_draft_n_max(spec_draft_n_max.parse::<i64>().unwrap_or(4));
    let _ = engine.set_spec_draft_n_min(spec_draft_n_min.parse::<i64>().unwrap_or(0));
    let _ = engine.set_spec_draft_p_split(spec_draft_p_split.parse::<f32>().unwrap_or(0.10));
    let _ = engine.set_spec_draft_p_min(spec_draft_p_min.parse::<f32>().unwrap_or(0.00));
    let _ = engine.set_spec_draft_backend_sampling(matches!(
        spec_draft_backend_sampling.as_str(),
        "1" | "true" | "on" | "yes"
    ));
    let _ = engine.set_spec_draft_ngl(spec_draft_ngl.parse::<i64>().unwrap_or(-1));
    if reasoning_mode != "off" {
        engine.charge_prop("--reasoning-budget".to_string(), "-1".to_string());
        engine.charge_prop(
            "--reasoning-budget-message".to_string(),
            "Responda em portugues de forma direta, completa e objetiva.".to_string(),
        );
    }
    match reasoning_mode.as_str() {
        "on" => {
            let _ = engine.set_enable_reasoning(1);
        }
        "off" => {
            let _ = engine.set_enable_reasoning(0);
        }
        _ => {
            let _ = engine.set_enable_reasoning(-1);
        }
    }

    if let Err(err) = engine.load() {
        eprintln!("load failed: {}", err);
        std::process::exit(1);
    }

    print_runtime_caps(&mut engine);
    eprintln!(
        "runtime.spec_type = {}",
        engine
            .speculative_type()
            .unwrap_or_else(|| "unknown".to_string())
    );
    eprintln!(
        "runtime.spec_draft_model = {}",
        engine
            .spec_draft_model()
            .unwrap_or_else(|| "none".to_string())
    );
    reset_stream_state();

    if !media_files.is_empty() {
        if let Err(err) = engine.add_media_files(&media_files) {
            eprintln!("failed to add image media: {}", err);
            std::process::exit(1);
        }
    }

    if !audio_files.is_empty() {
        if let Err(err) = engine.add_audio_files(&audio_files) {
            eprintln!("failed to add audio media: {}", err);
            std::process::exit(1);
        }
    }

    if !video_files.is_empty() {
        if let Err(err) = engine.add_video_files(&video_files) {
            eprintln!("failed to add video media: {}", err);
            std::process::exit(1);
        }
    }

    if !media_b64.is_empty() {
        if let Err(err) = engine.add_media_base64s(&media_b64) {
            eprintln!("failed to add image base64 media: {}", err);
            std::process::exit(1);
        }
    }

    if !audio_b64.is_empty() {
        if let Err(err) = engine.add_audio_base64s(&audio_b64) {
            eprintln!("failed to add audio base64 media: {}", err);
            std::process::exit(1);
        }
    }

    if !video_b64.is_empty() {
        if let Err(err) = engine.add_video_base64s(&video_b64) {
            eprintln!("failed to add video base64 media: {}", err);
            std::process::exit(1);
        }
    }

    engine.charge_prop("prompt".to_string(), prompt.clone());

    let result = engine.generate_stream(stream_stdout);

    match result {
        Ok(text) => {
            println!();
            drop(text);
        }
        Err(err) => {
            eprintln!("generation failed: {}", err);
            std::process::exit(1);
        }
    }
}
