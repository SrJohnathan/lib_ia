use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, I24, Sample, SampleFormat, SizedSample, SupportedStreamConfig, U24};
use lib_rust::{SSTEngine, SSTGenerationType, WhisperSst};
use std::collections::VecDeque;
use std::env;
use std::io;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::thread;
use std::time::Duration;

const RESET: &str = "\x1b[0m";
const CYAN: &str = "\x1b[36;1m";
const GREEN: &str = "\x1b[32;1m";
const YELLOW: &str = "\x1b[33;1m";
const MAGENTA: &str = "\x1b[35;1m";
const RED: &str = "\x1b[31;1m";
const DEFAULT_VAD_MODEL: &str = "/home/johnathan/CLionProjects/lib_ia/vendor/whisper.cpp/models/for-tests-silero-v6.2.0-ggml.bin";

#[derive(Debug, Clone)]
struct Opts {
    model_path: String,
    vad_model_path: Option<String>,
    language: Option<String>,
    initial_prompt: Option<String>,
    use_gpu: bool,
    flash_attn: bool,
    vad_enabled: bool,
    translate: bool,
    no_context: bool,
    no_timestamps: bool,
    print_special: bool,
    suppress_nst: bool,
    n_threads: Option<i64>,
    n_processors: Option<i64>,
    step_ms: Option<i64>,
    length_ms: Option<i64>,
    keep_ms: Option<i64>,
    vad_threshold: Option<f64>,
    vad_min_speech_duration_ms: Option<i64>,
    vad_min_silence_duration_ms: Option<i64>,
    vad_max_speech_duration_s: Option<f64>,
    vad_speech_pad_ms: Option<i64>,
    vad_samples_overlap: Option<f64>,
    device_name: Option<String>,
    list_devices: bool,
    chunk_ms: u64,
}

impl Opts {
    fn parse() -> Result<Self, String> {
        let mut args = env::args().skip(1).peekable();

        let mut opts = Self {
            model_path: env::var("LIBIA_SST_MODEL")
                .or_else(|_| env::var("WHISPER_MODEL"))
                .unwrap_or_else(|_| "/home/johnathan/models/ggml-base.bin".to_string()),
            vad_model_path: env::var("LIBIA_SST_VAD_MODEL")
                .ok()
                .or_else(|| Some(DEFAULT_VAD_MODEL.to_string()))
                .filter(|path| std::path::Path::new(path).exists()),
            language: env::var("LIBIA_SST_LANGUAGE")
                .ok()
                .or_else(|| Some("pt".to_string())),
            initial_prompt: env::var("LIBIA_SST_PROMPT").ok(),
            use_gpu: false,
            flash_attn: false,
            vad_enabled: false,
            translate: false,
            no_context: false,
            no_timestamps: false,
            print_special: false,
            suppress_nst: false,
            n_threads: None,
            n_processors: None,
            step_ms: Some(0),
            length_ms: Some(4000),
            keep_ms: Some(200),
            vad_threshold: None,
            vad_min_speech_duration_ms: None,
            vad_min_silence_duration_ms: None,
            vad_max_speech_duration_s: None,
            vad_speech_pad_ms: None,
            vad_samples_overlap: None,
            device_name: None,
            list_devices: false,
            chunk_ms: 250,
        };

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                "--model" => opts.model_path = take_value("--model", &mut args)?,
                "--vad-model" | "--vad-model-path" => {
                    opts.vad_model_path = Some(take_value(&arg, &mut args)?)
                }
                "--language" => opts.language = Some(take_value("--language", &mut args)?),
                "--initial-prompt" | "--prompt" => {
                    opts.initial_prompt = Some(take_value(&arg, &mut args)?)
                }
                "--device" => opts.device_name = Some(take_value("--device", &mut args)?),
                "--chunk-ms" => {
                    opts.chunk_ms = take_value("--chunk-ms", &mut args)?
                        .parse()
                        .map_err(|_| "invalid --chunk-ms".to_string())?
                }
                "--use-gpu" => opts.use_gpu = true,
                "--flash-attn" => opts.flash_attn = true,
                "--vad" | "--vad-enabled" => opts.vad_enabled = true,
                "--translate" => opts.translate = true,
                "--no-context" => opts.no_context = true,
                "--no-timestamps" => opts.no_timestamps = true,
                "--print-special" => opts.print_special = true,
                "--suppress-nst" => opts.suppress_nst = true,
                "--list-devices" => opts.list_devices = true,
                "--n-threads" => {
                    opts.n_threads = Some(
                        take_value("--n-threads", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --n-threads".to_string())?,
                    )
                }
                "--n-processors" => {
                    opts.n_processors = Some(
                        take_value("--n-processors", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --n-processors".to_string())?,
                    )
                }
                "--step-ms" => {
                    opts.step_ms = Some(
                        take_value("--step-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --step-ms".to_string())?,
                    )
                }
                "--length-ms" => {
                    opts.length_ms = Some(
                        take_value("--length-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --length-ms".to_string())?,
                    )
                }
                "--keep-ms" => {
                    opts.keep_ms = Some(
                        take_value("--keep-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --keep-ms".to_string())?,
                    )
                }
                "--vad-threshold" => {
                    opts.vad_threshold = Some(
                        take_value("--vad-threshold", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-threshold".to_string())?,
                    )
                }
                "--vad-min-speech-duration-ms" => {
                    opts.vad_min_speech_duration_ms = Some(
                        take_value("--vad-min-speech-duration-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-min-speech-duration-ms".to_string())?,
                    )
                }
                "--vad-min-silence-duration-ms" => {
                    opts.vad_min_silence_duration_ms = Some(
                        take_value("--vad-min-silence-duration-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-min-silence-duration-ms".to_string())?,
                    )
                }
                "--vad-max-speech-duration-s" => {
                    opts.vad_max_speech_duration_s = Some(
                        take_value("--vad-max-speech-duration-s", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-max-speech-duration-s".to_string())?,
                    )
                }
                "--vad-speech-pad-ms" => {
                    opts.vad_speech_pad_ms = Some(
                        take_value("--vad-speech-pad-ms", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-speech-pad-ms".to_string())?,
                    )
                }
                "--vad-samples-overlap" => {
                    opts.vad_samples_overlap = Some(
                        take_value("--vad-samples-overlap", &mut args)?
                            .parse()
                            .map_err(|_| "invalid --vad-samples-overlap".to_string())?,
                    )
                }
                other if other.starts_with('-') => return Err(format!("unknown flag: {other}")),
                other => {
                    if opts.model_path == "/home/johnathan/models/ggml-base.bin" {
                        opts.model_path = other.to_string();
                    } else {
                        return Err(format!("unexpected positional argument: {other}"));
                    }
                }
            }
        }

        Ok(opts)
    }
}

fn take_value<I>(flag: &str, args: &mut I) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn print_help() {
    println!(
        "sst-inference

Usage:
  cargo run -p sst-inference -- --model /path/model.bin [options]

Options:
  --model PATH
  --vad-model PATH | --vad-model-path PATH
  --language LANG
  --initial-prompt TEXT | --prompt TEXT
  --use-gpu
  --flash-attn
  --vad | --vad-enabled
  --translate
  --no-context
  --no-timestamps
  --print-special
  --suppress-nst
  --n-threads N
  --n-processors N
  --step-ms N
  --length-ms N
  --keep-ms N
  --vad-threshold F
  --vad-min-speech-duration-ms N
  --vad-min-silence-duration-ms N
  --vad-max-speech-duration-s F
  --vad-speech-pad-ms N
  --vad-samples-overlap F
  --device NAME
  --chunk-ms N
  --list-devices
"
    );
}

fn colorize(color: &str, text: &str) -> String {
    format!("{color}{text}{RESET}")
}

fn high_pass_filter(data: &mut [f32], cutoff: f32, sample_rate: f32) {
    if data.len() < 2 || cutoff <= 0.0 || sample_rate <= 0.0 {
        return;
    }

    let rc = 1.0f32 / (2.0f32 * std::f32::consts::PI * cutoff);
    let dt = 1.0f32 / sample_rate;
    let alpha = dt / (rc + dt);
    let mut y = data[0];

    for i in 1..data.len() {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

fn vad_simple(
    pcmf32: &mut [f32],
    sample_rate: usize,
    last_ms: usize,
    vad_thold: f32,
    freq_thold: f32,
) -> bool {
    let n_samples = pcmf32.len();
    let n_samples_last = (sample_rate * last_ms) / 1000;

    if n_samples_last >= n_samples {
        return false;
    }

    if freq_thold > 0.0 {
        high_pass_filter(pcmf32, freq_thold, sample_rate as f32);
    }

    let mut energy_all = 0.0f32;
    let mut energy_last = 0.0f32;

    for (i, sample) in pcmf32.iter().enumerate() {
        let abs = sample.abs();
        energy_all += abs;
        if i >= n_samples - n_samples_last {
            energy_last += abs;
        }
    }

    energy_all /= n_samples as f32;
    energy_last /= n_samples_last as f32;

    energy_last <= vad_thold * energy_all
}

fn audio_energy(pcmf32: &[f32]) -> f32 {
    if pcmf32.is_empty() {
        return 0.0;
    }

    let mut energy = 0.0f32;
    for sample in pcmf32 {
        energy += sample.abs();
    }
    energy / pcmf32.len() as f32
}

fn print_event(event: SSTGenerationType, _: i32) {
    match event {
        SSTGenerationType::Text(text) => {
            if !text.trim().is_empty() {
                println!("{}", colorize(GREEN, &text));
            }
        }
        SSTGenerationType::SpeechStart | SSTGenerationType::SpeechEnd => {}
    }
}

fn list_devices(host: &cpal::Host) -> Result<(), String> {
    println!("{}", colorize(MAGENTA, "Input devices:"));
    let devices = host.input_devices().map_err(|e| e.to_string())?;
    for device in devices {
        println!(" - {device}");
    }
    Ok(())
}

fn choose_input_device(
    host: &cpal::Host,
    device_name: Option<&str>,
) -> Result<cpal::Device, String> {
    if let Some(name) = device_name {
        let devices = host.input_devices().map_err(|e| e.to_string())?;
        for device in devices {
            if device.to_string() == name {
                return Ok(device);
            }
        }
        return Err(format!("input device not found: {name}"));
    }

    host.default_input_device()
        .ok_or_else(|| "no default input device available".to_string())
}

fn apply_config(engine: &mut WhisperSst, opts: &Opts) {
    engine.charge_prop("--model".to_string(), opts.model_path.clone());
    if let Some(value) = &opts.vad_model_path {
        engine.charge_prop("--vad-model-path".to_string(), value.clone());
    }
    if let Some(value) = &opts.language {
        engine.charge_prop("--language".to_string(), value.clone());
    }
    if let Some(value) = &opts.initial_prompt {
        engine.charge_prop("--initial-prompt".to_string(), value.clone());
    }
    engine.charge_prop("--use-gpu".to_string(), opts.use_gpu.to_string());
    engine.charge_prop("--flash-attn".to_string(), opts.flash_attn.to_string());
    engine.charge_prop("--vad-enabled".to_string(), opts.vad_enabled.to_string());
    engine.charge_prop("--translate".to_string(), opts.translate.to_string());
    engine.charge_prop("--no-context".to_string(), opts.no_context.to_string());
    engine.charge_prop(
        "--no-timestamps".to_string(),
        opts.no_timestamps.to_string(),
    );
    engine.charge_prop(
        "--print-special".to_string(),
        opts.print_special.to_string(),
    );
    engine.charge_prop("--suppress-nst".to_string(), opts.suppress_nst.to_string());
    if let Some(value) = opts.n_threads {
        engine.charge_prop("--n-threads".to_string(), value.to_string());
    }
    if let Some(value) = opts.n_processors {
        engine.charge_prop("--n-processors".to_string(), value.to_string());
    }
    if let Some(value) = opts.step_ms {
        engine.charge_prop("--step-ms".to_string(), value.to_string());
    }
    if let Some(value) = opts.length_ms {
        engine.charge_prop("--length-ms".to_string(), value.to_string());
    }
    if let Some(value) = opts.keep_ms {
        engine.charge_prop("--keep-ms".to_string(), value.to_string());
    }
    if let Some(value) = opts.vad_threshold {
        engine.charge_prop("--vad-threshold".to_string(), value.to_string());
    }
    if let Some(value) = opts.vad_min_speech_duration_ms {
        engine.charge_prop(
            "--vad-min-speech-duration-ms".to_string(),
            value.to_string(),
        );
    }
    if let Some(value) = opts.vad_min_silence_duration_ms {
        engine.charge_prop(
            "--vad-min-silence-duration-ms".to_string(),
            value.to_string(),
        );
    }
    if let Some(value) = opts.vad_max_speech_duration_s {
        engine.charge_prop("--vad-max-speech-duration-s".to_string(), value.to_string());
    }
    if let Some(value) = opts.vad_speech_pad_ms {
        engine.charge_prop("--vad-speech-pad-ms".to_string(), value.to_string());
    }
    if let Some(value) = opts.vad_samples_overlap {
        engine.charge_prop("--vad-samples-overlap".to_string(), value.to_string());
    }
}

fn build_stream_for_format<T>(
    device: &cpal::Device,
    config: &SupportedStreamConfig,
    channels: usize,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, String>
where
    T: Sample + SizedSample,
    f32: FromSample<T>,
{
    let err_fn = move |err| eprintln!("{}", colorize(RED, &format!("stream error: {err}")));
    let stream = device
        .build_input_stream(
            (*config).into(),
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if channels <= 1 {
                    let mut chunk = Vec::with_capacity(data.len());
                    for &sample in data {
                        chunk.push(f32::from_sample(sample));
                    }
                    let _ = tx.send(chunk);
                } else {
                    let frames = data.len() / channels;
                    let mut chunk = Vec::with_capacity(frames);
                    for frame in 0..frames {
                        let mut acc = 0.0f32;
                        for ch in 0..channels {
                            acc += f32::from_sample(data[frame * channels + ch]);
                        }
                        chunk.push(acc / channels as f32);
                    }
                    let _ = tx.send(chunk);
                }
            },
            err_fn,
            None,
        )
        .map_err(|e| e.to_string())?;
    Ok(stream)
}

fn start_audio_stream(
    device: &cpal::Device,
    config: SupportedStreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream, String> {
    let channels = config.channels() as usize;
    match config.sample_format() {
        SampleFormat::I8 => build_stream_for_format::<i8>(device, &config, channels, tx),
        SampleFormat::I16 => build_stream_for_format::<i16>(device, &config, channels, tx),
        SampleFormat::I24 => build_stream_for_format::<I24>(device, &config, channels, tx),
        SampleFormat::I32 => build_stream_for_format::<i32>(device, &config, channels, tx),
        SampleFormat::I64 => build_stream_for_format::<i64>(device, &config, channels, tx),
        SampleFormat::U8 => build_stream_for_format::<u8>(device, &config, channels, tx),
        SampleFormat::U16 => build_stream_for_format::<u16>(device, &config, channels, tx),
        SampleFormat::U24 => build_stream_for_format::<U24>(device, &config, channels, tx),
        SampleFormat::U32 => build_stream_for_format::<u32>(device, &config, channels, tx),
        SampleFormat::U64 => build_stream_for_format::<u64>(device, &config, channels, tx),
        SampleFormat::F32 => build_stream_for_format::<f32>(device, &config, channels, tx),
        SampleFormat::F64 => build_stream_for_format::<f64>(device, &config, channels, tx),
        other => Err(format!("unsupported sample format: {other:?}")),
    }
}

fn main() -> Result<(), String> {
    let opts = Opts::parse()?;
    let host = cpal::default_host();

    if opts.list_devices {
        return list_devices(&host);
    }

    println!(
        "{}",
        colorize(
            MAGENTA,
            &format!("sst-inference using model: {}", opts.model_path)
        )
    );

    let device = choose_input_device(&host, opts.device_name.as_deref())?;
    println!("{}", colorize(MAGENTA, &format!("input device: {device}")));

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    println!(
        "{}",
        colorize(
            MAGENTA,
            &format!(
                "default input config: channels={}, sample_rate={}, format={:?}",
                config.channels(),
                config.sample_rate(),
                config.sample_format()
            )
        )
    );

    let mut engine = WhisperSst::new(
        opts.model_path.clone(),
        opts.vad_model_path.clone(),
        opts.use_gpu,
        opts.flash_attn,
        opts.language.clone(),
    );
    apply_config(&mut engine, &opts);
    engine.set_vad_enabled(opts.vad_enabled);
    engine
        .start()
        .map_err(|err| format!("failed to start SST engine: {err}"))?;

    println!(
        "{}",
        colorize(CYAN, "engine loaded. speak into the microphone.")
    );
    println!("{}", colorize(CYAN, "press ENTER to stop."));
    println!(
        "{}",
        colorize(
            MAGENTA,
            &format!("mode: {}", if opts.vad_enabled { "vad" } else { "normal" })
        )
    );

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = start_audio_stream(&device, config.clone(), tx)?;
    stream.play().map_err(|e| e.to_string())?;

    let sample_rate = config.sample_rate() as usize;
    let voice_ms: usize = 10000;
    let probe_ms: usize = 1250;
    let keep_ms: usize = 250;
    let voice_samples = ((sample_rate as u64 * voice_ms as u64) / 1000).max(1) as usize;
    let probe_samples = ((sample_rate as u64 * probe_ms as u64) / 1000).max(1) as usize;
    let keep_samples = ((sample_rate as u64 * keep_ms as u64) / 1000).max(1) as usize;
    let history_limit = probe_samples
        .saturating_add(keep_samples)
        .max(probe_samples);

    let stop = Arc::new(AtomicBool::new(false));
    let stop_reader = Arc::clone(&stop);
    thread::spawn(move || {
        let mut stdin = String::new();
        let _ = io::stdin().read_line(&mut stdin);
        stop_reader.store(true, Ordering::SeqCst);
    });

    if !opts.vad_enabled {
        let mut started = false;

        loop {
            match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(chunk) => {
                    if !started && !chunk.is_empty() {
                        started = true;
                        println!("{}", colorize(CYAN, "[speech] start"));
                    }
                    if !chunk.is_empty() {
                        engine.push_audio(&chunk, sample_rate as u32, 1, false, print_event)?;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }

            if stop.load(Ordering::SeqCst) {
                break;
            }
        }

        drop(stream);

        let text = engine.end(print_event)?;
        let cleaned = text.trim().to_string();
        if !cleaned.is_empty() {
            println!("{}", colorize(GREEN, &cleaned));
        }
        if started {
            println!("{}", colorize(YELLOW, "[speech] end"));
        }
        println!("{}", colorize(CYAN, "stopped."));
        return Ok(());
    }

    let mut history: VecDeque<f32> = VecDeque::new();
    let mut recording: Vec<f32> = Vec::new();
    let mut speech_active = false;
    let mut speech_hits = 0usize;
    let mut silence_hits = 0usize;
    let mut noise_floor = 0.0015f32;
    let mut last_end_at = std::time::Instant::now() - Duration::from_millis(2000);

    let flush_recording =
        |engine: &mut WhisperSst, recording: &mut Vec<f32>| -> Result<(), String> {
            if recording.is_empty() {
                return Ok(());
            }

            let utterance = std::mem::take(recording);
            let text = engine
                .push_audio(&utterance, sample_rate as u32, 1, true, print_event)
                .map_err(|err| format!("failed to transcribe audio: {err}"))?;
            let cleaned = text.trim().to_string();
            if !cleaned.is_empty() {
                println!("{}", colorize(GREEN, &cleaned));
            }
            println!("{}", colorize(YELLOW, "[speech] end"));
            Ok(())
        };

    loop {
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(mut chunk) => {
                if speech_active {
                    recording.extend_from_slice(&chunk);
                }
                for sample in chunk.drain(..) {
                    history.push_back(sample);
                }
                while history.len() > history_limit {
                    history.pop_front();
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::SeqCst) {
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }

        if stop.load(Ordering::SeqCst) {
            if speech_active {
                flush_recording(&mut engine, &mut recording)?;
                speech_active = false;
            }
            break;
        }

        if speech_active {
            let mut probe: Vec<f32> = history.iter().copied().collect();
            let speech_still_there = vad_simple(
                &mut probe,
                sample_rate,
                probe_ms,
                opts.vad_threshold.unwrap_or(0.6) as f32,
                100.0,
            );

            if speech_still_there {
                silence_hits = 0;
            } else {
                silence_hits += 1;
            }

            if recording.len() >= voice_samples || silence_hits >= 4 {
                flush_recording(&mut engine, &mut recording)?;
                speech_active = false;
                speech_hits = 0;
                silence_hits = 0;
                history.clear();
                last_end_at = std::time::Instant::now();
            }
            continue;
        }

        if history.len() < probe_samples {
            continue;
        }

        let mut probe: Vec<f32> = history.iter().copied().collect();
        let probe_energy = audio_energy(&probe);
        noise_floor = noise_floor * 0.98f32 + probe_energy * 0.02f32;
        let min_energy = noise_floor.max(0.0035f32);
        if probe_energy < min_energy {
            speech_hits = 0;
            continue;
        }

        let speech_now = vad_simple(
            &mut probe,
            sample_rate,
            probe_ms,
            opts.vad_threshold.unwrap_or(0.6) as f32,
            100.0,
        );

        if speech_now {
            speech_hits += 1;
        } else {
            speech_hits = 0;
        }

        if speech_hits >= 2 && !speech_active && last_end_at.elapsed() >= Duration::from_millis(700)
        {
            speech_active = true;
            silence_hits = 0;
            recording.clear();
            recording.extend(history.iter().copied());
            println!("{}", colorize(CYAN, "[speech] start"));
            if recording.len() >= voice_samples {
                flush_recording(&mut engine, &mut recording)?;
                speech_active = false;
                speech_hits = 0;
                history.clear();
                last_end_at = std::time::Instant::now();
            }
        }
    }

    drop(stream);

    if speech_active {
        flush_recording(&mut engine, &mut recording)?;
    } else if !recording.is_empty() {
        flush_recording(&mut engine, &mut recording)?;
    }

    println!("{}", colorize(CYAN, "stopped."));
    Ok(())
}
