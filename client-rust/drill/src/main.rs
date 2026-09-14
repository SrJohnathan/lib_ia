mod agent;
mod config;
mod subagents;
mod tools;
mod tui;

use agent::{Agent, AgentEvent};
use lib_rust::agents::ManagerEvent;
use lib_rust::{GenerationType, RuntimeLlama};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

pub use config::DEFAULT_MODEL;

struct Cli {
    model: Option<String>,
    ngl: Option<usize>,
    ctx: Option<usize>,
    n_predict: Option<usize>,
    temp: Option<f32>,
    spec: Option<String>,
    tools: Option<bool>,
    dir: Option<PathBuf>,
    persona: Option<String>,
    headless: Option<String>,
}

impl Cli {
    fn parse() -> Self {
        let mut cli = Cli {
            model: None,
            ngl: None,
            ctx: None,
            n_predict: None,
            temp: None,
            spec: None,
            tools: None,
            dir: None,
            persona: None,
            headless: None,
        };

        let mut args: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < args.len() {
            let arg = args[i].clone();
            let next = |args: &mut Vec<String>, i: &mut usize| -> Option<String> {
                if *i + 1 < args.len() {
                    *i += 1;
                    Some(args[*i].clone())
                } else {
                    None
                }
            };
            match arg.as_str() {
                "--model" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.model = Some(v);
                    }
                }
                "--ngl" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.ngl = v.parse().ok();
                    }
                }
                "--ctx" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.ctx = v.parse().ok();
                    }
                }
                "--n-predict" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.n_predict = v.parse().ok();
                    }
                }
                "--temp" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.temp = v.parse().ok();
                    }
                }
                "--spec" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.spec = Some(v);
                    }
                }
                "--no-tools" => cli.tools = Some(false),
                "--headless" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.headless = Some(v);
                    }
                }
                "--dir" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.dir = Some(PathBuf::from(v));
                    }
                }
                "--persona" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.persona = Some(v);
                    }
                }
                "-h" | "--help" => {
                    println!(
                        "drill — agente de codigo para LLMs GGUF\n\n\
                         Uso: drill [opcoes]\n\n\
                         --model <path>    modelo GGUF (default: ver ~/.drill/config.json)\n\
                         --ngl <n>         camadas GPU (default 999)\n\
                         --ctx <n>         contexto (default 8192)\n\
                         --n-predict <n>   tokens por turno (default 1024)\n\
                         --temp <f>        temperatura (default 0.6)\n\
                         --spec <tipo>     draft-mtp | none\n\
                         --no-tools        desabilita tool calls\n\
                         --dir <path>      diretorio de trabalho (default: cwd)\n\
                         --persona <texto> system prompt\n\
                         --headless <txt>  roda um turno sem TUI\n\n\
                         Modos (TUI):\n\
                           solo     conversa com o drill (default)\n\
                           agents   Ctrl+A — drill vira maestro com subagentes\n\
                                    definidos por drill.<nome>.agent.json\n\
                                    (em ~/.drill/agents e na raiz do projeto)\n\n\
                         Config: ~/.drill/config.json (autocriado). CLI sobrescreve config.\n\
                         Variaveis: DRILL_CONFIG=<path> troca o arquivo de config.\n"
                    );
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }
        cli
    }
}

struct Effective {
    model: String,
    ngl: usize,
    ctx: usize,
    n_predict: usize,
    temp: f32,
    spec: String,
    tools: bool,
    dir: PathBuf,
    persona: String,
    headless: Option<String>,
    threads: i64,
    threads_batch: i64,
    flash_attn: String,
    cache_type_k: String,
    cache_type_v: String,
    config_path: String,
}

fn resolve(cli: Cli) -> Effective {
    let cfg_path = config::config_path();
    let config::DrillConfig {
        model,
        tools,
        ngl,
        ctx,
        n_predict,
        temp,
        spec,
        persona,
        dir,
        threads,
        threads_batch,
        flash_attn,
        cache_type_k,
        cache_type_v,
    } = config::load_or_create(&cfg_path).0;

    Effective {
        model: cli.model.clone().unwrap_or(model),
        ngl: cli.ngl.unwrap_or(ngl),
        ctx: cli.ctx.unwrap_or(ctx),
        n_predict: cli.n_predict.unwrap_or(n_predict),
        temp: cli.temp.unwrap_or(temp),
        spec: cli.spec.unwrap_or(spec),
        tools: cli.tools.unwrap_or(tools),
        dir: cli
            .dir
            .or_else(|| dir.map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        persona: cli.persona.clone().unwrap_or(persona),
        headless: cli.headless.clone(),
        threads,
        threads_batch,
        flash_attn,
        cache_type_k,
        cache_type_v,
        config_path: cfg_path.display().to_string(),
    }
}

fn main() {
    let eff = resolve(Cli::parse());

    let mut engine = match build_engine(&eff) {
        Ok(engine) => engine,
        Err(err) => {
            eprintln!("falha ao iniciar runtime: {}", err);
            std::process::exit(1);
        }
    };

    if eff.tools {
        if let Err(err) = engine.set_tools_json(tools::TOOLS_JSON) {
            eprintln!("set_tools_json falhou: {}", err);
            std::process::exit(1);
        }
    }

    println!("[drill] carregando modelo... aguarde;\n  {} ({} camadas GPU, tools {}, config {})",
             eff.model, eff.ngl, if eff.tools { "on" } else { "off" }, eff.config_path);

    let has_thinking = engine.supports_thinking().unwrap_or(false);
    let engine = Arc::new(Mutex::new(engine));
    let interrupt = Arc::new(AtomicBool::new(false));
    let mut agent = Agent::new(engine, eff.dir.clone(), eff.tools, eff.ctx, interrupt.clone());

    let session = tui::SessionInfo::new(
        eff.model.clone(),
        eff.ngl,
        eff.ctx,
        eff.tools,
        has_thinking,
        eff.dir.display().to_string(),
    );
    let mut state = tui::AppState::new(session, interrupt);
    state.push(
        tui::Kind::Muted,
        format!("modelo {} | ngl {} | tools: {} | raciocinio: {} | config {}",
                eff.model, eff.ngl, if eff.tools { "on" } else { "off" },
                if has_thinking { "on" } else { "off" }, eff.config_path),
    );

    if let Some(headless_prompt) = eff.headless.clone() {
        return run_headless(&mut agent, &headless_prompt);
    }

    let (ev_tx, ev_rx) = channel::<tui::UiEvent>();
    let (turn_tx, turn_rx) = channel::<agent::WorkerMsg>();

    thread::spawn(move || worker(&mut agent, turn_rx, ev_tx));

    if let Err(err) = tui::run(state, ev_rx, turn_tx) {
        eprintln!("tui error: {}", err);
        std::process::exit(1);
    }
}

fn run_headless(agent: &mut Agent, prompt: &str) {
    let result = agent.run_turn(prompt, |ev| match ev {
        AgentEvent::Stream(kind) => match kind {
            GenerationType::Content(text) => print!("{}", text),
            GenerationType::Reasoning(text) => {
                eprintln!("\x1b[90m[raciocinio]\x1b[0m {}", text);
            }
            GenerationType::CallTool(tool) => {
                eprintln!("\x1b[35mcall {} args={}\x1b[0m", tool.name, tool.args);
            }
            GenerationType::ToolPreview(_) => {}
        },
        AgentEvent::ToolStart(name) => eprintln!("\x1b[35m[tool] {}\x1b[0m", name),
        AgentEvent::ToolResult(name, result) => {
            eprintln!("\x1b[33m[resultado de {}]\x1b[0m\n{}\n\x1b[33m---\x1b[0m", name, result)
        }
        AgentEvent::Status(text) => eprintln!("\x1b[90m[status] {}\x1b[0m", text),
    });

    match result {
        Ok(()) => {
            if let Some((tokens, tps)) = agent.last_stats() {
                eprintln!("\x1b[90m[{:.1} tok/s, {} tokens]\x1b[0m", tps, tokens);
            }
        }
        Err(err) => eprintln!("\x1b[31merro: {}\x1b[0m", err),
    }
}

fn build_engine(eff: &Effective) -> Result<RuntimeLlama, String> {
    let mut engine = RuntimeLlama::try_new(
        eff.model.clone(),
        eff.ctx,
        eff.n_predict,
        512,
        512,
        eff.temp,
        0.95,
        40,
        None,
        None,
        Some(eff.ngl),
        Some(eff.persona.clone()),
    )
        .map_err(|err| err.to_string())?;

    engine.set_use_jinja(true).map_err(|err| err.to_string())?;
    engine.set_enable_reasoning(1).map_err(|err| err.to_string())?;

    for (prop, value) in [
        ("cache-type-k", &eff.cache_type_k),
        ("cache-type-v", &eff.cache_type_v),
    ] {
        engine.set_string_prop(prop, value).map_err(|err| err.to_string())?;
    }
    engine.set_int_prop("threads", eff.threads).map_err(|err| err.to_string())?;
    engine.set_int_prop("threads-batch", eff.threads_batch).map_err(|err| err.to_string())?;
    engine
        .set_string_prop("flash-attn", &eff.flash_attn)
        .map_err(|err| err.to_string())?;

    if eff.spec == "draft-mtp" {
        engine.set_speculative_type("draft-mtp").map_err(|err| err.to_string())?;
        engine.set_spec_draft_ngl(999).map_err(|err| err.to_string())?;
        engine.set_spec_draft_backend_sampling(false).map_err(|err| err.to_string())?;
        for prop in ["n-gpu-layers-draft", "spec-draft-ngl"] {
            engine.set_int_prop(prop, eff.ngl as i64).map_err(|err| err.to_string())?;
        }
        engine.set_string_prop("spec-draft-device", "Vulkan0").map_err(|err| err.to_string())?;
    } else {
        engine
            .set_speculative_type("none")
            .map_err(|err| err.to_string())?;
    }

    Ok(engine)
}

fn worker(agent: &mut Agent, turns: Receiver<agent::WorkerMsg>, events: Sender<tui::UiEvent>) {
    let send = |event: tui::UiEvent| {
        let _ = events.send(event);
    };

    // Modo agents: eventos da fila do manager chegam direto na UI.
    let sink_events = events.clone();
    agent.set_event_sink(Arc::new(move |ev: ManagerEvent| {
        let ui = match ev {
            ManagerEvent::Stream(kind) => match kind {
                GenerationType::Content(text) => tui::UiEvent::Content(text),
                GenerationType::Reasoning(text) => tui::UiEvent::Reasoning(text),
                GenerationType::CallTool(tool) => tui::UiEvent::ToolCall(tool.name),
                GenerationType::ToolPreview(_) => {
                    tui::UiEvent::Status("tool preview".to_string())
                }
            },
            ManagerEvent::ToolStart(name) => tui::UiEvent::ToolStart(name),
            ManagerEvent::ToolResult(name, result) => {
                tui::UiEvent::ToolResult { name, result }
            }
            ManagerEvent::Status(text) => tui::UiEvent::Status(text),
        };
        let _ = sink_events.send(ui);
    }));

    while let Ok(msg) = turns.recv() {
        let input = match msg {
            agent::WorkerMsg::SetMode(mode) => {
                agent.set_mode(mode);
                send(tui::UiEvent::ModeChanged(mode));
                continue;
            }
            agent::WorkerMsg::Turn(input) => input,
        };

        let result = agent.run_turn(&input, |ev| {
            let ui = match ev {
                AgentEvent::Stream(kind) => match kind {
                    GenerationType::Content(text) => tui::UiEvent::Content(text),
                    GenerationType::Reasoning(text) => tui::UiEvent::Reasoning(text),
                    GenerationType::CallTool(tool) => {
                        tui::UiEvent::ToolCall(tool.name)
                    }
                    GenerationType::ToolPreview(_) => tui::UiEvent::Status("tool preview".to_string()),
                },
                AgentEvent::ToolStart(name) => tui::UiEvent::ToolStart(name),
                AgentEvent::ToolResult(name, result) => {
                    tui::UiEvent::ToolResult { name, result }
                }
                AgentEvent::Status(text) => tui::UiEvent::Status(text),
            };
            send(ui);
        });

        match result {
            Ok(()) => {
                if let Some((tokens, tps)) = agent.last_stats() {
                    send(tui::UiEvent::Stats { tokens, tps });
                }
            }
            Err(err) => send(tui::UiEvent::Err(err)),
        }
        send(tui::UiEvent::ContextUsage {
            used: agent.context_used(),
            max: agent.ctx_max(),
        });
        send(tui::UiEvent::Done);
    }
}