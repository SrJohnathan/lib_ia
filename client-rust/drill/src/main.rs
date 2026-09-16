mod agent;
mod config;
mod history;
mod provider;
mod subagents;
mod tools;
mod tui;

use agent::Agent;
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
    mmproj: Option<String>,
    ngl: Option<usize>,
    ctx: Option<usize>,
    n_predict: Option<usize>,
    temp: Option<f32>,
    spec: Option<String>,
    tools: Option<bool>,
    dir: Option<PathBuf>,
    persona: Option<String>,
    headless: Option<String>,
    mode: Option<String>,
}

impl Cli {
    fn parse() -> Self {
        let mut cli = Cli {
            model: None,
            mmproj: None,
            ngl: None,
            ctx: None,
            n_predict: None,
            temp: None,
            spec: None,
            tools: None,
            dir: None,
            persona: None,
            headless: None,
            mode: None,
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
                "--mmproj" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.mmproj = Some(v);
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
                "--mode" => {
                    if let Some(v) = next(&mut args, &mut i) {
                        cli.mode = Some(v);
                    }
                }
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
                         --mmproj <path>   projecao multi-modal (llava/qwen2vl; opcional)\n\
                         --ngl <n>         camadas GPU (default 999)\n\
                         --ctx <n>         contexto (default 8192)\n\
                         --n-predict <n>   tokens por turno (default 1024)\n\
                         --temp <f>        temperatura (default 0.6)\n\
                         --spec <tipo>     draft-mtp | none\n\
                         --no-tools        desabilita tool calls\n\
                         --dir <path>      diretorio de trabalho (default: cwd)\n\
                         --persona <texto> system prompt\n\
                         --mode <m>        solo | agents (default solo)\n\
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
    mmproj: String,
    ngl: usize,
    ctx: usize,
    n_predict: usize,
    temp: f32,
    spec: String,
    tools: bool,
    dir: PathBuf,
    persona: String,
    headless: Option<String>,
    mode: String,
    threads: i64,
    threads_batch: i64,
    flash_attn: String,
    cache_type_k: String,
    cache_type_v: String,
    config_path: String,
    /// Provider ativo e remoto (cloud): sem modelo local obrigatorio.
    remote_only: bool,
}

fn resolve(cli: Cli) -> Effective {
    let cfg_path = config::config_path();
    let cfg = config::load_or_create(&cfg_path).0;

    let selected_local = cfg
        .providers
        .local
        .iter()
        .find(|p| Some(&p.id) == cfg.provider_selected_uuid.as_ref())
        .or_else(|| cfg.providers.local.first())
        .cloned();
    let remote_only = cfg
        .provider_selected_uuid
        .as_ref()
        .map(|id| cfg.providers.cloud.iter().any(|p| &p.id == id))
        .unwrap_or(false);

    let fallback = config::LocalProvider {
        model: if cfg.model.trim().is_empty() {
            config::DEFAULT_MODEL.to_string()
        } else {
            cfg.model.clone()
        },
        ..config::LocalProvider::default()
    };
    let local = selected_local.unwrap_or(fallback);

    Effective {
        model: cli.model.clone().unwrap_or(local.model),
        mmproj: cli.mmproj.clone().unwrap_or(local.mmproj),
        ngl: cli.ngl.unwrap_or(local.ngl),
        ctx: cli.ctx.unwrap_or(local.ctx),
        n_predict: cli.n_predict.unwrap_or(local.n_predict),
        temp: cli.temp.unwrap_or(local.temp),
        spec: cli.spec.unwrap_or(local.spec),
        tools: cli.tools.unwrap_or(cfg.tools),
        dir: cli
            .dir
            .or_else(|| cfg.dir.clone().map(PathBuf::from))
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        persona: cli.persona.clone().unwrap_or(cfg.persona),
        headless: cli.headless.clone(),
        mode: cli.mode.clone().unwrap_or_else(|| "solo".to_string()),
        threads: cfg.threads,
        threads_batch: cfg.threads_batch,
        flash_attn: cfg.flash_attn,
        cache_type_k: cfg.cache_type_k,
        cache_type_v: cfg.cache_type_v,
        config_path: cfg_path.display().to_string(),
        remote_only,
    }
}

fn main() {
    let eff = resolve(Cli::parse());
    let cfg = config::load_or_create(std::path::Path::new(&eff.config_path)).0;

    let remote_only = eff.remote_only;
    // TUI + provider local com arquivo ausente (ou mmproj apontando para um
    // arquivo inexistente): nao aborta — o modal pede o caminho no boot.
    let setup_tui = !remote_only
        && eff.headless.is_none()
        && (eff.model.trim().is_empty()
            || !std::path::Path::new(&eff.model).exists()
            || (!eff.mmproj.trim().is_empty() && !std::path::Path::new(eff.mmproj.trim()).exists()));

    let mut engine = if setup_tui {
        eprintln!(
            "[drill] modelo local nao encontrado: {} — o TUI vai pedir o caminho do GGUF",
            eff.model
        );
        None
    } else {
        match build_engine(&eff) {
            Ok(engine) => Some(engine),
            Err(err) => {
                if remote_only {
                    eprintln!(
                        "[drill] provider remoto ativo; sem modelo local, seguindo sem RuntimeLlama: {}",
                        err
                    );
                    None
                } else {
                    eprintln!("falha ao iniciar runtime: {}", err);
                    std::process::exit(1);
                }
            }
        }
    };

    if let Some(ref mut engine) = engine {
        if eff.tools {
            if let Err(err) = engine.set_tools_json(tools::TOOLS_JSON) {
                eprintln!("set_tools_json falhou: {}", err);
                std::process::exit(1);
            }
        }
    }

    if engine.is_some() {
        println!("[drill] carregando modelo... aguarde;\n  {} ({} camadas GPU, tools {}, config {})",
                 eff.model, eff.ngl, if eff.tools { "on" } else { "off" }, eff.config_path);
    } else {
        eprintln!(
            "[drill] runtime local nao iniciado no boot (modelo ausente ou provider remoto)."
        );
    }

    let has_thinking = engine
        .as_mut()
        .and_then(|e| e.supports_thinking())
        .unwrap_or(false);
    let engine = engine.map(|e| Arc::new(Mutex::new(e)));
    let interrupt = Arc::new(AtomicBool::new(false));

    // Provider compartilhado (TUI escolhe / worker executa). Subagentes não usam.
    let provider_store = Arc::new(provider::ProviderStore::new(&cfg, eff.config_path.clone().into()));

    let mut agent = Agent::new(
        engine,
        eff.dir.clone(),
        eff.tools,
        eff.ctx,
        interrupt.clone(),
        eff.persona.clone(),
        Arc::clone(&provider_store),
        eff.temp,
        eff.n_predict,
    );

    match eff.mode.as_str() {
        "agents" | "maestro" => agent.set_mode(agent::Mode::Agents),
        _ => {}
    }

    // Sessao: id unico por execucao, usado como chave da lista `/session`.
    let session_id = format!(
        "{}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        std::process::id()
    );
    agent.set_session(session_id.clone());

    let history = Arc::new(history::History::new(subagents::config_root()));
    agent.set_history(Arc::clone(&history));

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
        format!("modelo {} | ngl {} | tools: {} | raciocinio: {} | provider {} | config {}",
                eff.model, eff.ngl, if eff.tools { "on" } else { "off" },
                if has_thinking { "on" } else { "off" }, agent.provider_summary(), eff.config_path),
    );

    if setup_tui {
        state.push(
            tui::Kind::Err,
            format!(
                "modelo GGUF nao encontrado: {} — informe o caminho no modal abaixo.",
                eff.model
            ),
        );
        state.open_model_setup(eff.model.clone(), eff.mmproj.clone());
    }

    // Canal de permissao: tools sensiveis (ex.: fetch_url) bloqueiam a thread
    // de geracao ate o usuario responder aqui.
    let permission_rx = tools::install_permission_channel();

    if let Some(headless_prompt) = eff.headless.clone() {
        agent.set_event_sink(Arc::new(|ev: ManagerEvent| match ev {
            ManagerEvent::Stream(kind) => match kind {
                GenerationType::Content(text) => print!("{}", text),
                GenerationType::Reasoning(text) => {
                    eprintln!("\x1b[90m[raciocinio]\x1b[0m {}", text);
                }
                GenerationType::CallTool(tool) => {
                    eprintln!("\x1b[35mcall {} args={}\x1b[0m", tool.name, tool.args);
                }
                GenerationType::ToolPreview(_) => {}
            },
            ManagerEvent::ToolStart(name) => eprintln!("\x1b[35m[tool] {}\x1b[0m", name),
            ManagerEvent::ToolResult(name, result) => {
                eprintln!("\x1b[33m[resultado de {}]\x1b[0m\n{}\n\x1b[33m---\x1b[0m", name, result)
            }
            ManagerEvent::Status(text) => eprintln!("\x1b[90m[status] {}\x1b[0m", text),
            ManagerEvent::AgentWorking(name) => {
                eprintln!("\x1b[36mtrabalhando: {}\x1b[0m", name);
            }
        }));
        spawn_headless_permission_responder(permission_rx);
        return run_headless(&mut agent, &headless_prompt);
    }

    let (ev_tx, ev_rx) = channel::<tui::UiEvent>();
    let (turn_tx, turn_rx) = channel::<agent::WorkerMsg>();

    // Snapshot inicial dos providers para o overlay `/provider`.
    {
        let (entries, selected) = provider_store.snapshot();
        let _ = ev_tx.send(tui::UiEvent::Providers { entries, selected });
    }

    let engine_params = EngineParams::from_eff(&eff);
    thread::spawn(move || worker(&mut agent, turn_rx, ev_tx, engine_params));

    if let Err(err) = tui::run(state, ev_rx, turn_tx, history, session_id, permission_rx) {
        eprintln!("tui error: {}", err);
        std::process::exit(1);
    }
}

/// Sem UI (modo `--headless`): pede permissao via stdin (y/N) numa thread a
/// parte, ja que a thread principal fica bloqueada em `agent.run_turn`.
fn spawn_headless_permission_responder(rx: Receiver<tools::PermissionRequest>) {
    thread::spawn(move || {
        for req in rx {
            eprint!("\n\x1b[33m[permissao]\x1b[0m {} quer acessar: {} — permitir? [y/N] ", req.tool, req.detail);
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            let allow = matches!(line.trim().to_lowercase().as_str(), "y" | "yes" | "s" | "sim");
            let _ = req.reply.send(allow);
        }
    });
}

fn run_headless(agent: &mut Agent, prompt: &str) {
    match agent.run_turn(prompt) {
        Ok((turn, notices)) => {
            for notice in notices {
                eprintln!("\x1b[90m[imagem] {}\x1b[0m", notice);
            }
            println!();
            if let (Some(tokens), Some(tps)) = (turn.tokens, turn.tps) {
                eprintln!("\x1b[90m[{:.1} tok/s, {} tokens]\x1b[0m", tps, tokens);
            }
        }
        Err(err) => eprintln!("\x1b[31merro: {}\x1b[0m", err),
    }
}

/// Parametros de runtime locais compartilhados entre o boot (build_engine) e o
/// worker (SetLocalModel), quando o modelo e configurado via modal depois.
pub(crate) struct EngineParams {
    pub ngl: usize,
    pub ctx: usize,
    pub n_predict: usize,
    pub temp: f32,
    pub spec: String,
    pub threads: i64,
    pub threads_batch: i64,
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
}

impl EngineParams {
    fn from_eff(eff: &Effective) -> Self {
        Self {
            ngl: eff.ngl,
            ctx: eff.ctx,
            n_predict: eff.n_predict,
            temp: eff.temp,
            spec: eff.spec.clone(),
            threads: eff.threads,
            threads_batch: eff.threads_batch,
            flash_attn: eff.flash_attn.clone(),
            cache_type_k: eff.cache_type_k.clone(),
            cache_type_v: eff.cache_type_v.clone(),
        }
    }
}

fn build_engine(eff: &Effective) -> Result<RuntimeLlama, String> {
    build_runtime(&eff.model, &eff.mmproj, &EngineParams::from_eff(eff))
}

/// Constroi um `RuntimeLlama` para o modelo GGUF (com mmproj opcional). Usado
/// tanto no boot quanto no worker quando o modelo e definido pelo modal.
pub(crate) fn build_runtime(
    model: &str,
    mmproj: &str,
    p: &EngineParams,
) -> Result<RuntimeLlama, String> {
    let model_path = std::path::Path::new(model);
    if !model_path.exists() {
        return Err(format!(
            "modelo nao encontrado: {} — verifique o caminho no config.json (ou use --model)",
            model
        ));
    }
    let mmproj = mmproj.trim();
    if !mmproj.is_empty() && !std::path::Path::new(mmproj).exists() {
        return Err(format!("arquivo mmproj nao encontrado: {}", mmproj));
    }
    let mmproj_opt = if mmproj.is_empty() {
        None
    } else {
        Some(mmproj.to_string())
    };
    let mut engine = RuntimeLlama::try_new(
        model.to_string(),
        p.ctx,
        p.n_predict,
        128,
        128,
        p.temp,
        0.95,
        40,
        mmproj_opt,
        None,
        Some(p.ngl),
        None, // system prompt vem do historico de cada agente (sem excecao nativa)
    )
        .map_err(|err| err.to_string())?;

    engine.set_use_jinja(true).map_err(|err| err.to_string())?;
    engine.set_enable_reasoning(1).map_err(|err| err.to_string())?;

    for (prop, value) in [
        ("cache-type-k", &p.cache_type_k),
        ("cache-type-v", &p.cache_type_v),
    ] {
        engine.set_string_prop(prop, value).map_err(|err| err.to_string())?;
    }
    engine.set_int_prop("threads", p.threads).map_err(|err| err.to_string())?;
    engine.set_int_prop("threads-batch", p.threads_batch).map_err(|err| err.to_string())?;
    engine
        .set_string_prop("flash-attn", &p.flash_attn)
        .map_err(|err| err.to_string())?;

    if p.spec == "draft-mtp" {
        engine.set_speculative_type("draft-mtp").map_err(|err| err.to_string())?;
        engine.set_spec_draft_ngl(999).map_err(|err| err.to_string())?;
        engine.set_spec_draft_backend_sampling(false).map_err(|err| err.to_string())?;
        for prop in ["n-gpu-layers-draft", "spec-draft-ngl"] {
            engine.set_int_prop(prop, p.ngl as i64).map_err(|err| err.to_string())?;
        }
        engine.set_string_prop("spec-draft-device", "Vulkan0").map_err(|err| err.to_string())?;
    } else {
        engine
            .set_speculative_type("none")
            .map_err(|err| err.to_string())?;
    }

    Ok(engine)
}

fn worker(
    agent: &mut Agent,
    turns: Receiver<agent::WorkerMsg>,
    events: Sender<tui::UiEvent>,
    engine_params: EngineParams,
) {
    let send = |event: tui::UiEvent| {
        let _ = events.send(event);
    };

    // Eventos de progresso da fila (stream/tool/status) chegam direto na UI,
    // tanto em solo quanto em agents (mesmo caminho de geracao).
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
            ManagerEvent::AgentWorking(name) => tui::UiEvent::AgentWorking(name),
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
            agent::WorkerMsg::SetSession(id) => {
                agent.set_session(id);
                continue;
            }
            agent::WorkerMsg::SetProvider(uuid) => {
                match agent.switch_provider(uuid) {
                    Ok(summary) => send(tui::UiEvent::Muted(format!("[provider] {}", summary))),
                    Err(err) => send(tui::UiEvent::Err(err)),
                }
                let (entries, selected) = agent.provider.snapshot();
                send(tui::UiEvent::Providers { entries, selected });
                continue;
            }
            agent::WorkerMsg::ListModels => {
                match agent.remote_models() {
                    Ok(models) => {
                        let cloud = agent.provider.selected_cloud();
                        let (current, effort) = cloud
                            .map(|c| (c.model, c.reasoning_effort))
                            .unwrap_or_default();
                        send(tui::UiEvent::Models {
                            models,
                            current,
                            effort,
                        });
                    }
                    Err(err) => send(tui::UiEvent::Err(err)),
                }
                continue;
            }
            agent::WorkerMsg::SetModelAndEffort(model, effort) => {
                match agent.set_model_and_effort(model, effort) {
                    Ok(summary) => send(tui::UiEvent::Muted(format!("[provider] {}", summary))),
                    Err(err) => send(tui::UiEvent::Err(err)),
                }
                let (entries, selected) = agent.provider.snapshot();
                send(tui::UiEvent::Providers { entries, selected });
                continue;
            }
            agent::WorkerMsg::AddCloudProvider {
                kind,
                base_url,
                model,
                api_key,
            } => {
                match agent.provider.add_cloud(kind, base_url, model, api_key) {
                    Ok(summary) => {
                        send(tui::UiEvent::Muted(format!("[provider] adicionado: {}", summary)));
                    }
                    Err(err) => send(tui::UiEvent::Err(err)),
                }
                let (entries, selected) = agent.provider.snapshot();
                send(tui::UiEvent::Providers { entries, selected });
                continue;
            }
            agent::WorkerMsg::SetLocalModel { model, mmproj } => {
                // Constroi o engine local com o caminho informado no modal e o
                // instala na fila (se ainda nao estava rodando).
                match agent.set_local_model(model, mmproj, &engine_params) {
                    Ok((model, has_thinking)) => {
                        send(tui::UiEvent::Muted(format!(
                            "[modelo] local configurado: {} (raciocinio {})",
                            model,
                            if has_thinking { "on" } else { "off" }
                        )));
                        send(tui::UiEvent::ModelReady {
                            model,
                            thinking: has_thinking,
                        });
                        let (entries, selected) = agent.provider.snapshot();
                        send(tui::UiEvent::Providers { entries, selected });
                    }
                    Err(err) => send(tui::UiEvent::Err(format!("[modelo] {}", err))),
                }
                continue;
            }
            agent::WorkerMsg::Turn(input) => input,
        };

        let result = agent.run_turn(&input);

        match result {
            Ok((turn, notices)) => {
                for notice in notices {
                    send(tui::UiEvent::Muted(format!("[imagem] {}", notice)));
                }
                if let (Some(tokens), Some(tps)) = (turn.tokens, turn.tps) {
                    send(tui::UiEvent::Stats { tokens, tps });
                }
                let used = turn
                    .context_tokens
                    .filter(|&t| t > 0)
                    .map(|t| t as usize)
                    .unwrap_or_else(|| agent.context_estimate());
                send(tui::UiEvent::ContextUsage {
                    used,
                    max: agent.ctx_max(),
                });
            }
            Err(err) => send(tui::UiEvent::Err(err)),
        }
        send(tui::UiEvent::Done);
    }
}