use crate::subagents::{self, SubagentConfig};
use crate::tools;
use lib_rust::agents::{AgentManager, AgentRole, ConfigAgent, EventSink, ToolExecutor};
use lib_rust::{ChatMessage, GenerationType, RuntimeLlama};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const MAX_TOOL_RESULT: usize = 24_000;
const MAESTRO_AGENT_ID: &str = "drill";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    Solo,
    Agents,
}

impl Mode {
    pub fn toggle(&mut self) {
        *self = match self {
            Mode::Solo => Mode::Agents,
            Mode::Agents => Mode::Solo,
        };
    }
    pub fn label(&self) -> &'static str {
        match self {
            Mode::Solo => "solo",
            Mode::Agents => "maestro",
        }
    }
}

pub enum AgentEvent {
    Stream(GenerationType),
    ToolStart(String),
    ToolResult(String, String),
    Status(String),
}

pub enum WorkerMsg {
    Turn(String),
    SetMode(Mode),
}

/// Cancela a geração nativa em andamento, de qualquer thread — inclusive da
/// UI, sem precisar de acesso mutável à instância do Agent/engine que está
/// bloqueada rodando na thread worker. Chamada ao interromper (Esc) ou
/// direcionar (Ctrl+Enter).
pub fn cancel_generation() {
    RuntimeLlama::cancel_all_generations();
}

/// Ferramentas exclusivas do modo maestro.
const MAESTRO_TOOLS: &[&str] = &["list_agents", "read_agent", "create_agent", "ask_agent"];

fn is_maestro_tool(name: &str) -> bool {
    MAESTRO_TOOLS.contains(&name)
}

pub struct Agent {
    engine: Arc<Mutex<RuntimeLlama>>,
    history: Vec<ChatMessage>,
    project_dir: PathBuf,
    tools_enabled: bool,
    max_tool_rounds: usize,
    ctx_max: usize,
    interrupt: Arc<AtomicBool>,

    mode: Mode,
    /// Fila de agentes: o maestro ("drill", role Root) e os subagentes (role
    /// Agent). No modo agents o turno entra na fila como qualquer outro agente.
    manager: Arc<Mutex<AgentManager>>,
}

impl Agent {
    pub fn new(
        engine: Arc<Mutex<RuntimeLlama>>,
        project_dir: PathBuf,
        tools_enabled: bool,
        ctx_max: usize,
        interrupt: Arc<AtomicBool>,
    ) -> Self {
        let mut manager = AgentManager::new();
        manager.set_interrupt(interrupt.clone());
        manager.add(ConfigAgent {
            agent_id: MAESTRO_AGENT_ID.to_string(),
            role: AgentRole::Root,
            prompt: maestro_prompt(),
            tools_json: tools_enabled.then(|| tools::TOOLS_JSON.to_string()),
        });
        for cfg in subagents::discover_agents(&project_dir) {
            manager.add(cfg.to_config());
        }

        let manager = Arc::new(Mutex::new(manager));
        manager
            .lock()
            .unwrap()
            .set_executor(make_executor(Arc::clone(&manager), project_dir.clone()));
        manager.lock().unwrap().run(engine.clone());

        Self {
            engine,
            history: Vec::new(),
            project_dir,
            tools_enabled,
            max_tool_rounds: 8,
            ctx_max,
            interrupt,
            mode: Mode::Solo,
            manager,
        }
    }

    /// Direciona os eventos de progresso da fila (modo agents) para a UI.
    pub fn set_event_sink(&self, sink: EventSink) {
        self.manager.lock().unwrap().set_event_sink(sink);
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// Tokens reais de contexto usados, quando o runtime reportar (pos-generacao).
    pub fn context_used(&mut self) -> usize {
        let tokens = {
            let mut engine = self.engine.lock().unwrap();
            engine.last_context_tokens()
        };
        match tokens {
            Some(tokens) if tokens > 0 => tokens as usize,
            _ => self.context_estimate(),
        }
    }

    /// Estimativa grosseira de tokens de contexto usados (chars/4).
    pub fn context_estimate(&self) -> usize {
        let chars: usize = self
            .history
            .iter()
            .map(|m| {
                m.content.len()
                    + m.reasoning_content
                        .as_deref()
                        .map(str::len)
                        .unwrap_or(0)
            })
            .sum();
        chars / 4
    }

    pub fn last_stats(&mut self) -> Option<(i64, f64)> {
        let mut engine = self.engine.lock().unwrap();
        let tokens = engine.last_generated_tokens()?;
        let tps = engine.last_tokens_per_second()?;
        Some((tokens, tps))
    }

    fn push_user(&mut self, content: String) {
        self.history.push(ChatMessage {
            role: "user".to_string(),
            content,
            reasoning_content: None,
        });
    }

    fn push_assistant(&mut self, content: String, reasoning: Option<String>) {
        self.history.push(ChatMessage {
            role: "assistant".to_string(),
            content,
            reasoning_content: reasoning,
        });
    }

    fn tool_result_block(call: &tools::ToolCall, result: &str) -> String {
        format!(
            "\n<tool_result> name={} id={}\n{}\n</tool_result>",
            call.name, call.call_id, result
        )
    }

    pub fn run_turn<F>(&mut self, user_input: &str, on_event: F) -> Result<(), String>
    where
        F: FnMut(AgentEvent),
    {
        if self.mode == Mode::Agents {
            return self.run_agent_turn(user_input);
        }
        self.run_solo_turn(user_input, on_event)
    }

    /// Modo agents: o maestro entra na fila do AgentManager como qualquer
    /// agente. A geracao acontece na thread do manager; os eventos chegam via
    /// EventSink setado pela UI.
    fn run_agent_turn(&mut self, user_input: &str) -> Result<(), String> {
        let agent = {
            let manager = self.manager.lock().unwrap();
            manager
                .get(MAESTRO_AGENT_ID)
                .ok_or_else(|| "agente raiz nao registrado".to_string())?
        };
        agent.execute(user_input).map_err(|err| err)?;
        Ok(())
    }

    /// Modo solo: loop direto no engine compartilhado (mesmo runtime da fila).
    fn run_solo_turn<F>(&mut self, user_input: &str, mut on_event: F) -> Result<(), String>
    where
        F: FnMut(AgentEvent),
    {
        self.push_user(user_input.to_string());

        if !self.tools_enabled {
            let _ = self.generate_round(on_event)?;
            self.interrupt.store(false, Ordering::Relaxed);
            return Ok(());
        }

        for round in 0..self.max_tool_rounds {
            let tool_calls = {
                let mut cb = on_event;
                cb(AgentEvent::Status(format!("generating ({})", round + 1)));
                let calls = self.generate_round(|event| cb(event))?;
                on_event = cb;
                calls
            };

            if self.interrupt.swap(false, Ordering::Relaxed) {
                on_event(AgentEvent::Status("interrompido".to_string()));
                return Ok(());
            }

            if tool_calls.is_empty() {
                on_event(AgentEvent::Status("done".to_string()));
                return Ok(());
            }

            for call in tool_calls {
                on_event(AgentEvent::ToolStart(call.name.clone()));

                // Ferramentas do maestro sao executadas pelo mesmo executor da
                // fila (sem spawn separado), usando um snapshot que nao segura
                // o lock do manager durante a chamada.
                if is_maestro_tool(&call.name) {
                    let executor = self.manager.lock().unwrap().executor_snapshot();
                    let result = match executor {
                        Some(exec) => exec(&call.name, call.args.clone())
                            .unwrap_or_else(|err| format!("ERROR: {}", truncate(err, 4_000))),
                        None => "ERROR: executor de ferramentas indisponivel".to_string(),
                    };
                    on_event(AgentEvent::ToolResult(call.name.clone(), result.clone()));
                    self.push_user(Self::tool_result_block(&call, &result));
                    continue;
                }

                let (tx, rx) = std::sync::mpsc::channel();
                let project_dir = self.project_dir.to_string_lossy().to_string();
                let call_clone = call.clone();

                std::thread::spawn(move || {
                    let res = tools::execute(&call_clone, &project_dir);
                    let _ = tx.send(res);
                });

                let result = loop {
                    if self
                        .interrupt
                        .load(std::sync::atomic::Ordering::Relaxed)
                    {
                        on_event(AgentEvent::Status("interrompido".to_string()));
                        return Ok(());
                    }
                    if let Ok(res) = rx.try_recv() {
                        break res;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(10));
                };

                on_event(AgentEvent::ToolResult(call.name.clone(), result.clone()));
                self.push_user(Self::tool_result_block(&call, &result));
            }
        }

        Err("max tool rounds reached without final answer".to_string())
    }

    fn generate_round<F>(&mut self, mut on_event: F) -> Result<Vec<tools::ToolCall>, String>
    where
        F: FnMut(AgentEvent),
    {
        let mut tool_calls: Vec<tools::ToolCall> = Vec::new();
        let mut reasoning = String::new();
        let interrupt = self.interrupt.clone();

        let result = {
            let mut engine = self.engine.lock().unwrap();
            engine.generate_chat_messages_stream_with(&self.history, |chunk, _, _| {
                if interrupt.load(Ordering::Relaxed) {
                    return;
                }
                match &chunk {
                    GenerationType::Content(text) => {
                        on_event(AgentEvent::Stream(GenerationType::Content(text.clone())));
                    }
                    GenerationType::Reasoning(text) => {
                        reasoning.push_str(text);
                        on_event(AgentEvent::Stream(GenerationType::Reasoning(text.clone())));
                    }
                    GenerationType::CallTool(tool) => {
                        tool_calls.push(tools::ToolCall {
                            name: tool.name.clone(),
                            call_id: tool.call_id.clone(),
                            args: tool.args.clone(),
                        });
                        on_event(AgentEvent::Stream(GenerationType::CallTool(tool.clone())));
                    }
                    GenerationType::ToolPreview(_) => {}
                }
            })
        };

        if interrupt.load(Ordering::Relaxed) {
            if let Ok(output) = result {
                self.push_assistant(
                    output,
                    if reasoning.is_empty() {
                        None
                    } else {
                        Some(reasoning)
                    },
                );
            }
            return Ok(Vec::new());
        }

        let output = result.map_err(|err| err.to_string())?;
        self.push_assistant(
            output,
            if reasoning.is_empty() {
                None
            } else {
                Some(reasoning)
            },
        );
        Ok(tool_calls)
    }
}

/// Persona do maestro: unica mensagem system do historico do Root na fila.
fn maestro_prompt() -> String {
    format!(
        "[drill:maestro] Você é o agente MAESTRO de um time de IA. \
         Seus subagentes sao definidos por arquivos `drill.<nome>.agent.json`. \
         Ferramentas exclusivas:\n\
         - list_agents: lista os subagentes disponiveis com nome, descricao, ferramentas e temperatura\n\
         - read_agent <name>: exibe o perfil de um subagente\n\
         - create_agent <name, persona, ...>: cria um novo subagente\n\
         - ask_agent <name, prompt>: delega uma tarefa a um subagente e retorna a resposta\n\n\
         Ao responder o usuario, delegue o trabalho usando ask_agent e resuma os resultados. \
         Mantenha respostas concisas em portugues. {}\n",
        subagents::agents_root().display()
    )
}

/// Executor de ferramentas do time: base (read_file, patch_file, ...) via
/// tools::execute e ferramentas do maestro resolvidas aqui mesmo.
fn make_executor(manager: Arc<Mutex<AgentManager>>, project_dir: PathBuf) -> ToolExecutor {
    let project = project_dir.to_string_lossy().to_string();
    Arc::new(move |name: &str, args: Value| -> Result<String, String> {
        if is_maestro_tool(name) {
            return maestro_dispatch(name, args, &manager, &project_dir)
                .map(|text| truncate(text, MAX_TOOL_RESULT));
        }
        let call = tools::ToolCall {
            name: name.to_string(),
            call_id: String::new(),
            args,
        };
        Ok(tools::execute(&call, &project))
    })
}

fn maestro_dispatch(
    name: &str,
    args: Value,
    manager: &Arc<Mutex<AgentManager>>,
    project_dir: &Path,
) -> Result<String, String> {
    let arg_str = |key: &str| args.get(key).and_then(Value::as_str).map(str::to_string);
    let arg_i64 = |key: &str| args.get(key).and_then(Value::as_i64);
    let arg_f32 = |key: &str| args.get(key).and_then(Value::as_f64).map(|v| v as f32);

    match name {
        "list_agents" => {
            let agents = subagents::discover_agents(project_dir);
            if agents.is_empty() {
                Ok(
                    "Nenhum subagente encontrado. Crie com create_agent ou salve um arquivo \
                     drill.<nome>.agent.json em ~/.drill/agents ou no diretorio do projeto."
                        .to_string(),
                )
            } else {
                list_agents_table(&agents)
            }
        }
        "read_agent" => {
            let name = arg_str("name").ok_or("missing 'name'")?;
            subagents::read_agent(&name, project_dir)
        }
        "create_agent" => {
            let name = arg_str("name").ok_or("missing 'name'")?;
            let persona = arg_str("persona").ok_or("missing 'persona'")?;
            let description = arg_str("description").unwrap_or_default();
            let temp = arg_f32("temp").unwrap_or(0.6);
            let max_tool_rounds = arg_i64("max_tool_rounds")
                .map(|v| v.max(1) as usize)
                .unwrap_or(4);
            let allowed_tools = args
                .get("allowed_tools")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| {
                    vec![
                        "read_file".to_string(),
                        "write_file".to_string(),
                        "patch_file".to_string(),
                        "run_command".to_string(),
                    ]
                });

            let cfg = SubagentConfig {
                name: name.clone(),
                description: description.clone(),
                persona,
                temp: temp.clamp(0.0, 2.0),
                max_tool_rounds,
                allowed_tools,
            };

            let path = subagents::save_agent(&cfg)?;
            // Registrar na fila para ask_agent ja encontrar no mesmo turno.
            manager.lock().unwrap().add(cfg.to_config());
            Ok(format!(
                "Subagente '{}' criado e salvo em {}.\ndescricao: {}\ntemp: {:.1} | rounds: {} | ferramentas: {}",
                cfg.name,
                path.display(),
                description,
                cfg.temp,
                cfg.max_tool_rounds,
                cfg.allowed_tools.join(", ")
            ))
        }
        "ask_agent" => {
            let name = arg_str("name").ok_or("missing 'name'")?;
            let prompt = arg_str("prompt").ok_or("missing 'prompt'")?;
            let reply = manager.lock().unwrap().execute_now(&name, &prompt)?;
            Ok(format!("[{}]{}", name, if reply.is_empty() {
                "(resposta vazia)".to_string()
            } else {
                format!("\n{}", reply)
            }))
        }
        other => Err(format!("unknown maestro tool: {}", other)),
    }
}

fn list_agents_table(agents: &[SubagentConfig]) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(&format!("{:<12} {:<40} {}\n", "NOME", "DESCRICAO", "FERRAMENTAS"));
    out.push_str(&"-".repeat(100));
    out.push('\n');
    for cfg in agents {
        let tools_str = if cfg.allowed_tools.is_empty() {
            "(todas)".to_string()
        } else {
            cfg.allowed_tools.join(", ")
        };
        let desc = if cfg.description.len() > 38 {
            format!("{}...", &cfg.description[..35])
        } else {
            cfg.description.clone()
        };
        out.push_str(&format!(
            "{:<12} {:<40} temp={:.1} rounds={}\n",
            cfg.name, desc, cfg.temp, cfg.max_tool_rounds
        ));
        out.push_str(&format!("{:14}{}\n", "", tools_str));
    }
    Ok(out)
}

fn truncate(text: String, max: usize) -> String {
    if text.len() <= max {
        text
    } else {
        format!("{}...[TRUNCATED {} chars omitted]", &text[..max], text.len() - max)
    }
}