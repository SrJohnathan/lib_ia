use crate::history::History;
use crate::subagents::{self, SubagentConfig};
use crate::tools;
use lib_rust::agents::{AgentManager, AgentRole, ConfigAgent, EventSink, TurnResult};
use lib_rust::RuntimeLlama;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

const MAX_TOOL_RESULT: usize = 24_000;
const MAESTRO_AGENT_ID: &str = "drill";

/// Prompt de reflexão: ao final do trabalho, o agente produz o resumo que é
/// entregue ao maestro e também vai para o historico (kind = "resumo").
const SUMMARY_PROMPT: &str = "[drill] Trabalho concluído. Escreva um RESUMO curto \
     (maximo 4 linhas) do que você fez, arquivos/estado atual e pendencias. \
     Responda apenas com o resumo.";

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

pub enum WorkerMsg {
    Turn(String),
    SetMode(Mode),
}

/// Cancela a geração nativa em andamento, de qualquer thread.
pub fn cancel_generation() {
    RuntimeLlama::cancel_all_generations();
}

/// Ferramentas exclusivas do modo maestro.
const MAESTRO_TOOLS: &[&str] = &["list_agents", "read_agent", "create_agent", "ask_agent"];

fn is_maestro_tool(name: &str) -> bool {
    MAESTRO_TOOLS.contains(&name)
}

/// Frente do drill para a fila de agentes. Solo e agentes compartilham o
/// MESMO caminho de geracao (`AgentManager` == uma thread == RuntimeLlama).
pub struct Agent {
    manager: Arc<AgentManager>,
    ctx_max: usize,
    tools_enabled: bool,
    mode: Mode,
    persona: String,
}

impl Agent {
    pub fn new(
        engine: Arc<std::sync::Mutex<RuntimeLlama>>,
        project_dir: PathBuf,
        tools_enabled: bool,
        ctx_max: usize,
        interrupt: Arc<AtomicBool>,
        persona: String,
    ) -> Self {
        let mut manager = AgentManager::new();
        manager.set_interrupt(interrupt);
        let manager = Arc::new(manager);

        manager.add(ConfigAgent {
            agent_id: MAESTRO_AGENT_ID.to_string(),
            role: AgentRole::Root,
            prompt: persona.clone(),
            tools_json: tools_enabled.then(|| tools::TOOLS_JSON.to_string()),
            // Solo (modo padrao): nao forca tools; apenas agents/maestro.
            force_tools: false,
        });
        for cfg in subagents::discover_agents(&project_dir) {
            manager.add(cfg.to_config());
        }

        manager.set_executor(make_executor(Arc::clone(&manager), project_dir));
        manager.run(engine);

        Self {
            manager,
            ctx_max,
            tools_enabled,
            mode: Mode::Solo,
            persona,
        }
    }

    /// Direciona os eventos de progresso da fila (stream/tool/status) para a UI.
    pub fn set_event_sink(&self, sink: EventSink) {
        self.manager.set_event_sink(sink);
    }

    /// Registra o historico persistente (LanceDB): cada turno concluído vira
    /// uma linha gravada por uma thread separada.
    pub fn set_history(&self, history: Arc<History>) {
        self.manager.set_history(Arc::new(move |row| {
            history.record(row);
        }));
    }

    /// Persona/tools do agente "drill" mudam com o modo; o historico do root
    /// e reiniciado (um root, um contexto por modo).
    pub fn set_mode(&mut self, mode: Mode) {
        let prev = self.mode;
        self.mode = mode;
        if prev != mode {
            let prompt = match mode {
                Mode::Solo => self.persona.clone(),
                Mode::Agents => maestro_prompt(),
            };
            self.manager.add(ConfigAgent {
                agent_id: MAESTRO_AGENT_ID.to_string(),
                role: AgentRole::Root,
                prompt,
                tools_json: self.tools_enabled.then(|| tools::TOOLS_JSON.to_string()),
                force_tools: self.tools_enabled && mode == Mode::Agents,
            });
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

    /// Estimativa grosseira de tokens de contexto (chars/4) a partir do
    /// historico do root — usado apenas como fallback quando o runtime nao
    /// reporta tokens. Nao toca no engine.
    pub fn context_estimate(&self) -> usize {
        let chars: usize = self
            .manager
            .history_of(MAESTRO_AGENT_ID)
            .map(|history| {
                history
                    .iter()
                    .map(|m| {
                        m.content.len()
                            + m.reasoning_content
                                .as_deref()
                                .map(str::len)
                                .unwrap_or(0)
                    })
                    .sum::<usize>()
            })
            .unwrap_or(0);
        chars / 4
    }

    /// Turno unico para solo e agentes: enfileira no root e espera o
    /// TurnResult (stats já lidas na thread da geracao).
    pub fn run_turn(&mut self, user_input: &str) -> Result<TurnResult, String> {
        self.manager.enqueue(MAESTRO_AGENT_ID, user_input)
    }
}

/// Persona do maestro: mensagem system do historico do Root em modo agents.
/// As personas nunca vao via system_prompt do engine (o wrapper C++ o
/// pre-anexa ao historico e um system duplicado estoura excecao nativa),
/// entao solo (persona base) e agents (maestro) injetam sua persona como
/// primeira mensagem do historico.
fn maestro_prompt() -> String {
    format!(
        "[drill:maestro] Você é o agente MAESTRO de um time de IA. \
         Seus subagentes sao definidos por arquivos `drill.<nome>.agent.json`. \
         Ferramentas exclusivas:\n\
         - list_agents: lista os subagentes disponiveis com nome, descricao, ferramentas e temperatura\n\
         - read_agent <name>: exibe o perfil de um subagente\n\
         - create_agent <name, persona, ...>: cria um novo subagente\n\
         - ask_agent <name, prompt>: delega uma tarefa a um subagente e retorna TRABALHO + RESUMO\n\n\
         FLUXO DE TRABALHO EM EQUIPE:\n\
         1. Ao receber um pedido, liste os subagentes (list_agents) e identifique \
         quais especialistas sao necessarios (ex.: front, back, test).\n\
         2. Monte um plano e delegue a CADA especialista uma chamada ask_agent com \
         prompt objetivo, completo e em portugues.\n\
         3. Cada ask_agent responde com [NOME] TRABALHO e RESUMO. Se um trabalho \
         depende de outro, espere o resultado antes de delegar o seguinte.\n\
         4. Ao final, agregue os RESUMOS de todos e responda ao usuario com uma secao \
         final chamada 'RESUMO GERAL' listando por agente o que foi feito, decisoes e proximo passo.\n\n\
         REGRAS DE ACao:\n\
         - Sempre que a tarefa exigir ferramenta, EMITA a chamada de ferramenta imediatamente, \
         sem texto narrativo antes (nao escreva 'vou fazer x' e pare).\n\
         - Ao criar um subagente, chame create_agent com name, description e persona completos \
         logo na primeira rodada.\n\
         - Responda apenas ao final, agregando os resultados.\n\n\
         Mantenha as respostas concisas em portugues. {}\n",
        subagents::agents_root().display()
    )
}

/// Executor de ferramentas do time: base (read_file, patch_file, ...) via
/// tools::execute e ferramentas do maestro resolvidas aqui mesmo. Corre NA
/// thread da fila (a mesma do RuntimeLlama).
fn make_executor(manager: Arc<AgentManager>, project_dir: PathBuf) -> lib_rust::agents::ToolExecutor {
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

#[allow(clippy::too_many_arguments)]
fn maestro_dispatch(
    name: &str,
    args: Value,
    manager: &Arc<AgentManager>,
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
            let description = arg_str("description").unwrap_or_default();
            // Persona opcional: o JSON obrigatorio pequeno (so o nome) e muito
            // mais provavel de ser completado pelo modelo do que um prompt
            // grande embutido na chamada.
            let persona =
                arg_str("persona").filter(|p| !p.is_empty()).unwrap_or_else(|| {
                    if description.is_empty() {
                        format!("Voce e o agente {} do time do drill. Responda de forma concisa e objetiva, em portugues.", name)
                    } else {
                        format!("Voce e o agente {}, especialista em {}. Responda de forma concisa e objetiva, em portugues.", name, description)
                    }
                });
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
            // Registrar na fila para ask_agent ja achar no mesmo turno.
            manager.add(cfg.to_config());
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
            // Trabalho + resumo, executados NA MESMA THREAD (inline, sem
            // re-enfileirar). O resumo é a entrega do agente ao maestro.
            let (work, resumo) = manager
                .run_inline_with_summary(&name, &prompt, SUMMARY_PROMPT)
                .map_err(|err| format!("ask_agent falhou: {}", err))?;
            Ok(format!(
                "[{}]\nTRABALHO\n{}\nFIN_TRABALHO\nRESUMO\n{}",
                name, work.reply, resumo.reply
            ))
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