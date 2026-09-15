use crate::agents::agent::{AgentRole, ConfigAgent, TurnResult};
use crate::agents::agent_manager::{AgentManager, EventSink, HistoryListener, ToolExecutor};
use crate::agents::subagents::{self, SubagentConfig};
use crate::runtime::RuntimeLlama;
use crate::tools;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

pub const MAESTRO_AGENT_ID: &str = "drill";
pub const MAX_TOOL_RESULT: usize = 24_000;

/// Prompt de reflexão: ao final do trabalho, o agente produz o resumo que é
/// entregue ao maestro e também vai para o historico (kind = "resumo").
pub const SUMMARY_PROMPT: &str = "[drill] Trabalho concluído. Escreva um RESUMO curto \
     (maximo 4 linhas) do que você fez, arquivos/estado atual e pendencias. \
     Responda apenas com o resumo.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentMode {
    Solo,
    Agents,
}

impl AgentMode {
    pub fn toggle(&mut self) {
        *self = match self {
            AgentMode::Solo => AgentMode::Agents,
            AgentMode::Agents => AgentMode::Solo,
        };
    }

    pub fn label(&self) -> &'static str {
        match self {
            AgentMode::Solo => "solo",
            AgentMode::Agents => "agents",
        }
    }

    pub fn stored(&self) -> &'static str {
        self.label()
    }
}

pub fn maestro_prompt() -> String {
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

pub fn make_executor(manager: Arc<AgentManager>, project_dir: PathBuf) -> ToolExecutor {
    let project = project_dir.to_string_lossy().to_string();
    Arc::new(move |name: &str, args: Value| -> Result<String, String> {
        if tools::is_maestro_tool(name) {
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
pub fn maestro_dispatch(
    name: &str,
    args: Value,
    manager: &Arc<AgentManager>,
    project_dir: &Path,
) -> Result<String, String> {
    let arg_val = |key: &str| -> Option<Value> {
        let direct = |obj: &Value| -> Option<Value> {
            let v = obj.get(key).cloned();
            v.filter(|v| !v.is_null())
        };
        if let Some(v) = direct(&args) {
            return Some(v);
        }
        for wrap in ["arguments", "args", "parameters"] {
            if let Some(inner) = args.get(wrap) {
                if let Some(v) = direct(inner) {
                    return Some(v);
                }
            }
        }
        if key == "name" {
            for alias in ["agent", "agent_name", "subagent"] {
                for inner in [
                    args.get("arguments"),
                    args.get("args"),
                    args.get("parameters"),
                ]
                .into_iter()
                .flatten()
                .chain(std::iter::once(&args))
                {
                    if let Some(v) = inner.get(alias).cloned() {
                        return Some(v);
                    }
                }
            }
        }
        None
    };
    let arg_str = |key: &str| arg_val(key).and_then(|v| v.as_str().map(str::to_string));
    let arg_i64 = |key: &str| arg_val(key).and_then(|v| v.as_i64());
    let arg_f32 = |key: &str| arg_val(key).and_then(|v| v.as_f64().map(|v| v as f32));

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
            let persona =
                arg_str("persona").filter(|p| !p.is_empty()).unwrap_or_else(|| {
                    if description.is_empty() {
                        format!(
                            "Voce e o agente {} do time do drill. Responda de forma concisa e objetiva, em portugues.",
                            name
                        )
                    } else {
                        format!(
                            "Voce e o agente {}, especialista em {}. Responda de forma concisa e objetiva, em portugues.",
                            name, description
                        )
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

pub fn list_agents_table(agents: &[SubagentConfig]) -> Result<String, String> {
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

/// Orquestrador principal responsável pela criação do Root Agent,
/// descoberta e gerenciamento dos subagentes e integração de ferramentas.
pub struct AgentOrchestrator {
    manager: Arc<AgentManager>,
    ctx_max: usize,
    tools_enabled: bool,
    mode: AgentMode,
    persona: String,
    session_id: String,
    interrupt: Arc<AtomicBool>,
    temp: f32,
    n_predict: usize,
}

impl AgentOrchestrator {
    pub fn new(
        engine: Option<Arc<Mutex<RuntimeLlama>>>,
        project_dir: PathBuf,
        tools_enabled: bool,
        ctx_max: usize,
        interrupt: Arc<AtomicBool>,
        persona: String,
        temp: f32,
        n_predict: usize,
    ) -> Self {
        let mut manager = AgentManager::new();
        manager.set_interrupt(Arc::clone(&interrupt));
        let manager = Arc::new(manager);

        // 1. Criação e registro do Root Agent
        manager.add(ConfigAgent {
            agent_id: MAESTRO_AGENT_ID.to_string(),
            role: AgentRole::Root,
            prompt: persona.clone(),
            tools_json: tools_enabled.then(|| tools::TOOLS_JSON.to_string()),
            force_tools: false,
        });

        // 2. Carregamento e registro de todos os subagentes descobertos
        for cfg in subagents::discover_agents(&project_dir) {
            manager.add(cfg.to_config());
        }

        // 3. Configuração do Executor padrão de ferramentas (base + maestro)
        manager.set_executor(make_executor(Arc::clone(&manager), project_dir));

        if let Some(engine) = engine {
            manager.run(engine);
        }

        Self {
            manager,
            ctx_max,
            tools_enabled,
            mode: AgentMode::Solo,
            persona,
            session_id: String::new(),
            interrupt,
            temp,
            n_predict,
        }
    }

    pub fn manager(&self) -> &Arc<AgentManager> {
        &self.manager
    }

    pub fn interrupt(&self) -> &Arc<AtomicBool> {
        &self.interrupt
    }

    pub fn temp(&self) -> f32 {
        self.temp
    }

    pub fn n_predict(&self) -> usize {
        self.n_predict
    }

    pub fn tools_enabled(&self) -> bool {
        self.tools_enabled
    }

    pub fn set_event_sink(&self, sink: EventSink) {
        self.manager.set_event_sink(sink);
    }

    pub fn set_history(&self, listener: HistoryListener) {
        self.manager.set_history(listener);
    }

    pub fn set_mode(&mut self, mode: AgentMode) {
        let prev = self.mode;
        self.mode = mode;
        if prev != mode {
            let prompt = match mode {
                AgentMode::Solo => self.persona.clone(),
                AgentMode::Agents => maestro_prompt(),
            };
            self.manager.add(ConfigAgent {
                agent_id: MAESTRO_AGENT_ID.to_string(),
                role: AgentRole::Root,
                prompt,
                tools_json: self.tools_enabled.then(|| tools::TOOLS_JSON.to_string()),
                force_tools: self.tools_enabled && mode == AgentMode::Agents,
            });
            self.manager
                .set_session(self.session_id.clone(), mode.stored().to_string());
        }
    }

    pub fn set_session(&mut self, session_id: String) {
        self.session_id = session_id;
        self.manager
            .set_session(self.session_id.clone(), self.mode.stored().to_string());
    }

    pub fn mode(&self) -> AgentMode {
        self.mode
    }

    pub fn ctx_max(&self) -> usize {
        self.ctx_max
    }

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

    pub fn run_turn(&mut self, user_input: &str) -> Result<TurnResult, String> {
        self.manager.enqueue(MAESTRO_AGENT_ID, user_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lib-rust-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_orchestrator_initialization_and_mode() {
        let dir = temp_test_dir("init");
        let interrupt = Arc::new(AtomicBool::new(false));
        let mut orch = AgentOrchestrator::new(
            None,
            dir.clone(),
            true,
            4096,
            interrupt,
            "persona do root".to_string(),
            0.6,
            512,
        );

        assert_eq!(orch.mode(), AgentMode::Solo);
        assert!(orch.manager().get(MAESTRO_AGENT_ID).is_some());
        let root = orch.manager().get(MAESTRO_AGENT_ID).unwrap();
        assert_eq!(root.config.role, AgentRole::Root);
        assert_eq!(root.config.prompt, "persona do root");

        // Alterna para Agents
        orch.set_mode(AgentMode::Agents);
        assert_eq!(orch.mode(), AgentMode::Agents);
        let root_agents = orch.manager().get(MAESTRO_AGENT_ID).unwrap();
        assert!(root_agents.config.prompt.contains("[drill:maestro]"));
        assert!(root_agents.config.force_tools);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_maestro_subagent_creation_and_tools() {
        let dir = temp_test_dir("subagent_tools");
        let interrupt = Arc::new(AtomicBool::new(false));
        let orch = AgentOrchestrator::new(
            None,
            dir.clone(),
            true,
            4096,
            interrupt,
            "persona teste".to_string(),
            0.6,
            512,
        );

        let executor = make_executor(Arc::clone(orch.manager()), dir.clone());

        // 1. Cria um subagente via executor de tool (create_agent)
        let create_res = executor(
            "create_agent",
            json!({
                "name": "tester_agent",
                "description": "especialista em testes unitarios",
                "persona": "Voce escreve testes unitarios concisos.",
                "temp": 0.5,
                "max_tool_rounds": 3,
                "allowed_tools": ["read_file", "write_file"]
            }),
        );
        assert!(create_res.is_ok(), "create_res={:?}", create_res);
        assert!(create_res.unwrap().contains("tester_agent"));

        // O agente deve ter sido adicionado no manager da lib-rust
        let sub = orch.manager().get("tester_agent");
        assert!(sub.is_some());
        let sub_state = sub.unwrap();
        assert_eq!(sub_state.config.agent_id, "tester_agent");
        assert_eq!(sub_state.config.role, AgentRole::Agent);
        assert_eq!(sub_state.config.prompt, "Voce escreve testes unitarios concisos.");

        // 2. Lê a definição via tool read_agent
        let read_res = executor("read_agent", json!({ "name": "tester_agent" }));
        assert!(read_res.is_ok(), "read_res={:?}", read_res);
        let read_text = read_res.unwrap();
        assert!(read_text.contains("tester_agent"));

        // 3. Executa tool base write_file e read_file através do mesmo executor unificado
        let write_res = executor(
            "write_file",
            json!({
                "path": "hello.txt",
                "content": "conteudo do teste de ferramentas na lib-rust"
            }),
        );
        assert!(write_res.is_ok());

        let read_file_res = executor("read_file", json!({ "path": "hello.txt" }));
        assert!(read_file_res.is_ok());
        assert!(read_file_res.unwrap().contains("conteudo do teste de ferramentas na lib-rust"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
