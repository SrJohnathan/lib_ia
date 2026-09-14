use crate::traits::ChatMessage;
use std::sync::mpsc::Sender;

/// Papel do agente dentro do time.
/// - `Root`: o agente principal (ex.: drill/maestro).
/// - `Agent`: subagente delegado via `ask_agent`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Root,
    Agent,
}

#[derive(Debug, Clone)]
pub struct ConfigAgent {
    pub agent_id: String,
    pub role: AgentRole,
    /// Persona do agente, enviada como primeira mensagem do tipo "system" no
    /// proprio historico (nao altera o system_prompt global do RuntimeLlama).
    pub prompt: String,
    /// Ferramentas do agente: `Some` habilita o loop de tools (mesmo executor
    /// compartilhado); `None` responde em um unico round.
    pub tools_json: Option<String>,
    /// Quando `true`, uma rodada que termina sem chamada de ferramenta e
    /// tratada como falha: tenta-se extrair o tool call do texto (fallback
    /// JSON) e, se nao houver, aplica-se um empurrao ("chame a ferramenta
    /// agora"). Mantem acoes como `create_agent` independentes da vontade do
    /// modelo de narrar em vez de agir.
    pub force_tools: bool,
}

impl Default for ConfigAgent {
    fn default() -> Self {
        Self {
            agent_id: String::new(),
            role: AgentRole::Root,
            prompt: String::new(),
            tools_json: None,
            force_tools: false,
        }
    }
}

/// Estado vivo de um agente registrado na fila: config + historico proprio +
/// resumo do ultimo trabalho concluido.
#[derive(Debug, Clone)]
pub struct AgentState {
    pub config: ConfigAgent,
    pub history: Vec<ChatMessage>,
    /// Resumo do ultimo trabalho do agente (gerado por relexão em
    /// `run_inline_with_summary`). Usado para historico/UI.
    pub summary: Option<String>,
}

impl AgentState {
    pub fn new(config: ConfigAgent) -> Self {
        Self {
            config,
            history: Vec::new(),
            summary: None,
        }
    }
}

/// Linha de historico persistente gerada ao final de cada turno. O `kind`
/// indica a natureza do registro: "root" (turno do usuario no drill/maestro),
/// "main" (trabalho de um subagente) ou "resumo" (resumo reflexivo do agente).
/// E entregue ao listener registrado via `AgentManager::set_history`.
#[derive(Debug, Clone)]
pub struct HistoryRow {
    pub agent_id: String,
    pub kind: String,
    pub input: String,
    pub reply: String,
    pub tokens: Option<i64>,
    pub tps: Option<f64>,
    pub context_tokens: Option<i64>,
}

/// Resultado de um turno. Stats sao lidos na MESMA thread que gerou
/// (dentro de `run_turn_unified`) e devolvidos aqui; quem consumiu a fila
/// nunca volta ao engine.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnResult {
    pub reply: String,
    pub tokens: Option<i64>,
    pub tps: Option<f64>,
    pub context_tokens: Option<i64>,
}

/// Task enfileirada no AgentManager (um turno de um agente).
pub struct AgentTask {
    pub agent_id: String,
    pub input: String,
    pub reply: Sender<Result<TurnResult, String>>,
}

impl AgentTask {
    pub fn new(
        agent_id: String,
        input: String,
        reply: Sender<Result<TurnResult, String>>,
    ) -> Self {
        Self {
            agent_id,
            input,
            reply,
        }
    }
}