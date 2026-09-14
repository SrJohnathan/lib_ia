use crate::traits::ChatMessage;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

/// Papel do agente dentro do time.
/// - `Root`: o agente principal (ex.: drill/maestro), único com acesso ao
///   loop de ferramentas disparado pela fila.
/// - `Agent`: subagente delegado via `ask_agent`; responde em um turno unico.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Root,
    Agent,
}

#[derive(Debug, Clone)]
pub struct ConfigAgent {
    pub agent_id: String,
    pub role: AgentRole,
    /// Persona do agente, enviada como primeira mensagem do tipo "system" no proprio historico
    /// (nao altera o system_prompt global do RuntimeLlama).
    pub prompt: String,
    /// Ferramentas do agente (reservado para uso futuro; o RuntimeLlama nao e alterado).
    pub tools_json: Option<String>,
}

pub struct AgentTask {
    pub agent_id: String,
    pub input_prompt: String,
    pub reply_sender: Sender<std::result::Result<String, String>>,
}

#[derive(Clone)]
pub struct Agent {
    pub config: ConfigAgent,
    pub(crate) history: Arc<Mutex<Vec<ChatMessage>>>,
    queue_tx: Sender<AgentTask>,
}

impl Agent {
    pub fn new(config: ConfigAgent, queue_tx: Sender<AgentTask>) -> Self {
        Self {
            config,
            history: Arc::new(Mutex::new(Vec::new())),
            queue_tx,
        }
    }

    /// Envia uma tarefa para a fila do Manager e bloqueia aguardando a resposta.
    /// O Manager e o unico escritor do historico (adiciona a mensagem do usuario
    /// e a resposta do assistente), garantindo uma unica chamada ao RuntimeLlama por task.
    pub fn execute(&self, input_prompt: &str) -> std::result::Result<String, String> {
        let (tx, rx) = channel();
        let task = AgentTask {
            agent_id: self.config.agent_id.clone(),
            input_prompt: input_prompt.to_string(),
            reply_sender: tx,
        };

        self.queue_tx.send(task).map_err(|e| e.to_string())?;
        rx.recv().map_err(|e| e.to_string())?
    }

    /// Copia do historico de tokens do agente (persona + turnos anteriores).
    pub fn history(&self) -> Vec<ChatMessage> {
        self.history.lock().unwrap().clone()
    }

    pub fn clear_history(&self) {
        self.history.lock().unwrap().clear();
    }
}