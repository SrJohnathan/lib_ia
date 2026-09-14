use crate::agents::agent::{
    AgentRole, AgentState, AgentTask, ConfigAgent, HistoryRow, TurnResult,
};
use crate::runtime::RuntimeLlama;
use crate::traits::{ChatMessage, GenerationType, ToolCall};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// Rounds maximos do loop de ferramentas de um agente com tools habilitadas.
const MAX_TOOL_ROUNDS: usize = 8;

/// Executor de ferramentas do time: recebe (nome, args) e devolve o resultado.
pub type ToolExecutor =
    Arc<dyn Fn(&str, Value) -> Result<String, String> + Send + Sync>;

/// Sink de eventos de progresso (stream, ferramentas, status) para a UI.
pub type EventSink = Arc<dyn Fn(ManagerEvent) + Send + Sync>;

/// Receptor de linhas de historico persistente (ex.: gravacao em LanceDB).
pub type HistoryListener = Arc<dyn Fn(HistoryRow) + Send + Sync>;

#[derive(Debug, Clone)]
pub enum ManagerEvent {
    Stream(GenerationType),
    ToolStart(String),
    ToolResult(String, String),
    Status(String),
    /// Qual agente esta gerando neste momento (para o card da UI). Disparado
    /// SEMPRE, inclusive em turnos inline de subagentes.
    AgentWorking(String),
}

/// Fila serial de agentes + registro de estados. Nao conhece tools nem loop de
/// geracao: toda geracao roda em `run_turn_unified`, NA MESMA thread que
/// consome a fila (e que depois le as stats do engine).
pub struct AgentManager {
    states: Arc<Mutex<HashMap<String, AgentState>>>,
    queue_tx: Sender<AgentTask>,
    queue_rx: Mutex<Option<Receiver<AgentTask>>>,
    engine: Mutex<Option<Arc<Mutex<RuntimeLlama>>>>,
    executor: Arc<Mutex<Option<ToolExecutor>>>,
    events: Arc<Mutex<Option<EventSink>>>,
    interrupt: Arc<AtomicBool>,
    max_tool_rounds: usize,
    history: Arc<Mutex<Option<HistoryListener>>>,
}

impl AgentManager {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            states: Arc::new(Mutex::new(HashMap::new())),
            queue_tx: tx,
            queue_rx: Mutex::new(Some(rx)),
            engine: Mutex::new(None),
            executor: Arc::new(Mutex::new(None)),
            events: Arc::new(Mutex::new(None)),
            interrupt: Arc::new(AtomicBool::new(false)),
            max_tool_rounds: MAX_TOOL_ROUNDS,
            history: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_interrupt(&mut self, interrupt: Arc<AtomicBool>) {
        self.interrupt = interrupt;
    }

    pub fn set_max_tool_rounds(&mut self, rounds: usize) {
        self.max_tool_rounds = rounds.max(1);
    }

    pub fn add(&self, config: ConfigAgent) {
        self.states
            .lock()
            .unwrap()
            .insert(config.agent_id.clone(), AgentState::new(config));
    }

    pub fn get(&self, agent_id: &str) -> Option<AgentState> {
        self.states.lock().unwrap().get(agent_id).cloned()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        self.states.lock().unwrap().keys().cloned().collect()
    }

    pub fn root_ids(&self) -> Vec<String> {
        self.states
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, state)| state.config.role == AgentRole::Root)
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn is_root(&self, agent_id: &str) -> bool {
        self.states
            .lock()
            .unwrap()
            .get(agent_id)
            .map(|state| state.config.role == AgentRole::Root)
            .unwrap_or(false)
    }

    pub fn history_of(&self, agent_id: &str) -> Option<Vec<ChatMessage>> {
        self.get(agent_id).map(|state| state.history)
    }

    pub fn set_executor(&self, executor: ToolExecutor) {
        *self.executor.lock().unwrap() = Some(executor);
    }

    pub fn set_event_sink(&self, sink: EventSink) {
        *self.events.lock().unwrap() = Some(sink);
    }

    /// Registra o receptor de historico: para cada turno concluido, uma
    /// `HistoryRow` e entregue aqui (geralmente gravada em LanceDB pelo drill).
    pub fn set_history(&self, listener: HistoryListener) {
        *self.history.lock().unwrap() = Some(listener);
    }

    pub fn set_summary(&self, agent_id: &str, summary: String) {
        if let Ok(mut guard) = self.states.lock() {
            if let Some(state) = guard.get_mut(agent_id) {
                state.summary = Some(summary);
            }
        }
    }

    /// Envia uma task para a fila e bloqueia no reply. Serial: o runtime so e
    /// usado por uma thread por vez (o worker da fila).
    pub fn enqueue(&self, agent_id: &str, input: &str) -> Result<TurnResult, String> {
        let (tx, rx) = channel();
        let task = AgentTask::new(agent_id.to_string(), input.to_string(), tx);
        self.queue_tx.send(task).map_err(|err| err.to_string())?;
        rx.recv().map_err(|err| err.to_string())?
    }

    pub fn run(&self, engine: Arc<Mutex<RuntimeLlama>>) {
        let rx = self
            .queue_rx
            .lock()
            .unwrap()
            .take()
            .expect("AgentManager já está rodando");
        *self.engine.lock().unwrap() = Some(engine.clone());
        let states = Arc::clone(&self.states);
        let executor = Arc::clone(&self.executor);
        let events = Arc::clone(&self.events);
        let interrupt = Arc::clone(&self.interrupt);
        let history = Arc::clone(&self.history);
        let max_tool_rounds = self.max_tool_rounds;

        thread::spawn(move || {
            while let Ok(task) = rx.recv() {
                let mut state = {
                    let map = states.lock().unwrap();
                    map.get(&task.agent_id).cloned()
                };

                let Some(state) = state.as_mut() else {
                    let _ = task.reply.send(Err("Agente não encontrado".to_string()));
                    continue;
                };

                let executor = executor.lock().unwrap().clone();
                let history = history.lock().unwrap().clone();
                let result = Self::run_turn_unified(
                    state,
                    &engine,
                    executor,
                    &events,
                    &interrupt,
                    max_tool_rounds,
                    &task.input,
                    true,
                    history,
                    "root",
                );

                if let Ok(mut guard) = states.lock() {
                    guard.insert(task.agent_id.clone(), state.clone());
                }
                let _ = task.reply.send(result);
            }
        });
    }

    /// Executa um agente inline NA MESMA THREAD do chamador — usado por
    /// `ask_agent`, que ja roda dentro do turno do Root, no worker da fila.
    /// Nao re-enfileira (evita deadlock com o worker ocupado) e nao emite
    /// stream para a UI (a resposta volta como tool result do maestro).
    pub fn run_inline(&self, agent_id: &str, input: &str) -> Result<TurnResult, String> {
        self.run_inline_kind(agent_id, input, "main")
    }

    /// Trabalho completo de um subagente NA MESMA THREAD: um turno "main" com
    /// a tarefa e, ao final, um turno de reflexão ("resumo") que preserva o
    /// historico do agente. Retorna (trabalho, resumo). O resumo também fica
    /// gravado no estado do agente (`set_summary`).
    pub fn run_inline_with_summary(
        &self,
        agent_id: &str,
        work_input: &str,
        summary_input: &str,
    ) -> Result<(TurnResult, TurnResult), String> {
        let work = self.run_inline(agent_id, work_input)?;
        let summary = self.run_inline_kind(agent_id, summary_input, "resumo")?;
        self.set_summary(agent_id, summary.reply.clone());
        Ok((work, summary))
    }

    /// Variante interna de `run_inline` com o `kind` de historico controlado.
    fn run_inline_kind(
        &self,
        agent_id: &str,
        input: &str,
        kind: &str,
    ) -> Result<TurnResult, String> {
        let engine = self
            .engine
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| "runtime não iniciado".to_string())?;
        let executor = self.executor.lock().unwrap().clone();
        let events = Arc::clone(&self.events);
        let interrupt = Arc::clone(&self.interrupt);
        let history = self.history.lock().unwrap().clone();

        let mut state = self
            .get(agent_id)
            .ok_or_else(|| format!("agente '{}' não encontrado", agent_id))?;
        let result = Self::run_turn_unified(
            &mut state,
            &engine,
            executor,
            &events,
            &interrupt,
            self.max_tool_rounds,
            input,
            false,
            history,
            kind,
        );
        if result.is_ok() {
            if let Ok(mut guard) = self.states.lock() {
                guard.insert(agent_id.to_string(), state);
            }
        }
        result
    }

    /// Caminho unico de geracao (mesma logica do antigo solo): persona se o
    /// historico estiver vazio, mensagem do usuario, rounds de tools/stream,
    /// e stats do engine lidas AINDA nesta thread.
    #[allow(clippy::too_many_arguments)]
    fn run_turn_unified(
        state: &mut AgentState,
        engine: &Arc<Mutex<RuntimeLlama>>,
        executor: Option<ToolExecutor>,
        events: &Arc<Mutex<Option<EventSink>>>,
        interrupt: &Arc<AtomicBool>,
        max_tool_rounds: usize,
        input: &str,
        stream_events: bool,
        history: Option<HistoryListener>,
        kind: &str,
    ) -> Result<TurnResult, String> {
        // Card da UI: qual agente esta trabalhando agora (também em inline).
        if let Some(sink) = events.lock().unwrap().as_ref() {
            sink(ManagerEvent::AgentWorking(state.config.agent_id.clone()));
        }

        if state.history.is_empty() && !state.config.prompt.is_empty() {
            state.history.push(ChatMessage {
                role: "system".to_string(),
                content: state.config.prompt.clone(),
                reasoning_content: None,
            });
        }
        state.history.push(ChatMessage {
            role: "user".to_string(),
            content: input.to_string(),
            reasoning_content: None,
        });

        let use_tools = state.config.tools_json.is_some();
        let limit = if use_tools { max_tool_rounds } else { 1 };

        let emit = |event: ManagerEvent| {
            if stream_events {
                if let Some(sink) = events.lock().unwrap().as_ref() {
                    sink(event);
                }
            }
        };

        let mut reply = String::new();
        let mut finished = false;

        for round in 0..limit {
            emit(ManagerEvent::Status(format!("generating ({})", round + 1)));

            let mut calls: Vec<ToolCall> = Vec::new();
            let mut reasoning = String::new();
            let interrupted = Arc::clone(interrupt);

            let output = {
                let mut engine_guard = engine.lock().unwrap();
                engine_guard
                    .generate_chat_messages_stream_with(&state.history, |chunk, _, _| {
                        if interrupted.load(Ordering::Relaxed) {
                            return;
                        }
                        match &chunk {
                            GenerationType::Content(text) => {
                                emit(ManagerEvent::Stream(GenerationType::Content(
                                    text.clone(),
                                )));
                            }
                            GenerationType::Reasoning(text) => {
                                reasoning.push_str(text);
                                emit(ManagerEvent::Stream(GenerationType::Reasoning(
                                    text.clone(),
                                )));
                            }
                            GenerationType::CallTool(tool) => {
                                calls.push(tool.clone());
                                emit(ManagerEvent::Stream(GenerationType::CallTool(
                                    tool.clone(),
                                )));
                            }
                            GenerationType::ToolPreview(_) => {}
                        }
                    })
                    .map_err(|err| err.to_string())?
            };

            state.history.push(ChatMessage {
                role: "assistant".to_string(),
                content: output.clone(),
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
            });

            if interrupt.swap(false, Ordering::Relaxed) {
                reply = output;
                finished = true;
                emit(ManagerEvent::Status("interrompido".to_string()));
                break;
            }

            if !use_tools || calls.is_empty() {
                reply = output;
                finished = true;
                emit(ManagerEvent::Status("done".to_string()));
                break;
            }

            for call in &calls {
                emit(ManagerEvent::ToolStart(call.name.clone()));
                let result = match &executor {
                    Some(exec) => match exec(&call.name, call.args.clone()) {
                        Ok(text) => text,
                        Err(err) => format!("ERROR: {}", err),
                    },
                    None => "(sem executor de ferramentas)".to_string(),
                };
                emit(ManagerEvent::ToolResult(call.name.clone(), result.clone()));
                state.history.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "\n<tool_result> name={} id={}\n{}\n</tool_result>",
                        call.name, call.call_id, result
                    ),
                    reasoning_content: None,
                });
            }
        }

        if !finished {
            return Err("max tool rounds reached without final answer".to_string());
        }

        // Stats na MESMA thread que acabou de gerar.
        let (tokens, tps, context_tokens) = {
            let mut engine_guard = engine.lock().unwrap();
            (
                engine_guard.last_generated_tokens(),
                engine_guard.last_tokens_per_second(),
                engine_guard.last_context_tokens(),
            )
        };

        if let Some(listener) = history.as_ref() {
            listener(HistoryRow {
                agent_id: state.config.agent_id.clone(),
                kind: kind.to_string(),
                input: input.to_string(),
                reply: reply.clone(),
                tokens,
                tps,
                context_tokens,
            });
        }

        Ok(TurnResult {
            reply,
            tokens,
            tps,
            context_tokens,
        })
    }
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}