use crate::agents::agent::{Agent, AgentRole, AgentTask, ConfigAgent};
use crate::runtime::RuntimeLlama;
use crate::traits::{ChatMessage, GenerationType, ToolCall};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

/// Rounds maximos do loop de ferramentas de um agente Root.
const MAX_TOOL_ROUNDS: usize = 8;

/// Executor de ferramentas do time: chamado no loop de tools de um agente
/// Root, recebe (nome, args) e devolve o resultado da execucao.
pub type ToolExecutor =
    Arc<dyn Fn(&str, Value) -> Result<String, String> + Send + Sync>;

/// Sink de eventos de progresso de um turno (stream, ferramentas, status).
/// Setado pela camada de UI apos o runtime ja rodar; compartilhado via Mutex.
pub type EventSink = Arc<dyn Fn(ManagerEvent) + Send + Sync>;

#[derive(Debug, Clone)]
pub enum ManagerEvent {
    Stream(GenerationType),
    ToolStart(String),
    ToolResult(String, String),
    Status(String),
}

/// Fila de requisicoes de agentes. O `run()` consome uma task por vez, em
/// série, com UMA chamada ao RuntimeLlama por task (sempre pelo caminho de
/// stream), mantendo o modelo carregado e o historico/token por agente.
pub struct AgentManager {
    agents: Arc<Mutex<HashMap<String, Agent>>>,
    queue_tx: Sender<AgentTask>,
    queue_rx: Option<Receiver<AgentTask>>,
    executor: Option<ToolExecutor>,
    events: Arc<Mutex<Option<EventSink>>>,
    runtime: Option<Arc<Mutex<RuntimeLlama>>>,
    interrupt: Arc<AtomicBool>,
    max_tool_rounds: usize,
}

impl AgentManager {
    pub fn new() -> Self {
        let (tx, rx) = channel();
        Self {
            agents: Arc::new(Mutex::new(HashMap::new())),
            queue_tx: tx,
            queue_rx: Some(rx),
            executor: None,
            events: Arc::new(Mutex::new(None)),
            runtime: None,
            interrupt: Arc::new(AtomicBool::new(false)),
            max_tool_rounds: MAX_TOOL_ROUNDS,
        }
    }

    pub fn add(&self, config: ConfigAgent) {
        let agent = Agent::new(config.clone(), self.queue_tx.clone());
        let mut map = self.agents.lock().unwrap();
        map.insert(config.agent_id, agent);
    }

    pub fn remove(&self, agent_id: &str) -> Option<Agent> {
        let mut map = self.agents.lock().unwrap();
        map.remove(agent_id)
    }

    pub fn get(&self, agent_id: &str) -> Option<Agent> {
        let map = self.agents.lock().unwrap();
        map.get(agent_id).cloned()
    }

    pub fn agent_ids(&self) -> Vec<String> {
        let map = self.agents.lock().unwrap();
        map.keys().cloned().collect()
    }

    pub fn root_ids(&self) -> Vec<String> {
        self.agents
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, agent)| agent.config.role == AgentRole::Root)
            .map(|(id, _)| id.clone())
            .collect()
    }

    pub fn is_root(&self, agent_id: &str) -> bool {
        self.agents
            .lock()
            .unwrap()
            .get(agent_id)
            .map(|agent| agent.config.role == AgentRole::Root)
            .unwrap_or(false)
    }

    pub fn set_executor(&mut self, executor: ToolExecutor) {
        self.executor = Some(executor);
    }

    pub fn set_event_sink(&self, sink: EventSink) {
        *self.events.lock().unwrap() = Some(sink);
    }

    pub fn set_interrupt(&mut self, interrupt: Arc<AtomicBool>) {
        self.interrupt = interrupt;
    }

    /// Expoe o executor (clone barato) para uso fora da fila, sem segurar o
    /// lock do manager durante a chamada.
    pub fn executor_snapshot(&self) -> Option<ToolExecutor> {
        self.executor.clone()
    }

    /// Inicia o worker da fila em uma thread propria. O runtime fica
    /// compartilhado (Arc<Mutex<RuntimeLlama>>) para quem roda fora da fila
    /// (ex.: modo solo / execute_now), mas o portal de entrada do modelo e
    /// sempre UMA thread por vez.
    pub fn run(&mut self, runtime: Arc<Mutex<RuntimeLlama>>) {
        let rx = self.queue_rx.take().expect("AgentManager já está rodando");
        let agents_ref = Arc::clone(&self.agents);
        let executor = self.executor.clone();
        let events = Arc::clone(&self.events);
        let interrupt = Arc::clone(&self.interrupt);
        let max_tool_rounds = self.max_tool_rounds;
        self.runtime = Some(runtime.clone());

        thread::spawn(move || {
            while let Ok(task) = rx.recv() {
                let agent = {
                    let map = agents_ref.lock().unwrap();
                    map.get(&task.agent_id).cloned()
                };

                let Some(agent) = agent else {
                    let _ = task.reply_sender.send(Err(
                        "Agente não encontrado".to_string()
                    ));
                    continue;
                };

                let result = Self::process_turn(
                    &agent,
                    &task.input_prompt,
                    &runtime,
                    &interrupt,
                    executor.clone(),
                    &events,
                    max_tool_rounds,
                );
                let _ = task.reply_sender.send(result);
            }
        });
    }

    /// Executa um agente inline (fora da fila) no runtime compartilhado —
    /// usado por `ask_agent`, que roda dentro do turno do Root e portanto
    /// nao pode entrar na propria fila (seria deadlock com o worker ocupado).
    pub fn execute_now(&self, agent_id: &str, input_prompt: &str) -> Result<String, String> {
        let runtime = self
            .runtime
            .as_ref()
            .ok_or_else(|| "runtime não iniciado".to_string())?
            .clone();
        let agent = self
            .get(agent_id)
            .ok_or_else(|| format!("agente '{}' não encontrado", agent_id))?;
        Self::process_turn(
            &agent,
            input_prompt,
            &runtime,
            &self.interrupt,
            self.executor.clone(),
            &self.events,
            1,
        )
    }

    /// Turno de um agente: mesmo caminho (stream) para Root e subagentes —
    /// o Root faz o loop de ferramentas, o subagente responde em 1 round.
    fn process_turn(
        agent: &Agent,
        input_prompt: &str,
        runtime: &Arc<Mutex<RuntimeLlama>>,
        interrupt: &Arc<AtomicBool>,
        executor: Option<ToolExecutor>,
        events: &Arc<Mutex<Option<EventSink>>>,
        max_tool_rounds: usize,
    ) -> Result<String, String> {
        let history_slot = agent.history.clone();
        let mut history = {
            let mut slot = history_slot.lock().unwrap();
            if slot.is_empty() {
                slot.push(ChatMessage {
                    role: "system".to_string(),
                    content: agent.config.prompt.clone(),
                    reasoning_content: None,
                });
            }
            slot.push(ChatMessage {
                role: "user".to_string(),
                content: input_prompt.to_string(),
                reasoning_content: None,
            });
            slot.clone()
        };

        let use_tools = agent.config.tools_json.is_some();
        let limit = if use_tools { max_tool_rounds } else { 1 };

        let emit = |event: ManagerEvent| {
            if let Some(sink) = events.lock().unwrap().as_ref() {
                sink(event);
            }
        };

        let mut reply = String::new();
        let mut done = false;

        for round in 0..limit {
            emit(ManagerEvent::Status(format!("generating ({})", round + 1)));

            let mut calls: Vec<ToolCall> = Vec::new();
            let mut reasoning = String::new();
            let interrupted = Arc::clone(interrupt);

            let output = {
                let mut engine = runtime.lock().unwrap();
                engine
                    .generate_chat_messages_stream_with(&history, |chunk, _, _| {
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

            history.push(ChatMessage {
                role: "assistant".to_string(),
                content: output.clone(),
                reasoning_content: if reasoning.is_empty() {
                    None
                } else {
                    Some(reasoning)
                },
            });

            if interrupt.load(Ordering::Relaxed) {
                reply = output;
                done = true;
                break;
            }

            if !use_tools || calls.is_empty() {
                reply = output;
                done = true;
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
                history.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!(
                        "\n<tool_result> name={} id={}\n{}\n</tool_result>",
                        call.name, call.call_id, result
                    ),
                    reasoning_content: None,
                });
            }
        }

        if let Ok(mut slot) = history_slot.lock() {
            *slot = history;
        }

        if done {
            Ok(reply)
        } else {
            Err("max tool rounds reached without final answer".to_string())
        }
    }
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}