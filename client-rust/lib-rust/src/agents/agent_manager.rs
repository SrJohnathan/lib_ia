use crate::agents::agent::{
    AgentRole, AgentState, AgentTask, ConfigAgent, HistoryRow, TurnResult,
};
use crate::runtime::RuntimeLlama;
use crate::traits::{ChatMessage, GenerationType, ToolCall};
use serde_json::json;
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

        let use_tools_any = state.config.tools_json.is_some();
        // Modo forcado reserva UMA rodada final sem tools para a resposta ao
        // usuario: garante conclusao mesmo se o modelo insistir em ferramentas.
        let limit = if use_tools_any {
            if state.config.force_tools {
                max_tool_rounds + 1
            } else {
                max_tool_rounds
            }
        } else {
            1
        };
        let mut nudges = 0;
        let mut any_tool_executed = false;

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
            let use_tools = use_tools_any
                && (!state.config.force_tools || round + 1 < limit);
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

            // Fallback: o modelo escreveu o tool call como texto (bloco
            // `<tool_call>`/```json) em vez de emiti-lo nativamente. Recupera
            // a chamada para nao depender da vontade do modelo.
            if calls.is_empty() {
                if let Some(parsed) = parse_tool_json(&output) {
                    emit(ManagerEvent::Status(format!(
                        "tool call via fallback JSON: {}",
                        parsed.name
                    )));
                    calls.push(parsed);
                }
            }

            if interrupt.swap(false, Ordering::Relaxed) {
                reply = output;
                finished = true;
                emit(ManagerEvent::Status("interrompido".to_string()));
                break;
            }

            if !use_tools || calls.is_empty() {
                // Modo forcado: uma rodada que prometeu acao mas nao emitiu
                // ferramenta merece um empurrao em vez de encerrar a prosa.
                if use_tools
                    && state.config.force_tools
                    && !any_tool_executed
                    && round + 1 < limit
                    && nudges < 2
                {
                    nudges += 1;
                    emit(ManagerEvent::Status(format!(
                        "sem tool call em {}; forcando (round {})",
                        state.config.agent_id,
                        round + 2
                    )));
                    state.history.push(ChatMessage {
                        role: "user".to_string(),
                        content: "\nATENCAO: a sua resposta anterior NAO executou nenhuma ferramenta. \
                                  Se a tarefa do usuario exige uma ferramenta, essa acao NAO foi feita \
                                  e voce NAO deve reportar como feita. Chame AGORA a ferramenta \
                                  apropriada com os argumentos completos. Se a tarefa nao requer \
                                  ferramenta, responda a pergunta do usuario diretamente, sem falar \
                                  em ferramentas."
                            .to_string(),
                        reasoning_content: None,
                    });
                    continue;
                }
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
            any_tool_executed = true;
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

/// Extrai um tool call escrito como texto da resposta do assistente quando
/// o parsing nativo (llama.cpp/mtmd) nao classificou a chamada. Suporta:
/// - bloco Qwen `<tool_call> {...} </tool_call>`, incluindo `{"type":"function",...}`
/// - bloco de codigo ```json ``` com `{"name": ..., "arguments": {...}}`
/// - JSON simples puro `{"name": ..., "args": {...}}`
fn parse_tool_json(content: &str) -> Option<ToolCall> {
    let mut candidates: Vec<String> = Vec::new();

    let mut rest = content;
    while let Some(start) = rest.find("<tool_call>") {
        let body = &rest[start + "<tool_call>".len()..];
        match body.find("</tool_call>") {
            Some(end) => {
                candidates.push(body[..end].trim().to_string());
                rest = &body[end + "</tool_call>".len()..];
            }
            None => break,
        }
    }

    for start in content.match_indices("```json") {
        let after = &content[start.0 + "```json".len()..];
        if let Some(end) = after.find("```") {
            candidates.push(after[..end].trim().to_string());
        }
    }

    for cand in candidates {
        if let Ok(value) = serde_json::from_str::<Value>(&cand) {
            if let Some(call) = tool_call_from_value(&value) {
                return Some(call);
            }
        }
        // Pode haver texto na frente/atras do JSON dentro do bloco.
        if let Some(brace) = cand.find('{') {
            if let Some(close) = cand.rfind('}') {
                if let Ok(value) = serde_json::from_str::<Value>(&cand[brace..=close]) {
                    if let Some(call) = tool_call_from_value(&value) {
                        return Some(call);
                    }
                }
            }
        }
    }

    // JSON simples sem fence: `{"name": "...", "arguments": {...}}` solto.
    if let Some(brace) = content.find('{') {
        if let Some(close) = content.rfind('}') {
            if let Ok(value) = serde_json::from_str::<Value>(&content[brace..=close]) {
                if let Some(call) = tool_call_from_value(&value) {
                    return Some(call);
                }
            }
        }
    }

    None
}

fn tool_call_from_value(value: &Value) -> Option<ToolCall> {
    let obj = value.as_object()?;
    let (name, args) = if let Some(fun) = obj.get("function").and_then(Value::as_object) {
        let name = fun.get("name").and_then(Value::as_str)?;
        let args = fun
            .get("arguments")
            .or_else(|| fun.get("args"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        (name.to_string(), args)
    } else {
        let name = obj
            .get("name")
            .or_else(|| obj.get("function_name"))
            .and_then(Value::as_str)?;
        if name.is_empty() {
            return None;
        }
        let args = obj
            .get("arguments")
            .or_else(|| obj.get("args"))
            .cloned()
            .unwrap_or_else(|| json!({}));
        (name.to_string(), args)
    };

    let call_id = obj
        .get("call_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("fallback_{}", name));

    Some(ToolCall {
        call_id,
        command: name.clone(),
        name,
        args,
    })
}

impl Default for AgentManager {
    fn default() -> Self {
        Self::new()
    }
}