//! Providers do Root (solo/maestro).
//!
//! O config define uma lista de providers locais (llama.cpp nativo) e remotos
//! (`type-url`: openai/anthropic/google). O `provider-selected-uuid` escolhe o
//! ativo; a troca e feita pelo overlay `/provider` e persistida no config.json.
//!
//! Subagentes NAO seguem o provider: rodam sempre no runtime local.
//!
//! O contrato remoto da lib-rust e agnostico de vendor: um closure
//! `Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> RemoteModelOutput` recebe o
//! historico no shape OpenAI (IR) e cada generator traduz para o wire do seu
//! vendor. Nada na lib-rust muda.

use crate::config::{self, CloudKind, CloudProvider, LocalProvider};
use lib_rust::agents::{RemoteEvent, RemoteModelOutput};
use lib_rust::ToolCall;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Closure de um round remoto usado por `AgentManager::run_remote_turn`.
pub type CloudGenerator =
    Box<dyn Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String>>;

/// Provider ativo no momento.
#[derive(Debug, Clone)]
pub enum Selected {
    Local(LocalProvider),
    Cloud(CloudProvider),
}

impl Selected {
    pub fn is_cloud(&self) -> bool {
        matches!(self, Selected::Cloud(_))
    }

    pub fn model(&self) -> &str {
        match self {
            Selected::Local(p) => &p.model,
            Selected::Cloud(p) => &p.model,
        }
    }

    pub fn ctx(&self) -> usize {
        match self {
            Selected::Local(p) => p.ctx,
            Selected::Cloud(p) => p.ctx,
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Selected::Local(p) => format!("local: {} ({})", p.name, p.model),
            Selected::Cloud(p) => {
                let key = if resolve_cloud_key(p).is_some() {
                    "key ok"
                } else {
                    "SEM API KEY"
                };
                let mut s = format!(
                    "{}: {} [{}] @ {} ({})",
                    p.kind.label(),
                    p.model,
                    key,
                    p.base_url,
                    p.name
                );
                if !p.reasoning_effort.trim().is_empty() {
                    s.push_str(&format!(" | raciocinio {}", p.reasoning_effort));
                }
                s
            }
        }
    }
}

/// Item do overlay `/provider`.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    pub id: String,
    pub name: String,
    pub detail: String,
    pub cloud: bool,
}

struct State {
    local: Vec<LocalProvider>,
    cloud: Vec<CloudProvider>,
    selected: Option<String>,
}

/// Estado compartilhado dos providers entre a TUI (escolhe) e o worker
/// (turnos). Persiste a escolha em `~/.drill/config.json`.
pub struct ProviderStore {
    state: Arc<Mutex<State>>,
    pub config_path: PathBuf,
}

impl ProviderStore {
    pub fn new(cfg: &config::DrillConfig, config_path: PathBuf) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                local: cfg.providers.local.clone(),
                cloud: cfg.providers.cloud.clone(),
                selected: cfg.provider_selected_uuid.clone(),
            })),
            config_path,
        }
    }

    fn resolve<'a>(state: &'a State, uuid: &Option<String>) -> Option<Selected> {
        let uuid = uuid.as_deref()?;
        if let Some(p) = state.local.iter().find(|p| p.id == uuid) {
            return Some(Selected::Local(p.clone()));
        }
        state
            .cloud
            .iter()
            .find(|p| p.id == uuid)
            .map(|p| Selected::Cloud(p.clone()))
    }

    /// Provider ativo. Cai no primeiro local/cloud se o uuid persistido nao
    /// existir mais (o `ensure_defaults` costuma evitar isso).
    pub fn current(&self) -> Selected {
        let guard = self.state.lock().unwrap();
        Self::resolve(&guard, &guard.selected)
            .or_else(|| guard.local.first().cloned().map(Selected::Local))
            .or_else(|| guard.cloud.first().cloned().map(Selected::Cloud))
            .unwrap_or_else(|| Selected::Local(LocalProvider::default()))
    }

    pub fn selected_cloud(&self) -> Option<CloudProvider> {
        match self.current() {
            Selected::Cloud(p) => Some(p),
            _ => None,
        }
    }

    pub fn selected_local(&self) -> Option<LocalProvider> {
        match self.current() {
            Selected::Local(p) => Some(p),
            _ => None,
        }
    }

    pub fn is_cloud(&self) -> bool {
        self.current().is_cloud()
    }

    pub fn summary(&self) -> String {
        self.current().summary()
    }

    /// Snapshot para a TUI: entradas (local + cloud) e o uuid selecionado.
    pub fn snapshot(&self) -> (Vec<ProviderEntry>, Option<String>) {
        let guard = self.state.lock().unwrap();
        let selected = guard.selected.clone();
        let mut entries = Vec::new();
        for p in &guard.local {
            entries.push(ProviderEntry {
                id: p.id.clone(),
                name: p.name.clone(),
                detail: format!("local · {} (ngl {}, ctx {})", p.model, p.ngl, p.ctx),
                cloud: false,
            });
        }
        for p in &guard.cloud {
            let key = if resolve_cloud_key(p).is_some() {
                "key ok"
            } else {
                "sem key"
            };
            let router = if p.router.trim().is_empty() {
                default_router(p.kind, "")
            } else {
                p.router.trim().to_string()
            };
            entries.push(ProviderEntry {
                id: p.id.clone(),
                name: p.name.clone(),
                detail: format!(
                    "{} · {} · {} [{}, {}]",
                    p.kind.label(),
                    p.model,
                    router,
                    key,
                    p.base_url
                ),
                cloud: true,
            });
        }
        (entries, selected)
    }

    /// Troca o provider ativo pelo uuid. Persiste em config.json.
    pub fn select(&self, uuid: &str) -> Result<String, String> {
        let mut guard = self.state.lock().unwrap();
        if Self::resolve(&guard, &Some(uuid.to_string())).is_none() {
            return Err(format!("provider '{}' nao existe no config", uuid));
        }
        guard.selected = Some(uuid.to_string());
        let selected = uuid.to_string();
        drop(guard);

        let path = self.config_path.clone();
        config::update(&path, |cfg| {
            cfg.provider_selected_uuid = Some(selected);
        })?;
        Ok(self.summary())
    }

    /// Define modelo + forca do raciocinio do provider cloud ativo, persistindo.
    pub fn set_model_and_effort(
        &self,
        model: String,
        reasoning_effort: String,
    ) -> Result<(), String> {
        let uuid = {
            let mut guard = self.state.lock().unwrap();
            let Some(uuid) = guard.selected.clone() else {
                return Err("nenhum provider selecionado".to_string());
            };
            let Some(cloud) = guard.cloud.iter_mut().find(|p| p.id == uuid) else {
                return Err("provider ativo nao e remoto (modelo so vale para cloud)".to_string());
            };
            cloud.model = model.clone();
            cloud.reasoning_effort = reasoning_effort.clone();
            uuid
        };
        let path = self.config_path.clone();
        config::update(&path, move |cfg| {
            if let Some(cloud) = cfg.providers.cloud.iter_mut().find(|p| p.id == uuid) {
                cloud.model = model;
                cloud.reasoning_effort = reasoning_effort;
            }
        })
    }

    /// Atualiza o modelo e o mmproj do provider local ativo (ou o primeiro
    /// local configurado), gravando em config.json.
    pub fn set_local_model(
        &self,
        model: String,
        mmproj: String,
    ) -> Result<(), String> {
        let uuid = {
            let mut guard = self.state.lock().unwrap();
            let selected = guard.selected.clone();
            let idx = guard
                .local
                .iter()
                .position(|p| Some(&p.id) == selected.as_ref())
                .unwrap_or(0);
            let target = guard
                .local
                .get_mut(idx)
                .ok_or_else(|| "nenhum provider local configurado (/provider)".to_string())?;
            target.model = model.clone();
            target.mmproj = mmproj.clone();
            let uuid = target.id.clone();
            uuid
        };
        let path = self.config_path.clone();
        config::update(&path, move |cfg| {
            if let Some(local) = cfg.providers.local.iter_mut().find(|p| p.id == uuid) {
                local.model = model;
                local.mmproj = mmproj;
            } else {
                cfg.model = model;
            }
        })
    }

    /// Adiciona um provider cloud novo (wizard `a` do overlay /provider), grava
    /// em config.json e o deixa como provider ativo.
    pub fn add_cloud(
        &self,
        kind: CloudKind,
        base_url: String,
        model: String,
        api_key: String,
    ) -> Result<String, String> {
        let (provider, uuid) = {
            let mut guard = self.state.lock().unwrap();
            let n = guard.cloud.len() + 1;
            let provider = CloudProvider {
                id: config::new_id("cloud"),
                name: format!("{} {}", kind.label(), n),
                kind,
                base_url: base_url.trim().to_string(),
                model: model.trim().to_string(),
                router: String::new(),
                api_key: api_key.trim().to_string(),
                reasoning_effort: "medium".to_string(),
                ctx: 8192,
            };
            let uuid = provider.id.clone();
            guard.cloud.push(provider.clone());
            guard.selected = Some(uuid.clone());
            (provider, uuid)
        };

        let path = self.config_path.clone();
        let provider_for_cfg = provider.clone();
        config::update(&path, move |cfg| {
            cfg.providers.cloud.push(provider_for_cfg);
            cfg.provider_selected_uuid = Some(uuid);
        })?;
        Ok(self.summary())
    }
}

/// Resolve a chave do provider cloud: campo do config primeiro, depois a env
/// do vendor.
pub fn resolve_cloud_key(cloud: &CloudProvider) -> Option<String> {
    let trimmed = cloud.api_key.trim();
    if !trimmed.is_empty() {
        return Some(trimmed.to_string());
    }
    let var = match cloud.kind {
        CloudKind::OpenAi => "OPENAI_API_KEY",
        CloudKind::Anthropic => "ANTHROPIC_API_KEY",
        CloudKind::Google => "GOOGLE_API_KEY",
    };
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Prefere um id de modelo com vocacao de chat: ignora ASR/TTS/guard/audio.
pub fn pick_chat_model(models: &[String]) -> Option<String> {
    models
        .iter()
        .find(|id| {
            let low = id.to_lowercase();
            !(low.contains("whisper")
                || low.contains("allam")
                || low.contains("orph")
                || low.contains("stt")
                || low.contains("tts")
                || low.contains("audio")
                || low.contains("guard")
                || low.contains("safeguard"))
        })
        .cloned()
}

fn default_router(kind: CloudKind, router: &str) -> String {
    if !router.trim().is_empty() {
        return router.trim().to_string();
    }
    match kind {
        CloudKind::OpenAi => "/chat/completions".to_string(),
        CloudKind::Anthropic => "/v1/messages".to_string(),
        CloudKind::Google => "streamGenerateContent".to_string(),
    }
}

/// Junta base_url + router sem duplicar barras. `router` vazio usa o default.
fn endpoint(base_url: &str, router: &str, default: &str) -> String {
    let base = base_url.trim().trim_end_matches('/');
    let path = if router.trim().is_empty() {
        default
    } else {
        router.trim()
    };
    if path.starts_with('/') {
        format!("{}{}", base, path)
    } else {
        format!("{}/{}", base, path)
    }
}

fn has_tools(tools: &Value) -> bool {
    !tools.is_null() && !matches!(tools, Value::Array(v) if v.is_empty())
}

/// LISTA os modelos disponiveis no provider cloud ativo.
pub fn list_models(cloud: &CloudProvider) -> Result<Vec<String>, String> {
    let key = resolve_cloud_key(cloud);
    let client = http_client()?;
    match cloud.kind {
        CloudKind::OpenAi => {
            let url = endpoint(&cloud.base_url, "/models", "/models");
            let mut req = client.get(&url);
            if let Some(key) = &key {
                req = req.bearer_auth(key);
            }
            let body: Value = get_json(req, &url)?;
            let ids = body
                .get("data")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.get("id").and_then(Value::as_str))
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if ids.is_empty() {
                return Err(format!("nenhum modelo listado por {}", url));
            }
            Ok(ids)
        }
        CloudKind::Google => {
            let url = endpoint(&cloud.base_url, "/models", "/models");
            let mut req = client.get(&url);
            if let Some(key) = &key {
                req = req.header("x-goog-api-key", key);
            }
            let body: Value = get_json(req, &url)?;
            let ids = body
                .get("models")
                .and_then(Value::as_array)
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.get("name").and_then(Value::as_str))
                        .map(|s| s.strip_prefix("models/").unwrap_or(s).to_string())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if ids.is_empty() {
                return Err(format!("nenhum modelo listado por {}", url));
            }
            Ok(ids)
        }
        CloudKind::Anthropic => Err(
            "listagem de modelos nao suportada pela API anthropic; informe o modelo no config"
                .to_string(),
        ),
    }
}

fn get_json(req: reqwest::blocking::RequestBuilder, url: &str) -> Result<Value, String> {
    let resp = req
        .send()
        .map_err(|err| format!("falha ao chamar {}: {}", url, err))?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().unwrap_or_default();
        return Err(format!("GET {} -> HTTP {}: {}", url, status, text));
    }
    resp.json()
        .map_err(|err| format!("resposta invalida de {}: {}", url, err))
}

/// Monta o generator de um round remoto conforme o tipo do provider.
pub fn build_generator(
    cloud: &CloudProvider,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> CloudGenerator {
    match cloud.kind {
        CloudKind::OpenAi => {
            if cloud.router.to_lowercase().contains("responses") {
                Box::new(openai_responses_generator(
                    cloud.clone(),
                    tools,
                    temperature,
                    max_tokens,
                    interrupt,
                ))
            } else {
                Box::new(openai_chat_generator(
                    cloud.clone(),
                    tools,
                    temperature,
                    max_tokens,
                    interrupt,
                ))
            }
        }
        CloudKind::Anthropic => Box::new(anthropic_generator(
            cloud.clone(),
            tools,
            temperature,
            max_tokens,
            interrupt,
        )),
        CloudKind::Google => Box::new(google_generator(
            cloud.clone(),
            tools,
            temperature,
            max_tokens,
            interrupt,
        )),
    }
}

// ---------------------------------------------------------------------------
// Helpers de traducao do IR (shape OpenAI) para o wire de cada vendor.
// ---------------------------------------------------------------------------

fn text_of(message: &Value) -> String {
    message
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Parte normalizada do `content` de uma mensagem do IR: texto ou imagem
/// (data URI `data:<mime>;base64,...`).
enum IrPart {
    Text(String),
    Image { mime: String, data: String },
}

/// Separa o `content` de uma mensagem do IR em partes (texto e imagens). O
/// IR usa texto puro (`"...string..."`) ou um array de blocos OpenAI
/// (`{"type":"text"}`, `{"type":"image_url","image_url":{"url":...}}`).
fn ir_parts(message: &Value) -> Vec<IrPart> {
    let content = message.get("content").unwrap_or(&Value::Null);
    let mut out = Vec::new();
    match content {
        Value::String(s) => {
            if !s.is_empty() {
                out.push(IrPart::Text(s.clone()));
            }
        }
        Value::Array(arr) => {
            for block in arr {
                match block.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(Value::as_str) {
                            if !t.is_empty() {
                                out.push(IrPart::Text(t.to_string()));
                            }
                        }
                    }
                    "image_url" => {
                        let url = block
                            .get("image_url")
                            .and_then(|i| i.get("url"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if let Some((mime, data)) = split_data_uri(url) {
                            out.push(IrPart::Image { mime, data });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    out
}

/// Extrai `(mime, base64)` de uma data URI `data:<mime>;base64,<payload>`.
fn split_data_uri(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (mime, payload) = rest.split_once(";base64,")?;
    if mime.is_empty() || payload.is_empty() {
        return None;
    }
    Some((mime.to_string(), payload.to_string()))
}

/// Payloads SSE (`data: ...`), ignorando `[DONE]`, eventos e vazios.
fn sse_payloads(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("data:") else {
            continue;
        };
        let rest = rest.trim();
        if rest.is_empty() || rest == "[DONE]" {
            continue;
        }
        out.push(rest.to_string());
    }
    out
}

fn args_value(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn openai_chat_tools(tools: &Value) -> Value {
    tools.clone()
}

fn responses_tools(tools: &Value) -> Value {
    let Some(arr) = tools.as_array() else {
        return Value::Array(Vec::new());
    };
    Value::Array(
        arr.iter()
            .filter_map(|t| {
                let fun = t.get("function")?;
                Some(json!({
                    "type": "function",
                    "name": fun.get("name").cloned().unwrap_or(json!("")),
                    "description": fun.get("description").cloned().unwrap_or(json!("")),
                    "parameters": fun.get("parameters").cloned().unwrap_or(json!({})),
                }))
            })
            .collect(),
    )
}

fn anthropic_tools(tools: &Value) -> Value {
    let Some(arr) = tools.as_array() else {
        return Value::Array(Vec::new());
    };
    Value::Array(
        arr.iter()
            .filter_map(|t| {
                let fun = t.get("function")?;
                Some(json!({
                    "name": fun.get("name").cloned().unwrap_or(json!("")),
                    "description": fun.get("description").cloned().unwrap_or(json!("")),
                    "input_schema": fun.get("parameters").cloned().unwrap_or(json!({"type":"object"})),
                }))
            })
            .collect(),
    )
}

fn google_tools(tools: &Value) -> Value {
    let Some(arr) = tools.as_array() else {
        return Value::Array(Vec::new());
    };
    let decls: Vec<Value> = arr
        .iter()
        .filter_map(|t| {
            let fun = t.get("function")?;
            Some(json!({
                "name": fun.get("name").cloned().unwrap_or(json!("")),
                "description": fun.get("description").cloned().unwrap_or(json!("")),
                "parameters": fun.get("parameters").cloned().unwrap_or(json!({"type":"object"})),
            }))
        })
        .collect();
    if decls.is_empty() {
        Value::Array(Vec::new())
    } else {
        json!([{ "functionDeclarations": decls }])
    }
}

/// IR -> `(instructions, input)` da OpenAI Responses API.
fn responses_input(messages: &[Value]) -> (Option<String>, Vec<Value>) {
    let mut instructions = Vec::new();
    let mut input = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => instructions.push(text_of(m)),
            "assistant" => {
                let content = text_of(m);
                if !content.is_empty() {
                    input.push(json!({"role": "assistant", "content": content}));
                }
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    for tc in calls {
                        let fun = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                        input.push(json!({
                            "type": "function_call",
                            "call_id": tc.get("id").cloned().unwrap_or(json!("")),
                            "name": fun.get("name").cloned().unwrap_or(json!("")),
                            "arguments": fun.get("arguments").cloned().unwrap_or(json!("{}")),
                        }));
                    }
                }
            }
            "tool" => input.push(json!({
                "type": "function_call_output",
                "call_id": m.get("tool_call_id").cloned().unwrap_or(json!("")),
                "output": text_of(m),
            })),
            _ => {
                let mut text = String::new();
                let mut images = Vec::new();
                for part in ir_parts(m) {
                    match part {
                        IrPart::Text(t) => text.push_str(&t),
                        IrPart::Image { mime, data } => images.push(json!({
                            "type": "input_image",
                            "image_url": format!("data:{};base64,{}", mime, data),
                        })),
                    }
                }
                if !text.is_empty() {
                    input.push(json!({"role": role, "content": text}));
                }
                for img in images {
                    input.push(img);
                }
            }
        }
    }
    let instr = if instructions.is_empty() {
        None
    } else {
        Some(instructions.join("\n\n"))
    };
    (instr, input)
}

/// IR -> `(system, messages)` da Anthropic Messages API.
fn anthropic_messages(messages: &[Value]) -> (Option<String>, Vec<Value>) {
    let mut system = Vec::new();
    let mut out = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => system.push(text_of(m)),
            "assistant" => {
                let mut blocks = Vec::new();
                let content = text_of(m);
                if !content.is_empty() {
                    blocks.push(json!({"type": "text", "text": content}));
                }
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    for tc in calls {
                        let fun = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                        let raw = fun
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.get("id").cloned().unwrap_or(json!("")),
                            "name": fun.get("name").cloned().unwrap_or(json!("")),
                            "input": args_value(raw),
                        }));
                    }
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "assistant", "content": blocks}));
                }
            }
            "tool" => out.push(json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id").cloned().unwrap_or(json!("")),
                    "content": text_of(m),
                }],
            })),
            _ => {
                let mut text = String::new();
                let mut blocks = Vec::new();
                for part in ir_parts(m) {
                    match part {
                        IrPart::Text(t) => text.push_str(&t),
                        IrPart::Image { mime, data } => blocks.push(json!({
                            "type": "image",
                            "source": {"type": "base64", "media_type": mime, "data": data},
                        })),
                    }
                }
                if !text.is_empty() {
                    blocks.insert(0, json!({"type": "text", "text": text}));
                }
                if !blocks.is_empty() {
                    out.push(json!({"role": "user", "content": blocks}));
                }
            }
        }
    }
    let sys = if system.is_empty() {
        None
    } else {
        Some(system.join("\n\n"))
    };
    (sys, out)
}

/// IR -> `(systemInstruction, contents)` do Google Generative Language API.
fn google_contents(messages: &[Value]) -> (Option<Value>, Vec<Value>) {
    let mut system = Vec::new();
    let mut out = Vec::new();
    let mut names: BTreeMap<String, String> = BTreeMap::new();
    for m in messages {
        let role = m.get("role").and_then(Value::as_str).unwrap_or("");
        match role {
            "system" => system.push(text_of(m)),
            "assistant" => {
                let mut parts = Vec::new();
                let content = text_of(m);
                if !content.is_empty() {
                    parts.push(json!({"text": content}));
                }
                if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
                    for tc in calls {
                        let fun = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                        let name = fun
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let raw = fun
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        if let Some(id) = tc.get("id").and_then(Value::as_str) {
                            names.insert(id.to_string(), name.clone());
                        }
                        parts.push(json!({
                            "functionCall": {"name": name, "args": args_value(raw)},
                        }));
                    }
                }
                if !parts.is_empty() {
                    out.push(json!({"role": "model", "parts": parts}));
                }
            }
            "tool" => {
                let id = m
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = names.get(&id).cloned().unwrap_or_default();
                out.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": {"result": text_of(m)},
                        },
                    }],
                }));
            }
            _ => {
                let mut text = String::new();
                let mut parts = Vec::new();
                for part in ir_parts(m) {
                    match part {
                        IrPart::Text(t) => text.push_str(&t),
                        IrPart::Image { mime, data } => parts.push(json!({
                            "inlineData": {"mimeType": mime, "data": data},
                        })),
                    }
                }
                if !text.is_empty() {
                    parts.insert(0, json!({"text": text}));
                }
                if !parts.is_empty() {
                    out.push(json!({"role": "user", "parts": parts}));
                }
            }
        }
    }
    let sys = if system.is_empty() {
        None
    } else {
        Some(json!({"parts": [{"text": system.join("\n\n")}]}))
    };
    (sys, out)
}

// ---------------------------------------------------------------------------
// Generators por vendor.
// ---------------------------------------------------------------------------

/// OpenAI `/chat/completions` (e qualquer endpoint OpenAI-compativel).
pub fn openai_chat_generator(
    cloud: CloudProvider,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> impl Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String> {
    move |messages: Vec<Value>, emit: &mut dyn FnMut(RemoteEvent)| {
        let url = endpoint(&cloud.base_url, &cloud.router, "/chat/completions");
        let mut body = json!({
            "model": cloud.model,
            "messages": messages,
            "stream": true,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });
        let effort = cloud.reasoning_effort.trim();
        if !effort.is_empty() {
            body["reasoning_effort"] = Value::String(effort.to_string());
        }
        if has_tools(&tools) {
            body["tools"] = openai_chat_tools(&tools);
            body["tool_choice"] = Value::String("auto".to_string());
        }

        let client = http_client()?;
        let mut req = client.post(&url).json(&body);
        if let Some(key) = resolve_cloud_key(&cloud) {
            req = req.bearer_auth(key);
        }
        let started = Instant::now();
        let body_text = post_stream(req, &url)?;

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut tools_by_idx: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
        let mut usage: Option<(i64, i64, i64)> = None;

        for payload in sse_payloads(&body_text) {
            let Ok(chunk) = serde_json::from_str::<Value>(&payload) else {
                continue;
            };
            if let Some(u) = chunk.get("usage") {
                usage = Some((
                    u.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0),
                    u.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0),
                    u.get("total_tokens").and_then(Value::as_i64).unwrap_or(0),
                ));
            }
            let Some(choice) = chunk
                .get("choices")
                .and_then(Value::as_array)
                .and_then(|arr| arr.first())
            else {
                continue;
            };
            let delta = choice.get("delta").cloned().unwrap_or_else(|| json!({}));
            if let Some(c) = delta.get("content").and_then(Value::as_str) {
                if !c.is_empty() {
                    content.push_str(c);
                    emit(RemoteEvent::Content(c.to_string()));
                }
            }
            if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !r.is_empty() {
                    reasoning.push_str(r);
                    emit(RemoteEvent::Reasoning(r.to_string()));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for tc in calls {
                    let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let entry = tools_by_idx
                        .entry(index)
                        .or_insert_with(|| (String::new(), String::new(), String::new()));
                    if let Some(id) = tc.get("id").and_then(Value::as_str) {
                        entry.0 = id.to_string();
                    }
                    if let Some(fun) = tc.get("function") {
                        if let Some(n) = fun.get("name").and_then(Value::as_str) {
                            entry.1 = n.to_string();
                        }
                        if let Some(a) = fun.get("arguments").and_then(Value::as_str) {
                            entry.2.push_str(a);
                        }
                    }
                }
            }
        }

        let calls = collect_calls(tools_by_idx, emit);
        let output = finalize(content, reasoning, calls, usage, started);
        let _ = interrupt.load(Ordering::Relaxed);
        Ok(output)
    }
}

/// OpenAI `/responses`.
pub fn openai_responses_generator(
    cloud: CloudProvider,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> impl Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String> {
    move |messages: Vec<Value>, emit: &mut dyn FnMut(RemoteEvent)| {
        let url = endpoint(&cloud.base_url, &cloud.router, "/responses");
        let (instructions, input) = responses_input(&messages);
        let mut body = json!({
            "model": cloud.model,
            "input": input,
            "stream": true,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        });
        if let Some(instr) = instructions {
            body["instructions"] = Value::String(instr);
        }
        let effort = cloud.reasoning_effort.trim();
        if !effort.is_empty() {
            body["reasoning"] = json!({"effort": effort});
        }
        if has_tools(&tools) {
            body["tools"] = responses_tools(&tools);
            body["tool_choice"] = Value::String("auto".to_string());
        }

        let client = http_client()?;
        let mut req = client.post(&url).json(&body);
        if let Some(key) = resolve_cloud_key(&cloud) {
            req = req.bearer_auth(key);
        }
        let started = Instant::now();
        let body_text = post_stream(req, &url)?;

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut usage: Option<(i64, i64, i64)> = None;

        for payload in sse_payloads(&body_text) {
            let Ok(chunk) = serde_json::from_str::<Value>(&payload) else {
                continue;
            };
            match chunk.get("type").and_then(Value::as_str).unwrap_or("") {
                "response.output_text.delta" => {
                    if let Some(d) = chunk.get("delta").and_then(Value::as_str) {
                        if !d.is_empty() {
                            content.push_str(d);
                            emit(RemoteEvent::Content(d.to_string()));
                        }
                    }
                }
                "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                    if let Some(d) = chunk.get("delta").and_then(Value::as_str) {
                        if !d.is_empty() {
                            reasoning.push_str(d);
                            emit(RemoteEvent::Reasoning(d.to_string()));
                        }
                    }
                }
                "response.output_item.done" => {
                    let item = chunk.get("item").cloned().unwrap_or_else(|| json!({}));
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let raw = item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        if !name.is_empty() {
                            let tool = ToolCall {
                                call_id: if call_id.is_empty() {
                                    format!("call_{}", calls.len())
                                } else {
                                    call_id
                                },
                                command: name.clone(),
                                name,
                                args: args_value(raw),
                            };
                            emit(RemoteEvent::ToolCall(tool.clone()));
                            calls.push(tool);
                        }
                    }
                }
                "response.completed" => {
                    if let Some(u) = chunk.get("response").and_then(|r| r.get("usage")) {
                        usage = Some((
                            u.get("input_tokens").and_then(Value::as_i64).unwrap_or(0),
                            u.get("output_tokens").and_then(Value::as_i64).unwrap_or(0),
                            u.get("total_tokens").and_then(Value::as_i64).unwrap_or(0),
                        ));
                    }
                }
                _ => {}
            }
        }

        let output = finalize(content, reasoning, calls, usage, started);
        let _ = interrupt.load(Ordering::Relaxed);
        Ok(output)
    }
}

/// Anthropic Messages API.
pub fn anthropic_generator(
    cloud: CloudProvider,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> impl Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String> {
    move |messages: Vec<Value>, emit: &mut dyn FnMut(RemoteEvent)| {
        let url = endpoint(&cloud.base_url, &cloud.router, "/v1/messages");
        let (system, msgs) = anthropic_messages(&messages);
        let mut body = json!({
            "model": cloud.model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "stream": true,
            "temperature": temperature,
        });
        if let Some(sys) = system {
            body["system"] = Value::String(sys);
        }
        if has_tools(&tools) {
            let t = anthropic_tools(&tools);
            if has_tools(&t) {
                body["tools"] = t;
            }
        }

        let client = http_client()?;
        let mut req = client
            .post(&url)
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        if let Some(key) = resolve_cloud_key(&cloud) {
            req = req.header("x-api-key", key);
        }
        let started = Instant::now();
        let body_text = post_stream(req, &url)?;

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        // index do bloco -> (id, name, json parcial)
        let mut blocks: BTreeMap<usize, (String, String, String)> = BTreeMap::new();
        let mut usage: Option<(i64, i64, i64)> = None;

        for payload in sse_payloads(&body_text) {
            let Ok(chunk) = serde_json::from_str::<Value>(&payload) else {
                continue;
            };
            match chunk.get("type").and_then(Value::as_str).unwrap_or("") {
                "message_start" => {
                    if let Some(u) = chunk
                        .get("message")
                        .and_then(|m| m.get("usage"))
                        .and_then(|u| u.get("input_tokens"))
                        .and_then(Value::as_i64)
                    {
                        usage.get_or_insert((0, 0, 0)).0 = u;
                    }
                }
                "content_block_start" => {
                    let index = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let block = chunk.get("content_block").cloned().unwrap_or_else(|| json!({}));
                    if block.get("type").and_then(Value::as_str) == Some("tool_use") {
                        let id = block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        blocks.insert(index, (id, name, String::new()));
                    }
                }
                "content_block_delta" => {
                    let index = chunk.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                    let delta = chunk.get("delta").cloned().unwrap_or_else(|| json!({}));
                    match delta.get("type").and_then(Value::as_str).unwrap_or("") {
                        "text_delta" => {
                            if let Some(t) = delta.get("text").and_then(Value::as_str) {
                                if !t.is_empty() {
                                    content.push_str(t);
                                    emit(RemoteEvent::Content(t.to_string()));
                                }
                            }
                        }
                        "thinking_delta" => {
                            if let Some(t) = delta.get("thinking").and_then(Value::as_str) {
                                if !t.is_empty() {
                                    reasoning.push_str(t);
                                    emit(RemoteEvent::Reasoning(t.to_string()));
                                }
                            }
                        }
                        "input_json_delta" => {
                            if let Some(p) = delta.get("partial_json").and_then(Value::as_str) {
                                if let Some(entry) = blocks.get_mut(&index) {
                                    entry.2.push_str(p);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                "message_delta" => {
                    if let Some(out) = chunk
                        .get("usage")
                        .and_then(|u| u.get("output_tokens"))
                        .and_then(Value::as_i64)
                    {
                        let e = usage.get_or_insert((0, 0, 0));
                        e.1 = out;
                        e.2 = e.0 + e.1;
                    }
                }
                _ => {}
            }
        }

        for (_, (id, name, raw)) in blocks {
            if name.is_empty() {
                continue;
            }
            let tool = ToolCall {
                call_id: if id.is_empty() {
                    format!("call_{}", calls.len())
                } else {
                    id
                },
                command: name.clone(),
                name,
                args: args_value(&raw),
            };
            emit(RemoteEvent::ToolCall(tool.clone()));
            calls.push(tool);
        }

        let output = finalize(content, reasoning, calls, usage, started);
        let _ = interrupt.load(Ordering::Relaxed);
        Ok(output)
    }
}

/// Google Generative Language API (`streamGenerateContent`).
pub fn google_generator(
    cloud: CloudProvider,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> impl Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String> {
    move |messages: Vec<Value>, emit: &mut dyn FnMut(RemoteEvent)| {
        let router = if cloud.router.trim().is_empty() {
            "streamGenerateContent".to_string()
        } else {
            cloud.router.trim().trim_start_matches('/').to_string()
        };
        let base = cloud.base_url.trim().trim_end_matches('/');
        let url = format!("{}/models/{}:{}?alt=sse", base, cloud.model, router);
        let (system, contents) = google_contents(&messages);
        let mut body = json!({
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        });
        if let Some(sys) = system {
            body["systemInstruction"] = sys;
        }
        if has_tools(&tools) {
            let t = google_tools(&tools);
            if has_tools(&t) {
                body["tools"] = t;
            }
        }

        let client = http_client()?;
        let mut req = client.post(&url).json(&body);
        if let Some(key) = resolve_cloud_key(&cloud) {
            req = req.header("x-goog-api-key", key);
        }
        let started = Instant::now();
        let body_text = post_stream(req, &url)?;

        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls: Vec<ToolCall> = Vec::new();
        let mut usage: Option<(i64, i64, i64)> = None;

        for payload in sse_payloads(&body_text) {
            let Ok(chunk) = serde_json::from_str::<Value>(&payload) else {
                continue;
            };
            if let Some(u) = chunk.get("usageMetadata") {
                usage = Some((
                    u.get("promptTokenCount").and_then(Value::as_i64).unwrap_or(0),
                    u.get("candidatesTokenCount").and_then(Value::as_i64).unwrap_or(0),
                    u.get("totalTokenCount").and_then(Value::as_i64).unwrap_or(0),
                ));
            }
            let Some(parts) = chunk
                .get("candidates")
                .and_then(Value::as_array)
                .and_then(|c| c.first())
                .and_then(|c| c.get("content"))
                .and_then(|c| c.get("parts"))
                .and_then(Value::as_array)
            else {
                continue;
            };
            for part in parts {
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    if !t.is_empty() {
                        content.push_str(t);
                        emit(RemoteEvent::Content(t.to_string()));
                    }
                }
                if part.get("thought").and_then(Value::as_bool).unwrap_or(false) {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        if !t.is_empty() {
                            reasoning.push_str(t);
                            emit(RemoteEvent::Reasoning(t.to_string()));
                        }
                    }
                }
                if let Some(fc) = part.get("functionCall") {
                    let name = fc
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string();
                    if !name.is_empty() {
                        let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                        let tool = ToolCall {
                            call_id: format!("call_{}", calls.len()),
                            command: name.clone(),
                            name,
                            args,
                        };
                        emit(RemoteEvent::ToolCall(tool.clone()));
                        calls.push(tool);
                    }
                }
            }
        }

        let output = finalize(content, reasoning, calls, usage, started);
        let _ = interrupt.load(Ordering::Relaxed);
        Ok(output)
    }
}

fn collect_calls(
    tools_by_idx: BTreeMap<usize, (String, String, String)>,
    emit: &mut dyn FnMut(RemoteEvent),
) -> Vec<ToolCall> {
    let mut calls: Vec<ToolCall> = Vec::new();
    for (idx, (call_id, name, args_raw)) in tools_by_idx {
        if name.is_empty() {
            continue;
        }
        let tool = ToolCall {
            call_id: if call_id.is_empty() {
                format!("call_{}", idx)
            } else {
                call_id
            },
            command: name.clone(),
            name,
            args: args_value(&args_raw),
        };
        emit(RemoteEvent::ToolCall(tool.clone()));
        calls.push(tool);
    }
    calls
}

fn finalize(
    content: String,
    reasoning: String,
    calls: Vec<ToolCall>,
    usage: Option<(i64, i64, i64)>,
    started: Instant,
) -> RemoteModelOutput {
    let _ = reasoning;
    let (prompt_tokens, completion_tokens, total_tokens) = usage.unwrap_or((0, 0, 0));
    let tps = if completion_tokens > 0 {
        Some(completion_tokens as f64 / started.elapsed().as_secs_f64().max(0.001))
    } else {
        None
    };
    RemoteModelOutput {
        content,
        calls,
        tokens: (total_tokens > 0).then_some(total_tokens),
        tps,
        context_tokens: (prompt_tokens > 0).then_some(prompt_tokens),
    }
}

fn post_stream(
    req: reqwest::blocking::RequestBuilder,
    url: &str,
) -> Result<String, String> {
    let mut resp = req
        .send()
        .map_err(|err| format!("falha ao chamar {}: {}", url, err))
        .and_then(|resp| {
            if resp.status().is_success() {
                Ok(resp)
            } else {
                let status = resp.status().as_u16();
                let text = resp.text().unwrap_or_default();
                Err(format!("POST {} -> HTTP {}: {}", url, status, text))
            }
        })?;
    let mut body_text = String::new();
    resp.read_to_string(&mut body_text)
        .map_err(|err| format!("erro ao ler stream de {}: {}", url, err))?;
    Ok(body_text)
}

fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent("drill/0.1")
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(|err| format!("erro ao criar cliente HTTP: {}", err))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img_msg(text: &str) -> Value {
        json!({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": "data:image/png;base64,QUJDRA=="}},
            ],
        })
    }

    #[test]
    fn responses_translates_image() {
        let (instructions, input) = responses_input(&[img_msg("veja isso")]);
        assert!(instructions.is_none());
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "veja isso");
        assert_eq!(input[1]["type"], "input_image");
        assert_eq!(input[1]["image_url"], "data:image/png;base64,QUJDRA==");
    }

    #[test]
    fn anthropic_translates_image() {
        let (system, msgs) = anthropic_messages(&[img_msg("veja isso")]);
        assert!(system.is_none());
        assert_eq!(msgs.len(), 1);
        let blocks = msgs[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "veja isso");
        assert_eq!(blocks[1]["type"], "image");
        assert_eq!(blocks[1]["source"]["type"], "base64");
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
        assert_eq!(blocks[1]["source"]["data"], "QUJDRA==");
    }

    #[test]
    fn google_translates_image() {
        let (system, contents) = google_contents(&[img_msg("veja isso")]);
        assert!(system.is_none());
        assert_eq!(contents.len(), 1);
        let parts = contents[0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["text"], "veja isso");
        assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
        assert_eq!(parts[1]["inlineData"]["data"], "QUJDRA==");
    }

    #[test]
    fn text_content_unchanged_without_images() {
        let msg = json!({"role": "user", "content": "soma dois mais dois"});
        let (_, input) = responses_input(&[msg.clone()]);
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["content"], "soma dois mais dois");
        let (_, msgs) = anthropic_messages(&[msg.clone()]);
        assert_eq!(msgs[0]["content"][0]["text"], "soma dois mais dois");
        let (_, contents) = google_contents(&[msg]);
        assert_eq!(contents[0]["parts"][0]["text"], "soma dois mais dois");
    }
}
