//! Provider do Root (solo/maestro). Dois provedores:
//!
//! - `Local`  — llama.cpp nativo (RuntimeLlama) carregado no processo.
//! - `OpenAi` — qualquer API OpenAI-compativel (`/v1/chat/completions`),
//!   incluindo OpenAI, Ollama, LM Studio, llama.cpp server, vLLM, etc.
//!
//! Subagentes NAO seguem o provider: nao importam esta escolha e rodam sempre
//! no runtime local (queru `ask_agent`, quer modo solo/maestro).

use crate::config;
use lib_rust::agents::{RemoteEvent, RemoteModelOutput};
use lib_rust::ToolCall;
use serde_json::{json, Value};
use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Local,
    OpenAi,
}

impl ProviderKind {
    pub fn label(&self) -> &'static str {
        match self {
            ProviderKind::Local => "local",
            ProviderKind::OpenAi => "openai",
        }
    }
    pub fn from_str(s: &str) -> Self {
        if s == config::DEFAULT_PROVIDER_OPENAI {
            ProviderKind::OpenAi
        } else {
            ProviderKind::Local
        }
    }
}

/// Configuracao efetiva do provider no momento.
#[derive(Debug, Clone)]
pub struct ProviderSettings {
    pub kind: ProviderKind,
    pub base_url: String,
    pub model: String,
    /// Resolvido: env `OPENAI_API_KEY` quando presentes; senao o valor do
    /// config.json. None = sem chave configurada.
    pub api_key: Option<String>,
}

impl ProviderSettings {
    pub fn summary(&self) -> String {
        match self.kind {
            ProviderKind::Local => format!("local ({})", self.model),
            ProviderKind::OpenAi => {
                let key = if self.api_key.is_some() { "key ok" } else { "SEM API KEY" };
                format!("openai: {} [{}] @ {}", self.model, key, self.base_url)
            }
        }
    }
}

/// Estado compartilhado do provider entre a TUI (escolhe) e o worker (turnos).
/// Persiste a escolha em `~/.drill/config.json`.
pub struct ProviderStore {
    pub settings: Arc<Mutex<ProviderSettings>>,
    pub config_path: PathBuf,
}

impl ProviderStore {
    pub fn new(cfg: &config::DrillConfig, config_path: PathBuf) -> Self {
        let kind = ProviderKind::from_str(&cfg.provider);
        let base_url = if cfg.openai_base_url.trim().is_empty() {
            config::DEFAULT_OPENAI_BASE_URL.to_string()
        } else {
            cfg.openai_base_url.clone()
        };
        let settings = ProviderSettings {
            kind,
            base_url,
            model: cfg.openai_model.clone(),
            api_key: resolve_api_key(&cfg.openai_api_key),
        };
        Self {
            settings: Arc::new(Mutex::new(settings)),
            config_path,
        }
    }

    pub fn current(&self) -> ProviderSettings {
        self.settings.lock().unwrap().clone()
    }

    pub fn kind(&self) -> ProviderKind {
        self.current().kind
    }

    /// Troca o provider. Persiste em config.json. `openai_model` vazio e
    /// preenchido pelo chamador (worker) via `first_available_model` quando o
    /// usuario escolhe a API sem modelo definido.
    pub fn set_kind(&self, kind: ProviderKind) {
        let mut guard = self.settings.lock().unwrap();
        guard.kind = kind;
        drop(guard);
        let path = self.config_path.clone();
        let _ = config::update(&path, |cfg| {
            cfg.provider = kind.label().to_string();
        });
    }

    /// Define/limpa o modelo da API remota, persistindo.
    pub fn set_openai(&self, base_url: Option<String>, model: String) {
        let url_for_cfg = base_url.clone();
        let mut guard = self.settings.lock().unwrap();
        if let Some(url) = base_url {
            guard.base_url = url;
        }
        guard.model = model.clone();
        drop(guard);
        let path = self.config_path.clone();
        let _ = config::update(&path, |cfg| {
            cfg.provider = ProviderKind::OpenAi.label().to_string();
            if let Some(url) = url_for_cfg {
                cfg.openai_base_url = url;
            }
            cfg.openai_model = model;
        });
    }

    /// Chave configurada? (env vence o config.json.)
    pub fn has_api_key(&self) -> bool {
        let guard = self.settings.lock().unwrap();
        guard.api_key.is_some()
    }
}

/// Resolve a chave: env `OPENAI_API_KEY` primeiro, depois o campo do config.
pub fn resolve_api_key(config_key: &str) -> Option<String> {
    if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let trimmed = key.trim().to_string();
        if !trimmed.is_empty() {
            return Some(trimmed);
        }
    }
    let trimmed = config_key.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

pub fn base_url_with_env(configured: &str) -> String {
    std::env::var("OPENAI_BASE_URL")
        .map(|v| {
            let t = v.trim().trim_end_matches('/');
            if t.is_empty() {
                configured.to_string()
            } else {
                t.to_string()
            }
        })
        .unwrap_or_else(|_| configured.trim_end_matches('/').to_string())
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

/// LISTA os modelos disponiveis: `GET {base}/models`. Retorna os `id`s.
pub fn list_models(base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    let url = format!("{}/models", base_url_with_env(base_url));
    let client = http_client()?;
    let mut req = client.get(&url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req
        .send()
        .map_err(|err| format!("falha ao listar modelos em {}: {}", url, err))?;
    if !resp.status().is_success() {
        return Err(format!(
            "GET {} -> HTTP {}",
            url,
            resp.status().as_u16()
        ));
    }
    let body: Value = resp
        .json()
        .map_err(|err| format!("resposta invalida de {}: {}", url, err))?;
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

/// Monta o closure de UM round remoto usado por `AgentManager::run_remote_turn`.
///
/// Faz um POST streaming em `{base}/chat/completions`, emite `Content`/
/// `Reasoning`/`ToolCall` ao vivo e devolve a saida estruturada. Respeita o
/// `interrupt` compartilhado (Esc/steer) entre chunks.
#[allow(clippy::too_many_arguments)]
pub fn openai_generator(
    base_url: String,
    api_key: Option<String>,
    model: String,
    tools: Value,
    temperature: f32,
    max_tokens: usize,
    interrupt: Arc<AtomicBool>,
) -> impl Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent)) -> Result<RemoteModelOutput, String> {
    move |messages: Vec<Value>, emit: &mut dyn FnMut(RemoteEvent)| {
        let url = format!("{}/chat/completions", base_url_with_env(&base_url));
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "temperature": temperature,
            "max_tokens": max_tokens,
        });
        if !tools.is_null() && !matches!(&tools, Value::Array(v) if v.is_empty()) {
            body["tools"] = tools.clone();
            body["tool_choice"] = Value::String("auto".to_string());
        }

        let client = http_client()?;
        let mut req = client.post(&url).json(&body);
        if let Some(key) = &api_key {
            req = req.bearer_auth(key);
        }
        let started = Instant::now();
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

        let mut content = String::new();
        let mut reasoning = String::new();
        // Names/args dos tool calls acumulados por indice do delta do stream.
        let mut tools_by_idx: std::collections::BTreeMap<usize, (String, String, String)> =
            std::collections::BTreeMap::new();
        let mut usage: Option<(i64, i64, i64)> = None;

        let mut buffer = Vec::new();
        let mut done = false;
        let mut body_text = String::new();
        resp.read_to_string(&mut body_text)
            .map_err(|err| format!("erro ao ler stream de {}: {}", url, err))?;
        buffer.extend_from_slice(body_text.as_bytes());

        for line in buffer.split(|&b| b == b'\n') {
            if done {
                break;
            }
            let line = String::from_utf8_lossy(line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let payload = line
                .strip_prefix("data:")
                .map(str::trim)
                .unwrap_or(line);
            if payload == "[DONE]" {
                done = true;
                break;
            }
            let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
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
                    let entry = tools_by_idx.entry(index).or_insert_with(|| (String::new(), String::new(), String::new()));
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

        let mut calls: Vec<ToolCall> = Vec::new();
        for (idx, (call_id, name, args_raw)) in tools_by_idx {
            if name.is_empty() {
                continue;
            }
            let args: Value = match serde_json::from_str(&args_raw) {
                Ok(value) => value,
                Err(_) => Value::String(args_raw),
            };
            let tool = ToolCall {
                call_id: if call_id.is_empty() { format!("call_{}", idx) } else { call_id },
                command: name.clone(),
                name,
                args,
            };
            emit(RemoteEvent::ToolCall(tool.clone()));
            calls.push(tool);
        }

        let (prompt_tokens, completion_tokens, total_tokens) = usage.unwrap_or((0, 0, 0));
        let tps = if completion_tokens > 0 {
            Some(completion_tokens as f64 / started.elapsed().as_secs_f64().max(0.001))
        } else {
            None
        };
        let output = RemoteModelOutput {
            content,
            calls,
            tokens: (total_tokens > 0).then_some(total_tokens),
            tps,
            context_tokens: (prompt_tokens > 0).then_some(prompt_tokens),
        };
        if interrupt.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok(output); // interrupcao tratada pelo loop (round termina aqui)
        }
        Ok(output)
    }
}

fn http_client() -> Result<reqwest::blocking::Client, String> {
    reqwest::blocking::Client::builder()
        .user_agent("drill/0.1")
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(|err| format!("erro ao criar cliente HTTP: {}", err))
}