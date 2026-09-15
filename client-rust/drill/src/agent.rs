use crate::history::History;
use crate::provider::{self, ProviderKind, ProviderStore};
use lib_rust::agents::{AgentMode, AgentOrchestrator, EventSink, TurnResult, MAESTRO_AGENT_ID};
use lib_rust::RuntimeLlama;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub type Mode = AgentMode;

pub enum WorkerMsg {
    Turn(String),
    SetMode(Mode),
    SetSession(String),
    /// Escolhe o provider do Root (solo/maestro). Subagentes ignoram.
    SetProvider(ProviderKind),
    /// Lista modelos da API OpenAI-compativel (so com provider remoto).
    ListModels,
}

/// Cancela a geração nativa em andamento, de qualquer thread.
pub fn cancel_generation() {
    RuntimeLlama::cancel_all_generations();
}

/// Frente do drill para a fila de agentes. Solo e agentes compartilham o
/// MESMO caminho de geracao (`AgentOrchestrator` em `lib-rust`).
/// O provider escolhido (`/provider`) so redireciona o turno do ROOT: com
/// provider remoto, o turno nao enfileira no engine local e sim na API;
/// subagentes (ask_agent) continuam SEMPRE no runtime local.
pub struct Agent {
    orchestrator: AgentOrchestrator,
    provider: Arc<ProviderStore>,
}

impl Agent {
    pub fn new(
        engine: Option<Arc<std::sync::Mutex<RuntimeLlama>>>,
        project_dir: PathBuf,
        tools_enabled: bool,
        ctx_max: usize,
        interrupt: Arc<AtomicBool>,
        persona: String,
        provider: Arc<ProviderStore>,
        temp: f32,
        n_predict: usize,
    ) -> Self {
        let orchestrator = AgentOrchestrator::new(
            engine,
            project_dir,
            tools_enabled,
            ctx_max,
            interrupt,
            persona,
            temp,
            n_predict,
        );

        Self {
            orchestrator,
            provider,
        }
    }

    /// Direciona os eventos de progresso da fila (stream/tool/status) para a UI.
    pub fn set_event_sink(&self, sink: EventSink) {
        self.orchestrator.set_event_sink(sink);
    }

    /// Registra o historico persistente (LanceDB): cada turno concluído vira
    /// uma linha gravada por uma thread separada.
    pub fn set_history(&self, history: Arc<History>) {
        self.orchestrator.set_history(Arc::new(move |row| {
            history.record(row);
        }));
    }

    pub fn set_mode(&mut self, mode: Mode) {
        self.orchestrator.set_mode(mode);
    }

    pub fn set_session(&mut self, session_id: String) {
        self.orchestrator.set_session(session_id);
    }

    pub fn mode(&self) -> Mode {
        self.orchestrator.mode()
    }

    pub fn ctx_max(&self) -> usize {
        self.orchestrator.ctx_max()
    }

    pub fn context_estimate(&self) -> usize {
        self.orchestrator.context_estimate()
    }

    /// Turno unico para solo e agentes: enfileira no root ou resolve na API
    /// remota, conforme o provider escolhido na TUI (`/provider`). Stats ja
    /// lidas na thread da geracao.
    pub fn run_turn(&mut self, user_input: &str) -> Result<TurnResult, String> {
        let provider = self.provider.kind();
        match provider {
            ProviderKind::Local => self.orchestrator.run_turn(user_input),
            ProviderKind::OpenAi => {
                let settings = self.provider.current();
                let tools: Value = if self.orchestrator.tools_enabled() {
                    serde_json::from_str(lib_rust::tools::TOOLS_JSON).unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let generator = provider::openai_generator(
                    settings.base_url,
                    settings.api_key,
                    settings.model,
                    tools,
                    self.orchestrator.temp(),
                    self.orchestrator.n_predict(),
                    Arc::clone(self.orchestrator.interrupt()),
                );
                self.orchestrator.manager().run_remote_turn(
                    MAESTRO_AGENT_ID,
                    user_input,
                    "root",
                    true,
                    &generator,
                )
            }
        }
    }

    /// Resume do provider atual para a TUI/banner.
    pub fn provider_summary(&self) -> String {
        self.provider.current().summary()
    }

    /// Troca o provider do root. Com OpenAi: exige API key (env ou config) e,
    /// se nao houver modelo definido, adota o primeiro id de `/models`.
    pub fn switch_provider(&self, kind: ProviderKind) -> Result<String, String> {
        match kind {
            ProviderKind::Local => {
                self.provider.set_kind(ProviderKind::Local);
                Ok(format!("provider: {}", self.provider_summary()))
            }
            ProviderKind::OpenAi => {
                let settings = self.provider.current();
                if settings.api_key.is_none() {
                    return Err(
                        "OpenAI API sem chave: defina OPENAI_API_KEY (env) ou openai_api_key \
                         no config.json antes de usar o provider 'openai'."
                            .to_string(),
                    );
                }
                self.provider.set_kind(ProviderKind::OpenAi);
                if settings.model.is_empty() {
                    let models =
                        provider::list_models(&settings.base_url, settings.api_key.as_deref())?;
                    let picked = provider::pick_chat_model(&models)
                        .ok_or_else(|| "endpoint nao listou modelos de chat".to_string())?;
                    self.provider.set_openai(None, picked.clone());
                    Ok(format!(
                        "provider: openai (modelo '{}' adotado de /models)\n{}",
                        picked,
                        models.join("\n")
                    ))
                } else {
                    Ok(format!("provider: openai ({})", self.provider_summary()))
                }
            }
        }
    }

    /// Lista os ids de modelos do endpoint OpenAI-compativel (so valido quando
    /// o provider atual e `openai`). Retorna texto pronto para a UI.
    pub fn list_models_remote(&self) -> Result<String, String> {
        let settings = self.provider.current();
        if settings.kind != ProviderKind::OpenAi {
            return Err(
                "o endpoint de modelos so existe com provider 'openai' (use /provider openai)"
                    .to_string(),
            );
        }
        let models = provider::list_models(&settings.base_url, settings.api_key.as_deref())?;
        Ok(format!(
            "Modelos em {}\n\n{}",
            settings.base_url,
            models
                .iter()
                .map(|m| {
                    if *m == settings.model {
                        format!("  * {} (atual)", m)
                    } else {
                        format!("  {}", m)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_config(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("drill-test-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{"model":"","tools":true,"provider":"local",
               "openai_base_url":"https://api.groq.com/openai/v1",
               "openai_model":"","openai_api_key":""}"#,
        )
        .unwrap();
        (dir, path)
    }

    fn make_agent(config_path: &std::path::Path, dir: &std::path::Path) -> Arc<Agent> {
        let cfg = config::load_or_create(config_path).0;
        let store = Arc::new(provider::ProviderStore::new(&cfg, config_path.to_path_buf()));
        Arc::new(Agent::new(
            None,
            dir.to_path_buf(),
            true,
            4096,
            Arc::new(AtomicBool::new(false)),
            "voce e um agente de teste global".to_string(),
            store,
            0.7,
            512,
        ))
    }

    #[test]
    fn provider_local_default_and_summary() {
        let (dir, path) = temp_config("local");
        let agent = make_agent(&path, &dir);
        assert_eq!(provider::ProviderKind::Local, agent.provider.kind());
        assert!(agent.provider_summary().contains("local"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Soh roda com rede + chave valida (env OPENAI_API_KEY). Real: Groq.
    #[test]
    fn groq_switch_and_list() {
        let key = std::env::var("OPENAI_API_KEY").unwrap_or_default();
        if key.is_empty() {
            eprintln!("pulando: OPENAI_API_KEY nao definida");
            return;
        }
        let (dir, path) = temp_config("groq");
        let agent = make_agent(&path, &dir);
        let msg = agent.switch_provider(provider::ProviderKind::OpenAi).unwrap();
        assert!(msg.contains("provider: openai"), "msg={}", msg);
        let cur = agent.provider.current();
        assert!(
            !cur.model.is_empty(),
            "modelo deveria ser adotado de /models"
        );
        assert!(cur.base_url.contains("groq.com"));
        let cfg = config::load_or_create(&path).0;
        assert_eq!(cfg.provider, provider::ProviderKind::OpenAi.label());
        assert_eq!(cfg.openai_model, cur.model);
        let listing = agent.list_models_remote().unwrap();
        assert!(listing.contains("Modelos em"), "listing={}", listing);
        let _ = std::fs::remove_dir_all(&dir);
    }
}