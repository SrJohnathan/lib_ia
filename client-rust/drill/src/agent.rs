use crate::history::History;
use crate::provider::{self, ProviderStore, Selected};
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
    /// Troca o provider do Root pelo uuid (local ou cloud). Subagentes ignoram.
    SetProvider(String),
    /// Lista modelos do provider cloud ativo.
    ListModels,
    /// Define modelo + forca do raciocinio do provider cloud ativo (persiste).
    SetModelAndEffort(String, String),
    /// Configura o modelo local: constroi o engine e instala o runtime (persiste).
    SetLocalModel { model: String, mmproj: String },
    /// Adiciona um provider cloud novo (wizard `a` no overlay /provider):
    /// cria, persiste em config.json e o deixa ativo.
    AddCloudProvider {
        kind: crate::config::CloudKind,
        base_url: String,
        model: String,
        api_key: String,
    },
}

/// Cancela a geração nativa em andamento, de qualquer thread.
pub fn cancel_generation() {
    RuntimeLlama::cancel_all_generations();
}

/// Frente do drill para a fila de agentes. Solo e agentes compartilham o
/// MESMO caminho de geracao (`AgentOrchestrator` em `lib-rust`).
/// O provider escolhido (`/provider`) so redireciona o turno do ROOT: com
/// provider cloud, o turno vai para a API; subagentes (ask_agent) continuam
/// SEMPRE no runtime local.
pub struct Agent {
    orchestrator: AgentOrchestrator,
    pub provider: Arc<ProviderStore>,
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

    /// Turno unico para solo e agentes: enfileira no root (local) ou resolve na
    /// API remota, conforme o provider ativo (`/provider`).
    pub fn run_turn(&mut self, user_input: &str) -> Result<TurnResult, String> {
        match self.provider.current() {
            Selected::Local(_) => self.orchestrator.run_turn(user_input),
            Selected::Cloud(cloud) => {
                let tools: Value = if self.orchestrator.tools_enabled() {
                    serde_json::from_str(lib_rust::tools::TOOLS_JSON).unwrap_or(Value::Null)
                } else {
                    Value::Null
                };
                let generator = provider::build_generator(
                    &cloud,
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
                    generator.as_ref(),
                )
            }
        }
    }

    /// Resume do provider atual para a TUI/banner.
    pub fn provider_summary(&self) -> String {
        self.provider.summary()
    }

    /// Troca o provider ativo pelo uuid. Se o cloud escolhido nao tiver modelo,
    /// adota o primeiro id de `/models`.
    pub fn switch_provider(&self, uuid: String) -> Result<String, String> {
        self.provider.select(&uuid)?;
        if let Some(cloud) = self.provider.selected_cloud() {
            if cloud.model.trim().is_empty() {
                let models = provider::list_models(&cloud)?;
                let picked = provider::pick_chat_model(&models)
                    .ok_or_else(|| "endpoint nao listou modelos de chat".to_string())?;
                self.provider
                    .set_model_and_effort(picked.clone(), cloud.reasoning_effort.clone())?;
                return Ok(format!(
                    "{} (modelo '{}' adotado de /models)\n{}",
                    self.provider_summary(),
                    picked,
                    models.join("\n")
                ));
            }
        }
        Ok(self.provider_summary())
    }

    /// Lista os ids de modelos do provider cloud ativo.
    pub fn remote_models(&self) -> Result<Vec<String>, String> {
        let cloud = self
            .provider
            .selected_cloud()
            .ok_or_else(|| "o overlay /models so vale com provider cloud (/provider)".to_string())?;
        provider::list_models(&cloud)
    }

    /// Define o modelo e a forca do raciocinio do provider cloud, persistindo.
    pub fn set_model_and_effort(
        &self,
        model: String,
        reasoning_effort: String,
    ) -> Result<String, String> {
        self.provider.set_model_and_effort(model, reasoning_effort)?;
        Ok(self.provider_summary())
    }

    /// Constroi o engine local com o modelo e mmproj indicados e o instala
    /// no AgentManager (se ainda nao estiver rodando). Tambem persiste no
    /// config.json. Retorna (modelo, has_thinking).
    pub fn set_local_model(
        &self,
        model: String,
        mmproj: String,
        params: &crate::EngineParams,
    ) -> Result<(String, bool), String> {
        let mut engine = crate::build_runtime(&model, &mmproj, params)?;
        if self.orchestrator.tools_enabled() {
            engine
                .set_tools_json(lib_rust::tools::TOOLS_JSON)
                .map_err(|err| err.to_string())?;
        }
        let has_thinking = engine.supports_thinking().unwrap_or(false);
        let engine = std::sync::Arc::new(std::sync::Mutex::new(engine));
        self.orchestrator.manager().install_engine(engine);
        self.provider.set_local_model(model.clone(), mmproj)?;
        Ok((model, has_thinking))
    }

    /// Lista os modelos do provider cloud ativo em texto para a UI.
    pub fn list_models_remote(&self) -> Result<String, String> {
        let cloud = self
            .provider
            .selected_cloud()
            .ok_or_else(|| "o overlay /models so vale com provider cloud".to_string())?;
        let models = provider::list_models(&cloud)?;
        Ok(format!(
            "Modelos em {}\n\n{}",
            cloud.base_url,
            models
                .iter()
                .map(|m| {
                    if *m == cloud.model {
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
    use crate::config;
    use super::*;

    fn temp_config(tag: &str, cloud: bool) -> (std::path::PathBuf, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("drill-test-{}-{}", std::process::id(), tag));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        let text = if cloud {
            r#"{"model":"","tools":true,"providers":{"local":[],
               "cloud":[{"id":"c1","name":"groq","type-url":"openai",
                 "base_url":"https://api.groq.com/openai/v1","model":"",
                 "router":"/chat/completions","api_key":"","reasoning_effort":"medium","ctx":8192}]},
               "provider-selected-uuid":"c1"}"#
        } else {
            r#"{"model":"","tools":true,"providers":{"local":[],"cloud":[]},
               "provider-selected-uuid":null}"#
        };
        std::fs::write(&path, text).unwrap();
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
        let (dir, path) = temp_config("local", false);
        let agent = make_agent(&path, &dir);
        assert!(!agent.provider.is_cloud());
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
        let (dir, path) = temp_config("groq", true);
        let agent = make_agent(&path, &dir);
        let msg = agent.switch_provider("c1".to_string()).unwrap();
        assert!(msg.contains("groq") || msg.contains("openai"), "msg={}", msg);
        let cloud = agent.provider.selected_cloud().unwrap();
        assert!(!cloud.model.is_empty(), "modelo deveria ser adotado de /models");
        assert!(cloud.base_url.contains("groq.com"));
        let cfg = config::load_or_create(&path).0;
        assert_eq!(cfg.provider_selected_uuid.as_deref(), Some("c1"));
        assert_eq!(cfg.providers.cloud[0].model, cloud.model);
        let listing = agent.list_models_remote().unwrap();
        assert!(listing.contains("Modelos em"), "listing={}", listing);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
