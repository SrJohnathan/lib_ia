use crate::history::History;
use crate::provider::{self, ProviderStore, Selected};
use base64::Engine as _;
use lib_rust::agents::{AgentMode, AgentOrchestrator, EventSink, TurnResult, MAESTRO_AGENT_ID};
use lib_rust::RuntimeLlama;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

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
    /// Engine local (RuntimeLlama) — espelho do que a fila usa; permite anexar
    /// imagens (add_media_file) antes do turno local. `None` = sem runtime.
    engine: Option<Arc<Mutex<RuntimeLlama>>>,
    /// Diretorio de trabalho: resolve as mencoes `@arquivo` do prompt.
    project_dir: PathBuf,
}

/// Imagem resolvida de uma menção `@arquivo` no prompt.
struct PublishedImage {
    /// Token como apareceu no prompt (ex.: `rel/foto.png`).
    display: String,
    /// Caminho absoluto do arquivo.
    path: PathBuf,
    mime: String,
    base64: String,
}

/// Resultado do preparo de um turno: prompt final + imagens anexáveis.
struct PreparedTurn {
    prompt: String,
    /// Imagens que serao anexadas ao modelo (multimodal ativo).
    images: Vec<PublishedImage>,
    /// Avisos de imagem enviados à UI (anexadas ou ignoradas).
    ui_notices: Vec<String>,
}

const IMAGE_EXTS: [&str; 3] = ["png", "jpg", "jpeg"];

fn is_image_ext(name: &str) -> bool {
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

fn mime_of(name: &str) -> String {
    let ext = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext.to_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg".to_string(),
        _ => "image/png".to_string(),
    }
}

impl Agent {
    pub fn new(
        engine: Option<Arc<Mutex<RuntimeLlama>>>,
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
            engine.clone(),
            project_dir.clone(),
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
            engine,
            project_dir,
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
    /// API remota, conforme o provider ativo (`/provider`). Retorna o resultado
    /// e avisos de anexos de imagem (anexadas ou ignoradas) para a UI.
    pub fn run_turn(
        &mut self,
        user_input: &str,
    ) -> Result<(TurnResult, Vec<String>), String> {
        match self.provider.current() {
            Selected::Local(p) => {
                let multimodal = self.engine.is_some() && !p.mmproj.trim().is_empty();
                let prep = self.prepare_turn(user_input, multimodal);
                if !prep.images.is_empty() {
                    self.attach_local_media(&prep.images)?;
                }
                let result = self.orchestrator.run_turn(&prep.prompt);
                if !prep.images.is_empty() {
                    self.detach_local_media();
                }
                Ok((result?, prep.ui_notices))
            }
            Selected::Cloud(cloud) => {
                let prep = self.prepare_turn(user_input, true);
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
                let media: Vec<Value> = prep
                    .images
                    .iter()
                    .map(|img| {
                        json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{};base64,{}", img.mime, img.base64),
                            },
                        })
                    })
                    .collect();
                let result = self.orchestrator.manager().run_remote_turn(
                    MAESTRO_AGENT_ID,
                    &prep.prompt,
                    "root",
                    true,
                    &media,
                    generator.as_ref(),
                );
                Ok((result?, prep.ui_notices))
            }
        }
    }

    /// Anexa as imagens resolvidas ao engine local (runtime) antes do turno.
    fn attach_local_media(&self, images: &[PublishedImage]) -> Result<(), String> {
        let Some(engine) = &self.engine else {
            return Ok(());
        };
        let mut guard = engine.lock().map_err(|e| e.to_string())?;
        let _ = guard.clear_media();
        for img in images {
            guard
                .add_media_file(&img.path.display().to_string())
                .map_err(|err| format!("falha ao anexar imagem {}: {}", img.display, err))?;
        }
        Ok(())
    }

    /// Remove os anexos do engine local apos o turno.
    fn detach_local_media(&self) {
        if let Some(engine) = &self.engine {
            let _ = engine.lock().map(|mut g| g.clear_media());
        }
    }

    /// Resolve um caminho de menção `@arquivo` (relativo ao diretorio de
    /// trabalho, ou absoluto) e confirma que o arquivo existe.
    fn resolve_mention(&self, token: &str) -> Option<PathBuf> {
        let p = PathBuf::from(token);
        let candidate = if p.is_absolute() {
            p
        } else {
            self.project_dir.join(&p)
        };
        candidate.is_file().then_some(candidate)
    }

    /// Varre o input em busca de menções `@arquivo`:
    /// - não-imagem (ou inexistente): mantém o token literal (fluxo atual);
    /// - imagem + multimodal ativo: anexa ao modelo (local/cloud);
    /// - imagem + multimodal desativado: mantém o token e injeta um aviso no
    ///   prompt (a imagem não é enviada).
    fn prepare_turn(&self, user_input: &str, multimodal: bool) -> PreparedTurn {
        const AVISO: &str = "[AVISO] O usuario anexou imagens que o modelo atual NAO processa \
                             (multimodal desativado); elas nao foram enviadas ao modelo:";
        let mut prompt = String::with_capacity(user_input.len() + 128);
        let mut images = Vec::new();
        let mut ui_notices = Vec::new();
        let mut skipped = Vec::new();

        let chars: Vec<char> = user_input.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] != '@' {
                prompt.push(chars[i]);
                i += 1;
                continue;
            }
            let mut j = i + 1;
            while j < chars.len() && !chars[j].is_whitespace() {
                j += 1;
            }
            if j == i + 1 || chars[i + 1] == '@' {
                prompt.push('@');
                i += 1;
                continue;
            }
            let token: String = chars[i + 1..j].iter().collect();
            let rel: String = chars[i..j].iter().collect();
            let is_image = is_image_ext(&token) && self.resolve_mention(&token).is_some();
            if is_image && multimodal {
                if let Some(path) = self.resolve_mention(&token) {
                    let bytes = std::fs::read(&path).unwrap_or_default();
                    let base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    images.push(PublishedImage {
                        display: token.clone(),
                        mime: mime_of(&token),
                        base64,
                        path,
                    });
                    ui_notices
                        .push(format!("imagem adicionada ao contexto: {}", token));
                }
            } else if is_image {
                skipped.push(token.clone());
                ui_notices
                    .push(format!("imagem ignorada (modelo sem suporte a imagens): {}", token));
            }
            prompt.push_str(&rel);
            i = j;
        }

        if !skipped.is_empty() {
            prompt.push('\n');
            prompt.push_str(AVISO);
            prompt.push(' ');
            prompt.push_str(&skipped.join(", "));
            prompt.push('.');
        }

        PreparedTurn {
            prompt,
            images,
            ui_notices,
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

        // Solo + cloud: não faz sentido manter o GGUF na memória
        if self.provider.is_cloud() && matches!(self.mode(), Mode::Solo) {
            cancel_generation();
            self.orchestrator.manager().unload_engine();
        }

        if let Some(cloud) = self.provider.selected_cloud() {
            if cloud.model.trim().is_empty() {
                let models = provider::list_models(&cloud)?;
                let picked = provider::pick_chat_model(&models)
                    .ok_or_else(|| "endpoint nao listou modelos de chat".to_string())?;
                self.provider
                    .set_model_and_effort(picked.clone(), cloud.reasoning_effort.clone())?;
                return Ok(format!(
                    "{} (modelo '{}' adotado de /models; local descarregado)\n{}",
                    self.provider_summary(),
                    picked,
                    models.join("\n")
                ));
            }
            return Ok(format!(
                "{} (modelo local descarregado)",
                self.provider_summary()
            ));
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
        &mut self,
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
        let engine = Arc::new(Mutex::new(engine));
        self.engine = Some(Arc::clone(&engine));
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

    fn write_img(dir: &std::path::Path, name: &str, bytes: &[u8]) {
        std::fs::write(dir.join(name), bytes).unwrap();
    }

    #[test]
    fn mention_non_image_stays_literal() {
        let (dir, path) = temp_config("img1", false);
        let agent = make_agent(&path, &dir);
        std::fs::write(dir.join("main.rs"), b"fn main() {}").unwrap();
        let prep = agent.prepare_turn("leia @main.rs e me diga", true);
        assert!(prep.prompt.contains("@main.rs"));
        assert!(prep.images.is_empty());
        assert!(prep.ui_notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mention_image_multimodal_attaches() {
        let (dir, path) = temp_config("img2", false);
        let agent = make_agent(&path, &dir);
        write_img(&dir, "foto.png", b"\x89PNG fake bytes");
        let prep = agent.prepare_turn("descreva @foto.png", true);
        assert!(prep.prompt.contains("@foto.png"), "prompt={}", prep.prompt);
        assert_eq!(prep.images.len(), 1);
        let img = &prep.images[0];
        assert_eq!(img.mime, "image/png");
        assert_eq!(
            img.base64,
            base64::engine::general_purpose::STANDARD.encode(b"\x89PNG fake bytes")
        );
        assert!(img.path.is_file());
        assert_eq!(prep.ui_notices.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mention_image_no_multimodal_injects_notice() {
        let (dir, path) = temp_config("img3", false);
        let agent = make_agent(&path, &dir);
        write_img(&dir, "foto.png", b"\x89PNG fake bytes");
        let prep = agent.prepare_turn("descreva @foto.png", false);
        assert!(prep.images.is_empty());
        assert!(prep.prompt.contains("@foto.png"));
        assert!(
            prep.prompt.contains("[AVISO]"),
            "prompt deveria conter aviso: {}",
            prep.prompt
        );
        assert_eq!(prep.ui_notices.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mention_missing_image_treated_as_text() {
        let (dir, path) = temp_config("img4", false);
        let agent = make_agent(&path, &dir);
        let prep = agent.prepare_turn("e se @naoexiste.png?", true);
        assert!(prep.prompt.contains("@naoexiste.png"));
        assert!(prep.images.is_empty());
        assert!(prep.ui_notices.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mention_jpeg_and_uppercase_ext() {
        let (dir, path) = temp_config("img5", false);
        let agent = make_agent(&path, &dir);
        write_img(&dir, "foto.JPG", b"jpeg bytes");
        write_img(&dir, "scan.jpeg", b"jpeg bytes 2");
        let prep = agent.prepare_turn("veja @foto.JPG e @scan.jpeg", true);
        assert_eq!(prep.images.len(), 2);
        assert!(prep.images.iter().all(|i| i.mime == "image/jpeg"));
        assert_eq!(prep.ui_notices.len(), 2);
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
