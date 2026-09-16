use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "/run/media/johnathan/Novo volume/Qwen3.8-27B-MTP-Q4_K_M.gguf";
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
pub const DEFAULT_GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_PERSONA: &str = "Voce e o drill, um agente de codigo que trabalha em um repositorio local. \
    Voce recebeu um conjunto de ferramentas (read_file, write_file, list_dir, run_command). \
    Sempre que precisar inspecionar arquivos, editar codigo ou executar comandos, use as ferramentas \
    em vez de inventar respostas. Quando terminar a tarefa, responda ao usuario com um resumo \
    claro do que fez. Responda em portugues, de forma concisa e objetiva.";

/// Formato de wire de um provider remoto (campo `type-url` do config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudKind {
    OpenAi,
    Anthropic,
    Google,
}

impl Default for CloudKind {
    fn default() -> Self {
        CloudKind::OpenAi
    }
}

impl CloudKind {
    pub fn label(&self) -> &'static str {
        match self {
            CloudKind::OpenAi => "openai",
            CloudKind::Anthropic => "anthropic",
            CloudKind::Google => "google",
        }
    }

    pub fn base_url_default(&self) -> &'static str {
        match self {
            CloudKind::OpenAi => DEFAULT_OPENAI_BASE_URL,
            CloudKind::Anthropic => DEFAULT_ANTHROPIC_BASE_URL,
            CloudKind::Google => DEFAULT_GOOGLE_BASE_URL,
        }
    }
}

/// Provider local (llama.cpp nativo). Um GGUF + parametros de runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LocalProvider {
    pub id: String,
    pub name: String,
    pub model: String,
    pub ngl: usize,
    pub ctx: usize,
    pub n_predict: usize,
    pub temp: f32,
    pub spec: String,
    /// Multi-modal (llava/qwen2vl): projecao do modelo (arquivo mmproj). Vazio = sem.
    pub mmproj: String,
}

impl Default for LocalProvider {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "local (GGUF)".to_string(),
            model: DEFAULT_MODEL.to_string(),
            ngl: 999,
            ctx: 8192,
            n_predict: 1024,
            temp: 0.6,
            spec: "draft-mtp".to_string(),
            mmproj: String::new(),
        }
    }
}

/// Provider remoto. `router` e o path/acao do endpoint; `ctx` so serve para
/// exibir o uso de contexto na UI.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CloudProvider {
    pub id: String,
    pub name: String,
    #[serde(rename = "type-url")]
    pub kind: CloudKind,
    pub base_url: String,
    pub model: String,
    pub router: String,
    /// Opcional: se vazio, le a env do vendor (OPENAI/ANTHROPIC/GOOGLE_API_KEY).
    pub api_key: String,
    /// Forca do raciocinio: "" / "low" / "medium" / "high".
    pub reasoning_effort: String,
    pub ctx: usize,
}

impl Default for CloudProvider {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: "cloud".to_string(),
            kind: CloudKind::OpenAi,
            base_url: DEFAULT_OPENAI_BASE_URL.to_string(),
            model: String::new(),
            router: String::new(),
            api_key: String::new(),
            reasoning_effort: "medium".to_string(),
            ctx: 8192,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Providers {
    pub local: Vec<LocalProvider>,
    pub cloud: Vec<CloudProvider>,
}

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct DrillConfig {
    /// Modelo local padrao (usado se nenhum provider local for definido).
    pub model: String,
    pub tools: bool,
    pub persona: String,
    pub dir: Option<String>,
    pub threads: i64,
    pub threads_batch: i64,
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub providers: Providers,
    /// UUID do provider ativo (`providers.local` ou `providers.cloud`).
    #[serde(rename = "provider-selected-uuid")]
    pub provider_selected_uuid: Option<String>,
}

impl Default for DrillConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            tools: true,
            persona: DEFAULT_PERSONA.to_string(),
            dir: None,
            threads: 12,
            threads_batch: 12,
            flash_attn: "on".to_string(),
            cache_type_k: "q4_0".to_string(),
            cache_type_v: "q4_0".to_string(),
            providers: Providers::default(),
            provider_selected_uuid: None,
        }
    }
}

/// Gera um id (uuid-like) para um provider novo. Estavel o bastante para o
/// `provider-selected-uuid` persistido; nao usa crate `uuid`.
pub fn new_id(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{:x}-{}", prefix, nanos, std::process::id())
}

/// Garante invariantes minimas do config: ids preenchidos, pelo menos um
/// provider e `provider-selected-uuid` apontando para um id valido. Retorna
/// `true` se algo mudou (precisa regravar).
pub fn ensure_defaults(cfg: &mut DrillConfig) -> bool {
    let mut changed = false;

    if cfg.providers.local.is_empty() && cfg.providers.cloud.is_empty() {
        let mut local = LocalProvider::default();
        if !cfg.model.trim().is_empty() {
            local.model = cfg.model.clone();
        }
        local.id = new_id("local");
        cfg.providers.local.push(local);
        changed = true;
    }

    for (i, local) in cfg.providers.local.iter_mut().enumerate() {
        if local.id.trim().is_empty() {
            local.id = new_id(&format!("local{}", i));
            changed = true;
        }
        if local.name.trim().is_empty() {
            local.name = format!("local {}", i + 1);
            changed = true;
        }
    }
    for (i, cloud) in cfg.providers.cloud.iter_mut().enumerate() {
        if cloud.id.trim().is_empty() {
            cloud.id = new_id(&format!("cloud{}", i));
            changed = true;
        }
        if cloud.name.trim().is_empty() {
            cloud.name = format!("{} {}", cloud.kind.label(), i + 1);
            changed = true;
        }
        if cloud.base_url.trim().is_empty() {
            cloud.base_url = cloud.kind.base_url_default().to_string();
            changed = true;
        }
    }

    let selected_valid = cfg
        .provider_selected_uuid
        .as_deref()
        .map(|id| {
            cfg.providers.local.iter().any(|p| p.id == id)
                || cfg.providers.cloud.iter().any(|p| p.id == id)
        })
        .unwrap_or(false);
    if !selected_valid {
        let next = cfg
            .providers
            .local
            .first()
            .map(|p| p.id.clone())
            .or_else(|| cfg.providers.cloud.first().map(|p| p.id.clone()));
        cfg.provider_selected_uuid = next;
        changed = true;
    }

    changed
}

pub fn home_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Some(dir) = std::env::var_os("USERPROFILE") {
            return Some(PathBuf::from(dir));
        }
        if let (Some(drive), Some(path)) = (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
            return Some(PathBuf::from(drive).join(path));
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Some(dir) = std::env::var_os("HOME") {
            return Some(PathBuf::from(dir));
        }
    }
    None
}

pub fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("DRILL_CONFIG") {
        return PathBuf::from(path);
    }
    home_dir()
        .map(|home| home.join(".drill").join("config.json"))
        .unwrap_or_else(|| PathBuf::from(".drill").join("config.json"))
}

/// Loads ~/.drill/config.json. Creates it with defaults if missing or invalid.
/// Returns (config, created).
pub fn load_or_create(path: &Path) -> (DrillConfig, bool) {
    let mut created = false;
    let mut cfg = if path.is_file() {
        match fs::read_to_string(path).map_err(|err| err.to_string()).and_then(|text| {
            serde_json::from_str::<DrillConfig>(&text).map_err(|err| err.to_string())
        }) {
            Ok(cfg) => cfg,
            Err(err) => {
                eprintln!(
                    "[drill] config invalido em {} ({}); usando defaults",
                    path.display(),
                    err
                );
                created = true;
                DrillConfig::default()
            }
        }
    } else {
        created = true;
        DrillConfig::default()
    };

    let changed = ensure_defaults(&mut cfg);
    if created || changed {
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        match serde_json::to_string_pretty(&cfg) {
            Ok(text) => {
                if fs::write(path, text).is_ok() && created {
                    eprintln!("[drill] config criado em {}", path.display());
                }
            }
            Err(err) => eprintln!("[drill] erro ao serializar config: {}", err),
        }
    }
    (cfg, created)
}

/// Le `~/.drill/config.json`, aplica `f` sobre ele e grava de volta. Se o
/// arquivo nao existir/for invalido, parte dos defaults. Usado para persistir
/// a escolha de provider e o modelo da API.
pub fn update<F: FnOnce(&mut DrillConfig)>(path: &Path, f: F) -> Result<(), String> {
    let (mut cfg, _) = load_or_create(path);
    f(&mut cfg);
    let text = serde_json::to_string_pretty(&cfg).map_err(|err| err.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(path, text).map_err(|err| err.to_string())
}