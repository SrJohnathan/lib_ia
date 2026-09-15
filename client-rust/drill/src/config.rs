use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "/run/media/johnathan/Novo volume/Qwen3.8-27B-MTP-Q4_K_M.gguf";
pub const DEFAULT_PROVIDER_LOCAL: &str = "local";
pub const DEFAULT_PROVIDER_OPENAI: &str = "openai";
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
pub const DEFAULT_PERSONA: &str = "Voce e o drill, um agente de codigo que trabalha em um repositorio local. \
    Voce recebeu um conjunto de ferramentas (read_file, write_file, list_dir, run_command). \
    Sempre que precisar inspecionar arquivos, editar codigo ou executar comandos, use as ferramentas \
    em vez de inventar respostas. Quando terminar a tarefa, responda ao usuario com um resumo \
    claro do que fez. Responda em portugues, de forma concisa e objetiva.";

#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct DrillConfig {
    pub model: String,
    pub tools: bool,
    pub ngl: usize,
    pub ctx: usize,
    pub n_predict: usize,
    pub temp: f32,
    pub spec: String,
    pub persona: String,
    pub dir: Option<String>,
    pub threads: i64,
    pub threads_batch: i64,
    pub flash_attn: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
    /// Provider da geracao do Root (solo/maestro): "local" (llama.cpp nativo)
    /// ou "openai" (API OpenAI-compativel). Subagentes sao sempre "local".
    pub provider: String,
    pub openai_base_url: String,
    pub openai_model: String,
    /// Opcional: se vazio, le a env OPENAI_API_KEY.
    pub openai_api_key: String,
}

impl Default for DrillConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MODEL.to_string(),
            tools: true,
            ngl: 999,
            ctx: 8192,
            n_predict: 1024,
            temp: 0.6,
            spec: "draft-mtp".to_string(),
            persona: DEFAULT_PERSONA.to_string(),
            dir: None,
            threads: 12,
            threads_batch: 12,
            flash_attn: "on".to_string(),
            cache_type_k: "q4_0".to_string(),
            cache_type_v: "q4_0".to_string(),
            provider: DEFAULT_PROVIDER_LOCAL.to_string(),
            openai_base_url: DEFAULT_OPENAI_BASE_URL.to_string(),
            openai_model: "".to_string(),
            openai_api_key: "".to_string(),
        }
    }
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
    if path.is_file() {
        match fs::read_to_string(path).map_err(|err| err.to_string()).and_then(|text| {
            serde_json::from_str::<DrillConfig>(&text).map_err(|err| err.to_string())
        }) {
            Ok(cfg) => return (cfg, false),
            Err(err) => {
                eprintln!(
                    "[drill] config invalido em {} ({}); usando defaults",
                    path.display(),
                    err
                );
            }
        }
    }

    let cfg = DrillConfig::default();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match serde_json::to_string_pretty(&cfg) {
        Ok(text) => {
            if fs::write(path, text).is_ok() {
                eprintln!("[drill] config criado em {}", path.display());
            } else {
                eprintln!("[drill] config nao pode ser criado em {}", path.display());
            }
        }
        Err(err) => eprintln!("[drill] erro ao serializar config: {}", err),
    }
    (cfg, true)
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