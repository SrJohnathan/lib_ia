use crate::tools;
use lib_rust::agents::{AgentRole, ConfigAgent};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub const AGENT_FILE_PREFIX: &str = "drill.";
pub const AGENT_FILE_SUFFIX: &str = ".agent.json";

/// Ferramentas base do time (disponiveis para subagentes). Ferramentas do
/// maestro (list/read/create/ask_agent) sao exclusivas do drill/maestro.
const BASE_TOOLS: &[&str] = &["read_file", "write_file", "list_dir", "patch_file", "run_command"];

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SubagentConfig {
    pub name: String,
    pub description: String,
    pub persona: String,
    pub temp: f32,
    pub max_tool_rounds: usize,
    pub allowed_tools: Vec<String>,
}

impl Default for SubagentConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            description: String::new(),
            persona: String::new(),
            temp: 0.6,
            max_tool_rounds: 4,
            allowed_tools: vec![
                "read_file".to_string(),
                "write_file".to_string(),
                "patch_file".to_string(),
                "run_command".to_string(),
            ],
        }
    }
}

impl SubagentConfig {
    pub fn file_name(&self) -> String {
        format!("{}{}{}", AGENT_FILE_PREFIX, self.name, AGENT_FILE_SUFFIX)
    }

    pub fn is_valid(&self) -> Result<(), String> {
        if self.name.trim().is_empty() {
            return Err("nome do agente vazio".into());
        }
        if !self
            .name
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return Err(format!("nome invalido '{}': use apenas letras, numeros, '_', '-' ou '.'", self.name));
        }
        if self.persona.trim().is_empty() {
            return Err("persona do agente vazia".into());
        }
        Ok(())
    }

    pub fn load_from_file(path: &Path) -> Result<SubagentConfig, String> {
        let text = fs::read_to_string(path).map_err(|err| err.to_string())?;
        let cfg: SubagentConfig = serde_json::from_str(&text).map_err(|err| err.to_string())?;
        cfg.is_valid()?;
        Ok(cfg)
    }

    /// Converte para o ConfigAgent da fila do AgentManager (sempre role Agent).
    pub fn to_config(&self) -> ConfigAgent {
        ConfigAgent {
            agent_id: self.name.clone(),
            role: AgentRole::Agent,
            prompt: self.persona.clone(),
            tools_json: tools_json_for(&self.allowed_tools),
            force_tools: false,
        }
    }
}

/// Diretorio padrao dos agentes: <config_dir>/agents (mesmo ~/.drill do config.json).
pub fn agents_root() -> PathBuf {
    config_root().join("agents")
}

pub fn config_root() -> PathBuf {
    crate::config::config_path()
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(".drill"))
}

fn is_agent_file(name: &str) -> bool {
    name.starts_with(AGENT_FILE_PREFIX) && name.ends_with(AGENT_FILE_SUFFIX)
}

/// Procura por arquivos `drill.<nome>.agent.json` no diretorio de agentes do
/// usuario e na raiz do projeto. O projeto tem precedencia sobre nomes iguais.
pub fn discover_agents(project_dir: &Path) -> Vec<SubagentConfig> {
    let mut all: Vec<SubagentConfig> = Vec::new();

    for root in [agents_root(), project_dir.to_path_buf()] {
        let entries = match fs::read_dir(&root) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        let mut files: Vec<PathBuf> = entries
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .map(|n| is_agent_file(&n.to_string_lossy()))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        for file in files {
            if let Ok(cfg) = SubagentConfig::load_from_file(&file) {
                all.push(cfg);
            }
        }
    }

    // Mantem a ultima ocorrencia por nome (projeto vence). Depois restaura a ordem.
    let mut out: Vec<SubagentConfig> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for cfg in all.into_iter().rev() {
        if seen.insert(cfg.name.clone()) {
            out.push(cfg);
        }
    }
    out.reverse();
    out
}

/// Salva o agente em <agents_root>/drill.<nome>.agent.json (cria o diretorio).
pub fn save_agent(cfg: &SubagentConfig) -> Result<PathBuf, String> {
    cfg.is_valid()?;
    let root = agents_root();
    fs::create_dir_all(&root).map_err(|err| err.to_string())?;
    let path = root.join(cfg.file_name());
    let text = serde_json::to_string_pretty(cfg).map_err(|err| err.to_string())?;
    fs::write(&path, text).map_err(|err| err.to_string())?;
    Ok(path)
}

/// Conteudo (json pretty) do arquivo de um agente, se existir.
pub fn read_agent(name: &str, project_dir: &Path) -> Result<String, String> {
    let file_name = format!("{}{}{}", AGENT_FILE_PREFIX, name, AGENT_FILE_SUFFIX);
    for root in [agents_root(), project_dir.to_path_buf()] {
        let path = root.join(&file_name);
        if path.is_file() {
            return fs::read_to_string(&path).map_err(|err| err.to_string());
        }
    }
    Err(format!("agente '{}' nao encontrado (procurei por {})", name, file_name))
}

pub fn find_agent(name: &str, project_dir: &Path) -> Result<SubagentConfig, String> {
    discover_agents(project_dir)
        .into_iter()
        .find(|cfg| cfg.name == name)
        .ok_or_else(|| format!("agente '{}' nao encontrado", name))
}

/// Monta o JSON de ferramentas de um subagente filtrando `TOOLS_JSON` pelo
/// `allowed_tools` (vazio = todas as ferramentas base; ferramentas do maestro
/// nunca sao incluidas). Retorna `None` sem nenhuma ferramenta habilitada
/// (agente responde em um unico round, sem loop de tools).
pub fn tools_json_for(allowed_tools: &[String]) -> Option<String> {
    let all: Value = serde_json::from_str(tools::TOOLS_JSON).ok()?;
    let array = all.as_array()?;

    let want: Vec<&str> = if allowed_tools.is_empty() {
        BASE_TOOLS.to_vec()
    } else {
        allowed_tools
            .iter()
            .map(String::as_str)
            .filter(|name| BASE_TOOLS.contains(name))
            .collect()
    };

    let filtered: Vec<&Value> = array
        .iter()
        .filter(|entry| {
            entry["function"]["name"]
                .as_str()
                .map(|name| want.contains(&name))
                .unwrap_or(false)
        })
        .collect();

    if filtered.is_empty() {
        None
    } else {
        serde_json::to_string(&filtered).ok()
    }
}