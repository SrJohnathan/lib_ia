use crate::agent;
use crate::history::{History, StoredSession};
use crate::tools::PermissionRequest;
use crate::tui::helpers::{char_to_byte, centered_rect, wrap_text, draw_provider_overlay, draw_models_overlay};
use crate::tui::highlight_code::CodeHighlighter;
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEvent, KeyEventKind,
    KeyModifiers, KeyboardEnhancementFlags, MouseEventKind,
};
use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use std::collections::VecDeque;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use crate::tui::welcome::{draw_welcome};

mod helpers;
mod provider_ui;

mod highlight_code;
mod openia;

mod welcome;

// --- Cores e Temas ---
const COLOR_APP_BG: Color = Color::Rgb(12, 12, 14);
const COLOR_CARD_BG: Color = Color::Rgb(24, 24, 28);
const COLOR_CODE_BG: Color = Color::Rgb(16, 16, 20); // Card de Código (Mais Escuro)
const COLOR_INPUT_BG: Color = Color::Rgb(32, 32, 38);
const COLOR_BORDER: Color = Color::Rgb(45, 45, 52);
const COLOR_TEXT_MAIN: Color = Color::Rgb(220, 220, 230);
const COLOR_TEXT_MUTED: Color = Color::Rgb(130, 130, 145);
const COLOR_ACCENT: Color = Color::Rgb(90, 165, 255);

// Highlight de Código Syntax
const COLOR_KEYWORD: Color = Color::Rgb(205, 120, 245);
const COLOR_STRING: Color = Color::Rgb(150, 225, 140);
const COLOR_COMMENT: Color = Color::Rgb(110, 115, 130);

// Cores de Diff
const COLOR_DIFF_ADD_BG: Color = Color::Rgb(20, 50, 35);
const COLOR_DIFF_ADD_FG: Color = Color::Rgb(150, 235, 170);
const COLOR_DIFF_REMOVE_BG: Color = Color::Rgb(55, 22, 28);
const COLOR_DIFF_REMOVE_FG: Color = Color::Rgb(245, 150, 150);

const COLOR_LINE_NUM: Color = Color::Rgb(90, 95, 110);

// Popup de autocomplete (/ comandos e @ arquivos)
const COLOR_POPUP_BG: Color = Color::Rgb(20, 20, 26);
const COLOR_POPUP_SELECTED_BG: Color = Color::Rgb(34, 34, 42);

/// Comandos "/" reconhecidos e sua descricao no menu de autocomplete.
const SLASH_COMMANDS: &[(&str, &str)] = &[
    ("/provider", "escolhe o provider ativo (local ou cloud)"),
    ("/models", "escolhe modelo e forca do raciocinio (provider cloud)"),
    ("/model", "configura/recarrega o modelo local (GGUF + mmproj)"),
    ("/session", "lista/retoma conversas salvas"),
    ("/clear", "limpa a tela do chat"),
    ("/help", "mostra atalhos de teclado"),
];

/// Forca do raciocinio do overlay `/models` (label, valor enviado na API).
/// Valor vazio = nao envia o campo (padrao do servidor).
pub  const REASONING_EFFORTS: &[(&str, &str)] = &[
    ("padrao", ""),
    ("baixo", "low"),
    ("medio", "medium"),
    ("alto", "high"),
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kind {
    User,
    Assistant,
    Reasoning,
    Tool,
    ToolResult,
    Err,
    Muted,
}

#[derive(Debug)]
pub struct UiLine {
    pub kind: Kind,
    pub text: String,
}

/// Tipo de autocomplete ativo no input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PopupKind {
    /// "/" no inicio da mensagem -> comandos.
    Slash,
    /// "@" em qualquer ponto -> arquivos do projeto.
    Mention,
}

/// Etapa do modal de configuracao do modelo local (aberto no boot quando o
/// GGUF nao existe, ou via `/model`).
#[derive(Debug, Clone)]
pub enum ModelSetup {
    /// Primeiro modal: caminho do GGUF. `multimodal` marca que o modelo
    /// precisa do mmproj (segundo modal em sequencia).
    Gguf {
        input: String,
        pos: usize,
        multimodal: bool,
    },
    /// Segundo modal (modelo multimodal): caminho do mmproj. Enter com o
    /// campo vazio = sem mmproj.
    Mmproj { input: String, pos: usize },
}

/// Tipos de API suportados no wizard de adicionar provider cloud (Ctrl+A no
/// overlay `/provider` → Cloud), na ordem ciclada por Tab.
pub(crate) const CLOUD_KINDS: [crate::config::CloudKind; 3] = [
    crate::config::CloudKind::OpenAi,
    crate::config::CloudKind::Anthropic,
    crate::config::CloudKind::Google,
];

/// Etapa do wizard de adicionar um provider cloud (aberto pelo overlay
/// `/provider`). Tab cicla o tipo; nos campos de texto, Enter avanca.
#[derive(Debug, Clone)]
pub enum ProviderSetup {
    /// Escolha do tipo de API (openai/anthropic/google). Tab cicla, Enter avanca.
    Kind { pos: usize },
    /// URL base do endpoint (pre-enchida com o default do tipo escolhido).
    BaseUrl { input: String, pos: usize },
    /// Modelo da API (ex.: gpt-5 / claude-sonnet-4 / gemini-2.5-flash).
    Model { input: String, pos: usize },
    /// API key (opcional; Enter vazio = usar a env do vendor). Enter cria.
    ApiKey { input: String, pos: usize },
}

/// Um item do menu de autocomplete: `label` e o texto inserido ao selecionar,
/// `detail` e o texto secundario (descricao do comando ou caminho completo).
#[derive(Debug, Clone)]
pub struct PopupItem {
    pub label: String,
    pub detail: String,
}

/// Estado do menu de autocomplete aberto sobre o input.
#[derive(Debug)]
pub struct InputPopup {
    pub kind: PopupKind,
    /// Posicao (em chars) do caractere gatilho ("/" ou "@") dentro do input.
    pub trigger_pos: usize,
    pub items: Vec<PopupItem>,
    pub selected: usize,
}

#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub model: String,
    pub ngl: usize,
    pub ctx_max: usize,
    pub ctx_used: usize,
    pub tools_enabled: bool,
    pub reasoning_enabled: bool,
    pub tps: Option<f64>,
    pub last_tool: Option<String>,
    pub last_diff: Option<String>,
    pub project_dir: String,
}

impl SessionInfo {
    pub fn new(
        model: String,
        ngl: usize,
        ctx_max: usize,
        tools_enabled: bool,
        reasoning_enabled: bool,
        project_dir: String,
    ) -> Self {
        Self {
            model,
            ngl,
            ctx_max,
            ctx_used: 0,
            tools_enabled,
            reasoning_enabled,
            tps: None,
            last_tool: None,
            last_diff: None,
            project_dir,
        }
    }
}

pub enum UiEvent {
    Content(String),
    Reasoning(String),
    ToolCall(String),
    ToolStart(String),
    ToolResult { name: String, result: String },
    Status(String),
    Stats { tokens: i64, tps: f64 },
    ContextUsage { used: usize, max: usize },
    ModeChanged(crate::agent::Mode),
    AgentWorking(String),
    Muted(String),
    Err(String),
    Done,
    /// Lista de modelos do endpoint cloud ativo para o overlay `/models`.
    Models { models: Vec<String>, current: String, effort: String },
    /// Snapshot dos providers (local + cloud) e do uuid selecionado. Enviado no
    /// boot e apos cada troca/edicao, para o overlay `/provider`.
    Providers {
        entries: Vec<crate::provider::ProviderEntry>,
        selected: Option<String>,
    },
    /// Engine local instalado pelo modal de configuracao (`/model` ou boot
    /// sem GGUF): atualiza modelo/raciocinio exibidos.
    ModelReady { model: String, thinking: bool },
}

pub struct AppState {
    pub lines: Vec<UiLine>,
    pub status: String,
    pub input: String,
    pub input_pos: usize,
    pub scroll: usize,
    pub max_scroll: usize,
    pub scrollbar_state: ScrollbarState,
    pub gen_active: bool,
    pub tps: Option<f64>,
    pub session: SessionInfo,
    pub queue: VecDeque<String>,
    pub interrupt: Arc<AtomicBool>,
    pub highlighter: CodeHighlighter,
    pub mode: crate::agent::Mode,
    pub working_agent: Option<String>,
    /// Sessao atual: todo turno gravado aponta para esse id.
    pub session_id: String,
    /// Histórico persistente (LanceDB) — usado pela lista `/session`.
    pub history: Option<Arc<History>>,
    /// Overlay `/session` aberto.
    pub session_view: bool,
    pub sessions: Vec<StoredSession>,
    pub session_cursor: usize,
    pub session_scroll: usize,
    /// Overlay `/provider` (escolha do provider ativo). So o Root (solo/maestro).
    pub provider_view: bool,
    pub provider_cursor: usize,
    /// Nivel de navegacao do overlay /provider (Root | LocalList | CloudList).
    pub provider_nav: provider_ui::ProviderNav,
    /// Providers disponiveis (local + cloud), atualizados via
    /// `UiEvent::Providers`.
    pub providers: Vec<crate::provider::ProviderEntry>,
    pub provider_selected: Option<String>,
    /// Overlay `/models`: escolhe o modelo da API remota e a forca do
    /// raciocinio; persiste tudo no config.json.
    pub models_view: bool,
    pub models: Vec<String>,
    pub models_cursor: usize,
    pub models_scroll: usize,
    pub models_current: String,
    pub effort_cursor: usize,
    /// Menu de autocomplete aberto sobre o input ("/" ou "@"), se houver.
    pub popup: Option<InputPopup>,
    /// Cache dos arquivos do projeto para a mencao "@" (populado sob demanda).
    pub mention_files: Vec<String>,
    /// Pedido de permissao (ex.: fetch_url) aguardando resposta do usuario.
    pub pending_permission: Option<PermissionRequest>,
    /// Pedidos que chegaram enquanto ja havia um pendente.
    pub permission_queue: VecDeque<PermissionRequest>,
    /// Tool em execucao (nome + inicio) para o cronometro de status.
    pub tool_running: Option<String>,
    pub tool_at: Option<Instant>,
    /// Marca que o conteudo do chat mudou e a cache de linhas renderizadas
    /// precisa ser recalculada (tools de subagentes empilhando diffs grandes
    /// faziam a TUI congelar re-backtle-highlighting tudo a cada frame).
    pub render_dirty: bool,
    pub render_cache: Vec<ratatui::text::Line<'static>>,

    pub show_welcome: bool,
    /// Modal de configuracao do modelo local aberto (boot sem GGUF ou /model).
    pub model_setup: Option<ModelSetup>,
    /// Modelo GGUF ja confirmado no primeiro modal — usado quando o segundo
    /// modal (mmproj) conclui o fluxo.
    pub setup_model: String,
    /// Ultimo mmproj configurado (prefill do modal).
    pub local_mmproj: String,
    /// Wizard de adicionar provider cloud aberto (do overlay `/provider`).
    pub provider_setup: Option<ProviderSetup>,
    /// Dados acumulados do provider cloud em criacao pelo wizard.
    pub new_cloud_kind: usize,
    pub new_cloud_base_url: String,
    pub new_cloud_model: String,
}

impl AppState {
    pub fn new(session: SessionInfo, interrupt: Arc<AtomicBool>) -> Self {
        Self {
            lines: Vec::new(),
            status: "ready".to_string(),
            input: String::new(),
            input_pos: 0,
            scroll: 0,
            max_scroll: 0,
            scrollbar_state: ScrollbarState::default(),
            gen_active: false,
            tps: None,
            session,
            queue: VecDeque::new(),
            interrupt,
            highlighter: CodeHighlighter::new(),
            mode: crate::agent::Mode::Solo,
            working_agent: None,
            session_id: String::new(),
            history: None,
            session_view: false,
            sessions: Vec::new(),
            session_cursor: 0,
            session_scroll: 0,
            provider_view: false,
            provider_cursor: 0,
            provider_nav: provider_ui::ProviderNav::Root,
            providers: Vec::new(),
            provider_selected: None,
            models_view: false,
            models: Vec::new(),
            models_cursor: 0,
            models_scroll: 0,
            models_current: String::new(),
            effort_cursor: 1,
            popup: None,
            mention_files: Vec::new(),
            pending_permission: None,
            permission_queue: VecDeque::new(),
            tool_running: None,
            tool_at: None,
            render_dirty: true,
            render_cache: Vec::new(),
            show_welcome: true,
            model_setup: None,
            setup_model: String::new(),
            local_mmproj: String::new(),
            provider_setup: None,
            new_cloud_kind: 0,
            new_cloud_base_url: String::new(),
            new_cloud_model: String::new(),
        }
    }

    /// Abre o modal de configuracao do modelo local. Sai da welcome (o modal
    /// renderiza sobre o chat normal) e pre-enche com o caminho atual.
    pub fn open_model_setup(&mut self, model: String, mmproj: String) {
        let multimodal = !mmproj.trim().is_empty();
        self.model_setup = Some(ModelSetup::Gguf {
            pos: model.chars().count(),
            input: model,
            multimodal,
        });
        self.local_mmproj = mmproj;
        self.popup = None;
        self.show_welcome = false;
    }

    /// Abre o wizard de adicionar um provider cloud (tecla `a` no overlay
    /// `/provider`). O popup fica aberto para o novo provider aparecer na lista.
    pub fn open_provider_setup(&mut self) {
        let kind = self.new_cloud_kind.min(CLOUD_KINDS.len() - 1);
        self.provider_setup = Some(ProviderSetup::Kind { pos: kind });
        self.popup = None;
        self.status = "tipo de API (Tab alterna, Enter confirma)".to_string();
    }

    pub fn push(&mut self, kind: Kind, text: String) {
        self.lines.push(UiLine { kind, text });
        self.render_dirty = true;
    }

    pub fn append_stream(&mut self, kind: Kind, text: String) {
        if let Some(last) = self.lines.last_mut() {
            if last.kind == kind {
                last.text.push_str(&text);
                self.render_dirty = true;
                return;
            }
        }
        self.push(kind, text);
    }

    /// Abre o overlay `/session` recarregando a lista do LanceDB.
    pub fn open_session_view(&mut self) {
        self.sessions = match self.history.as_ref() {
            Some(history) => history.list_sessions(),
            None => Vec::new(),
        };
        self.session_cursor = 0;
        self.session_scroll = 0;
        self.session_view = true;
        self.status = "sessoes".to_string();
    }

    fn selected_session(&self) -> Option<StoredSession> {
        self.sessions.get(self.session_cursor).cloned()
    }

    fn delete_selected_session(&mut self) {
        if let Some(history) = self.history.as_ref() {
            if let Some(session) = self.selected_session() {
                if history.delete_session(&session.id).is_ok() {
                    self.sessions.remove(self.session_cursor);
                    if !self.sessions.is_empty() && self.session_cursor >= self.sessions.len() {
                        self.session_cursor = self.sessions.len() - 1;
                    }
                }
            }
        }
    }

    /// Entra numa sessao: carrega a transcricao (visual) do LanceDB e passa a
    /// gravar os proximos turnos sob esse id.
    pub fn enter_session(&mut self, id: String) {
        if let Some(history) = self.history.as_ref() {
            let rows = history.get_session(&id);
            for row in rows {
                match row.kind.as_str() {
                    "root" => {
                        self.push(Kind::User, row.input);
                        if !row.reply.trim().is_empty() {
                            self.push(Kind::Assistant, row.reply);
                        }
                    }
                    "main" => {
                        self.push(Kind::Muted, format!("← {}: {}", row.agent_id, row.input));
                        if !row.reply.trim().is_empty() {
                            self.push(Kind::Muted, format!("{} → {}", row.agent_id, row.reply));
                        }
                    }
                    _ => {
                        self.push(
                            Kind::Muted,
                            format!("[resumo {}] {}", row.agent_id, row.reply),
                        );
                    }
                }
            }
            self.scroll_to_bottom();
        }
        self.session_id = id;
        self.session_view = false;
        self.status = "sessao carregada".to_string();
    }

    /// Recalcula o menu de autocomplete a partir do texto e do cursor atuais.
    /// Chamado apos qualquer edicao do input. "/" so dispara comandos quando
    /// e o primeiro caractere da mensagem; "@" dispara mencao de arquivo em
    /// qualquer posicao, olhando para tras ate o ultimo "@" sem espaco.
    pub fn update_popup(&mut self) {
        let before_cursor: String = self.input.chars().take(self.input_pos).collect();

        if let Some(rest) = before_cursor.strip_prefix('/') {
            if !rest.contains(char::is_whitespace) {
                let query = rest;
                let items: Vec<PopupItem> = SLASH_COMMANDS
                    .iter()
                    .filter(|(name, _)| name[1..].starts_with(query))
                    .map(|(name, desc)| PopupItem {
                        label: name.to_string(),
                        detail: desc.to_string(),
                    })
                    .collect();
                self.popup = if items.is_empty() {
                    None
                } else {
                    let selected = self
                        .popup
                        .as_ref()
                        .filter(|p| p.kind == PopupKind::Slash)
                        .map(|p| p.selected.min(items.len() - 1))
                        .unwrap_or(0);
                    Some(InputPopup {
                        kind: PopupKind::Slash,
                        trigger_pos: 0,
                        items,
                        selected,
                    })
                };
                return;
            }
        }

        if let Some(at_rel) = before_cursor.rfind('@') {
            let after_at = &before_cursor[at_rel + '@'.len_utf8()..];
            if !after_at.contains(char::is_whitespace) {
                if self.mention_files.is_empty() {
                    self.mention_files =
                        scan_project_files(std::path::Path::new(&self.session.project_dir));
                }
                let query = after_at.to_lowercase();
                let mut matches: Vec<&String> = self
                    .mention_files
                    .iter()
                    .filter(|p| query.is_empty() || p.to_lowercase().contains(&query))
                    .collect();
                matches.sort_by_key(|p| {
                    let name = p.rsplit('/').next().unwrap_or(p);
                    (!name.to_lowercase().starts_with(&query), p.len())
                });
                matches.truncate(30);
                self.popup = if matches.is_empty() {
                    None
                } else {
                    let trigger_pos = before_cursor[..at_rel].chars().count();
                    let items: Vec<PopupItem> = matches
                        .into_iter()
                        .map(|p| PopupItem {
                            label: p.clone(),
                            detail: String::new(),
                        })
                        .collect();
                    let selected = self
                        .popup
                        .as_ref()
                        .filter(|p| p.kind == PopupKind::Mention)
                        .map(|p| p.selected.min(items.len() - 1))
                        .unwrap_or(0);
                    Some(InputPopup {
                        kind: PopupKind::Mention,
                        trigger_pos,
                        items,
                        selected,
                    })
                };
                return;
            }
        }

        self.popup = None;
    }

    /// Aplica o item selecionado do popup, substituindo o gatilho ("/..." ou
    /// "@...") pelo texto escolhido, e fecha o menu.
    pub fn apply_popup_selection(&mut self) {
        let Some(popup) = self.popup.take() else {
            return;
        };
        let Some(item) = popup.items.get(popup.selected).cloned() else {
            return;
        };
        match popup.kind {
            PopupKind::Slash => {
                self.input = item.label;
                self.input_pos = self.input.chars().count();
            }
            PopupKind::Mention => {
                let chars: Vec<char> = self.input.chars().collect();
                let start = popup.trigger_pos.min(chars.len());
                let end = self.input_pos.min(chars.len());
                let mut new_input: String = chars[..start].iter().collect();
                new_input.push('@');
                new_input.push_str(&item.label);
                new_input.push(' ');
                let new_pos = new_input.chars().count();
                new_input.extend(chars[end..].iter());
                self.input = new_input;
                self.input_pos = new_pos;
            }
        }
    }

    /// Responde ao pedido de permissao pendente e promove o proximo da fila,
    /// se houver.
    pub fn answer_permission(&mut self, allow: bool) {
        if let Some(req) = self.pending_permission.take() {
            self.push(
                Kind::Muted,
                format!(
                    "[permissao] {} para '{}': {}",
                    if allow { "concedida" } else { "negada" },
                    req.tool,
                    req.detail
                ),
            );
            let _ = req.reply.send(allow);
        }
        self.pending_permission = self.permission_queue.pop_front();
    }

    pub fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Content(text) => {
                self.append_stream(Kind::Assistant, text);
                self.scroll_to_bottom();
            }
            UiEvent::Reasoning(text) => {
                let starting_new_block =
                    !matches!(self.lines.last(), Some(l) if l.kind == Kind::Reasoning);
                let chunk = if starting_new_block {
                    format!("\n[thinking]\n{}", text)
                } else {
                    text
                };
                self.append_stream(Kind::Reasoning, chunk);
                self.scroll_to_bottom();
            }
            UiEvent::ToolCall(text) => {
                self.push(Kind::Tool, text);
                self.scroll_to_bottom();
            }
            UiEvent::ToolStart(name) => {
                self.status = format!("running {}", name);
                self.session.last_tool = Some(name.clone());
                self.tool_running = Some(name);
                self.tool_at = Some(Instant::now());
            }
            UiEvent::ToolResult { name, result } => {
                self.push(Kind::ToolResult, format!("[{}]\n{}", name, result));
                self.session.last_tool = Some(name);
                self.session.last_diff = Some(result);
                self.scroll_to_bottom();
            }
            UiEvent::Status(text) => self.status = text,
            UiEvent::Stats { tokens, tps } => {
                self.tps = Some(tps);
                self.session.tps = Some(tps);
                self.status = format!("{:.1} tok/s ({} tokens)", tps, tokens);
            }
            UiEvent::ContextUsage { used, max } => {
                self.session.ctx_used = used;
                self.session.ctx_max = max;
            }
            UiEvent::ModeChanged(mode) => {
                self.mode = mode;
            }
            UiEvent::AgentWorking(name) => {
                if name == "drill" {
                    if self.mode == crate::agent::Mode::Agents {
                        self.working_agent = Some("maestro".to_string());
                        self.status = "trabalhando: maestro".to_string();
                    } else {
                        self.working_agent = Some("drill".to_string());
                        self.status = "trabalhando: drill".to_string();
                    }
                } else {
                    self.working_agent = Some(name.clone());
                    self.status = format!("trabalhando: {}", name);
                }
            }
            UiEvent::Muted(text) => {
                self.push(Kind::Muted, text);
                self.scroll_to_bottom();
            }
            UiEvent::Err(text) => {
                self.push(Kind::Err, format!("error: {}", text));
                self.status = "error".to_string();
                // Falha ao carregar modelos: fecha o overlay pra o erro nao
                // ficar preso atras do modal.
                if self.models_view {
                    self.models_view = false;
                    self.models = Vec::new();
                }
            }
            UiEvent::Models { models, current, effort } => {
                self.models = models;
                self.models_cursor = 0;
                self.models_scroll = 0;
                self.models_current = current;
                self.effort_cursor = REASONING_EFFORTS
                    .iter()
                    .position(|(_, v)| *v == effort.trim())
                    .unwrap_or(1);
                self.status = if self.models.is_empty() {
                    "nenhum modelo listado".to_string()
                } else {
                    "escolha o modelo...".to_string()
                };
            }
            UiEvent::Providers { entries, selected } => {
                self.providers = entries;
                self.provider_selected = selected;
                if self.provider_cursor >= self.providers.len() {
                    self.provider_cursor = self.providers.len().saturating_sub(1);
                }
            }
            UiEvent::ModelReady { model, thinking } => {
                self.session.model = model;
                self.session.reasoning_enabled = thinking;
                self.status = "modelo local pronto".to_string();
            }
            UiEvent::Done => {
                self.gen_active = false;
                self.interrupt.store(false, Ordering::Relaxed);
                self.working_agent = None;
                self.tool_running = None;
                self.tool_at = None;
                self.status = "ready".to_string();
            }
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll = 0;
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll = (self.scroll + amount).min(self.max_scroll);
    }

    pub fn scroll_down(&mut self, amount: usize) {
        self.scroll = self.scroll.saturating_sub(amount);
    }

    /// Limite de sub-linhas renderizadas por bloco de resultado de tool.
    /// Resultados grandes (read_file de arquivo longo, saida de run_command)
    /// empilhavam diffs de dezenas de milhares de linhas e travavam a TUI.
    const TOOL_RESULT_MAX_LINES: usize = 60;

    pub fn rendered_lines(&mut self, area: Rect) -> Vec<Line<'static>> {
        let content_height = area.height.saturating_sub(2) as usize;
        let cap = content_height.max(1);

        // Cache: re-renderizar todo o chat (highlight de cada linha) a cada frame
        // custava caro quando tools de subagentes acumulavam resultados grandes.
        if !self.render_dirty && !self.render_cache.is_empty() {
            self.max_scroll = self.render_cache.len().saturating_sub(cap);
            return self.render_cache.clone();
        }

        let mut rendered: Vec<Line<'static>> = Vec::new();
        let mut in_code_block = false;

        for line in &self.lines {
            match line.kind {
                Kind::User => {
                    rendered.push(Line::from(vec![
                        Span::styled(
                            "> ",
                            Style::default()
                                .fg(COLOR_ACCENT)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            line.text.clone(),
                            Style::default()
                                .fg(COLOR_TEXT_MAIN)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                Kind::Tool => {
                    rendered.push(Line::from(vec![
                        Span::styled("← ", Style::default().fg(COLOR_ACCENT)),
                        Span::styled(
                            line.text.clone(),
                            Style::default()
                                .fg(COLOR_TEXT_MAIN)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                Kind::ToolResult => {
                    let lines_vec: Vec<&str> = line.text.split('\n').collect();
                    let target_width = area.width.saturating_sub(2) as usize;
                    let mut line_counter = 1; // Pode ser sincronizado com o offset real do arquivo
                    let mut num_rendered = 0usize;

                    for (i, sub) in lines_vec.iter().enumerate() {
                        if sub.starts_with("---") {
                            rendered.push(Line::from(Span::styled(
                                sub.to_string(),
                                Style::default()
                                    .fg(COLOR_TEXT_MUTED)
                                    .add_modifier(Modifier::BOLD),
                            )));
                            continue;
                        }

                        if num_rendered >= Self::TOOL_RESULT_MAX_LINES {
                            let remaining = lines_vec.len() - i;
                            rendered.push(Line::from(Span::styled(
                                format!("  ... ({} linhas omitidas)", remaining),
                                Style::default()
                                    .fg(COLOR_TEXT_MUTED)
                                    .add_modifier(Modifier::ITALIC),
                            )));
                            break;
                        }

                        let (bg_color, prefix_sign, text_content) =
                            if let Some(stripped) = sub.strip_prefix('+') {
                                (COLOR_DIFF_ADD_BG, "+ ", stripped)
                            } else if let Some(stripped) = sub.strip_prefix('-') {
                                (COLOR_DIFF_REMOVE_BG, "- ", stripped)
                            } else {
                                let content = sub.strip_prefix(' ').unwrap_or(sub);
                                (COLOR_CODE_BG, "  ", content)
                            };

                        let mut spans = Vec::new();

                        // 1. Número da linha
                        let num_str = format!("{:4} ", line_counter);
                        spans.push(Span::styled(
                            num_str,
                            Style::default().fg(COLOR_LINE_NUM).bg(bg_color),
                        ));

                        // 2. Indicador de mudança (+ / -)
                        spans.push(Span::styled(
                            prefix_sign,
                            Style::default()
                                .fg(if prefix_sign == "+ " {
                                    COLOR_ACCENT
                                } else {
                                    COLOR_DIFF_REMOVE_FG
                                })
                                .bg(bg_color),
                        ));

                        // 3. Syntax Highlighting do código com o fundo do diff
                        let highlighted_line =
                            self.highlighter.highlight_line(text_content, "rs", 0);
                        for mut span in highlighted_line.spans {
                            span.style = span.style.bg(bg_color); // Força a cor de fundo do diff no highlight
                            spans.push(span);
                        }

                        // 4. Preenchimento de fundo até o final do painel
                        let current_len: usize =
                            spans.iter().map(|s| s.content.chars().count()).sum();
                        if current_len < target_width {
                            let padding = " ".repeat(target_width - current_len);
                            spans.push(Span::styled(padding, Style::default().bg(bg_color)));
                        }

                        rendered.push(Line::from(spans));
                        line_counter += 1;
                        num_rendered += 1;
                    }
                }
                Kind::Reasoning => {
                    for sub in line.text.split('\n') {
                        rendered.push(Line::from(Span::styled(
                            sub.to_string(),
                            Style::default()
                                .fg(COLOR_TEXT_MUTED)
                                .add_modifier(Modifier::ITALIC),
                        )));
                    }
                }
                Kind::Assistant => {
                    let mut current_lang = String::from("rs");
                    let max_w = area.width.saturating_sub(2) as usize;

                    for sub in line.text.split('\n') {
                        let trimmed = sub.trim_start();
                        if trimmed.starts_with("```") {
                            // ... seu código de code-block continua igual
                            continue;
                        }

                        if in_code_block {
                            let highlighted =
                                self.highlighter.highlight_line(sub, &current_lang, max_w);
                            rendered.push(highlighted);
                        } else {
                            // ← aqui a quebra de linha
                            for wrapped in wrap_text(sub, max_w) {
                                rendered.push(Line::from(Span::styled(
                                    wrapped,
                                    Style::default().fg(COLOR_TEXT_MAIN),
                                )));
                            }
                        }
                    }
                }

                Kind::Muted | Kind::Err | _ => {
                    let max_w = area.width.saturating_sub(2) as usize;
                    let style = match line.kind {
                        Kind::Muted => Style::default().fg(COLOR_TEXT_MUTED),
                        Kind::Err => Style::default().fg(Color::Red),
                        _ => Style::default().fg(COLOR_TEXT_MAIN),
                    };
                    for wrapped in wrap_text(&line.text, max_w) {
                        rendered.push(Line::from(Span::styled(wrapped, style)));
                    }
                }

                _ => {
                    rendered.push(Line::from(Span::styled(
                        line.text.clone(),
                        Style::default().fg(COLOR_TEXT_MAIN),
                    )));
                }
            }
        }

        let n = rendered.len();
        self.max_scroll = n.saturating_sub(cap);

        self.render_cache = rendered.clone();
        self.render_dirty = false;

        rendered
    }
}

// Auxiliar para destacar o código
fn format_code_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();

    if line.trim_start().starts_with("//") || line.trim_start().starts_with('#') {
        return Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(COLOR_COMMENT).bg(COLOR_CODE_BG),
        ));
    }

    let words = line.split_whitespace();
    let mut current_pos = 0;

    for word in words {
        if let Some(idx) = line[current_pos..].find(word) {
            let space_slice = &line[current_pos..current_pos + idx];
            if !space_slice.is_empty() {
                spans.push(Span::styled(
                    space_slice.to_string(),
                    Style::default().bg(COLOR_CODE_BG),
                ));
            }
            current_pos += idx + word.len();

            let style = match word {
                "fn" | "let" | "mut" | "pub" | "use" | "const" | "return" | "if" | "else"
                | "match" | "struct" | "enum" => Style::default()
                    .fg(COLOR_KEYWORD)
                    .add_modifier(Modifier::BOLD)
                    .bg(COLOR_CODE_BG),
                w if w.starts_with('"') || w.ends_with('"') || w.starts_with('\'') => {
                    Style::default().fg(COLOR_STRING).bg(COLOR_CODE_BG)
                }
                _ => Style::default().fg(COLOR_TEXT_MAIN).bg(COLOR_CODE_BG),
            };

            spans.push(Span::styled(word.to_string(), style));
        }
    }

    if current_pos < line.len() {
        spans.push(Span::styled(
            line[current_pos..].to_string(),
            Style::default().bg(COLOR_CODE_BG),
        ));
    }

    Line::from(spans)
}

/// Varre `root` recursivamente e devolve caminhos relativos (com "/"), para
/// popular o autocomplete de mencao "@". Ignora diretorios pesados/irrelevantes
/// e para em limites de profundidade/quantidade para nao travar a UI.
fn scan_project_files(root: &std::path::Path) -> Vec<String> {
    const SKIP_DIRS: &[&str] = &[
        ".git",
        "target",
        "node_modules",
        ".idea",
        ".vscode",
        "dist",
        "build",
        ".cache",
    ];
    const MAX_FILES: usize = 4000;
    const MAX_DEPTH: usize = 14;

    let mut out = Vec::new();
    let mut stack: Vec<(std::path::PathBuf, usize)> = vec![(root.to_path_buf(), 0)];

    while let Some((dir, depth)) = stack.pop() {
        if out.len() >= MAX_FILES {
            break;
        }
        if depth > MAX_DEPTH {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name_str = entry.file_name().to_string_lossy().to_string();
            if name_str.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                if SKIP_DIRS.contains(&name_str.as_str()) {
                    continue;
                }
                stack.push((path, depth + 1));
            } else if out.len() < MAX_FILES {
                if let Ok(rel) = path.strip_prefix(root) {
                    out.push(rel.to_string_lossy().replace('\\', "/"));
                }
            }
        }
    }

    out.sort();
    out
}

pub fn run(
    mut state: AppState,
    events: Receiver<UiEvent>,
    turns: Sender<crate::agent::WorkerMsg>,
    history: Arc<History>,
    session_id: String,
    permission_rx: Receiver<PermissionRequest>,
) -> Result<(), io::Error> {
    state.history = Some(history);
    state.session_id = session_id;
    let mut terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), EnableMouseCapture);

    let kitty_keyboard = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    {
        let _ = crossterm::execute!(std::io::stdout(), event::EnableBracketedPaste);
        if kitty_keyboard {
            let _ = crossterm::execute!(
                std::io::stdout(),
                event::PushKeyboardEnhancementFlags(
                    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                )
            );
        }
    }

    let result = loop_result(&mut terminal, &mut state, &events, &turns, &permission_rx);

    {
        let _ = crossterm::execute!(std::io::stdout(), DisableMouseCapture);
        if kitty_keyboard {
            let _ = crossterm::execute!(std::io::stdout(), event::PopKeyboardEnhancementFlags);
        }
        let _ = crossterm::execute!(std::io::stdout(), event::DisableBracketedPaste);
    }
    ratatui::restore();
    result
}

fn loop_result(
    terminal: &mut ratatui::DefaultTerminal,
    state: &mut AppState,
    events: &Receiver<UiEvent>,
    turns: &Sender<crate::agent::WorkerMsg>,
    permission_rx: &Receiver<PermissionRequest>,
) -> Result<(), io::Error> {
    loop {
        let mut has_new_events = false;
        while let Ok(event) = events.try_recv() {
            state.apply(event);
            has_new_events = true;
        }

        while let Ok(req) = permission_rx.try_recv() {
            state.permission_queue.push_back(req);
        }
        if state.pending_permission.is_none() {
            state.pending_permission = state.permission_queue.pop_front();
        }

        if !state.gen_active {
            if let Some(text) = state.queue.pop_front() {
                state.push(Kind::User, text.clone());
                state.status = "generating".to_string();
                state.gen_active = true;
                let _ = turns.send(agent::WorkerMsg::Turn(text));
            }
        }

        // Cronometro da tool em execucao: mostra que a UI segue viva mesmo que
        // a ferramenta demore (ask_agent/run_command) — antes parecia travada.
        if let (Some(name), Some(at)) = (&state.tool_running, state.tool_at) {
            state.status = format!("running {} ({}s)", name, at.elapsed().as_secs());
        }

        terminal.draw(|frame| draw(frame, state))?;

        if event::poll(Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if state.pending_permission.is_some() {
                        match key.code {
                            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                                state.answer_permission(true);
                            }
                            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                                state.answer_permission(false);
                            }
                            _ => {}
                        }
                        continue;
                    }

                    if let Some(action) = read_key(key, state) {
                        match action {
                            Action::ChooseProviderId(id) => {
                                state.provider_view = false;
                                state.provider_nav = provider_ui::ProviderNav::Root;
                                state.status = "definindo provider...".to_string();
                                let _ = turns.send(agent::WorkerMsg::SetProvider(id));
                            }
                            Action::OpenLocalModelSetup => {
                                // Fecha o overlay e abre o modal de GGUF existente.
                                state.provider_view = false;
                                state.provider_nav = provider_ui::ProviderNav::Root;
                                let model = state.session.model.clone();
                                let mmproj = state.local_mmproj.clone();
                                state.open_model_setup(model, mmproj);
                                state.status = "configure o modelo local (GGUF)...".to_string();
                            }

                            Action::Quit => return Ok(()),
                            Action::SetupCancel => {
                                state.model_setup = None;
                                state.status =
                                    "modelo local nao configurado; use /model ou /provider"
                                        .to_string();
                            }
                            Action::SetupConfirm => {
                                handle_setup_confirm(state, turns);
                            }
                            Action::OpenProviderSetup => {
                                state.open_provider_setup();
                            }
                            Action::ProviderSetupConfirm => {
                                handle_provider_setup_confirm(state, turns);
                            }
                            Action::ProviderSetupCancel => {
                                state.provider_setup = None;
                                state.status =
                                    "provider cloud nao adicionado; Ctrl+A adiciona".to_string();
                            }
                            Action::NavSession(delta) => {
                                let len = state.sessions.len() as isize;
                                let pos = state.session_cursor as isize;
                                let target = if delta == isize::MIN {
                                    0
                                } else if delta == isize::MAX {
                                    len - 1
                                } else {
                                    (pos + delta).clamp(0, len - 1)
                                };
                                state.session_cursor = if target < 0 { 0 } else { target as usize };
                                if state.session_scroll >= state.sessions.len() {
                                    state.session_scroll = state.sessions.len().saturating_sub(1);
                                }
                            }
                            Action::DeleteSession => {
                                state.delete_selected_session();
                            }
                            Action::OpenSession => {
                                if let Some(session) = state.selected_session() {
                                    state.enter_session(session.id.clone());
                                    let _ = turns
                                        .send(agent::WorkerMsg::SetSession(session.id.clone()));
                                }
                            }
                            Action::CloseSessionView => {
                                state.session_view = false;
                                state.status = "ready".to_string();
                            }
                            Action::CloseProviderView => {
                                state.provider_view = false;
                                state.provider_nav = provider_ui::ProviderNav::Root;
                                state.provider_cursor = 0;
                                state.status = "ready".to_string();
                            }
                            Action::ChooseModel => {
                                if let Some(model) = state.models.get(state.models_cursor).cloned()
                                {
                                    let effort =
                                        REASONING_EFFORTS[state.effort_cursor].1.to_string();
                                    state.models_view = false;
                                    state.status = "definindo modelo...".to_string();
                                    let _ = turns.send(agent::WorkerMsg::SetModelAndEffort(
                                        model, effort,
                                    ));
                                }
                                continue;
                            }
                            Action::CloseModelsView => {
                                state.models_view = false;
                                state.status = "ready".to_string();
                            }
                            Action::Submit => {
                                state.show_welcome = false;

                                let text = if !state.input.is_empty() {
                                    std::mem::take(&mut state.input)
                                } else {
                                    state.queue.pop_front().unwrap_or_default()
                                };
                                state.input_pos = 0;
                                match text.trim() {
                                    "/session" => {
                                        state.open_session_view();
                                        continue;
                                    }
                                    "/provider" => {
                                        state.provider_view = true;
                                        state.provider_cursor = 0;
                                        state.provider_nav = provider_ui::ProviderNav::Root;
                                        state.status = "escolha Local ou Cloud...".to_string();
                                        continue;
                                    }
                                    "/models" => {
                                        state.models_view = true;
                                        state.models_cursor = 0;
                                        state.models_scroll = 0;
                                        state.status = "carregando modelos...".to_string();
                                        let _ = turns.send(agent::WorkerMsg::ListModels);
                                        continue;
                                    }
                                    "/clear" => {
                                        state.lines.clear();
                                        state.scroll = 0;
                                        state.status = "chat limpo".to_string();
                                        continue;
                                    }
                                    "/help" => {
                                        state.push(Kind::Muted, help_text());
                                        continue;
                                    }
                                    "/model" => {
                                        state.open_model_setup(
                                            state.session.model.clone(),
                                            state.local_mmproj.clone(),
                                        );
                                        state.status =
                                            "informe o caminho do modelo GGUF".to_string();
                                        continue;
                                    }
                                    _ => {}
                                }
                                state.popup = None;
                                state.push(Kind::User, text.clone());
                                state.status = "generating".to_string();
                                state.gen_active = true;
                                let _ = turns.send(agent::WorkerMsg::Turn(text));
                            }
                            Action::Queue => {
                                let text = std::mem::take(&mut state.input);
                                state.input_pos = 0;
                                state.queue.push_back(text);
                            }
                            Action::Steer => {
                                if !state.input.is_empty() {
                                    let text = std::mem::take(&mut state.input);
                                    state.input_pos = 0;
                                    state.queue.push_front(text);
                                }
                                state.interrupt.store(true, Ordering::Relaxed);
                                agent::cancel_generation();
                                state.status = "interrupting...".to_string();
                            }
                            Action::Interrupt => {
                                state.interrupt.store(true, Ordering::Relaxed);
                                agent::cancel_generation();
                                state.status = "interrupting...".to_string();
                            }
                            Action::ToggleMode => {
                                if state.gen_active {
                                    state.status = "aguarde terminar a geracao".to_string();
                                } else {
                                    let new_mode = match state.mode {
                                        crate::agent::Mode::Solo => crate::agent::Mode::Agents,
                                        crate::agent::Mode::Agents => crate::agent::Mode::Solo,
                                    };
                                    let _ = turns.send(agent::WorkerMsg::SetMode(new_mode));
                                    state.mode = new_mode;
                                    state.status = format!("modo {}", new_mode.label());
                                    state.push(
                                        Kind::Muted,
                                        format!("[modo] {}: o drill eh o agente maestro com subagentes; use Ctrl+A para voltar a solo.", new_mode.label()),
                                    );
                                }
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => match mouse.kind {
                    MouseEventKind::ScrollUp => state.scroll_up(3),
                    MouseEventKind::ScrollDown => state.scroll_down(3),
                    _ => {}
                },
                Event::Paste(text) => {
                    if state.pending_permission.is_some() || state.session_view {
                        // não cola em overlays de permissão/sessão
                    } else if state.provider_setup.is_some() {
                        // cola no campo atual do wizard de cloud
                        let sanitized: String = text
                            .chars()
                            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                            .collect();
                        if !sanitized.is_empty() {
                            provider_setup_edit(state, |input, pos| {
                                let byte = char_to_byte(input, *pos);
                                input.insert_str(byte, &sanitized);
                                *pos += sanitized.chars().count();
                            });
                        }
                    } else if state.model_setup.is_some() {
                        // cola no campo do modal de modelo local
                        let sanitized: String = text
                            .chars()
                            .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
                            .collect();
                        if !sanitized.is_empty() {
                            match state.model_setup.as_mut() {
                                Some(ModelSetup::Gguf { input, pos, .. })
                                | Some(ModelSetup::Mmproj { input, pos }) => {
                                    let byte = char_to_byte(input, *pos);
                                    input.insert_str(byte, &sanitized);
                                    *pos += sanitized.chars().count();
                                }
                                None => {}
                            }
                        }
                    } else {
                        insert_text_at_cursor(state, &text);
                        state.update_popup();
                    }
                }
                _ => {}
            }
        }
    }
}

enum Action {
    Quit,
    Submit,
    Queue,
    Steer,
    Interrupt,
    ToggleMode,
    NavSession(isize),
    OpenSession,
    DeleteSession,
    CloseSessionView,
    ChooseProviderId(String),
    CloseProviderView,
    /// Ctrl+A na lista Local: abre modal de GGUF.
    OpenLocalModelSetup,
    ChooseModel,
    CloseModelsView,
    /// Enter no modal de configuracao do modelo local.
    SetupConfirm,
    /// Esc no modal de configuracao do modelo local.
    SetupCancel,
    /// Ctrl+A na lista Cloud: abre o wizard de adicionar provider cloud.
    OpenProviderSetup,
    /// Enter no wizard de provider cloud: avanca etapa ou cria o provider.
    ProviderSetupConfirm,
    /// Esc no wizard de provider cloud: cancela.
    ProviderSetupCancel,

}

fn read_key(key: KeyEvent, state: &mut AppState) -> Option<Action> {
    // Modal de configuracao do modelo local: bloqueia todo o resto da UI.
    if state.model_setup.is_some() {
        return match key.code {
            KeyCode::Char(c) => {
                setup_edit(state, |input, pos| {
                    let byte = char_to_byte(input, *pos);
                    input.insert(byte, c);
                    *pos += 1;
                });
                None
            }
            KeyCode::Backspace => {
                setup_edit(state, |input, pos| {
                    if *pos > 0 {
                        let end = char_to_byte(input, *pos);
                        let start = char_to_byte(input, *pos - 1);
                        input.drain(start..end);
                        *pos -= 1;
                    }
                });
                None
            }
            KeyCode::Delete => {
                setup_edit(state, |input, pos| {
                    if *pos < input.chars().count() {
                        let start = char_to_byte(input, *pos);
                        let end = char_to_byte(input, *pos + 1);
                        input.drain(start..end);
                    }
                });
                None
            }
            KeyCode::Left => {
                setup_edit(state, |_, pos| {
                    if *pos > 0 {
                        *pos -= 1;
                    }
                });
                None
            }
            KeyCode::Right => {
                setup_edit(state, |input, pos| {
                    if *pos < input.chars().count() {
                        *pos += 1;
                    }
                });
                None
            }
            KeyCode::Home => {
                setup_edit(state, |_, pos| *pos = 0);
                None
            }
            KeyCode::End => {
                setup_edit(state, |input, pos| *pos = input.chars().count());
                None
            }
            KeyCode::Tab => {
                if let Some(ModelSetup::Gguf { multimodal, .. }) = state.model_setup.as_mut() {
                    *multimodal = !*multimodal;
                }
                None
            }
            KeyCode::Enter => Some(Action::SetupConfirm),
            KeyCode::Esc => Some(Action::SetupCancel),
            _ => None,
        };
    }

    // Wizard de adicionar provider cloud (aberto pelo overlay /provider).
    if state.provider_setup.is_some() {
        return match key.code {
            // Navegação do tipo de API (Kind) — setas / j/k / Tab
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                if let Some(ProviderSetup::Kind { pos }) = state.provider_setup.as_mut() {
                    *pos = if *pos == 0 {
                        CLOUD_KINDS.len() - 1
                    } else {
                        *pos - 1
                    };
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                if let Some(ProviderSetup::Kind { pos }) = state.provider_setup.as_mut() {
                    *pos = (*pos + 1) % CLOUD_KINDS.len();
                }
                None
            }
            KeyCode::Tab => {
                if let Some(ProviderSetup::Kind { pos }) = state.provider_setup.as_mut() {
                    *pos = (*pos + 1) % CLOUD_KINDS.len();
                }
                None
            }

            // Edição de texto (BaseUrl / Model / ApiKey)
            KeyCode::Char(c) => {
                provider_setup_edit(state, |input, pos| {
                    let byte = char_to_byte(input, *pos);
                    input.insert(byte, c);
                    *pos += 1;
                });
                None
            }
            KeyCode::Backspace => {
                provider_setup_edit(state, |input, pos| {
                    if *pos > 0 {
                        let end = char_to_byte(input, *pos);
                        let start = char_to_byte(input, *pos - 1);
                        input.drain(start..end);
                        *pos -= 1;
                    }
                });
                None
            }
            KeyCode::Delete => {
                provider_setup_edit(state, |input, pos| {
                    if *pos < input.chars().count() {
                        let start = char_to_byte(input, *pos);
                        let end = char_to_byte(input, *pos + 1);
                        input.drain(start..end);
                    }
                });
                None
            }
            KeyCode::Left => {
                provider_setup_edit(state, |_, pos| {
                    if *pos > 0 {
                        *pos -= 1;
                    }
                });
                None
            }
            KeyCode::Right => {
                provider_setup_edit(state, |input, pos| {
                    if *pos < input.chars().count() {
                        *pos += 1;
                    }
                });
                None
            }
            KeyCode::Home => {
                provider_setup_edit(state, |_, pos| *pos = 0);
                None
            }
            KeyCode::End => {
                provider_setup_edit(state, |input, pos| *pos = input.chars().count());
                None
            }

            KeyCode::Enter => Some(Action::ProviderSetupConfirm),
            KeyCode::Esc => Some(Action::ProviderSetupCancel),
            _ => None,
        };
    }

    if state.session_view {
        return match key.code {
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => Some(Action::NavSession(-1)),
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => Some(Action::NavSession(1)),
            KeyCode::PageUp => Some(Action::NavSession(-20)),
            KeyCode::PageDown => Some(Action::NavSession(20)),
            KeyCode::Home => Some(Action::NavSession(isize::MIN)),
            KeyCode::End => Some(Action::NavSession(isize::MAX)),
            KeyCode::Enter => Some(Action::OpenSession),
            KeyCode::Char('d') | KeyCode::Char('D') | KeyCode::Delete => {
                Some(Action::DeleteSession)
            }
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                Some(Action::CloseSessionView)
            }
            _ => None,
        };
    }
    // Overlay `/provider`: hierarquia Root → Local/Cloud list.
    // 'a' solto NAO faz nada (evita bug ao digitar). Ctrl+A adiciona no contexto.
    if state.provider_view {
        use provider_ui::ProviderKeyResult;
        let result = provider_ui::handle_key(
            key.code,
            key.modifiers,
            state.provider_nav,
            &mut state.provider_cursor,
            &state.providers,
        );
        return match result {
            ProviderKeyResult::None => None,
            ProviderKeyResult::Close => Some(Action::CloseProviderView),
            ProviderKeyResult::Back => {
                state.provider_nav = provider_ui::ProviderNav::Root;
                state.provider_cursor = 0;
                state.status = "escolha Local ou Cloud...".to_string();
                None
            }
            ProviderKeyResult::EnterGroup(nav) => {
                state.provider_nav = nav;
                state.provider_cursor = 0;
                state.status = match nav {
                    provider_ui::ProviderNav::LocalList => "providers locais...".to_string(),
                    provider_ui::ProviderNav::CloudList => "providers cloud...".to_string(),
                    _ => "escolha o provider...".to_string(),
                };
                None
            }
            ProviderKeyResult::Choose(id) => Some(Action::ChooseProviderId(id)),
            ProviderKeyResult::Add => {
                if matches!(state.provider_nav, provider_ui::ProviderNav::CloudList) {
                    Some(Action::OpenProviderSetup)
                } else {
                    Some(Action::OpenLocalModelSetup)
                }
            }
        };
    }
    // Overlay `/models`: escolhe modelo (j/k) e forca do raciocinio
    // (h/l ou setas). Enter define e persiste; Esc/q fecha sem mudar.
    if state.models_view {
        return match key.code {
            KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
                state.models_cursor = state.models_cursor.saturating_sub(1);
                if state.models_cursor < state.models_scroll {
                    state.models_scroll = state.models_cursor;
                }
                None
            }
            KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
                let max = state.models.len().saturating_sub(1);
                state.models_cursor = (state.models_cursor + 1).min(max);
                None
            }
            KeyCode::Left | KeyCode::Char('h') | KeyCode::Char('H') => {
                state.effort_cursor = state.effort_cursor.saturating_sub(1);
                None
            }
            KeyCode::Right | KeyCode::Char('l') | KeyCode::Char('L') => {
                let max = REASONING_EFFORTS.len().saturating_sub(1);
                state.effort_cursor = (state.effort_cursor + 1).min(max);
                None
            }
            KeyCode::Enter => Some(Action::ChooseModel),
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                Some(Action::CloseModelsView)
            }
            _ => None,
        };
    }
    // Menu de autocomplete ("/" comandos ou "@" arquivos) aberto: setas navegam,
    // Tab/Enter seleciona, Esc fecha. Qualquer outra tecla cai para a edicao
    // normal do input, que reavalia o popup no final da funcao.
    if state.popup.is_some() {
        match key.code {
            KeyCode::Up => {
                if let Some(popup) = state.popup.as_mut() {
                    if popup.selected > 0 {
                        popup.selected -= 1;
                    }
                }
                return None;
            }
            KeyCode::Down => {
                if let Some(popup) = state.popup.as_mut() {
                    if popup.selected + 1 < popup.items.len() {
                        popup.selected += 1;
                    }
                }
                return None;
            }
            KeyCode::Tab => {
                state.apply_popup_selection();
                return None;
            }
            KeyCode::Enter => {
                let kind = state.popup.as_ref().map(|p| p.kind);
                state.apply_popup_selection();
                return match kind {
                    // "/comando" + Enter: aceita e ja executa (o proprio
                    // Action::Submit trata /session, /clear e /help).
                    Some(PopupKind::Slash) if !state.gen_active => Some(Action::Submit),
                    // "@arquivo" + Enter: so insere a mencao; a mensagem
                    // continua sendo digitada.
                    _ => None,
                };
            }
            KeyCode::Esc => {
                state.popup = None;
                return None;
            }
            _ => {}
        }
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('a') {
        return Some(Action::ToggleMode);
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Enter {
        return Some(Action::Steer);
    }
    let action = match key.code {
        KeyCode::Esc => {
            if state.gen_active {
                Some(Action::Interrupt)
            } else {
                Some(Action::Quit)
            }
        }
        KeyCode::Enter => {
            if state.gen_active {
                if !state.input.is_empty() {
                    Some(Action::Queue)
                } else {
                    None
                }
            } else if !state.input.is_empty() || !state.queue.is_empty() {
                Some(Action::Submit)
            } else {
                None
            }
        }
        KeyCode::PageUp => {
            state.scroll_up(20);
            None
        }
        KeyCode::PageDown => {
            state.scroll_down(20);
            None
        }
        KeyCode::Backspace => {
            if !state.input.is_empty() && state.input_pos > 0 {
                let end = char_to_byte(&state.input, state.input_pos);
                let start = char_to_byte(&state.input, state.input_pos - 1);
                state.input.drain(start..end);
                state.input_pos -= 1;
            }
            None
        }
        KeyCode::Delete => {
            if state.input_pos < state.input.chars().count() {
                let start = char_to_byte(&state.input, state.input_pos);
                let end = char_to_byte(&state.input, state.input_pos + 1);
                state.input.drain(start..end);
            }
            None
        }
        KeyCode::Left => {
            if state.input_pos > 0 {
                state.input_pos -= 1;
            }
            None
        }
        KeyCode::Right => {
            if state.input_pos < state.input.chars().count() {
                state.input_pos += 1;
            }
            None
        }
        KeyCode::Home => {
            state.input_pos = 0;
            None
        }
        KeyCode::End => {
            state.input_pos = state.input.chars().count();
            None
        }
        KeyCode::Char(c) => {
            let byte = char_to_byte(&state.input, state.input_pos);
            state.input.insert(byte, c);
            state.input_pos += 1;
            None
        }
        _ => None,
    };

    // So faz sentido reavaliar o popup quando o input pode ter mudado (texto
    // ou posicao do cursor); Submit/Queue ja limpam o input antes de chegar aqui.
    state.update_popup();
    action
}


/// Edita o campo do modal de configuracao ativo (GGUF ou mmproj).
fn setup_edit(state: &mut AppState, f: impl FnOnce(&mut String, &mut usize)) {
    match state.model_setup.as_mut() {
        Some(ModelSetup::Gguf { input, pos, .. }) => f(input, pos),
        Some(ModelSetup::Mmproj { input, pos }) => f(input, pos),
        None => f(&mut state.input, &mut state.input_pos),
    }
}

/// Enter no modal de configuracao: valida o caminho do fa; avanca para o
/// segundo modal (mmproj) se o modelo for multimodal, ou dispara o load do
/// engine local (`WorkerMsg::SetLocalModel`).
fn handle_setup_confirm(state: &mut AppState, turns: &Sender<crate::agent::WorkerMsg>) {
    use crate::agent::WorkerMsg;
    let stage = state.model_setup.take();
    match stage {
        Some(ModelSetup::Gguf { input, multimodal, .. }) => {
            let path = input.trim().to_string();
            if path.is_empty() {
                state.model_setup = Some(ModelSetup::Gguf {
                    input: path,
                    pos: 0,
                    multimodal,
                });
                state.status = "informe o caminho do modelo GGUF".to_string();
            } else if !std::path::Path::new(&path).exists() {
                let pos = path.chars().count();
                state.model_setup = Some(ModelSetup::Gguf {
                    input: path.clone(),
                    pos,
                    multimodal,
                });
                state.push(Kind::Err, format!("arquivo nao encontrado: {}", path));
                state.status = "arquivo nao encontrado".to_string();
            } else if multimodal {
                let prefill = state.local_mmproj.clone();
                let prefill_pos = prefill.chars().count();
                state.setup_model = path;
                state.model_setup = Some(ModelSetup::Mmproj {
                    input: prefill,
                    pos: prefill_pos,
                });
                state.status =
                    "modelo multimodal: informe o mmproj (Enter vazio = sem)".to_string();
            } else {
                let mmproj = state.local_mmproj.clone();
                state.setup_model = path.clone();
                state.status = "carregando modelo local...".to_string();
                let _ = turns.send(WorkerMsg::SetLocalModel {
                    model: path,
                    mmproj,
                });
            }
        }
        Some(ModelSetup::Mmproj { input, .. }) => {
            let path = input.trim().to_string();
            if !path.is_empty() && !std::path::Path::new(&path).exists() {
                let pos = path.chars().count();
                state.model_setup = Some(ModelSetup::Mmproj { input: path.clone(), pos });
                state.push(Kind::Err, format!("arquivo nao encontrado: {}", path));
                state.status = "arquivo nao encontrado".to_string();
            } else {
                state.status = "carregando modelo local...".to_string();
                let _ = turns.send(WorkerMsg::SetLocalModel {
                    model: state.setup_model.clone(),
                    mmproj: path,
                });
            }
        }
        None => {}
    }
}

/// Edita o campo de texto do wizard de provider cloud ativo (URL/modelo/key).
fn provider_setup_edit(state: &mut AppState, f: impl FnOnce(&mut String, &mut usize)) {
    match state.provider_setup.as_mut() {
        Some(ProviderSetup::BaseUrl { input, pos })
        | Some(ProviderSetup::Model { input, pos })
        | Some(ProviderSetup::ApiKey { input, pos }) => f(input, pos),
        _ => {}
    }
}

/// Enter no wizard de provider cloud: valida a etapa atual e avanca; na ultima
/// etapa envia `WorkerMsg::AddCloudProvider`, que persiste em config.json.
fn handle_provider_setup_confirm(state: &mut AppState, turns: &Sender<crate::agent::WorkerMsg>) {
    use crate::agent::WorkerMsg;
    let stage = state.provider_setup.take();
    match stage {
        Some(ProviderSetup::Kind { pos }) => {
            let kind = CLOUD_KINDS.get(pos).cloned().unwrap_or_default();
            state.new_cloud_kind = pos;
            let prefill = kind.base_url_default().to_string();
            let prefill_pos = prefill.chars().count();
            state.new_cloud_base_url = prefill.clone();
            state.provider_setup = Some(ProviderSetup::BaseUrl {
                input: prefill,
                pos: prefill_pos,
            });
            state.status = "URL base do endpoint (Enter confirma)".to_string();
        }
        Some(ProviderSetup::BaseUrl { input, .. }) => {
            let url = input.trim().to_string();
            let url_pos = url.chars().count();
            if url.is_empty() || !(url.starts_with("http://") || url.starts_with("https://")) {
                state.push(Kind::Err, "URL base deve comecar com http:// ou https://".to_string());
                state.provider_setup = Some(ProviderSetup::BaseUrl {
                    input: url,
                    pos: url_pos,
                });
                state.status = "URL base invalida".to_string();
            } else {
                state.new_cloud_base_url = url;
                state.provider_setup = Some(ProviderSetup::Model {
                    input: String::new(),
                    pos: 0,
                });
                state.status = "modelo da API (ex.: gpt-5, claude-sonnet-4, gemini-2.5-flash)"
                    .to_string();
            }
        }
        Some(ProviderSetup::Model { input, .. }) => {
            let model = input.trim().to_string();
            let model_pos = model.chars().count();
            if model.is_empty() {
                state.push(Kind::Err, "informe o nome do modelo da API".to_string());
                state.provider_setup = Some(ProviderSetup::Model {
                    input: model,
                    pos: model_pos,
                });
                state.status = "modelo da API obrigatorio".to_string();
            } else {
                state.new_cloud_model = model;
                state.provider_setup = Some(ProviderSetup::ApiKey {
                    input: String::new(),
                    pos: 0,
                });
                state.status = "API key (opcional; Enter vazio = env do vendor)".to_string();
            }
        }
        Some(ProviderSetup::ApiKey { input, .. }) => {
            let kind = CLOUD_KINDS
                .get(state.new_cloud_kind)
                .cloned()
                .unwrap_or_default();
            let api_key = input.trim().to_string();
            let _ = turns.send(WorkerMsg::AddCloudProvider {
                kind,
                base_url: state.new_cloud_base_url.clone(),
                model: state.new_cloud_model.clone(),
                api_key,
            });
            state.status = "adicionando provider cloud...".to_string();
        }
        None => {}
    }
}

/// Insere um bloco de texto colado na posicao atual do cursor. O input e de
/// uma linha so, entao quebras de linha do texto colado viram espaco.
fn insert_text_at_cursor(state: &mut AppState, text: &str) {
    let sanitized: String = text
        .chars()
        .map(|c| if c == '\n' || c == '\r' { ' ' } else { c })
        .collect();
    if sanitized.is_empty() {
        return;
    }
    let byte = char_to_byte(&state.input, state.input_pos);
    state.input.insert_str(byte, &sanitized);
    state.input_pos += sanitized.chars().count();
}

fn draw(frame: &mut Frame, state: &mut AppState) {

    if state.show_welcome {
        draw_welcome(frame, state);

        return;
    }



    let app_background = Block::default().style(Style::default().bg(COLOR_APP_BG));
    frame.render_widget(app_background, frame.area());

    let main_layout = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]);
    let [body_area, footer_area] = main_layout.areas(frame.area());

    let cols = Layout::horizontal([Constraint::Percentage(74), Constraint::Percentage(26)]);
    let [left_area, right_area] = cols.areas(body_area);

    draw_left_panel(frame, state, left_area);
    draw_right_panel(frame, state, right_area);
    draw_footer(frame, state, footer_area);

    if state.session_view {
        draw_session_overlay(frame, state);
    }

    if state.provider_view {
        draw_provider_overlay(frame, state);
    }

    if state.models_view {
        draw_models_overlay(frame, state);
    }

    if let Some(req) = &state.pending_permission {
        draw_permission_overlay(frame, req);
    }

    if state.model_setup.is_some() {
        draw_model_setup_overlay(frame, state);
    }

    if state.provider_setup.is_some() {
        draw_provider_setup_overlay(frame, state);
    }
}

/// Modal de configuracao do modelo local: pede o caminho do GGUF e, se o
/// modelo for multimodal (Tab no primeiro modal), o caminho do mmproj.
fn draw_model_setup_overlay(frame: &mut Frame, state: &AppState) {
    let area = centered_rect(64, 40, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Yellow))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " configurar modelo local ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let (title, hint, input_text, pos, is_mmproj) = match &state.model_setup {
        Some(ModelSetup::Gguf { input, pos, multimodal }) => (
            "caminho do modelo GGUF (*.gguf)",
            if *multimodal {
                "modelo multimodal: [x] (Tab alterna) — o mmproj sera pedido a seguir"
            } else {
                "modelo multimodal: [ ] (Tab alterna)"
            },
            input.clone(),
            *pos,
            false,
        ),
        Some(ModelSetup::Mmproj { input, pos }) => (
            "arquivo mmproj — projetor multi-modal (Enter vazio = sem mmproj)",
            "",
            input.clone(),
            *pos,
            true,
        ),
        None => ("", "", String::new(), 0, false),
    };

    // Janela do caminho: exibe a cauda quando o texto excede a largura.
    let width = inner.width.saturating_sub(6) as usize;
    let chars: Vec<char> = input_text.chars().collect();
    let tail = chars.len().saturating_sub(width);
    let shown: String = if tail > 0 {
        let mut s = String::from("…");
        s.extend(chars[tail..].iter());
        s
    } else {
        chars.iter().collect()
    };
    let cursor_char = pos.saturating_sub(tail).min(width + 1) + if tail > 0 { 1 } else { 0 };

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!(" {}", title),
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                " > ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            if shown.is_empty() {
                Span::styled(
                    if is_mmproj {
                        "(deixe vazio se nao tiver mmproj)"
                    } else {
                        "informe o caminho completo do arquivo..."
                    },
                    Style::default().fg(COLOR_TEXT_MUTED),
                )
            } else {
                Span::styled(shown.clone(), Style::default().fg(COLOR_TEXT_MAIN))
            },
        ]),
        Line::from(""),
        Line::from(Span::styled(hint, Style::default().fg(COLOR_TEXT_MUTED))),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "Enter",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" confirmar     ", Style::default().fg(COLOR_TEXT_MUTED)),
            Span::styled(
                "Esc",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" cancelar", Style::default().fg(COLOR_TEXT_MUTED)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(lines).wrap(ratatui::widgets::Wrap { trim: true }),
        inner,
    );
    frame.set_cursor_position((inner.x + 3 + cursor_char as u16, inner.y + 3));
}

/// Modal do wizard de adicionar provider cloud: tipo (Tab cicla), URL base,
/// modelo e API key. A ultima etapa cria e persiste o provider.
fn draw_provider_setup_overlay(frame: &mut Frame, state: &AppState) {
    let area = centered_rect(64, 42, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Cyan))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " adicionar provider cloud ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = vec![Line::from("")];
    let mut cursor_pos: Option<(u16, u16)> = None;

    // Etapa do tipo de API: rende 3 opcoes selecionaveis via Tab.
    if let Some(ProviderSetup::Kind { pos }) = &state.provider_setup {
        lines.push(Line::from(Span::styled(
            " tipo de API (Tab alterna)",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        for (i, kind) in CLOUD_KINDS.iter().enumerate() {
            let selected = i == *pos;
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(COLOR_TEXT_MAIN)
            };
            let marker = if selected { "[x]" } else { "[ ]" };
            lines.push(Line::from(Span::styled(
                format!("  {} {}   {}", marker, kind.label(), kind.base_url_default()),
                style,
            )));
        }
        lines.push(Line::from(""));
    } else {
        // Etapas de texto: URL base / modelo / API key.
        let (label, hint, input_text, pos) = match &state.provider_setup {
            Some(ProviderSetup::BaseUrl { input, pos }) => (
                "URL base do endpoint",
                "ex.: https://api.openai.com/v1 (ou qualquer endpoint OpenAI-compativel)",
                input.clone(),
                *pos,
            ),
            Some(ProviderSetup::Model { input, pos }) => (
                "modelo da API",
                "ex.: gpt-5, claude-sonnet-4, gemini-2.5-flash",
                input.clone(),
                *pos,
            ),
            Some(ProviderSetup::ApiKey { input, pos }) => (
                "API key",
                "deixe vazio para usar a env do vendor (OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)",
                input.clone(),
                *pos,
            ),
            _ => ("", "", String::new(), 0),
        };

        let width = inner.width.saturating_sub(6) as usize;
        let chars: Vec<char> = input_text.chars().collect();
        let tail = chars.len().saturating_sub(width);
        let shown: String = if tail > 0 {
            let mut s = String::from("…");
            s.extend(chars[tail..].iter());
            s
        } else {
            chars.iter().collect()
        };
        let cursor_char = pos.saturating_sub(tail).min(width + 1) + if tail > 0 { 1 } else { 0 };

        lines.push(Line::from(Span::styled(
            format!(" {}", label),
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )));
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(
                " > ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            if shown.is_empty() {
                Span::styled(hint, Style::default().fg(COLOR_TEXT_MUTED))
            } else {
                Span::styled(shown.clone(), Style::default().fg(COLOR_TEXT_MAIN))
            },
        ]));
        lines.push(Line::from(""));
        cursor_pos = Some((inner.x + 3 + cursor_char as u16, inner.y + 3));
    }

    lines.push(Line::from(vec![
        Span::styled(
            "Enter",
            Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" confirmar     ", Style::default().fg(COLOR_TEXT_MUTED)),
        Span::styled(
            "Esc",
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        ),
        Span::styled(" cancelar", Style::default().fg(COLOR_TEXT_MUTED)),
    ]));

    frame.render_widget(
        Paragraph::new(lines).wrap(ratatui::widgets::Wrap { trim: true }),
        inner,
    );

    if let Some((x, y)) = cursor_pos {
        frame.set_cursor_position((x, y));
    }
}

/// Modal bloqueante: aparece enquanto uma tool sensivel (ex.: fetch_url)
/// aguarda a confirmacao do usuario.
fn draw_permission_overlay(frame: &mut Frame, req: &PermissionRequest) {
    let area = centered_rect(60, 30, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::Yellow))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " permissao necessaria ",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let detail: String = req.detail.chars().take(inner.width as usize * 3).collect();
    let lines = vec![
        Line::from(Span::styled(
            format!("o agente quer executar: {}", req.tool),
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(detail, Style::default().fg(COLOR_TEXT_MUTED))),
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "y",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                " / Enter: permitir    ",
                Style::default().fg(COLOR_TEXT_MUTED),
            ),
            Span::styled(
                "n",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            Span::styled(" / Esc: negar", Style::default().fg(COLOR_TEXT_MUTED)),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(lines).wrap(ratatui::widgets::Wrap { trim: true }),
        inner,
    );
}

fn draw_session_overlay(frame: &mut Frame, state: &mut AppState) {
    let area = centered_rect(72, 72, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " Sessoes ( /session ) ",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let last_line = inner.y + inner.height.saturating_sub(1);
    let hint = Line::from(Span::styled(
        " j/k ou setas: navegar   Enter: abrir   d: apagar   q/Esc: fechar",
        Style::default().fg(COLOR_TEXT_MUTED),
    ));
    frame.render_widget(hint, Rect::new(inner.x, last_line, inner.width, 1));

    if state.sessions.is_empty() {
        let msg = Paragraph::new(Line::from(Span::styled(
            "(nenhuma sessao gravada ainda)",
            Style::default().fg(COLOR_TEXT_MUTED),
        )))
        .style(Style::default().bg(COLOR_CARD_BG));
        frame.render_widget(msg, Rect::new(inner.x, inner.y, inner.width, 3));
        return;
    }

    let list_height = inner.height.saturating_sub(2) as usize;
    // Mantem o cursor visivel na janela.
    if state.session_cursor < state.session_scroll {
        state.session_scroll = state.session_cursor;
    } else if state.session_cursor >= state.session_scroll + list_height {
        state.session_scroll = state.session_cursor + 1 - list_height;
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    let start = state.session_scroll;
    let end = (state.session_scroll + list_height).min(state.sessions.len());
    for (i, session) in state.sessions[start..end].iter().enumerate() {
        let selected = start + i == state.session_cursor;
        let badge_color = if session.mode == "solo" {
            COLOR_TEXT_MUTED
        } else {
            Color::Magenta
        };
        let preview: String = session.preview.chars().take(48).collect();
        if preview.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("  ", Style::default()),
                Span::styled(
                    if selected { "» " } else { "  " },
                    Style::default().fg(if selected { COLOR_ACCENT } else { COLOR_BORDER }),
                ),
                Span::styled(
                    format!("[{}]", session.mode),
                    Style::default()
                        .fg(badge_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" (sem mensagens)", Style::default().fg(COLOR_TEXT_MUTED)),
            ]));
        } else {
            let spans = vec![
                Span::styled(
                    if selected { "» " } else { "  " },
                    Style::default().fg(if selected { COLOR_ACCENT } else { COLOR_BORDER }),
                ),
                Span::styled(
                    format!("[{}]", session.mode),
                    Style::default()
                        .fg(badge_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" ", Style::default()),
                Span::styled(
                    preview,
                    Style::default()
                        .fg(COLOR_TEXT_MAIN)
                        .add_modifier(if selected {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                Span::styled(
                    format!(
                        "  {} msgs · {}",
                        session.messages,
                        fmt_ago(session.last_created)
                    ),
                    Style::default().fg(COLOR_TEXT_MUTED),
                ),
            ];
            lines.push(Line::from(spans));
        }
    }
    let list = Paragraph::new(lines).style(Style::default().bg(COLOR_CARD_BG));
    frame.render_widget(
        list,
        Rect::new(inner.x, inner.y, inner.width, list_height as u16),
    );
}






fn help_text() -> String {
    "atalhos:\n\
     Enter        envia (ou enfileira, se ja gerando)\n\
     Esc          cancela a geracao / sai\n\
     Ctrl+A       alterna solo <-> maestro\n\
     Ctrl+Enter   interrompe e reenvia (steer)\n\
     /session     lista/retoma conversas salvas\n\
     /provider    escolhe local ou API OpenAI-compativel (so o ROOT)\n\
     /models      lista modelos da API (provider openai)\n\
     /clear       limpa a tela do chat\n\
     @arquivo     menciona um arquivo do projeto (autocomplete)\n\
     ↑/↓ no menu  navega; Tab/Enter seleciona; Esc fecha"
        .to_string()
}

fn fmt_ago(ts: f64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let diff = (now - ts).max(0.0) as i64;
    if diff < 60 {
        "agora".to_string()
    } else if diff < 3600 {
        format!("{} min atras", diff / 60)
    } else if diff < 86400 {
        format!("{} h atras", diff / 3600)
    } else {
        format!("{} d atras", diff / 86400)
    }
}

fn draw_left_panel(frame: &mut Frame, state: &mut AppState, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]);
    let [chat_area, input_area] = rows.areas(area);

    let lines = state.rendered_lines(chat_area);
    let top = state
        .max_scroll
        .saturating_sub(state.scroll.min(state.max_scroll));

    let chat_title = if state.gen_active {
        Span::styled(
            format!(
                " ● {} ",
                state.working_agent.as_deref().unwrap_or("gerando...")
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(" Chat ", Style::default().fg(COLOR_TEXT_MUTED))
    };

    let chat_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_BORDER))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(chat_title));

    let body = Paragraph::new(lines.clone())
        .scroll((top as u16, 0))
        .block(chat_block);
    frame.render_widget(body, chat_area);

    if state.max_scroll > 0 {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"))
            .thumb_symbol("█")
            .style(Style::default().fg(COLOR_BORDER))
            .thumb_style(Style::default().fg(COLOR_TEXT_MUTED));

        let scrollbar_area = chat_area.inner(ratatui::layout::Margin {
            vertical: 1,
            horizontal: 0,
        });
        state.scrollbar_state = state
            .scrollbar_state
            .content_length(lines.len())
            .position(top);
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut state.scrollbar_state);
    }

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if state.gen_active {
            COLOR_BORDER
        } else {
            COLOR_ACCENT
        }))
        .style(Style::default().bg(COLOR_INPUT_BG));

    let input_line = if state.input.is_empty() {
        Line::from(Span::styled(
            " Digite sua mensagem...  ( / comandos   @ arquivos )",
            Style::default().fg(COLOR_TEXT_MUTED),
        ))
    } else {
        Line::from(Span::styled(
            format!(" {}", state.input),
            Style::default().fg(COLOR_TEXT_MAIN),
        ))
    };
    let input_widget = Paragraph::new(input_line).block(input_block);
    frame.render_widget(input_widget, input_area);

    let cursor_x = (1 + state.input_pos) as u16 + input_area.x;
    let cursor_y = input_area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));

    if let Some(popup) = &state.popup {
        draw_input_popup(frame, popup, input_area, chat_area);
    }

    let cursor_x = (1 + state.input_pos) as u16 + input_area.x;
    let cursor_y = input_area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}

/// Menu flutuante de autocomplete, ancorado logo acima do input.
 pub fn draw_input_popup(frame: &mut Frame, popup: &InputPopup, input_area: Rect, chat_area: Rect) {
    let visible = popup.items.len().min(8);
    let height = visible as u16 + 2;
    let y = input_area.y.saturating_sub(height).max(chat_area.y);
    let popup_area = Rect::new(input_area.x, y, input_area.width, height);

    let title = match popup.kind {
        PopupKind::Slash => " comandos ",
        PopupKind::Mention => " arquivos ",
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_POPUP_BG))
        .title(Line::from(Span::styled(
            title,
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(popup_area);
    frame.render_widget(Clear, popup_area);
    frame.render_widget(block, popup_area);

    // Mantem o item selecionado visivel dentro da janela de `visible` linhas.
    let start = popup
        .selected
        .saturating_sub(visible.saturating_sub(1))
        .min(popup.items.len().saturating_sub(visible));
    let end = (start + visible).min(popup.items.len());

    let mut lines: Vec<Line<'static>> = Vec::new();
    for (i, item) in popup.items[start..end].iter().enumerate() {
        let idx = start + i;
        let selected = idx == popup.selected;
        let bg = if selected {
            COLOR_POPUP_SELECTED_BG
        } else {
            COLOR_POPUP_BG
        };
        let mut spans = vec![
            Span::styled(
                if selected { "» " } else { "  " },
                Style::default()
                    .fg(if selected { COLOR_ACCENT } else { COLOR_BORDER })
                    .bg(bg),
            ),
            Span::styled(
                item.label.clone(),
                Style::default()
                    .fg(COLOR_TEXT_MAIN)
                    .bg(bg)
                    .add_modifier(if selected {
                        Modifier::BOLD
                    } else {
                        Modifier::empty()
                    }),
            ),
        ];
        if !item.detail.is_empty() {
            spans.push(Span::styled(
                format!("  {}", item.detail),
                Style::default().fg(COLOR_TEXT_MUTED).bg(bg),
            ));
        }
        lines.push(Line::from(spans));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn draw_right_panel(frame: &mut Frame, state: &AppState, area: Rect) {
    let side_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_BORDER))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " Painel ",
            Style::default().fg(COLOR_TEXT_MUTED),
        )));

    let inner_area = side_block.inner(area);
    frame.render_widget(side_block, area);

    let s = &state.session;
    let pct = if s.ctx_max > 0 {
        (s.ctx_used as f64 / s.ctx_max as f64 * 100.0).min(100.0)
    } else {
        0.0
    };
    let model_short = s.model.split('/').last().unwrap_or(&s.model);

    let mut lines = vec![
        Line::from(Span::styled(
            model_short,
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("status: {}", state.status),
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Modo",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("{} (Ctrl+A)", state.mode.label()),
            Style::default().fg(if state.mode == crate::agent::Mode::Agents {
                Color::Magenta
            } else {
                COLOR_TEXT_MUTED
            }),
        )),
        Line::from(Span::styled(
            crate::subagents::discover_agents(std::path::Path::new(&s.project_dir))
                .len()
                .to_string()
                + " subagente(s)",
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Context",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("{} / {} tokens", s.ctx_used, s.ctx_max),
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
        Line::from(Span::styled(
            format!("{:.1}% used", pct),
            Style::default().fg(if pct > 80.0 { Color::Red } else { COLOR_ACCENT }),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Speed",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            s.tps
                .map(|v| format!("{:.1} tok/s", v))
                .unwrap_or_else(|| "—".to_string()),
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Tools",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            if s.tools_enabled {
                "enabled"
            } else {
                "disabled"
            },
            Style::default().fg(if s.tools_enabled {
                Color::Green
            } else {
                COLOR_TEXT_MUTED
            }),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Equipe",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )),
    ];

    let root_label = if state.mode == crate::agent::Mode::Agents {
        "maestro".to_string()
    } else {
        "drill (solo)".to_string()
    };
    let root_active = state.working_agent.as_deref() == Some(root_label.as_str());
    lines.push(Line::from(vec![
        Span::styled(
            if root_active { "● " } else { "  " },
            Style::default()
                .fg(if root_active {
                    Color::Cyan
                } else {
                    COLOR_BORDER
                })
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            root_label.clone(),
            Style::default()
                .fg(if root_active {
                    COLOR_ACCENT
                } else {
                    COLOR_TEXT_MUTED
                })
                .add_modifier(if root_active {
                    Modifier::BOLD
                } else {
                    Modifier::empty()
                }),
        ),
    ]));

    let team: Vec<String> = crate::subagents::discover_agents(std::path::Path::new(&s.project_dir))
        .into_iter()
        .map(|cfg| cfg.name)
        .collect();
    if team.is_empty() {
        lines.push(Line::from(Span::styled(
            "(nenhum subagente)",
            Style::default().fg(COLOR_TEXT_MUTED),
        )));
    } else {
        for name in &team {
            let active = state.working_agent.as_deref() == Some(name.as_str());
            lines.push(Line::from(vec![
                Span::styled(
                    if active { "● " } else { "  " },
                    Style::default()
                        .fg(if active { Color::Cyan } else { COLOR_BORDER })
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    name.clone(),
                    Style::default()
                        .fg(if active {
                            COLOR_ACCENT
                        } else {
                            COLOR_TEXT_MUTED
                        })
                        .add_modifier(if active {
                            Modifier::BOLD
                        } else {
                            Modifier::empty()
                        }),
                ),
                if active {
                    Span::styled(
                        " (trabalhando)",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::ITALIC),
                    )
                } else {
                    Span::styled("", Style::default())
                },
            ]));
        }
    }

    if !state.queue.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            format!("Queue ({})", state.queue.len()),
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD),
        )));
        for (i, item) in state.queue.iter().take(3).enumerate() {
            let preview: String = item.chars().take(18).collect();
            lines.push(Line::from(Span::styled(
                format!("{}. {}...", i + 1, preview),
                Style::default().fg(COLOR_TEXT_MUTED),
            )));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner_area);
}

fn draw_footer(frame: &mut Frame, state: &AppState, area: Rect) {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    let footer_text = Line::from(vec![
        Span::styled(
            format!(" {} ", cwd),
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(" │ ", Style::default().fg(COLOR_BORDER).bg(COLOR_APP_BG)),
        Span::styled(
            format!("{:.1}K tokens", state.session.ctx_used as f64 / 1000.0),
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(" │ ", Style::default().fg(COLOR_BORDER).bg(COLOR_APP_BG)),
        Span::styled(
            "Enter ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "send ",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(
            "Esc ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "cancel/exit ",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(
            "Ctrl+A ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "maestro ",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(
            "Ctrl+Enter ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "steer",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(" │ ", Style::default().fg(COLOR_BORDER).bg(COLOR_APP_BG)),
        Span::styled(
            "/ ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "comandos ",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
        Span::styled(
            "@ ",
            Style::default()
                .fg(COLOR_TEXT_MAIN)
                .add_modifier(Modifier::BOLD)
                .bg(COLOR_APP_BG),
        ),
        Span::styled(
            "arquivos",
            Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG),
        ),
    ]);

    frame.render_widget(
        Paragraph::new(footer_text).style(Style::default().bg(COLOR_APP_BG)),
        area,
    );
}
