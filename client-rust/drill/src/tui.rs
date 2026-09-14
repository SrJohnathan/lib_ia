use crate::agent;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseEventKind, EnableMouseCapture, DisableMouseCapture, KeyboardEnhancementFlags};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState};
use ratatui::Frame;
use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;
use crate::tui::highlight_code::CodeHighlighter;

mod highlight_code;

// --- Cores e Temas ---
const COLOR_APP_BG: Color = Color::Rgb(12, 12, 14);
const COLOR_CARD_BG: Color = Color::Rgb(24, 24, 28);
const COLOR_CODE_BG: Color = Color::Rgb(16, 16, 20);         // Card de Código (Mais Escuro)
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
    pub fn new(model: String, ngl: usize, ctx_max: usize, tools_enabled: bool, reasoning_enabled: bool, project_dir: String) -> Self {
        Self {
            model, ngl, ctx_max, ctx_used: 0, tools_enabled, reasoning_enabled,
            tps: None, last_tool: None, last_diff: None, project_dir,
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
    Err(String),
    Done,
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
        }
    }

    pub fn push(&mut self, kind: Kind, text: String) {
        self.lines.push(UiLine { kind, text });
    }

    pub fn append_stream(&mut self, kind: Kind, text: String) {
        if let Some(last) = self.lines.last_mut() {
            if last.kind == kind {
                last.text.push_str(&text);
                return;
            }
        }
        self.push(kind, text);
    }

    pub fn apply(&mut self, event: UiEvent) {
        match event {
            UiEvent::Content(text) => {
                self.append_stream(Kind::Assistant, text);
                self.scroll_to_bottom();
            }
            UiEvent::Reasoning(text) => {
                let starting_new_block = !matches!(self.lines.last(), Some(l) if l.kind == Kind::Reasoning);
                let chunk = if starting_new_block { format!("\n[thinking]\n{}", text) } else { text };
                self.append_stream(Kind::Reasoning, chunk);
                self.scroll_to_bottom();
            }
            UiEvent::ToolCall(text) => {
                self.push(Kind::Tool, text);
                self.scroll_to_bottom();
            }
            UiEvent::ToolStart(name) => {
                self.status = format!("running {}", name);
                self.session.last_tool = Some(name);
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
            UiEvent::Err(text) => {
                self.push(Kind::Err, format!("error: {}", text));
                self.status = "error".to_string();
            }
            UiEvent::Done => {
                self.gen_active = false;
                self.interrupt.store(false, Ordering::Relaxed);
                self.status = "ready".to_string();
            }
        }
    }

    pub fn scroll_to_bottom(&mut self) {
        self.scroll = 0;
        self.scrollbar_state = self.scrollbar_state.position(self.max_scroll);
    }

    pub fn scroll_up(&mut self, amount: usize) {
        self.scroll = (self.scroll + amount).min(self.max_scroll);
        let pos = self.max_scroll.saturating_sub(self.scroll);
        self.scrollbar_state = self.scrollbar_state.position(pos);
    }

    pub fn scroll_down(&mut self, amount: usize) {
        self.scroll = self.scroll.saturating_sub(amount);
        let pos = self.max_scroll.saturating_sub(self.scroll);
        self.scrollbar_state = self.scrollbar_state.position(pos);
    }

    pub fn rendered_lines(&mut self, area: Rect) -> Vec<Line<'static>> {
        let mut rendered: Vec<Line<'static>> = Vec::new();
        let mut in_code_block = false;

        for line in &self.lines {
            match line.kind {
                Kind::User => {
                    rendered.push(Line::from(vec![
                        Span::styled("> ", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
                        Span::styled(line.text.clone(), Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD)),
                    ]));
                }
                Kind::Tool => {
                    rendered.push(Line::from(vec![
                        Span::styled("← ", Style::default().fg(COLOR_ACCENT)),
                        Span::styled(line.text.clone(), Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD)),
                    ]));
                }
                Kind::ToolResult => {
                    let lines_vec: Vec<&str> = line.text.split('\n').collect();
                    let target_width = area.width.saturating_sub(2) as usize;
                    let mut line_counter = 1; // Pode ser sincronizado com o offset real do arquivo

                    for sub in lines_vec {
                        if sub.starts_with("---") {
                            rendered.push(Line::from(Span::styled(
                                sub.to_string(),
                                Style::default().fg(COLOR_TEXT_MUTED).add_modifier(Modifier::BOLD),
                            )));
                            continue;
                        }

                        let (bg_color, prefix_sign, text_content) = if let Some(stripped) = sub.strip_prefix('+') {
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
                        spans.push(Span::styled(num_str, Style::default().fg(COLOR_LINE_NUM).bg(bg_color)));

                        // 2. Indicador de mudança (+ / -)
                        spans.push(Span::styled(prefix_sign, Style::default().fg(if prefix_sign == "+ " { COLOR_ACCENT } else { COLOR_DIFF_REMOVE_FG }).bg(bg_color)));

                        // 3. Syntax Highlighting do código com o fundo do diff
                        let highlighted_line = self.highlighter.highlight_line(text_content, "rs", 0);
                        for mut span in highlighted_line.spans {
                            span.style = span.style.bg(bg_color); // Força a cor de fundo do diff no highlight
                            spans.push(span);
                        }

                        // 4. Preenchimento de fundo até o final do painel
                        let current_len: usize = spans.iter().map(|s| s.content.chars().count()).sum();
                        if current_len < target_width {
                            let padding = " ".repeat(target_width - current_len);
                            spans.push(Span::styled(padding, Style::default().bg(bg_color)));
                        }

                        rendered.push(Line::from(spans));
                        line_counter += 1;
                    }
                }
                Kind::Reasoning => {
                    for sub in line.text.split('\n') {
                        rendered.push(Line::from(Span::styled(sub.to_string(), Style::default().fg(COLOR_TEXT_MUTED).add_modifier(Modifier::ITALIC))));
                    }
                }
                Kind::Assistant => {
                    let mut current_lang = String::from("rs"); // Linguagem padrão caso não especificada

                    for sub in line.text.split('\n') {
                        let trimmed = sub.trim_start();
                        if trimmed.starts_with("```") {
                            in_code_block = !in_code_block;

                            if in_code_block {
                                // Captura a linguagem após os 3 Backticks (ex: ```rust -> "rust")
                                let lang = trimmed.trim_start_matches("```").trim();
                                if !lang.is_empty() {
                                    current_lang = lang.to_string();
                                }
                            }

                            rendered.push(Line::from(Span::styled(
                                "───",
                                Style::default().fg(COLOR_BORDER).bg(COLOR_CODE_BG),
                            )));
                            continue;
                        }

                        if in_code_block {
                            let card_width = area.width.saturating_sub(2) as usize;
                            // Reutiliza o highlighter já instanciado e passa a linha + linguagem
                            let highlighted_line = self.highlighter.highlight_line(sub, &current_lang,card_width);
                            rendered.push(highlighted_line);
                        } else {
                            rendered.push(Line::from(Span::styled(
                                sub.to_string(),
                                Style::default().fg(COLOR_TEXT_MAIN),
                            )));
                        }
                    }
                }
                _ => {
                    rendered.push(Line::from(Span::styled(line.text.clone(), Style::default().fg(COLOR_TEXT_MAIN))));
                }
            }
        }

        let content_height = area.height.saturating_sub(2) as usize;
        let cap = content_height.max(1);
        let n = rendered.len();
        self.max_scroll = n.saturating_sub(cap);

        let start = n.saturating_sub(cap + self.scroll.min(self.max_scroll));
        let end = n.saturating_sub(self.scroll.min(self.max_scroll));

        rendered[start..end].to_vec()
    }
}

// Auxiliar para destacar o código
fn format_code_line(line: &str) -> Line<'static> {
    let mut spans = Vec::new();

    if line.trim_start().starts_with("//") || line.trim_start().starts_with('#') {
        return Line::from(Span::styled(line.to_string(), Style::default().fg(COLOR_COMMENT).bg(COLOR_CODE_BG)));
    }

    let words = line.split_whitespace();
    let mut current_pos = 0;

    for word in words {
        if let Some(idx) = line[current_pos..].find(word) {
            let space_slice = &line[current_pos..current_pos + idx];
            if !space_slice.is_empty() {
                spans.push(Span::styled(space_slice.to_string(), Style::default().bg(COLOR_CODE_BG)));
            }
            current_pos += idx + word.len();

            let style = match word {
                "fn" | "let" | "mut" | "pub" | "use" | "const" | "return" | "if" | "else" | "match" | "struct" | "enum" => {
                    Style::default().fg(COLOR_KEYWORD).add_modifier(Modifier::BOLD).bg(COLOR_CODE_BG)
                }
                w if w.starts_with('"') || w.ends_with('"') || w.starts_with('\'') => {
                    Style::default().fg(COLOR_STRING).bg(COLOR_CODE_BG)
                }
                _ => Style::default().fg(COLOR_TEXT_MAIN).bg(COLOR_CODE_BG),
            };

            spans.push(Span::styled(word.to_string(), style));
        }
    }

    if current_pos < line.len() {
        spans.push(Span::styled(line[current_pos..].to_string(), Style::default().bg(COLOR_CODE_BG)));
    }

    Line::from(spans)
}

pub fn run(mut state: AppState, events: Receiver<UiEvent>, turns: Sender<crate::agent::WorkerMsg>) -> Result<(), io::Error> {
    let mut terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), EnableMouseCapture);

    let kitty_keyboard = crossterm::terminal::supports_keyboard_enhancement().unwrap_or(false);
    {
        let _ = crossterm::execute!(std::io::stdout(), event::EnableBracketedPaste);
        if kitty_keyboard {
            let _ = crossterm::execute!(std::io::stdout(), event::PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES));
        }
    }

    let result = loop_result(&mut terminal, &mut state, &events, &turns);

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
) -> Result<(), io::Error> {
    loop {
        let mut has_new_events = false;
        while let Ok(event) = events.try_recv() {
            state.apply(event);
            has_new_events = true;
        }

        if !state.gen_active {
            if let Some(text) = state.queue.pop_front() {
                state.push(Kind::User, text.clone());
                state.status = "generating".to_string();
                state.gen_active = true;
                let _ = turns.send(agent::WorkerMsg::Turn(text));
            }
        }

        terminal.draw(|frame| draw(frame, state))?;

        if event::poll(Duration::from_millis(80))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    if let Some(action) = read_key(key, state) {
                        match action {
                            Action::Quit => return Ok(()),
                            Action::Submit => {
                                let text = if !state.input.is_empty() {
                                    std::mem::take(&mut state.input)
                                } else {
                                    state.queue.pop_front().unwrap_or_default()
                                };
                                state.input_pos = 0;
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
                _ => {}
            }
        }
    }
}

enum Action { Quit, Submit, Queue, Steer, Interrupt, ToggleMode }

fn read_key(key: KeyEvent, state: &mut AppState) -> Option<Action> {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Some(Action::Quit);
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('a') {
        return Some(Action::ToggleMode);
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Enter {
        return Some(Action::Steer);
    }
    match key.code {
        KeyCode::Esc => {
            if state.gen_active { Some(Action::Interrupt) } else { Some(Action::Quit) }
        }
        KeyCode::Enter => {
            if state.gen_active {
                if !state.input.is_empty() { Some(Action::Queue) } else { None }
            } else if !state.input.is_empty() || !state.queue.is_empty() {
                Some(Action::Submit)
            } else { None }
        }
        KeyCode::PageUp => { state.scroll_up(20); None }
        KeyCode::PageDown => { state.scroll_down(20); None }
        KeyCode::Backspace => {
            if !state.input.is_empty() && state.input_pos > 0 {
                state.input.remove(state.input_pos - 1);
                state.input_pos -= 1;
            }
            None
        }
        KeyCode::Delete => {
            if state.input_pos < state.input.len() { state.input.remove(state.input_pos); }
            None
        }
        KeyCode::Left => { if state.input_pos > 0 { state.input_pos -= 1; } None }
        KeyCode::Right => { if state.input_pos < state.input.len() { state.input_pos += 1; } None }
        KeyCode::Home => { state.input_pos = 0; None }
        KeyCode::End => { state.input_pos = state.input.len(); None }
        KeyCode::Char(c) => {
            state.input.insert(state.input_pos, c);
            state.input_pos += 1;
            None
        }
        _ => None,
    }
}

fn draw(frame: &mut Frame, state: &mut AppState) {
    let app_background = Block::default().style(Style::default().bg(COLOR_APP_BG));
    frame.render_widget(app_background, frame.area());

    let main_layout = Layout::vertical([Constraint::Min(3), Constraint::Length(1)]);
    let [body_area, footer_area] = main_layout.areas(frame.area());

    let cols = Layout::horizontal([Constraint::Percentage(74), Constraint::Percentage(26)]);
    let [left_area, right_area] = cols.areas(body_area);

    draw_left_panel(frame, state, left_area);
    draw_right_panel(frame, state, right_area);
    draw_footer(frame, state, footer_area);
}

fn draw_left_panel(frame: &mut Frame, state: &mut AppState, area: Rect) {
    let rows = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]);
    let [chat_area, input_area] = rows.areas(area);

    let lines = state.rendered_lines(chat_area);
    state.scrollbar_state = state.scrollbar_state.content_length(state.max_scroll);

    let chat_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_BORDER))
        .style(Style::default().bg(COLOR_CARD_BG));

    let body = Paragraph::new(lines).block(chat_block);
    frame.render_widget(body, chat_area);

    if state.max_scroll > 0 {
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some("│"))
            .thumb_symbol("█")
            .style(Style::default().fg(COLOR_BORDER))
            .thumb_style(Style::default().fg(COLOR_TEXT_MUTED));

        let scrollbar_area = chat_area.inner(ratatui::layout::Margin { vertical: 1, horizontal: 0 });
        frame.render_stateful_widget(scrollbar, scrollbar_area, &mut state.scrollbar_state);
    }

    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(if state.gen_active { COLOR_BORDER } else { COLOR_ACCENT }))
        .style(Style::default().bg(COLOR_INPUT_BG));

    let input = format!(" {}", state.input);
    let input_widget = Paragraph::new(Line::from(Span::styled(input, Style::default().fg(COLOR_TEXT_MAIN))))
        .block(input_block);
    frame.render_widget(input_widget, input_area);

    let cursor_x = (1 + state.input_pos) as u16 + input_area.x;
    let cursor_y = input_area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn draw_right_panel(frame: &mut Frame, state: &AppState, area: Rect) {
    let side_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_BORDER))
        .style(Style::default().bg(COLOR_CARD_BG));

    let inner_area = side_block.inner(area);
    frame.render_widget(side_block, area);

    let s = &state.session;
    let pct = if s.ctx_max > 0 { (s.ctx_used as f64 / s.ctx_max as f64 * 100.0).min(100.0) } else { 0.0 };
    let model_short = s.model.split('/').last().unwrap_or(&s.model);

    let mut lines = vec![
        Line::from(Span::styled(model_short, Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(format!("status: {}", state.status), Style::default().fg(COLOR_TEXT_MUTED))),
        Line::from(""),
        Line::from(Span::styled("Modo", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(
            format!("{} (Ctrl+A)", state.mode.label()),
            Style::default().fg(if state.mode == crate::agent::Mode::Agents { Color::Magenta } else { COLOR_TEXT_MUTED }),
        )),
        Line::from(Span::styled(
            crate::subagents::discover_agents(std::path::Path::new(&s.project_dir))
                .len()
                .to_string() + " subagente(s)",
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
        Line::from(""),
        Line::from(Span::styled("Context", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(format!("{} / {} tokens", s.ctx_used, s.ctx_max), Style::default().fg(COLOR_TEXT_MUTED))),
        Line::from(Span::styled(format!("{:.1}% used", pct), Style::default().fg(if pct > 80.0 { Color::Red } else { COLOR_ACCENT }))),
        Line::from(""),
        Line::from(Span::styled("Speed", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(s.tps.map(|v| format!("{:.1} tok/s", v)).unwrap_or_else(|| "—".to_string()), Style::default().fg(COLOR_TEXT_MUTED))),
        Line::from(""),
        Line::from(Span::styled("Tools", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))),
        Line::from(Span::styled(if s.tools_enabled { "enabled" } else { "disabled" }, Style::default().fg(if s.tools_enabled { Color::Green } else { COLOR_TEXT_MUTED }))),
    ];

    if !state.queue.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(format!("Queue ({})", state.queue.len()), Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD))));
        for (i, item) in state.queue.iter().take(3).enumerate() {
            let preview: String = item.chars().take(18).collect();
            lines.push(Line::from(Span::styled(format!("{}. {}...", i + 1, preview), Style::default().fg(COLOR_TEXT_MUTED))));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner_area);
}

fn draw_footer(frame: &mut Frame, state: &AppState, area: Rect) {
    let cwd = std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_else(|_| ".".to_string());

    let footer_text = Line::from(vec![
        Span::styled(format!(" {} ", cwd), Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
        Span::styled(" │ ", Style::default().fg(COLOR_BORDER).bg(COLOR_APP_BG)),
        Span::styled(format!("{:.1}K tokens", state.session.ctx_used as f64 / 1000.0), Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
        Span::styled(" │ ", Style::default().fg(COLOR_BORDER).bg(COLOR_APP_BG)),
        Span::styled("Enter ", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD).bg(COLOR_APP_BG)),
        Span::styled("send ", Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
        Span::styled("Esc ", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD).bg(COLOR_APP_BG)),
        Span::styled("cancel/exit ", Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
        Span::styled("Ctrl+A ", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD).bg(COLOR_APP_BG)),
        Span::styled("maestro ", Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
        Span::styled("Ctrl+Enter ", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD).bg(COLOR_APP_BG)),
        Span::styled("steer", Style::default().fg(COLOR_TEXT_MUTED).bg(COLOR_APP_BG)),
    ]);

    frame.render_widget(Paragraph::new(footer_text).style(Style::default().bg(COLOR_APP_BG)), area);
}