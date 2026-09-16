//! Overlay hierárquico de `/provider`.
//!
//!   /provider
//!     ├── Local  → lista GGUF; Ctrl+A adiciona local
//!     └── Cloud  → lista APIs; Ctrl+A wizard cloud
//!
//! Esc na lista volta para Local/Cloud. Esc no root fecha.
//! Tecla `a` solta NÃO faz nada (não bugava o input).

use crate::provider::ProviderEntry;
use crate::tui::helpers::centered_rect;
use crossterm::event::{KeyCode, KeyModifiers};
use ratatui::Frame;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};

const COLOR_CARD_BG: Color = Color::Rgb(24, 24, 28);
const COLOR_TEXT_MAIN: Color = Color::Rgb(220, 220, 230);
const COLOR_TEXT_MUTED: Color = Color::Rgb(130, 130, 145);
const COLOR_ACCENT: Color = Color::Rgb(90, 165, 255);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderNav {
    #[default]
    Root,
    LocalList,
    CloudList,
}

impl ProviderNav {
    pub fn title(self) -> &'static str {
        match self {
            ProviderNav::Root => " Providers ( /provider ) ",
            ProviderNav::LocalList => " Providers · Local ",
            ProviderNav::CloudList => " Providers · Cloud ",
        }
    }

    pub fn is_list(self) -> bool {
        matches!(self, ProviderNav::LocalList | ProviderNav::CloudList)
    }
}

pub fn filtered_providers<'a>(
    all: &'a [ProviderEntry],
    nav: ProviderNav,
) -> Vec<(usize, &'a ProviderEntry)> {
    match nav {
        ProviderNav::Root => Vec::new(),
        ProviderNav::LocalList => all.iter().enumerate().filter(|(_, e)| !e.cloud).collect(),
        ProviderNav::CloudList => all.iter().enumerate().filter(|(_, e)| e.cloud).collect(),
    }
}

pub fn draw(
    frame: &mut Frame,
    nav: ProviderNav,
    providers: &[ProviderEntry],
    selected_uuid: Option<&str>,
    cursor: usize,
) {
    let area = centered_rect(72, 48, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            nav.title(),
            Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line<'static>> = Vec::new();

    match nav {
        ProviderNav::Root => {
            lines.push(Line::from(Span::styled(
                "Escolha o tipo. Subagentes sempre usam runtime local.",
                Style::default().fg(COLOR_TEXT_MUTED),
            )));
            lines.push(Line::from(""));
            let items = [
                "Local  — modelos GGUF no disco",
                "Cloud  — OpenAI / Anthropic / Google",
            ];
            for (i, label) in items.iter().enumerate() {
                let on = i == cursor;
                let style = if on {
                    Style::default().fg(Color::Black).bg(COLOR_ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(COLOR_TEXT_MAIN)
                };
                lines.push(Line::from(Span::styled(
                    format!("  {} {}", if on { "»" } else { " " }, label),
                    style,
                )));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("j/k ou setas: navegar   ", Style::default().fg(COLOR_TEXT_MUTED)),
                Span::styled("Enter: abrir", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("   Esc/q: fechar", Style::default().fg(COLOR_TEXT_MUTED)),
            ]));
        }
        ProviderNav::LocalList | ProviderNav::CloudList => {
            let cloud = matches!(nav, ProviderNav::CloudList);
            let filtered = filtered_providers(providers, nav);
            lines.push(Line::from(Span::styled(
                if cloud {
                    "Providers remotos. Ctrl+A adiciona um novo."
                } else {
                    "Modelos locais (GGUF). Ctrl+A adiciona/configura."
                },
                Style::default().fg(COLOR_TEXT_MUTED),
            )));
            lines.push(Line::from(""));
            if filtered.is_empty() {
                lines.push(Line::from(Span::styled(
                    "  (nenhum provider neste grupo)",
                    Style::default().fg(COLOR_TEXT_MUTED),
                )));
            }
            for (i, (_idx, entry)) in filtered.iter().enumerate() {
                let on = i == cursor;
                let active = selected_uuid == Some(entry.id.as_str());
                let style = if on {
                    Style::default().fg(Color::Black).bg(COLOR_ACCENT).add_modifier(Modifier::BOLD)
                } else if active {
                    Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(COLOR_TEXT_MAIN)
                };
                lines.push(Line::from(Span::styled(
                    format!(
                        "  {} {}{}",
                        if on { "»" } else { " " },
                        entry.name,
                        if active { "  (ativo)" } else { "" }
                    ),
                    style,
                )));
                lines.push(Line::from(Span::styled(
                    format!("      {}", entry.detail),
                    Style::default().fg(COLOR_TEXT_MUTED),
                )));
            }
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("  Ctrl+A: ", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled(
                    if cloud { "adicionar provider cloud" } else { "adicionar / configurar modelo local" },
                    Style::default().fg(COLOR_TEXT_MAIN),
                ),
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::styled("j/k ou setas: navegar   ", Style::default().fg(COLOR_TEXT_MUTED)),
                Span::styled("Enter: usar", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
                Span::styled("   Esc: voltar   q: fechar", Style::default().fg(COLOR_TEXT_MUTED)),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(lines)
            .wrap(ratatui::widgets::Wrap { trim: true })
            .style(Style::default().bg(COLOR_CARD_BG)),
        inner,
    );
}

#[derive(Debug, Clone)]
pub enum ProviderKeyResult {
    None,
    Close,
    Back,
    EnterGroup(ProviderNav),
    Choose(String),
    Add,
}

pub fn handle_key(
    code: KeyCode,
    modifiers: KeyModifiers,
    nav: ProviderNav,
    cursor: &mut usize,
    providers: &[ProviderEntry],
) -> ProviderKeyResult {
    if modifiers.contains(KeyModifiers::CONTROL)
        && matches!(code, KeyCode::Char('a') | KeyCode::Char('A'))
    {
        return if nav.is_list() {
            ProviderKeyResult::Add
        } else {
            ProviderKeyResult::None
        };
    }

    match code {
        KeyCode::Up | KeyCode::Char('k') | KeyCode::Char('K') => {
            *cursor = cursor.saturating_sub(1);
            ProviderKeyResult::None
        }
        KeyCode::Down | KeyCode::Char('j') | KeyCode::Char('J') => {
            let max = match nav {
                ProviderNav::Root => 1,
                other => filtered_providers(providers, other).len().saturating_sub(1),
            };
            *cursor = (*cursor + 1).min(max);
            ProviderKeyResult::None
        }
        KeyCode::Enter => match nav {
            ProviderNav::Root => {
                let next = if *cursor == 0 {
                    ProviderNav::LocalList
                } else {
                    ProviderNav::CloudList
                };
                ProviderKeyResult::EnterGroup(next)
            }
            list_nav => {
                let filtered = filtered_providers(providers, list_nav);
                if let Some((_, entry)) = filtered.get(*cursor) {
                    ProviderKeyResult::Choose(entry.id.clone())
                } else {
                    ProviderKeyResult::None
                }
            }
        },
        KeyCode::Esc => {
            if nav.is_list() {
                ProviderKeyResult::Back
            } else {
                ProviderKeyResult::Close
            }
        }
        KeyCode::Char('q') | KeyCode::Char('Q') => ProviderKeyResult::Close,
        _ => ProviderKeyResult::None,
    }
}