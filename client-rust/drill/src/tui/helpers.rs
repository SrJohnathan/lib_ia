//! Utilitários compartilhados da TUI (texto, layout, cursor).

use ratatui::Frame;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::prelude::{Color, Line, Modifier, Span, Style};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use crate::tui::{provider_ui, AppState, COLOR_ACCENT, COLOR_CARD_BG, COLOR_TEXT_MUTED, COLOR_TEXT_MAIN, REASONING_EFFORTS};

pub fn wrap_text(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut lines = Vec::new();
    for paragraph in text.split('\n') {
        if paragraph.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        for word in paragraph.split_whitespace() {
            if current.is_empty() {
                current = word.to_string();
            } else if current.len() + 1 + word.len() <= width {
                current.push(' ');
                current.push_str(word);
            } else {
                lines.push(std::mem::take(&mut current));
                current = word.to_string();
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub fn char_to_byte(text: &str, char_pos: usize) -> usize {
    text.char_indices()
        .nth(char_pos)
        .map(|(i, _)| i)
        .unwrap_or(text.len())
}

pub fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
        .split(area);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
        .split(popup_layout[1])[1]
}

pub  fn draw_provider_overlay(frame: &mut Frame, state: &mut AppState) {
    provider_ui::draw(
        frame,
        state.provider_nav,
        &state.providers,
        state.provider_selected.as_deref(),
        state.provider_cursor,
    );
}

pub  fn draw_models_overlay(frame: &mut Frame, state: &mut AppState) {
    let area = centered_rect(72, 62, frame.area());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            " Modelos da API ( /models ) ",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        )));
    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    if state.models.is_empty() {
        let msg = Paragraph::new(Line::from(Span::styled(
            "(carregando modelos...)",
            Style::default().fg(COLOR_TEXT_MUTED),
        )))
            .style(Style::default().bg(COLOR_CARD_BG));
        frame.render_widget(msg, Rect::new(inner.x, inner.y + 1, inner.width, 3));
        return;
    }

    // Area do footer (raciocinio + hints) = 4 linhas.
    let list_height = inner.height.saturating_sub(4) as usize;
    if state.models_cursor < state.models_scroll {
        state.models_scroll = state.models_cursor;
    } else if list_height > 0 && state.models_cursor >= state.models_scroll + list_height {
        state.models_scroll = state.models_cursor + 1 - list_height;
    }

    let mut lines: Vec<Line<'static>> = Vec::new();
    let start = state.models_scroll;
    let end = (state.models_scroll + list_height).min(state.models.len());
    for i in start..end {
        let selected = i == state.models_cursor;
        let is_current = state.models[i] == state.models_current;
        let style = if selected {
            Style::default()
                .fg(Color::Black)
                .bg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else if is_current {
            Style::default().fg(COLOR_ACCENT)
        } else {
            Style::default().fg(COLOR_TEXT_MAIN)
        };
        lines.push(Line::from(Span::styled(
            format!(
                "  {} {} {}",
                if selected { "»" } else { " " },
                state.models[i],
                if is_current { "(atual)" } else { "" }
            ),
            style,
        )));
    }
    // Garante o espaco do footer dentro da janela de rolagem.
    for _ in end..(start + list_height).min(state.models.len()) {
        lines.push(Line::from(""));
    }
    lines.push(Line::from(""));

    let mut effort_line = Line::from(vec![Span::styled(
        "  raciocinio: ",
        Style::default().fg(COLOR_TEXT_MUTED),
    )]);
    for (i, (label, _)) in REASONING_EFFORTS.iter().enumerate() {
        let sel = i == state.effort_cursor;
        let style = if sel {
            Style::default()
                .fg(Color::Black)
                .bg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(COLOR_TEXT_MAIN)
        };
        effort_line.spans.push(Span::styled(format!(" {} ", label), style));
        effort_line.spans.push(Span::styled(" ", Style::default()));
    }
    lines.push(effort_line);
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled(
            "j/k: modelo   ",
            Style::default().fg(COLOR_TEXT_MUTED),
        ),
        Span::styled(
            "h/l: raciocinio   ",
            Style::default().fg(COLOR_TEXT_MUTED),
        ),
        Span::styled(
            "Enter: definir",
            Style::default()
                .fg(COLOR_ACCENT)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("   q/Esc: fechar", Style::default().fg(COLOR_TEXT_MUTED)),
    ]));

    let list = Paragraph::new(lines)
        .wrap(ratatui::widgets::Wrap { trim: true })
        .style(Style::default().bg(COLOR_CARD_BG));
    frame.render_widget(list, inner);
}