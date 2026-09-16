use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph};
use ratatui::Frame;

// reutiliza as cores e o centered_rect que já estão em tui.rs
use super::{centered_rect, COLOR_ACCENT, COLOR_CARD_BG, COLOR_TEXT_MAIN, COLOR_TEXT_MUTED};

pub enum ConfigModal {
    None,
    OpenAiBaseUrl { input: String, cursor: usize },
    OpenAiApiKey  { input: String, cursor: usize },
    // OpenAiModel { ... }  // se quiser depois
}



pub fn draw_config_modal(frame: &mut Frame, title: &str, input: &str, masked: bool, cursor: usize) {
    let area = centered_rect(70, 22, frame.area());

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_CARD_BG))
        .title(Line::from(Span::styled(
            format!(" {} ", title),
            Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD),
        )));

    let inner = block.inner(area);
    frame.render_widget(Clear, area);
    frame.render_widget(block, area);

    // texto exibido: mascarado ou normal
    let display = if masked {
        "•".repeat(input.chars().count())
    } else {
        input.to_string()
    };

    let lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("  {}", display),
            Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(vec![
            Span::styled("  Enter", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(" confirma   ", Style::default().fg(COLOR_TEXT_MUTED)),
            Span::styled("Esc", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
            Span::styled(" cancela", Style::default().fg(COLOR_TEXT_MUTED)),
        ]),
    ];

    frame.render_widget(
        Paragraph::new(lines).style(Style::default().bg(COLOR_CARD_BG)),
        inner,
    );


    let cursor_x = inner.x + 2 + cursor as u16;
    let cursor_y = inner.y + 1; // linha do input
    frame.set_cursor_position((cursor_x, cursor_y));


}


