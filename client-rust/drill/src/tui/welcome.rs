use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
};
use ratatui::text::{Line, Span};
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use crate::tui::{draw_input_popup, AppState, COLOR_ACCENT, COLOR_APP_BG, COLOR_BORDER, COLOR_INPUT_BG, COLOR_TEXT_MAIN, COLOR_TEXT_MUTED};

pub fn draw_welcome(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    frame.render_widget(Block::default().style(Style::default().bg(COLOR_APP_BG)), area);

    // --- logo ---
    let logo_lines = vec![
        Line::from(Span::styled(
            "d r i l l",
            Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "agente de código no terminal",
            Style::default().fg(COLOR_TEXT_MUTED),
        )),
    ];
    let logo_h = logo_lines.len() as u16;
    let logo_y = area.height.saturating_sub(12) / 2;
    frame.render_widget(
        Paragraph::new(logo_lines)
            .alignment(ratatui::layout::Alignment::Center)
            .style(Style::default().bg(COLOR_APP_BG)),
        Rect::new(area.x, area.y + logo_y, area.width, logo_h),
    );

    // --- janelinha de input (centro) ---
    let box_w = area.width.saturating_sub(16).min(72);
    let box_x = (area.width.saturating_sub(box_w)) / 2;
    let box_y = area.y + logo_y + logo_h + 2;
    let input_area = Rect::new(box_x, box_y, box_w, 4);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(COLOR_ACCENT))
        .style(Style::default().bg(COLOR_INPUT_BG));

    let model_short = state
        .session
        .model
        .split('/')
        .last()
        .unwrap_or("—");

    let placeholder = if state.input.is_empty() {
        " Digite sua mensagem..."
    } else {
        // mostra o que já digitou
        ""
    };

    let lines = vec![
        if state.input.is_empty() {
            Line::from(Span::styled(placeholder, Style::default().fg(COLOR_TEXT_MUTED)))
        } else {
            Line::from(Span::styled(
                format!(" {}", state.input),
                Style::default().fg(COLOR_TEXT_MAIN),
            ))
        },
        Line::from(vec![
            Span::styled(" ", Style::default()),
            Span::styled(state.mode.label(), Style::default().fg(COLOR_ACCENT)),
            Span::styled(" · ", Style::default().fg(COLOR_BORDER)),
            Span::styled(model_short, Style::default().fg(COLOR_TEXT_MUTED)),
            Span::styled("    ", Style::default()),
            Span::styled("/", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD)),
            Span::styled(" comandos  ", Style::default().fg(COLOR_TEXT_MUTED)),
            Span::styled("@", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD)),
            Span::styled(" arquivos  ", Style::default().fg(COLOR_TEXT_MUTED)),
            Span::styled("Ctrl+A", Style::default().fg(COLOR_TEXT_MAIN).add_modifier(Modifier::BOLD)),
            Span::styled(" maestro", Style::default().fg(COLOR_TEXT_MUTED)),
        ]),
    ];

    frame.render_widget(Paragraph::new(lines).block(block), input_area);

    // cursor dentro da janelinha
    let cursor_x = input_area.x + 1 + state.input_pos as u16;
    let cursor_y = input_area.y + 1;
    frame.set_cursor_position((cursor_x, cursor_y));

    // --- tip embaixo ---
    let tip_y = box_y + 6;
    let tip = Line::from(vec![
        Span::styled("tip  ", Style::default().fg(Color::Yellow)),
        Span::styled("/provider", Style::default().fg(COLOR_ACCENT).add_modifier(Modifier::BOLD)),
        Span::styled("  escolhe local ou API", Style::default().fg(COLOR_TEXT_MUTED)),
    ]);
    frame.render_widget(
        Paragraph::new(tip).alignment(ratatui::layout::Alignment::Center),
        Rect::new(area.x, tip_y, area.width, 1),
    );

    // popup de autocomplete (/ e @) também na welcome
    if let Some(popup) = &state.popup {
   
        let popup_height = (popup.items.len().min(8) as u16) + 2;
        let popup_y = input_area.y.saturating_sub(popup_height);
        let popup_area = Rect::new(input_area.x, popup_y, input_area.width, popup_height);

        // reutiliza a mesma função, passando áreas compatíveis
        draw_input_popup(frame, popup, input_area, Rect::new(area.x, area.y, area.width, popup_y));
    }
}

