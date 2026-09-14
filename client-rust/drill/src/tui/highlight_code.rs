use syntect::easy::HighlightLines;
use syntect::parsing::SyntaxSet;
use syntect::highlighting::{ThemeSet, Style as SyntectStyle};
use syntect::util::LinesWithEndings;
use ratatui::text::{Line, Span};
use ratatui::style::{Color, Style};
use crate::tui::COLOR_CODE_BG;

pub struct CodeHighlighter {
    ps: SyntaxSet,
    ts: ThemeSet,
}

impl CodeHighlighter {
    pub fn new() -> Self {
        Self {
            ps: SyntaxSet::load_defaults_newlines(),
            ts: ThemeSet::load_defaults(),
        }
    }
    pub fn highlight_line(&self, line: &str, lang: &str, target_width: usize) -> Line<'static> {
        let syntax = self
            .ps
            .find_syntax_by_extension(lang)
            .or_else(|| self.ps.find_syntax_by_token(lang))
            .unwrap_or_else(|| self.ps.find_syntax_plain_text());

        let theme = &self.ts.themes["base16-ocean.dark"];
        let mut highlighter = HighlightLines::new(syntax, theme);

        let ranges = highlighter
            .highlight_line(line, &self.ps)
            .unwrap_or_default();

        let mut spans = Vec::new();
        let mut current_len = 0;

        for (style, text) in ranges {
            let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
            current_len += text.chars().count();
            spans.push(Span::styled(
                text.to_string(),
                Style::default().fg(fg).bg(COLOR_CODE_BG),
            ));
        }

        // Completa o restante da linha com espaços coloridos até a borda do painel
        if current_len < target_width {
            let padding = " ".repeat(target_width - current_len);
            spans.push(Span::styled(padding, Style::default().bg(COLOR_CODE_BG)));
        }

        Line::from(spans)
    }
    pub fn highlight_code(&self, code: &str, lang: &str) -> Vec<Line<'static>> {
        // Encontra a sintaxe correspondente (ex: "rust", "js", "json") ou usa Plain Text como fallback
        let syntax = self.ps.find_syntax_by_token(lang)
            .unwrap_or_else(|| self.ps.find_syntax_plain_text());

        // Usa o tema "base16-ocean.dark" ou "Solarized (dark)" para alinhar com o fundo escuro
        let theme = &self.ts.themes["base16-ocean.dark"];
        let mut highlighter = HighlightLines::new(syntax, theme);

        let mut lines = Vec::new();

        for line in LinesWithEndings::from(code) {
            let ranges: Vec<(SyntectStyle, &str)> = highlighter.highlight_line(line, &self.ps).unwrap_or_default();
            let mut spans = Vec::new();

            for (style, text) in ranges {
                let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
                // Define o fundo customizado dos cards escuros (COLOR_CODE_BG)
                let bg = Color::Rgb(16, 16, 20);

                spans.push(Span::styled(
                    text.to_string(),
                    Style::default().fg(fg).bg(bg),
                ));
            }
            lines.push(Line::from(spans));
        }

        lines
    }
}