

pub enum ConfigModal {
    None,
    OpenAiBaseUrl { input: String, cursor: usize },
    OpenAiApiKey  { input: String, cursor: usize },
    // OpenAiModel { ... }  // se quiser depois
}



pub fn draw_config_modal(frame: &mut Frame, title: &str, input: &str, masked: bool) {
    let area = centered_rect(70, 22, frame.area());
    // Block + título
    // linha de input (se masked: "••••••")
    // dica: Enter confirma · Esc cancela
}


