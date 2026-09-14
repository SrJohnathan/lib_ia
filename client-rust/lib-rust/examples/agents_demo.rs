use lib_rust::agents::{AgentManager, AgentRole, ConfigAgent};
use lib_rust::RuntimeLlama;
use std::sync::{Arc, Mutex};

fn build_engine() -> RuntimeLlama {
    let mut engine = RuntimeLlama::try_new(
        "/run/media/johnathan/Novo volume1/Qwen3.5-9B-Q4_K_M.gguf".to_string(),
        8192,
        1024,
        512,
        512,
        0.6,
        0.95,
        40,
        None,
        None,
        Some(999),
        None,
    )
    .expect("runtime");

    engine.set_int_prop("threads", 12).unwrap();
    engine.set_int_prop("threads-batch", 12).unwrap();
    engine.set_string_prop("cache-type-k", "q4_0").unwrap();
    engine.set_string_prop("cache-type-v", "q4_0").unwrap();
    engine
}

fn main() {
    let mut manager = AgentManager::new();

    manager.add(ConfigAgent {
        agent_id: "codigo".to_string(),
        role: AgentRole::Agent,
        prompt: "Voce e um assistente de codigo conciso. Responda em portugues, no maximo 3 linhas.".to_string(),
        tools_json: None,
    });
    manager.add(ConfigAgent {
        agent_id: "poeta".to_string(),
        role: AgentRole::Agent,
        prompt: "Voce e um poeta. Responda apenas com um pequeno poema de 4 versos, em portugues.".to_string(),
        tools_json: None,
    });

    manager.run(Arc::new(Mutex::new(build_engine())));

    let codigo = manager.get("codigo").expect("agente codigo");
    let poeta = manager.get("poeta").expect("agente poeta");

    turn("codigo", &codigo, "Memorize que meu projeto se chama Cronos.");
    turn("poeta", &poeta, "Faca um verso curto sobre a neve.");

    println!("\n--- segundo turno (testa memoria por agente) ---");
    turn("codigo", &codigo, "Qual o nome do meu projeto?");
    turn("poeta", &poeta, "Agora sobre a lua.");

    for id in manager.agent_ids() {
        let agent = manager.get(&id).unwrap();
        println!("\n[{}] historico: {} mensagens, {} tokens (chars/4 estimados)",
            id, agent.history().len(),
            agent.history().iter().map(|m| m.content.len()).sum::<usize>() / 4);
    }
}

fn turn(id: &str, agent: &lib_rust::agents::Agent, prompt: &str) {
    println!("\n>>> [{}] {} \n  {}", id, prompt, "-".repeat(40));
    match agent.execute(prompt) {
        Ok(text) => println!("`{}`", text),
        Err(err) => println!("ERRO: {}", err),
    }
}