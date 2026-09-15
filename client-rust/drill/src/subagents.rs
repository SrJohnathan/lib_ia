//! Re-exportações do gerenciamento de subagentes da `lib_rust`.
//! O `drill` atua puramente como frontend e a persistência/gerenciamento vive na `lib-rust`.

#[allow(unused_imports)]
pub use lib_rust::agents::subagents::{
    agents_root, config_root, discover_agents, find_agent, read_agent, save_agent,
    tools_json_for, SubagentConfig, AGENT_FILE_PREFIX, AGENT_FILE_SUFFIX,
};