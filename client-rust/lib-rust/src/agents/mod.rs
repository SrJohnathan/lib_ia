pub mod agent;
pub mod agent_manager;
pub mod orchestrator;
pub mod subagents;

pub use agent::{AgentRole, AgentState, AgentTask, ConfigAgent, HistoryRow, TurnResult};
pub use agent_manager::{
    AgentManager, EventSink, HistoryListener, ManagerEvent, RemoteEvent, RemoteModelOutput,
    ToolExecutor,
};
pub use orchestrator::{AgentMode, AgentOrchestrator, MAESTRO_AGENT_ID, SUMMARY_PROMPT};
pub use subagents::{discover_agents, find_agent, read_agent, save_agent, SubagentConfig};