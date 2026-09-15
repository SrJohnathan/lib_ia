//! Re-exportações das ferramentas e permissões vindas da `lib_rust`.
//! O `drill` atua apenas como frontend consumindo esta interface.

#[allow(unused_imports)]
pub use lib_rust::tools::{
    ask_permission, execute, install_permission_channel, is_maestro_tool,
    PermissionRequest, ToolCall, BASE_TOOLS, MAESTRO_TOOLS, MAX_RESULT_CHARS,
    TOOLS_JSON, TOOLS_REQUIRING_PERMISSION,
};