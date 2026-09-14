use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

pub const TOOLS_JSON: &str = r#"[
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read a text file from disk. Use offset/limit to read a slice of long files.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "absolute or relative file path" },
          "offset": { "type": "integer", "description": "line number to start from (0-based)" },
          "limit": { "type": "integer", "description": "max number of lines to read" }
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Create or overwrite a text file with the given content.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "absolute or relative file path" },
          "content": { "type": "string", "description": "full new file content" }
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_dir",
      "description": "List directory entries with sizes and kind.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": { "type": "string", "description": "directory path (default '.')" }
        }
      }
    }
  },
  {
  "type": "function",
  "function": {
    "name": "patch_file",
    "description": "Replace a target block of text in a file with a new block of text.",
    "parameters": {
      "type": "object",
      "properties": {
        "path": { "type": "string", "description": "relative or absolute path to the file" },
        "old_str": { "type": "string", "description": "exact block of text to be replaced" },
        "new_str": { "type": "string", "description": "new block of text to insert in place of old_str" }
      },
      "required": ["path", "old_str", "new_str"]
    }
  }
},
  {
    "type": "function",
    "function": {
      "name": "run_command",
      "description": "Run a shell command (bash -lc) in the project directory. Returns stdout+stderr. Capable of running build tools, git, ls, grep, etc.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": { "type": "string", "description": "the shell command to execute" },
          "timeout_ms": { "type": "integer", "description": "timeout in milliseconds (default 60000)" }
        },
        "required": ["command"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_agents",
      "description": "List the available subagents (files drill.<nome>.agent.json found in ~/.drill/agents and the project dir). Returns name, description, tools and temperature of each.",
      "parameters": {
        "type": "object",
        "properties": {}
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "read_agent",
      "description": "Read the full definition (JSON file) of one subagent by name.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": { "type": "string", "description": "subagent name (e.g. lucy)" }
        },
        "required": ["name"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "create_agent",
      "description": "Create a new subagent and save it as ~/.drill/agents/drill.<name>.agent.json.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": { "type": "string", "description": "unique agent name, alphanumeric plus _ - ." },
          "description": { "type": "string", "description": "what this agent is specialized in" },
          "persona": { "type": "string", "description": "system prompt/persona of the subagent" },
          "temp": { "type": "number", "description": "sampling temperature (default 0.6)" },
          "max_tool_rounds": { "type": "integer", "description": "max tool rounds (default 4)" },
          "allowed_tools": { "type": "array", "items": { "type": "string" }, "description": "tools this agent may use" }
        },
        "required": ["name", "persona"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "ask_agent",
      "description": "Delegate a task to a subagent by name. Runs the subagent on the same model with its own persona and conversation history, then returns its reply.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": { "type": "string", "description": "subagent name" },
          "prompt": { "type": "string", "description": "task/prompt to send to the subagent" }
        },
        "required": ["name", "prompt"]
      }
    }
  }
]"#;

const MAX_RESULT_CHARS: usize = 24_000;

#[derive(Clone)]
pub struct ToolCall {
    pub name: String,
    pub call_id: String,
    pub args: Value,
}

pub fn execute(call: &ToolCall, project_dir: &str) -> String {
    let result = match call.name.as_str() {
        "read_file" => do_read_file(call),
        "write_file" => do_write_file(call),
        "list_dir" => do_list_dir(call),
        "patch_file" => do_patch_file(call),
        "run_command" => do_run_command(call, project_dir),
        other => Err(format!("unknown tool: {}", other)),
    };

    match result {
        Ok(text) => truncate(text),
        Err(err) => format!("ERROR: {}", truncate(err)),
    }
}

fn truncate(text: String) -> String {
    if text.len() <= MAX_RESULT_CHARS {
        text
    } else {
        let cut = MAX_RESULT_CHARS;
        format!("{}...[TRUNCATED {} chars omitted]", &text[..cut], text.len() - cut)
    }
}

fn arg_str(call: &ToolCall, key: &str) -> Option<String> {
    call.args.get(key).and_then(Value::as_str).map(str::to_string)
}

fn arg_i64(call: &ToolCall, key: &str) -> Option<i64> {
    call.args.get(key).and_then(Value::as_i64)
}

fn resolve(path: &str, project_dir: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        Path::new(project_dir).join(p).to_string_lossy().to_string()
    }
}
use similar::{ChangeTag, TextDiff};
fn do_patch_file(call: &ToolCall) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let old_str = arg_str(call, "old_str").ok_or("missing 'old_str'")?;
    let new_str = arg_str(call, "new_str").ok_or("missing 'new_str'")?;

    let full_path = resolve(&path, ".");
    let raw = fs::read_to_string(&full_path).map_err(|err| err.to_string())?;

    if !raw.contains(&old_str) {
        return Err(format!("'old_str' não encontrado em {}", path));
    }

    let updated = raw.replace(&old_str, &new_str);
    fs::write(&full_path, &updated).map_err(|err| err.to_string())?;

    // Gera a saída estruturada em formato Diff com contexto
    let diff = TextDiff::from_lines(&old_str, &new_str);
    let mut diff_output = format!("--- Edit {} ---\n", path);

    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            ChangeTag::Delete => "-",
            ChangeTag::Insert => "+",
            ChangeTag::Equal => " ",
        };
        diff_output.push_str(&format!("{}{}", sign, change));
    }

    Ok(diff_output)
}

fn do_read_file(call: &ToolCall) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let raw = fs::read_to_string(resolve(&path, ".")).map_err(|err| err.to_string())?;
    let offset = arg_i64(call, "offset").unwrap_or(0).max(0) as usize;
    let limit = arg_i64(call, "limit").map(|n| n.max(0) as usize);

    let lines: Vec<&str> = raw.lines().skip(offset).collect();
    let take = limit.map(|n| lines.iter().take(n)).unwrap_or_else(|| lines.iter().take(usize::MAX));
    let slice: Vec<&str> = take.cloned().collect();

    if slice.len() == lines.len() {
        Ok(format!("--- {} ({} lines) ---\n{}", path, lines.len(), slice.join("\n")))
    } else {
        Ok(format!(
            "--- {} lines {}-{} of {} ---\n{}",
            path,
            offset,
            offset + slice.len(),
            lines.len(),
            slice.join("\n")
        ))
    }
}

fn do_write_file(call: &ToolCall) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let content = arg_str(call, "content").ok_or("missing 'content'")?;
    let full = resolve(&path, ".");
    if let Some(parent) = Path::new(&full).parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(&full, &content).map_err(|err| err.to_string())?;
    Ok(format!("wrote {} bytes to {}", content.len(), full))
}

fn do_list_dir(call: &ToolCall) -> Result<String, String> {
    let path = arg_str(call, "path").unwrap_or_else(|| ".".to_string());
    let full = resolve(&path, ".");
    let mut out = String::new();
    for entry in fs::read_dir(&full).map_err(|err| err.to_string())? {
        let entry = entry.map_err(|err| err.to_string())?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
        out.push_str(&format!("{:<6} {} {}\n", if is_dir { "DIR" } else { "FILE" }, size, name));
    }
    Ok(out)
}

fn do_run_command(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let command = arg_str(call, "command").ok_or("missing 'command'")?;
    let timeout_ms = arg_i64(call, "timeout_ms").unwrap_or(60_000).clamp(1_000, 600_000) as u64;

    let pty_system = native_pty_system();
    let pair = pty_system.openpty(PtySize { rows: 24, cols: 120, pixel_width: 0, pixel_height: 0 })
        .map_err(|err| err.to_string())?;

    let mut cmd = CommandBuilder::new("bash");
    cmd.args(["-lc", &command]);
    cmd.cwd(project_dir);
    let mut child = pair.slave.spawn_command(cmd).map_err(|err| err.to_string())?;
    drop(pair.slave);

    let mut reader = pair.master.try_clone_reader().map_err(|err| err.to_string())?;

    let reader_thread = thread::spawn(move || {
        let mut buf = String::new();
        let _ = reader.read_to_string(&mut buf);
        buf
    });

    let start = Instant::now();
    let status = loop {
        if let Some(exit) = child.try_wait().map_err(|err| err.to_string())? {
            break format!("exit code: {}", exit.exit_code());
        }
        if start.elapsed() > Duration::from_millis(timeout_ms) {
            let _ = child.kill();
            break format!("TIMEOUT after {}ms (process killed)", timeout_ms);
        }
        thread::sleep(Duration::from_millis(20));
    };

    let _ = child.wait().ok();
    drop(pair.master);
    let output = reader_thread.join().unwrap_or_default();

    let trimmed = output.trim_end();
    if trimmed.is_empty() {
        Ok(status)
    } else {
        let suffix = if status.starts_with("exit code: 0") {
            String::new()
        } else {
            format!("\n--- {} ---\n", status)
        };
        Ok(format!("{}{}", trimmed, suffix))
    }
}