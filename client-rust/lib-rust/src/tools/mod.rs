use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde_json::Value;
use similar::{ChangeTag, TextDiff};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

pub const BASE_TOOLS: &[&str] = &["read_file", "write_file", "list_dir", "patch_file", "run_command"];
pub const MAESTRO_TOOLS: &[&str] = &["list_agents", "read_agent", "create_agent", "ask_agent"];

pub fn is_maestro_tool(name: &str) -> bool {
    MAESTRO_TOOLS.contains(&name)
}

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
      "name": "fetch_url",
      "description": "Fetch the raw text content of a URL via HTTP GET. Requires explicit permission from the user before it runs (a confirmation prompt is shown). Only http/https URLs are allowed and only text-like responses (text/*, json, xml, javascript) are returned, truncated if large. IMPORTANT for GitHub files: always use the raw content URL in the exact format https://raw.githubusercontent.com/<owner>/<repo>/<branch>/<path-to-file> — NEVER the rendered 'colorful' page at github.com/<owner>/<repo>/blob/<branch>/<path-to-file> (a github.com/blob URL is auto-converted, but pass raw directly when you can). BRANCH NAME WARNING: GitHub repos use either 'main' or 'master' as the default branch and there is NO way to know which one without checking — do not assume 'main' just because it is more common today, many repos still use 'master'. You do not need to get this right on the first try: if the branch guessed in the URL is wrong, this tool automatically detects the 404 and retries with the repository's real default branch, so just pick one ('main' is a reasonable first guess) and let the tool self-correct.",
      "parameters": {
        "type": "object",
        "properties": {
          "url": { "type": "string", "description": "http(s) URL to fetch; prefer raw/plain-text endpoints (e.g. raw.githubusercontent.com for GitHub files)" },
          "timeout_ms": { "type": "integer", "description": "timeout in milliseconds (default 15000)" }
        },
        "required": ["url"]
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
      "description": "Create a new subagent and save it as ~/.drill/agents/drill.<name>.agent.json. Only 'name' is required; persona is optional and auto-generated from name+description when omitted.",
      "parameters": {
        "type": "object",
        "properties": {
          "name": { "type": "string", "description": "unique agent name, alphanumeric plus _ - ." },
          "description": { "type": "string", "description": "what this agent is specialized in" },
          "persona": { "type": "string", "description": "system prompt/persona of the subagent (optional)" },
          "temp": { "type": "number", "description": "sampling temperature (default 0.6)" },
          "max_tool_rounds": { "type": "integer", "description": "max tool rounds (default 4)" },
          "allowed_tools": { "type": "array", "items": { "type": "string" }, "description": "tools this agent may use" }
        },
        "required": ["name"]
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

pub const MAX_RESULT_CHARS: usize = 24_000;

/// Tools que exigem confirmacao explicita do usuario antes de rodar.
pub const TOOLS_REQUIRING_PERMISSION: &[&str] = &["fetch_url"];

/// Pedido de permissao: a thread de geracao envia um `PermissionRequest` e
/// bloqueia em `reply_rx.recv()` ate o frontend (TUI ou headless) responder.
#[derive(Debug)]
pub struct PermissionRequest {
    pub tool: String,
    pub detail: String,
    pub reply: Sender<bool>,
}

static PERMISSION_TX: OnceLock<Sender<PermissionRequest>> = OnceLock::new();

/// Chamado pelo frontend no startup: cria o canal e devolve o Receiver
/// para quem for exibir o prompt (TUI ou o responder headless).
pub fn install_permission_channel() -> Receiver<PermissionRequest> {
    let (tx, rx) = channel();
    let _ = PERMISSION_TX.set(tx);
    rx
}

/// Pede permissao de forma sincrona. Bloqueia a thread atual ate a resposta.
/// Sem canal instalado, nega por padrao.
pub fn ask_permission(tool: &str, detail: &str) -> bool {
    let Some(tx) = PERMISSION_TX.get() else {
        return false;
    };
    let (reply_tx, reply_rx) = channel();
    let req = PermissionRequest {
        tool: tool.to_string(),
        detail: detail.to_string(),
        reply: reply_tx,
    };
    if tx.send(req).is_err() {
        return false;
    }
    reply_rx.recv().unwrap_or(false)
}

#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub call_id: String,
    pub args: Value,
}

pub fn execute(call: &ToolCall, project_dir: &str) -> String {
    if TOOLS_REQUIRING_PERMISSION.contains(&call.name.as_str()) {
        let detail = arg_str(call, "url").unwrap_or_else(|| call.args.to_string());
        if !ask_permission(&call.name, &detail) {
            return format!("ERROR: acesso negado pelo usuario para '{}' ({})", call.name, detail);
        }
    }

    let result = match call.name.as_str() {
        "read_file" => do_read_file(call, project_dir),
        "write_file" => do_write_file(call, project_dir),
        "list_dir" => do_list_dir(call, project_dir),
        "patch_file" => do_patch_file(call, project_dir),
        "run_command" => do_run_command(call, project_dir),
        "fetch_url" => do_fetch_url(call),
        other => Err(format!("unknown tool: {}", other)),
    };

    match result {
        Ok(text) => truncate(text),
        Err(err) => format!("ERROR: {}", truncate(err)),
    }
}

pub fn truncate(text: String) -> String {
    if text.len() <= MAX_RESULT_CHARS {
        text
    } else {
        let cut = MAX_RESULT_CHARS;
        format!("{}...[TRUNCATED {} chars omitted]", &text[..cut], text.len() - cut)
    }
}

pub fn unwrap_value(args: &Value) -> Option<&Value> {
    if args.is_object() {
        let keys: Vec<&str> = args.as_object()?.keys().map(|k| k.as_str()).collect();
        let has_inner = keys.iter().any(|k| matches!(*k, "arguments" | "args" | "parameters"));
        let looks_wrapped = has_inner && keys.iter().any(|k| matches!(*k, "name" | "tool" | "tool_name"));
        if looks_wrapped {
            for k in &["arguments", "args", "parameters"] {
                if let Some(v) = args.get(*k) {
                    if v.is_object() {
                        return Some(v);
                    }
                }
            }
        }
    }
    Some(args)
}

pub fn arg_str(call: &ToolCall, key: &str) -> Option<String> {
    let root = unwrap_value(&call.args).unwrap_or(&call.args);
    root.get(key)
        .or_else(|| if key == "path" { root.get("file_path") } else { None })
        .and_then(Value::as_str)
        .map(str::to_string)
}

pub fn arg_i64(call: &ToolCall, key: &str) -> Option<i64> {
    let root = unwrap_value(&call.args).unwrap_or(&call.args);
    root.get(key).and_then(Value::as_i64)
}

fn resolve(path: &str, project_dir: &str) -> String {
    let p = Path::new(path);
    if p.is_absolute() {
        path.to_string()
    } else {
        Path::new(project_dir).join(p).to_string_lossy().to_string()
    }
}

fn do_patch_file(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let old_str = arg_str(call, "old_str").ok_or("missing 'old_str'")?;
    let new_str = arg_str(call, "new_str").ok_or("missing 'new_str'")?;

    let full_path = resolve(&path, project_dir);
    let raw = fs::read_to_string(&full_path).map_err(|err| err.to_string())?;

    if !raw.contains(&old_str) {
        return Err(format!("'old_str' não encontrado em {}", path));
    }

    let updated = raw.replace(&old_str, &new_str);
    fs::write(&full_path, &updated).map_err(|err| err.to_string())?;

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

fn do_read_file(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let raw = fs::read_to_string(resolve(&path, project_dir)).map_err(|err| err.to_string())?;
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

fn do_write_file(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let path = arg_str(call, "path").ok_or("missing 'path'")?;
    let content = arg_str(call, "content").ok_or("missing 'content'")?;
    let full = resolve(&path, project_dir);
    if let Some(parent) = Path::new(&full).parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(&full, &content).map_err(|err| err.to_string())?;
    Ok(format!("wrote {} bytes to {}", content.len(), full))
}

fn do_list_dir(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let path = arg_str(call, "path").unwrap_or_else(|| ".".to_string());
    let full = resolve(&path, project_dir);
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

fn normalize_github_url(url: &str) -> String {
    let rest = url
        .strip_prefix("https://github.com/")
        .or_else(|| url.strip_prefix("http://github.com/"));
    let Some(rest) = rest else { return url.to_string() };

    let parts: Vec<&str> = rest.splitn(5, '/').collect();
    if parts.len() == 5 && parts[2] == "blob" {
        format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}",
            parts[0], parts[1], parts[3], parts[4]
        )
    } else {
        url.to_string()
    }
}

fn parse_raw_github_url(url: &str) -> Option<(String, String, String, String)> {
    let rest = url
        .strip_prefix("https://raw.githubusercontent.com/")
        .or_else(|| url.strip_prefix("http://raw.githubusercontent.com/"))?;
    let parts: Vec<&str> = rest.splitn(4, '/').collect();
    if parts.len() == 4 {
        Some((parts[0].to_string(), parts[1].to_string(), parts[2].to_string(), parts[3].to_string()))
    } else {
        None
    }
}

fn fetch_github_default_branch(client: &reqwest::blocking::Client, owner: &str, repo: &str) -> Option<String> {
    let api_url = format!("https://api.github.com/repos/{}/{}", owner, repo);
    let resp = client
        .get(&api_url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let text = resp.text().ok()?;
    let json: Value = serde_json::from_str(&text).ok()?;
    json.get("default_branch")?.as_str().map(|s| s.to_string())
}

fn do_fetch_url(call: &ToolCall) -> Result<String, String> {
    let raw_url = arg_str(call, "url").ok_or("missing 'url'")?;
    let timeout_ms = arg_i64(call, "timeout_ms").unwrap_or(15_000).clamp(1_000, 60_000) as u64;

    if !(raw_url.starts_with("http://") || raw_url.starts_with("https://")) {
        return Err("apenas URLs http/https sao permitidas".to_string());
    }

    let mut url = normalize_github_url(&raw_url);
    let mut note = if url != raw_url {
        format!("(github.com/.../blob convertido para raw: {})\n", url)
    } else {
        String::new()
    };

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_millis(timeout_ms))
        .user_agent("drill-agent/1.0")
        .build()
        .map_err(|err| err.to_string())?;

    let mut resp = client.get(&url).send().map_err(|err| err.to_string())?;

    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        if let Some((owner, repo, guessed_branch, path)) = parse_raw_github_url(&url) {
            let alt_branch = fetch_github_default_branch(&client, &owner, &repo)
                .filter(|b| *b != guessed_branch)
                .or_else(|| match guessed_branch.as_str() {
                    "main" => Some("master".to_string()),
                    "master" => Some("main".to_string()),
                    _ => None,
                });

            if let Some(alt_branch) = alt_branch {
                let retry_url = format!(
                    "https://raw.githubusercontent.com/{}/{}/{}/{}",
                    owner, repo, alt_branch, path
                );
                if let Ok(retry_resp) = client.get(&retry_url).send() {
                    if retry_resp.status() != reqwest::StatusCode::NOT_FOUND {
                        note.push_str(&format!(
                            "(branch '{}' nao existe nesse repo; tentei '{}' e funcionou: {})\n",
                            guessed_branch, alt_branch, retry_url
                        ));
                        url = retry_url;
                        resp = retry_resp;
                    }
                }
            }
        }
    }

    let status = resp.status();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let is_texty = content_type.is_empty()
        || content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript");

    if !is_texty {
        return Err(format!(
            "content-type '{}' nao e texto; fetch_url so devolve conteudo textual",
            content_type
        ));
    }

    let body = resp.text().map_err(|err| err.to_string())?;
    Ok(format!("{}--- GET {} [{}] ---\n{}", note, url, status.as_u16(), body))
}

fn do_run_command(call: &ToolCall, project_dir: &str) -> Result<String, String> {
    let command = arg_str(call, "command").ok_or("missing 'command'")?;
    let timeout_ms = arg_i64(call, "timeout_ms").unwrap_or(60_000).clamp(1_000, 600_000) as u64;

    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: 24,
            cols: 120,
            pixel_width: 0,
            pixel_height: 0,
        })
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
