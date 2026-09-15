use futures::StreamExt;
use lib_rust::agents::HistoryRow;
use lancedb::arrow::arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::arrow::SendableRecordBatchStream;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::runtime::Runtime;

/// Tabela LanceDB do historico. v2: adiciona `session_id` e `mode` (a v1
/// antiga sem esses campos nao e lida pela lista de sessoes).
const HISTORY_TABLE: &str = "historico_v2";

/// Resumo de uma conversa para a lista `/session`.
#[derive(Debug, Clone)]
pub struct StoredSession {
    pub id: String,
    pub mode: String,
    pub messages: usize,
    pub last_created: f64,
    pub preview: String,
}

/// Grava o historico de conversas em LanceDB (local), via thread dedicada:
/// o turno de geracao nunca bloqueia na escrita e uma falha aqui jamais
/// derruba a geracao.
pub struct History {
    tx: mpsc::Sender<HistoryRow>,
    pub path: PathBuf,
}

impl History {
    pub fn new(db_dir: PathBuf) -> Self {
        let path = db_dir.join("conversas.lance");
        let (tx, rx) = mpsc::channel::<HistoryRow>();
        let rt = Arc::new(Runtime::new().expect("tokio runtime para o history"));
        let writer_path = path.clone();

        thread::Builder::new()
            .name("history-writer".to_string())
            .spawn(move || {
                for row in rx {
                    if let Err(err) = rt.block_on(write_row(&writer_path, &row)) {
                        eprintln!("[history] erro ao gravar: {}", err);
                    }
                }
            })
            .ok();

        Self { tx, path }
    }

    pub fn record(&self, row: HistoryRow) {
        let _ = self.tx.send(row);
    }

    /// Lista todas as sessoes (uma por conversa), por ultima atividade.
    pub fn list_sessions(&self) -> Vec<StoredSession> {
        list_sessions(&self.path)
    }

    /// Todos os turnos de uma sessao, em ordem cronologica.
    pub fn get_session(&self, session_id: &str) -> Vec<HistoryRow> {
        get_session(&self.path, session_id)
    }

    /// Apaga todos os turnos de uma sessao.
    pub fn delete_session(&self, session_id: &str) -> Result<(), String> {
        let rt = Runtime::new().expect("tokio runtime");
        rt.block_on(delete_session(&self.path, session_id))
    }
}

async fn write_row(db_dir: &PathBuf, row: &HistoryRow) -> Result<(), String> {
    let batch = row_to_batch(row)?;
    let uri = db_dir.to_string_lossy().to_string();
    let db = lancedb::connect(&uri)
        .execute()
        .await
        .map_err(|err| err.to_string())?;

    match db.open_table(HISTORY_TABLE).execute().await {
        Ok(table) => {
            table
                .add(vec![batch])
                .execute()
                .await
                .map_err(|e| e.to_string())?;
            Ok(())
        }
        Err(_) => {
            // tabela ainda não existe: create já insere o batch
            db.create_table(HISTORY_TABLE, vec![batch])
                .execute()
                .await
                .map_err(|e| e.to_string())?;
            Ok(())
        }
    }
}

fn now_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn row_to_batch(row: &HistoryRow) -> Result<RecordBatch, String> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("created_at", DataType::Float64, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("mode", DataType::Utf8, false),
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("input", DataType::Utf8, false),
        Field::new("reply", DataType::Utf8, false),
        Field::new("tokens", DataType::Int32, false),
        Field::new("tps", DataType::Float64, false),
        Field::new("context_tokens", DataType::Int32, false),
    ]));

    let created_at: ArrayRef = Arc::new(Float64Array::from(vec![now_f64()]));
    let session_id: ArrayRef = Arc::new(StringArray::from(vec![row.session_id.as_str()]));
    let mode: ArrayRef = Arc::new(StringArray::from(vec![row.mode.as_str()]));
    let agent_id: ArrayRef = Arc::new(StringArray::from(vec![row.agent_id.as_str()]));
    let kind: ArrayRef = Arc::new(StringArray::from(vec![row.kind.as_str()]));
    let input: ArrayRef = Arc::new(StringArray::from(vec![row.input.as_str()]));
    let reply: ArrayRef = Arc::new(StringArray::from(vec![row.reply.as_str()]));
    let tokens: ArrayRef = Arc::new(Int32Array::from(vec![row.tokens.unwrap_or(0) as i32]));
    let tps: ArrayRef = Arc::new(Float64Array::from(vec![row.tps.unwrap_or(0.0)]));
    let context_tokens: ArrayRef =
        Arc::new(Int32Array::from(vec![row.context_tokens.unwrap_or(0) as i32]));

    RecordBatch::try_new(
        schema,
        vec![
            created_at, session_id, mode, agent_id, kind, input, reply, tokens, tps, context_tokens,
        ],
    )
    .map_err(|err| err.to_string())
}

/// Le as ultimas `n` linhas do historico (para verificacao/ferramentas).
pub fn last_rows(db_dir: &PathBuf, n: usize) -> Vec<HistoryRow> {
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let table = match open(&db_dir.join("conversas.lance")).await {
            Some(t) => t,
            None => return Vec::new(),
        };
        let mut stream = match table.query().limit(n).execute().await {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let mut out = read_sink(&mut stream).await
            .into_iter()
            .map(|(row, _)| row)
            .collect::<Vec<_>>();
        out.reverse();
        out
    })
}

fn list_sessions(db_path: &PathBuf) -> Vec<StoredSession> {
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let table = match open(db_path).await {
            Some(t) => t,
            None => return Vec::new(),
        };
        let mut stream = match table
            .query()
            .select(Select::columns(&["created_at", "session_id", "mode", "kind", "input"]))
            .execute()
            .await
        {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut sessions: std::collections::HashMap<String, StoredSession> = Default::default();
        while let Some(batch) = stream.next().await {
            let batch = match batch {
                Ok(batch) => batch,
                Err(_) => continue,
            };
            let created_at = batch.column_by_name("created_at").and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let session_id = batch.column_by_name("session_id").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let mode = batch.column_by_name("mode").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let kind = batch.column_by_name("kind").and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let input = batch.column_by_name("input").and_then(|c| c.as_any().downcast_ref::<StringArray>());

            if let (Some(created_at), Some(session_id), Some(mode), Some(kind), Some(input)) =
                (created_at, session_id, mode, kind, input)
            {
                for i in 0..session_id.len() {
                    let id = session_id.value(i).to_string();
                    if id.is_empty() {
                        continue;
                    }
                    let entry = sessions.entry(id.clone()).or_insert(StoredSession {
                        id: id.clone(),
                        mode: "solo".to_string(),
                        messages: 0,
                        last_created: 0.0,
                        preview: String::new(),
                    });
                    entry.messages += 1;
                    entry.mode = mode.value(i).to_string();
                    let ts = created_at.value(i);
                    if ts > entry.last_created {
                        entry.last_created = ts;
                        entry.preview = if kind.value(i) == "root" {
                            input.value(i).to_string()
                        } else {
                            entry.preview.clone()
                        };
                    }
                }
            }
        }

        let mut out: Vec<StoredSession> = sessions.into_values().collect();
        out.sort_by(|a, b| {
            b.last_created
                .partial_cmp(&a.last_created)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    })
}

fn get_session(db_path: &PathBuf, session_id: &str) -> Vec<HistoryRow> {
    let filter = format!("session_id = '{}'", escape_filter(session_id));
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let table = match open(db_path).await {
            Some(t) => t,
            None => return Vec::new(),
        };
        let mut stream = match table.query().only_if(&filter).execute().await {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let mut rows = read_sink(&mut stream).await;
        rows.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        rows.into_iter().map(|(row, _)| row).collect()
    })
}

async fn delete_session(db_path: &PathBuf, session_id: &str) -> Result<(), String> {
    let uri = db_path.to_string_lossy().to_string();
    let db = lancedb::connect(&uri)
        .execute()
        .await
        .map_err(|err| err.to_string())?;
    let table = db
        .open_table(HISTORY_TABLE)
        .execute()
        .await
        .map_err(|err| err.to_string())?;
    let filter = format!("session_id = '{}'", escape_filter(session_id));
    table
        .delete(&filter)
        .await
        .map(|_| ())
        .map_err(|err| err.to_string())
}

fn escape_filter(v: &str) -> String {
    v.replace('\'', "''")
}

/// Ordena sessoes por criacao (ver `get_session`).
async fn open(db_path: &PathBuf) -> Option<lancedb::Table> {
    let uri = db_path.to_string_lossy().to_string();
    let db = lancedb::connect(&uri).execute().await.ok()?;
    db.open_table(HISTORY_TABLE).execute().await.ok()
}

/// Le todas as linhas do stream, devolvendo (linha, created_at) para permitir
/// ordenacao cronologica fora (a consulta LanceDB nao garante ordem).
async fn read_sink(stream: &mut SendableRecordBatchStream) -> Vec<(HistoryRow, f64)> {
    let mut rows: Vec<(HistoryRow, f64)> = Vec::new();
    while let Some(batch) = stream.next().await {
        let batch = match batch {
            Ok(batch) => batch,
            Err(_) => continue,
        };
        let created_at = batch
            .column_by_name("created_at")
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
        let session_id = batch
            .column_by_name("session_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let mode = batch
            .column_by_name("mode")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let agent_id = batch
            .column_by_name("agent_id")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let kind = batch
            .column_by_name("kind")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let input = batch
            .column_by_name("input")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let reply = batch
            .column_by_name("reply")
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());

        if let (Some(created_at), Some(session_id), Some(mode), Some(agent_id), Some(kind), Some(input), Some(reply)) =
            (created_at, session_id, mode, agent_id, kind, input, reply)
        {
            for i in 0..agent_id.len() {
                rows.push((
                    HistoryRow {
                        agent_id: agent_id.value(i).to_string(),
                        kind: kind.value(i).to_string(),
                        input: input.value(i).to_string(),
                        reply: reply.value(i).to_string(),
                        tokens: None,
                        tps: None,
                        context_tokens: None,
                        session_id: session_id.value(i).to_string(),
                        mode: mode.value(i).to_string(),
                    },
                    created_at.value(i),
                ));
            }
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use lib_rust::agents::HistoryRow;
    use std::thread;

    fn row(id: &str, mode: &str, n: i64) -> HistoryRow {
        HistoryRow {
            agent_id: "drill".to_string(),
            kind: "root".to_string(),
            input: format!("msg {}", n),
            reply: format!("reply {}", n),
            tokens: None,
            tps: None,
            context_tokens: None,
            session_id: id.to_string(),
            mode: mode.to_string(),
        }
    }

    #[test]
    fn session_crud() {
        let dir = std::env::temp_dir().join(format!("drill-history-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let h = History::new(dir.clone());
        h.record(row("s1", "solo", 1));
        h.record(row("s1", "solo", 2));
        h.record(row("s2", "agents", 3));
        thread::sleep(std::time::Duration::from_millis(600));

        let sessions = h.list_sessions();
        assert_eq!(sessions.len(), 2);
        let s1 = sessions.iter().find(|s| s.id == "s1").unwrap();
        assert_eq!(s1.mode, "solo");
        assert_eq!(s1.messages, 2);
        assert_eq!(s1.preview, "msg 2");

        let rows = h.get_session("s1");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].input, "msg 1");

        assert!(h.delete_session("s1").is_ok());
        thread::sleep(std::time::Duration::from_millis(300));

        let sessions = h.list_sessions();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].id, "s2");
        assert_eq!(sessions[0].mode, "agents");

        let _ = std::fs::remove_dir_all(&dir);
    }
}