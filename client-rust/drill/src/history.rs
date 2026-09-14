use futures::StreamExt;
use lib_rust::agents::HistoryRow;
use lancedb::arrow::arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use lancedb::query::{ExecutableQuery, QueryBase};
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::runtime::Runtime;

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
}

async fn write_row(db_dir: &PathBuf, row: &HistoryRow) -> Result<(), String> {
    let batch = row_to_batch(row)?;
    let uri = db_dir.to_string_lossy().to_string();
    let db = lancedb::connect(&uri)
        .execute()
        .await
        .map_err(|err| err.to_string())?;

    let table = match db.open_table("historico").execute().await {
        Ok(table) => table,
        Err(_) => db
            .create_table("historico", vec![batch.clone()])
            .execute()
            .await
            .map_err(|err| err.to_string())?,
    };

    table
        .add(vec![batch])
        .execute()
        .await
        .map_err(|err| err.to_string())
        .map(|_| ())
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
        Field::new("agent_id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("input", DataType::Utf8, false),
        Field::new("reply", DataType::Utf8, false),
        Field::new("tokens", DataType::Int32, false),
        Field::new("tps", DataType::Float64, false),
        Field::new("context_tokens", DataType::Int32, false),
    ]));

    let created_at: ArrayRef = Arc::new(Float64Array::from(vec![now_f64()]));
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
            created_at, agent_id, kind, input, reply, tokens, tps, context_tokens,
        ],
    )
    .map_err(|err| err.to_string())
}

/// Le as ultimas `n` linhas do historico (para verificacao/ferramentas).
pub fn last_rows(db_dir: &PathBuf, n: usize) -> Vec<HistoryRow> {
    let db_path = db_dir.join("conversas.lance");
    let rt = Runtime::new().expect("tokio runtime");
    rt.block_on(async move {
        let uri = db_path.to_string_lossy().to_string();
        let db = match lancedb::connect(&uri).execute().await {
            Ok(db) => db,
            Err(_) => return Vec::new(),
        };
        let table = match db.open_table("historico").execute().await {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut stream = match table.query().limit(n).execute().await {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut rows: Vec<HistoryRow> = Vec::new();
        while let Some(batch) = stream.next().await {
            let batch = match batch {
                Ok(batch) => batch,
                Err(_) => continue,
            };
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

            if let (Some(agent_id), Some(kind), Some(input), Some(reply)) =
                (agent_id, kind, input, reply)
            {
                for i in 0..agent_id.len() {
                    rows.push(HistoryRow {
                        agent_id: agent_id.value(i).to_string(),
                        kind: kind.value(i).to_string(),
                        input: input.value(i).to_string(),
                        reply: reply.value(i).to_string(),
                        tokens: None,
                        tps: None,
                        context_tokens: None,
                    });
                }
            }
        }
        rows
    })
}