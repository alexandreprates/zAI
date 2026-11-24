//! Training/evaluation scalars stored in SQLite.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};

pub const MODELS: [&str; 3] = ["human", "zombie", "arena"];

/// Body of `POST /api/metrics`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricsPost {
    pub run_id: String,
    pub model: String,
    pub step: i64,
    /// Seconds since the Unix epoch; the server time is used when absent.
    #[serde(default)]
    pub wall_time: Option<f64>,
    pub scalars: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct MetricPoint {
    pub model: String,
    pub key: String,
    pub step: i64,
    pub wall_time: f64,
    pub value: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RunSummary {
    pub run_id: String,
    pub models: Vec<String>,
    pub last_step: i64,
    pub updated: f64,
}

pub struct MetricsStore {
    conn: Mutex<Connection>,
}

impl MetricsPost {
    pub fn validate(&self) -> Result<(), String> {
        if self.run_id.is_empty() || self.run_id.len() > 128 {
            return Err("run_id must be 1-128 characters".into());
        }
        if !MODELS.contains(&self.model.as_str()) {
            return Err(format!("model must be one of {MODELS:?}"));
        }
        if self.scalars.is_empty() {
            return Err("scalars must not be empty".into());
        }
        if self.scalars.keys().any(|k| k.is_empty() || k.len() > 128) {
            return Err("scalar keys must be 1-128 characters".into());
        }
        Ok(())
    }
}

impl MetricsStore {
    /// Opens (or creates) the database at `path`; `None` uses an in-memory database.
    pub fn open(path: Option<&Path>) -> rusqlite::Result<Self> {
        let conn = match path {
            Some(p) => Connection::open(p)?,
            None => Connection::open_in_memory()?,
        };
        conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS metrics (
                 run_id TEXT NOT NULL,
                 model TEXT NOT NULL,
                 key TEXT NOT NULL,
                 step INTEGER NOT NULL,
                 wall_time REAL NOT NULL,
                 value REAL NOT NULL
             );
             CREATE INDEX IF NOT EXISTS metrics_lookup ON metrics (run_id, model, key, step);",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    pub fn insert(&self, post: &MetricsPost, now: f64) -> rusqlite::Result<()> {
        let mut conn = self.conn.lock().expect("metrics lock");
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare_cached(
                "INSERT INTO metrics (run_id, model, key, step, wall_time, value)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            )?;
            for (key, value) in &post.scalars {
                stmt.execute(params![
                    post.run_id,
                    post.model,
                    key,
                    post.step,
                    post.wall_time.unwrap_or(now),
                    value
                ])?;
            }
        }
        tx.commit()
    }

    /// Points of one run, optionally filtered by model, keys and `step > since`, by step.
    pub fn query(
        &self,
        run_id: &str,
        model: Option<&str>,
        keys: Option<&[String]>,
        since: Option<i64>,
    ) -> rusqlite::Result<Vec<MetricPoint>> {
        let conn = self.conn.lock().expect("metrics lock");
        let mut stmt = conn.prepare_cached(
            "SELECT model, key, step, wall_time, value FROM metrics
             WHERE run_id = ?1 AND (?2 IS NULL OR model = ?2) AND step > ?3
             ORDER BY model, key, step",
        )?;
        let rows = stmt.query_map(params![run_id, model, since.unwrap_or(i64::MIN)], |r| {
            Ok(MetricPoint {
                model: r.get(0)?,
                key: r.get(1)?,
                step: r.get(2)?,
                wall_time: r.get(3)?,
                value: r.get(4)?,
            })
        })?;
        let mut out = Vec::new();
        for row in rows {
            let p = row?;
            if keys.is_none_or(|ks| ks.contains(&p.key)) {
                out.push(p);
            }
        }
        Ok(out)
    }

    pub fn runs(&self) -> rusqlite::Result<Vec<RunSummary>> {
        let conn = self.conn.lock().expect("metrics lock");
        let mut stmt = conn.prepare_cached(
            "SELECT run_id, group_concat(DISTINCT model), max(step), max(wall_time)
             FROM metrics GROUP BY run_id ORDER BY max(wall_time) DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            let models: String = r.get(1)?;
            Ok(RunSummary {
                run_id: r.get(0)?,
                models: models.split(',').map(str::to_owned).collect(),
                last_step: r.get(2)?,
                updated: r.get(3)?,
            })
        })?;
        rows.collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post(step: i64, reward: f64) -> MetricsPost {
        MetricsPost {
            run_id: "run-1".into(),
            model: "human".into(),
            step,
            wall_time: None,
            scalars: BTreeMap::from([("reward".into(), reward), ("loss".into(), 0.5)]),
        }
    }

    #[test]
    fn insert_and_query_by_key_and_step() {
        let store = MetricsStore::open(None).unwrap();
        store.insert(&post(1, 0.1), 100.0).unwrap();
        store.insert(&post(2, 0.2), 101.0).unwrap();
        let all = store.query("run-1", Some("human"), None, None).unwrap();
        assert_eq!(all.len(), 4);
        let rewards = store
            .query("run-1", None, Some(&["reward".into()]), Some(1))
            .unwrap();
        assert_eq!(rewards.len(), 1);
        assert_eq!((rewards[0].step, rewards[0].value), (2, 0.2));
        let runs = store.runs().unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].last_step, 2);
    }

    #[test]
    fn validation_rejects_unknown_models_and_empty_scalars() {
        let mut p = post(1, 0.0);
        assert!(p.validate().is_ok());
        p.model = "critic".into();
        assert!(p.validate().is_err());
        let mut p = post(1, 0.0);
        p.scalars.clear();
        assert!(p.validate().is_err());
    }
}
