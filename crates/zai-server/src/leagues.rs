//! League reports written by the M6 trainer and arena (`<ZAI_DATA_DIR>/leagues/<run_id>/`):
//! `pool.json` (snapshot pool, PFSP weights) and `arena.json` (Elo ratings, cross-play
//! matrix, tactic metrics). The server only reads and merges them.

use std::path::PathBuf;

use serde::Serialize;

use crate::replay::valid_id;

#[derive(Clone, Debug)]
pub struct LeagueStore {
    dir: PathBuf,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct LeagueSummary {
    pub id: String,
    /// Unix time of the newest of the two files.
    pub updated: u64,
}

const FILES: [&str; 2] = ["pool.json", "arena.json"];

impl LeagueStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    /// Leagues with at least one report, newest first.
    pub fn list(&self) -> Vec<LeagueSummary> {
        let mut out: Vec<LeagueSummary> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let id = e.file_name().to_str()?.to_string();
                if !valid_id(&id) {
                    return None;
                }
                let updated = FILES
                    .iter()
                    .filter_map(|f| e.path().join(f).metadata().ok()?.modified().ok())
                    .max()?;
                let updated = updated
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                Some(LeagueSummary { id, updated })
            })
            .collect();
        out.sort_by(|a, b| b.updated.cmp(&a.updated).then(a.id.cmp(&b.id)));
        out
    }

    /// `{"id", "pool", "arena"}` with each report as written (`null` if missing or invalid).
    pub fn get(&self, id: &str) -> Option<serde_json::Value> {
        if !valid_id(id) {
            return None;
        }
        let dir = self.dir.join(id);
        let read = |f: &str| -> serde_json::Value {
            std::fs::read(dir.join(f))
                .ok()
                .and_then(|b| serde_json::from_slice(&b).ok())
                .unwrap_or(serde_json::Value::Null)
        };
        let (pool, arena) = (read(FILES[0]), read(FILES[1]));
        if pool.is_null() && arena.is_null() {
            return None;
        }
        Some(serde_json::json!({ "id": id, "pool": pool, "arena": arena }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_and_merges_league_reports() {
        let tmp = tempfile::tempdir().unwrap();
        let store = LeagueStore::new(tmp.path().to_path_buf());
        assert!(store.list().is_empty());
        let dir = tmp.path().join("lg1");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("pool.json"), r#"{"pfsp": "hard"}"#).unwrap();
        std::fs::create_dir_all(tmp.path().join("empty")).unwrap();
        assert_eq!(
            store
                .list()
                .iter()
                .map(|l| l.id.as_str())
                .collect::<Vec<_>>(),
            ["lg1"]
        );
        let v = store.get("lg1").unwrap();
        assert_eq!(v["pool"]["pfsp"], "hard");
        assert!(v["arena"].is_null());
        assert!(store.get("empty").is_none());
        assert!(store.get("../lg1").is_none());
    }
}
