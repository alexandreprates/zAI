//! ZAI game server: live bot matches streamed to the web viewer, replays, and a metrics API.

pub mod api;
pub mod frames;
pub mod leagues;
pub mod matches;
pub mod metrics;
pub mod policy;
pub mod replay;
pub mod ws;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::routing::{get, post};
use axum::Router;
use tokio::sync::broadcast;
use tower_http::services::{ServeDir, ServeFile};
use zai_sim::Config;

use crate::leagues::LeagueStore;
use crate::matches::MatchHandle;
use crate::metrics::MetricsStore;
use crate::policy::ModelStore;
use crate::replay::ReplayStore;

pub struct AppState {
    pub base_config: Config,
    pub matches: RwLock<HashMap<String, MatchHandle>>,
    pub replays: ReplayStore,
    /// Trained models, in `models/` next to the replay directory.
    pub models: ModelStore,
    /// League reports, in `leagues/` next to the replay directory.
    pub leagues: LeagueStore,
    pub metrics: MetricsStore,
    pub metrics_tx: broadcast::Sender<String>,
    next_id: AtomicU64,
}

impl AppState {
    /// `data_dir` holds `replays/` and `metrics.db`; `None` keeps metrics in memory (tests).
    /// Models are read from the `models` directory next to `replays_dir`.
    pub fn new(base_config: Config, replays_dir: PathBuf, metrics_db: Option<PathBuf>) -> Self {
        let models = ModelStore::new(replays_dir.with_file_name("models"));
        let leagues = LeagueStore::new(replays_dir.with_file_name("leagues"));
        let replays = ReplayStore::new(replays_dir).expect("replay directory is writable");
        let metrics = MetricsStore::open(metrics_db.as_deref()).expect("metrics database opens");
        let (metrics_tx, _) = broadcast::channel(1024);
        Self {
            base_config,
            matches: RwLock::new(HashMap::new()),
            replays,
            models,
            leagues,
            metrics,
            metrics_tx,
            next_id: AtomicU64::new(1),
        }
    }

    pub fn get_match(&self, id: &str) -> Option<MatchHandle> {
        self.matches.read().expect("matches lock").get(id).cloned()
    }

    fn next_match_id(&self) -> String {
        format!(
            "m{}-{}",
            unix_now(),
            self.next_id.fetch_add(1, Ordering::Relaxed)
        )
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Merges a partial JSON config over `base` and validates the result.
pub fn merge_config(base: &Config, overrides: &serde_json::Value) -> Result<Config, String> {
    fn merge(a: &mut serde_json::Value, b: &serde_json::Value) {
        match (a, b) {
            (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
                for (k, v) in b {
                    merge(a.entry(k.clone()).or_insert(serde_json::Value::Null), v);
                }
            }
            (a, b) => *a = b.clone(),
        }
    }
    let mut value = serde_json::to_value(base).map_err(|e| e.to_string())?;
    if !overrides.is_null() {
        merge(&mut value, overrides);
    }
    let cfg: Config = serde_json::from_value(value).map_err(|e| format!("invalid config: {e}"))?;
    cfg.validate().map_err(|e| e.to_string())?;
    Ok(cfg)
}

/// HTTP router. `web_dir`, when set, is served as the viewer (SPA fallback to index.html).
pub fn app(state: Arc<AppState>, web_dir: Option<PathBuf>) -> Router {
    let router = Router::new()
        .route("/healthz", get(|| async { zai_sim::VERSION }))
        .route("/api/config/default", get(api::default_config))
        .route(
            "/api/matches",
            get(api::list_matches).post(api::create_match),
        )
        .route("/api/matches/{id}", get(api::get_match))
        .route("/api/matches/{id}/control", post(api::control_match))
        .route("/api/models", get(api::list_models))
        .route("/api/replays", get(api::list_replays))
        .route("/api/replays/{id}", get(api::get_replay))
        .route(
            "/api/metrics",
            get(api::get_metrics).post(api::post_metrics),
        )
        .route("/api/metrics/stream", get(api::metrics_stream))
        .route("/api/runs", get(api::list_runs))
        .route("/api/leagues", get(api::list_leagues))
        .route("/api/leagues/{id}", get(api::get_league))
        .route("/ws/match/{id}", get(ws::live))
        .route("/ws/replay/{id}", get(ws::replay))
        .with_state(state);
    match web_dir {
        Some(dir) => {
            let index = dir.join("index.html");
            router.fallback_service(ServeDir::new(dir).fallback(ServeFile::new(index)))
        }
        None => router,
    }
}
