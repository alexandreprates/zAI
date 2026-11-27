//! HTTP API.

use std::convert::Infallible;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

use crate::matches::{self, Control, MatchModels, MatchRequest};
use crate::metrics::MetricsPost;
use crate::policy::Side;
use crate::{merge_config, unix_now, AppState};

fn error(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "error": msg.into() }))).into_response()
}

pub async fn default_config(State(state): State<Arc<AppState>>) -> Response {
    Json(&state.base_config).into_response()
}

pub async fn list_matches(State(state): State<Arc<AppState>>) -> Response {
    let mut infos: Vec<_> = state
        .matches
        .read()
        .expect("matches lock")
        .values()
        .map(|h| h.info.lock().expect("match info lock").clone())
        .collect();
    infos.sort_by(|a, b| b.created_at.cmp(&a.created_at).then(b.id.cmp(&a.id)));
    Json(infos).into_response()
}

pub async fn create_match(
    State(state): State<Arc<AppState>>,
    Json(req): Json<MatchRequest>,
) -> Response {
    let cfg = match merge_config(&state.base_config, &req.config) {
        Ok(cfg) => cfg,
        Err(e) => return error(StatusCode::BAD_REQUEST, e),
    };
    let store = state.models.clone();
    let (human, zombies, zombie_count) = (req.human.clone(), req.zombies.clone(), cfg.zombie_count);
    let loaded = tokio::task::spawn_blocking(move || -> Result<MatchModels, String> {
        Ok(MatchModels {
            human: human
                .model_id()
                .map(|id| store.load(id, Side::Human, 1))
                .transpose()?,
            zombies: zombies
                .model_id()
                .map(|id| store.load(id, Side::Zombie, zombie_count))
                .transpose()?,
        })
    })
    .await;
    let models = match loaded {
        Ok(Ok(m)) => m,
        Ok(Err(e)) => return error(StatusCode::BAD_REQUEST, e),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let id = state.next_match_id();
    let seed = req.seed.unwrap_or_else(crate::random_seed);
    let handle = matches::start(
        id.clone(),
        cfg,
        &req,
        seed,
        unix_now(),
        state.replays.clone(),
        models,
    );
    let info = handle.info.lock().expect("match info lock").clone();
    state
        .matches
        .write()
        .expect("matches lock")
        .insert(id, handle);
    (StatusCode::CREATED, Json(info)).into_response()
}

pub async fn get_match(Path(id): Path<String>, State(state): State<Arc<AppState>>) -> Response {
    match state.get_match(&id) {
        Some(h) => Json(h.info.lock().expect("match info lock").clone()).into_response(),
        None => error(StatusCode::NOT_FOUND, "match not found"),
    }
}

#[derive(Debug, Deserialize)]
pub struct ControlRequest {
    speed: Option<f32>,
    paused: Option<bool>,
    #[serde(default)]
    stop: bool,
    auto_restart: Option<bool>,
}

pub async fn control_match(
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<ControlRequest>,
) -> Response {
    let Some(h) = state.get_match(&id) else {
        return error(StatusCode::NOT_FOUND, "match not found");
    };
    if let Some(s) = req.speed {
        h.control(Control::Speed(s));
    }
    if let Some(p) = req.paused {
        h.control(Control::Pause(p));
    }
    if let Some(a) = req.auto_restart {
        h.info.lock().expect("match info lock").auto_restart = a;
    }
    if req.stop {
        h.control(Control::Stop);
    }
    StatusCode::NO_CONTENT.into_response()
}

/// Trained models available for matches (`policy.rs`), newest first.
pub async fn list_models(State(state): State<Arc<AppState>>) -> Response {
    let store = state.models.clone();
    match tokio::task::spawn_blocking(move || store.list()).await {
        Ok(list) => Json(list).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// League runs with a pool or arena report (`leagues.rs`), newest first.
pub async fn list_leagues(State(state): State<Arc<AppState>>) -> Response {
    let store = state.leagues.clone();
    match tokio::task::spawn_blocking(move || store.list()).await {
        Ok(list) => Json(list).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `{"id", "pool", "arena"}` of one league run.
pub async fn get_league(Path(id): Path<String>, State(state): State<Arc<AppState>>) -> Response {
    let store = state.leagues.clone();
    match tokio::task::spawn_blocking(move || store.get(&id)).await {
        Ok(Some(v)) => Json(v).into_response(),
        Ok(None) => error(StatusCode::NOT_FOUND, "league not found"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn list_replays(State(state): State<Arc<AppState>>) -> Response {
    let store = state.replays.clone();
    match tokio::task::spawn_blocking(move || store.list()).await {
        Ok(list) => Json(list).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn get_replay(Path(id): Path<String>, State(state): State<Arc<AppState>>) -> Response {
    let store = state.replays.clone();
    match tokio::task::spawn_blocking(move || store.load(&id)).await {
        Ok(Some(r)) => Json(r).into_response(),
        Ok(None) => error(StatusCode::NOT_FOUND, "replay not found"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn post_metrics(
    State(state): State<Arc<AppState>>,
    Json(post): Json<MetricsPost>,
) -> Response {
    if let Err(e) = post.validate() {
        return error(StatusCode::BAD_REQUEST, e);
    }
    let now = unix_now() as f64;
    let st = state.clone();
    let p = post.clone();
    match tokio::task::spawn_blocking(move || st.metrics.insert(&p, now)).await {
        Ok(Ok(())) => {
            // Live listeners are optional.
            let _ = state.metrics_tx.send(
                serde_json::to_string(&json!({
                    "run_id": post.run_id,
                    "model": post.model,
                    "step": post.step,
                    "wall_time": post.wall_time.unwrap_or(now),
                    "scalars": post.scalars,
                }))
                .expect("metrics serialize"),
            );
            StatusCode::NO_CONTENT.into_response()
        }
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct MetricsQuery {
    run_id: String,
    model: Option<String>,
    /// Comma-separated keys.
    keys: Option<String>,
    since: Option<i64>,
}

pub async fn get_metrics(
    State(state): State<Arc<AppState>>,
    Query(q): Query<MetricsQuery>,
) -> Response {
    let keys: Option<Vec<String>> = q.keys.map(|k| {
        k.split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    });
    let st = state.clone();
    let res = tokio::task::spawn_blocking(move || {
        st.metrics
            .query(&q.run_id, q.model.as_deref(), keys.as_deref(), q.since)
    })
    .await;
    match res {
        Ok(Ok(points)) => Json(points).into_response(),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn list_runs(State(state): State<Arc<AppState>>) -> Response {
    let st = state.clone();
    match tokio::task::spawn_blocking(move || st.metrics.runs()).await {
        Ok(Ok(runs)) => Json(runs).into_response(),
        Ok(Err(e)) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// Server-sent events with every posted metrics batch (JSON).
pub async fn metrics_stream(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.metrics_tx.subscribe())
        .filter_map(|m| m.ok().map(|data| Ok(Event::default().data(data))));
    Sse::new(stream).keep_alive(KeepAlive::default())
}
