//! Viewer WebSockets: binary MessagePack frames from the server, JSON commands from the client
//! (`{"cmd": "speed", "value": 4}`, `{"cmd": "pause"}`, `{"cmd": "play"}`,
//! `{"cmd": "seek", "tick": 120}` for replays).

use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::broadcast::error::RecvError;

use crate::frames;
use crate::matches::{clamp_speed, Control, MatchHandle};
use crate::replay::Replay;
use crate::AppState;

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum ClientCmd {
    Speed { value: f32 },
    Pause,
    Play,
    Seek { tick: u32 },
}

fn parse(msg: &Message) -> Option<ClientCmd> {
    match msg {
        Message::Text(t) => serde_json::from_str(t.as_str()).ok(),
        _ => None,
    }
}

pub async fn live(
    ws: WebSocketUpgrade,
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Response {
    match state.get_match(&id) {
        Some(h) => ws.on_upgrade(move |socket| live_session(socket, h)),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn live_session(socket: WebSocket, handle: MatchHandle) {
    let (mut tx, mut rx) = socket.split();
    let mut frames = handle.frames.subscribe();
    let (init, snap) = handle.latest.lock().expect("latest frames lock").clone();
    if tx.send(Message::Binary(init)).await.is_err()
        || tx.send(Message::Binary(snap)).await.is_err()
    {
        return;
    }
    loop {
        tokio::select! {
            f = frames.recv() => match f {
                Ok(frame) => {
                    if tx.send(Message::Binary(frame)).await.is_err() {
                        break;
                    }
                }
                Err(RecvError::Lagged(_)) => continue,
                Err(RecvError::Closed) => break,
            },
            m = rx.next() => match m {
                Some(Ok(msg)) => match parse(&msg) {
                    Some(ClientCmd::Speed { value }) => handle.control(Control::Speed(value)),
                    Some(ClientCmd::Pause) => handle.control(Control::Pause(true)),
                    Some(ClientCmd::Play) => handle.control(Control::Pause(false)),
                    Some(ClientCmd::Seek { .. }) | None => {
                        if matches!(msg, Message::Close(_)) {
                            break;
                        }
                    }
                },
                Some(Err(_)) | None => break,
            },
        }
    }
}

pub async fn replay(
    ws: WebSocketUpgrade,
    Path(id): Path<String>,
    State(state): State<Arc<AppState>>,
) -> Response {
    let store = state.replays.clone();
    let loaded = tokio::task::spawn_blocking(move || store.load(&id))
        .await
        .ok()
        .flatten();
    match loaded {
        Some(r) => ws.on_upgrade(move |socket| replay_session(socket, r)),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn maybe_sleep(delay: Option<Duration>) {
    match delay {
        Some(d) => tokio::time::sleep(d).await,
        None => std::future::pending().await,
    }
}

async fn replay_session(socket: WebSocket, replay: Replay) {
    let (mut tx, mut rx) = socket.split();
    let total = replay.actions.len();
    let mut env = replay.simulate(0);
    let mut tick = 0usize;
    let mut speed = 1.0f32;
    let mut paused = false;
    let decision_dt = replay.config.decision_dt();
    let init = frames::init(
        &env,
        "replay",
        &replay.id,
        replay.episode,
        Some(total as u32),
    );
    if tx.send(Message::Binary(init)).await.is_err()
        || tx
            .send(Message::Binary(frames::snapshot(&env, speed, paused)))
            .await
            .is_err()
    {
        return;
    }
    loop {
        let delay = if paused || tick >= total {
            None
        } else if speed > 0.0 {
            Some(Duration::from_secs_f32(decision_dt / speed))
        } else {
            Some(Duration::ZERO)
        };
        tokio::select! {
            _ = maybe_sleep(delay) => {
                replay.apply(&mut env, tick);
                tick += 1;
            }
            m = rx.next() => match m {
                Some(Ok(msg)) => {
                    match parse(&msg) {
                        Some(ClientCmd::Speed { value }) => speed = clamp_speed(value),
                        Some(ClientCmd::Pause) => paused = true,
                        Some(ClientCmd::Play) => paused = false,
                        Some(ClientCmd::Seek { tick: t }) => {
                            tick = (t as usize).min(total);
                            env = replay.simulate(tick);
                        }
                        None if matches!(msg, Message::Close(_)) => break,
                        None => continue,
                    }
                }
                Some(Err(_)) | None => break,
            },
        }
        if tx
            .send(Message::Binary(frames::snapshot(&env, speed, paused)))
            .await
            .is_err()
        {
            break;
        }
    }
}
