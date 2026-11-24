//! End-to-end: HTTP API, live match WebSocket, replay storage and replay playback.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio_tungstenite::tungstenite::Message;
use tower::ServiceExt;
use zai_server::{app, AppState};
use zai_sim::Config;

async fn call(
    router: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json")
        .body(body.map_or(Body::empty(), |b| Body::from(b.to_string())))
        .unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, value)
}

fn decode(msg: Message) -> Value {
    match msg {
        Message::Binary(b) => rmp_serde::from_slice(&b).expect("msgpack frame"),
        other => panic!("expected a binary frame, got {other:?}"),
    }
}

async fn next_frame<S>(ws: &mut S) -> Value
where
    S: StreamExt<Item = Result<Message, tokio_tungstenite::tungstenite::Error>> + Unpin,
{
    let msg = tokio::time::timeout(Duration::from_secs(10), ws.next())
        .await
        .expect("frame within 10 s")
        .expect("stream open")
        .expect("valid message");
    decode(msg)
}

#[tokio::test]
async fn metrics_matches_and_replays_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState::new(
        Config::default(),
        dir.path().join("replays"),
        None,
    ));
    let router = app(state, None);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = router.clone();
    tokio::spawn(async move { axum::serve(listener, server).await.unwrap() });

    // Metrics.
    let post = json!({"run_id": "r1", "model": "human", "step": 10, "scalars": {"reward": 0.5}});
    assert_eq!(
        call(&router, "POST", "/api/metrics", Some(post)).await.0,
        StatusCode::NO_CONTENT
    );
    let bad = json!({"run_id": "r1", "model": "nope", "step": 1, "scalars": {"x": 1.0}});
    assert_eq!(
        call(&router, "POST", "/api/metrics", Some(bad)).await.0,
        StatusCode::BAD_REQUEST
    );
    let (status, points) = call(&router, "GET", "/api/metrics?run_id=r1&keys=reward", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(points[0]["value"], 0.5);
    let (_, runs) = call(&router, "GET", "/api/runs", None).await;
    assert_eq!(runs[0]["run_id"], "r1");

    // Invalid match config is rejected.
    let (status, _) = call(
        &router,
        "POST",
        "/api/matches",
        Some(json!({"config": {"zombie_count": 0}})),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // A short match at x16.
    let (status, info) = call(
        &router,
        "POST",
        "/api/matches",
        Some(json!({"config": {"episode_cap_s": 20.0}, "seed": 5, "speed": 16})),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED);
    let id = info["id"].as_str().unwrap().to_owned();

    let (mut ws, _) = tokio_tungstenite::connect_async(format!("ws://{addr}/ws/match/{id}"))
        .await
        .unwrap();
    let init = next_frame(&mut ws).await;
    assert_eq!(init["type"], "init");
    assert_eq!(init["houses"].as_array().unwrap().len(), 10);
    let snap = next_frame(&mut ws).await;
    assert_eq!(snap["type"], "snapshot");
    assert_eq!(snap["zombies"].as_array().unwrap().len(), 20);

    // Speed up through the WebSocket control channel and wait for the episode to end.
    ws.send(Message::Text(r#"{"cmd":"speed","value":0}"#.into()))
        .await
        .unwrap();
    let last_tick = loop {
        let f = next_frame(&mut ws).await;
        if f["type"] == "snapshot" && f["outcome"] != "running" {
            break f["tick"].as_u64().unwrap();
        }
    };
    assert!(last_tick > 0 && last_tick <= 40);

    let mut replay_id = None;
    for _ in 0..100 {
        let (_, m) = call(&router, "GET", &format!("/api/matches/{id}"), None).await;
        if m["status"] == "finished" {
            replay_id = m["replays"][0].as_str().map(str::to_owned);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let replay_id = replay_id.expect("match finishes and stores a replay");

    let (status, replay) = call(&router, "GET", &format!("/api/replays/{replay_id}"), None).await;
    assert_eq!(status, StatusCode::OK);
    let parsed: zai_server::replay::Replay = serde_json::from_value(replay).unwrap();
    assert_eq!(parsed.actions.len() as u64, last_tick);
    assert!(
        parsed.verify(),
        "replay re-simulates to the recorded final state"
    );
    let (_, list) = call(&router, "GET", "/api/replays", None).await;
    assert_eq!(list.as_array().unwrap().len(), 1);

    // Replay playback with seek.
    let (mut rws, _) =
        tokio_tungstenite::connect_async(format!("ws://{addr}/ws/replay/{replay_id}"))
            .await
            .unwrap();
    let init = next_frame(&mut rws).await;
    assert_eq!(init["total_ticks"].as_u64(), Some(last_tick));
    assert_eq!(next_frame(&mut rws).await["tick"], 0);
    rws.send(Message::Text(r#"{"cmd":"pause"}"#.into()))
        .await
        .unwrap();
    rws.send(Message::Text(r#"{"cmd":"seek","tick":10}"#.into()))
        .await
        .unwrap();
    let mut seen = false;
    for _ in 0..5 {
        let f = next_frame(&mut rws).await;
        if f["tick"] == 10 && f["paused"] == true {
            seen = true;
            break;
        }
    }
    assert!(seen, "seek moves the replay to tick 10");

    // Unknown ids.
    assert_eq!(
        call(&router, "GET", "/api/matches/nope", None).await.0,
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        call(&router, "GET", "/api/replays/..%2Fx", None).await.0,
        StatusCode::NOT_FOUND
    );
}
