//! ZAI game server entrypoint.
//!
//! Environment: `ZAI_HTTP_ADDR` (default `0.0.0.0:8080`), `ZAI_GRPC_ADDR` (agent API, default
//! `0.0.0.0:50051`), `ZAI_DATA_DIR` (default `./data`,
//! holds `replays/` and `metrics.db`), `ZAI_WEB_DIR` (built viewer, optional),
//! `ZAI_ZOMBIE_COUNT` / `ZAI_FOOD_COUNT` (base config overrides), `RUST_LOG`.

use std::path::PathBuf;
use std::sync::Arc;

use zai_server::{app, grpc_service, AppState};
use zai_sim::Config;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let base_config = match Config::default().with_env_overrides() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("{e}");
            std::process::exit(2);
        }
    };
    if let Err(e) = base_config.validate() {
        tracing::error!("{e}");
        std::process::exit(2);
    }
    let data_dir = PathBuf::from(std::env::var("ZAI_DATA_DIR").unwrap_or_else(|_| "data".into()));
    if let Err(e) = std::fs::create_dir_all(&data_dir) {
        tracing::error!("cannot create data dir {}: {e}", data_dir.display());
        std::process::exit(2);
    }
    let web_dir = std::env::var("ZAI_WEB_DIR").ok().map(PathBuf::from);
    let state = Arc::new(AppState::new(
        base_config,
        data_dir.join("replays"),
        Some(data_dir.join("metrics.db")),
    ));

    let addr = std::env::var("ZAI_HTTP_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("failed to bind {addr}: {e}");
            std::process::exit(2);
        }
    };
    let grpc_addr = std::env::var("ZAI_GRPC_ADDR").unwrap_or_else(|_| "0.0.0.0:50051".to_string());
    let grpc_listener = match tokio::net::TcpListener::bind(&grpc_addr).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("failed to bind {grpc_addr}: {e}");
            std::process::exit(2);
        }
    };
    tracing::info!(
        "zai-server {} listening on {addr} (HTTP) and {grpc_addr} (gRPC)",
        zai_sim::VERSION
    );
    // One shutdown signal for both listeners.
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = stop_tx.send(true);
    });
    let stopped = |mut rx: tokio::sync::watch::Receiver<bool>| async move {
        let _ = rx.wait_for(|s| *s).await;
    };
    let http = axum::serve(listener, app(state.clone(), web_dir))
        .with_graceful_shutdown(stopped(stop_rx.clone()));
    let grpc = tonic::transport::Server::builder()
        .add_service(grpc_service(state))
        .serve_with_incoming_shutdown(
            tokio_stream::wrappers::TcpListenerStream::new(grpc_listener),
            stopped(stop_rx),
        );
    let result = tokio::try_join!(async { http.await.map_err(|e| e.to_string()) }, async {
        grpc.await.map_err(|e| e.to_string())
    },);
    if let Err(e) = result {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
