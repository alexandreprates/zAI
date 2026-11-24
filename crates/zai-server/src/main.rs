//! ZAI game server entrypoint.
//!
//! Environment: `ZAI_HTTP_ADDR` (default `0.0.0.0:8080`), `ZAI_DATA_DIR` (default `./data`,
//! holds `replays/` and `metrics.db`), `ZAI_WEB_DIR` (built viewer, optional),
//! `ZAI_ZOMBIE_COUNT` / `ZAI_FOOD_COUNT` (base config overrides), `RUST_LOG`.

use std::path::PathBuf;
use std::sync::Arc;

use zai_server::{app, AppState};
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
    tracing::info!("zai-server {} listening on {addr}", zai_sim::VERSION);
    let shutdown = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    if let Err(e) = axum::serve(listener, app(state, web_dir))
        .with_graceful_shutdown(shutdown)
        .await
    {
        tracing::error!("server error: {e}");
        std::process::exit(1);
    }
}
