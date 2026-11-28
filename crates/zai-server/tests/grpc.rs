//! gRPC agent API against a real listener (docs/ARCHITECTURE.md section 4.9).

use std::sync::Arc;

use tokio_stream::wrappers::TcpListenerStream;
use zai_server::grpc::pb::agent_service_client::AgentServiceClient;
use zai_server::grpc::pb::{GetInfoRequest, SideSpec};
use zai_server::grpc::PROTOCOL_VERSION;
use zai_server::{grpc_service, AppState};
use zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{Block, HUMAN_LAYOUT, ZOMBIE_LAYOUT};
use zai_sim::{Config, HUMAN_OBS_DIM, OBS_VERSION, ZOMBIE_OBS_DIM};

/// Serves the agent API on a free port; returns a connected client and the temp dir.
async fn serve() -> (
    AgentServiceClient<tonic::transport::Channel>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let state = Arc::new(AppState::new(
        Config::default(),
        dir.path().join("replays"),
        None,
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(grpc_service(state))
            .serve_with_incoming(TcpListenerStream::new(listener)),
    );
    let client = AgentServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    (client, dir)
}

fn assert_layout(spec: &SideSpec, obs_dim: usize, layout: &[Block], action_dims: &[usize]) {
    assert_eq!(spec.obs_dim as usize, obs_dim);
    assert_eq!(spec.blocks.len(), layout.len());
    for (got, want) in spec.blocks.iter().zip(layout) {
        assert_eq!(got.name, want.name);
        assert_eq!(got.start as usize, want.start);
        assert_eq!(got.len as usize, want.len());
        assert_eq!(got.slots as usize, want.slots);
        assert_eq!(got.fields, want.fields);
    }
    let end = layout.last().map_or(0, |b| b.start + b.len());
    assert_eq!(end, obs_dim, "the blocks cover the observation");
    let dims: Vec<usize> = spec.action_dims.iter().map(|&d| d as usize).collect();
    assert_eq!(dims, action_dims);
}

#[tokio::test]
async fn get_info_reports_versions_layouts_and_the_base_config() {
    let (mut client, _dir) = serve().await;
    let info = client
        .get_info(GetInfoRequest {})
        .await
        .unwrap()
        .into_inner();
    assert_eq!(info.version, zai_sim::VERSION);
    assert_eq!(info.protocol, PROTOCOL_VERSION);
    assert_eq!(info.obs_version, OBS_VERSION);
    assert_layout(
        info.human.as_ref().unwrap(),
        HUMAN_OBS_DIM,
        &HUMAN_LAYOUT,
        &[MOVE_ACTIONS, INTERACT_ACTIONS],
    );
    assert_layout(
        info.zombie.as_ref().unwrap(),
        ZOMBIE_OBS_DIM,
        &ZOMBIE_LAYOUT,
        &[MOVE_ACTIONS],
    );
    let base: Config = serde_json::from_str(&info.base_config_json).unwrap();
    assert_eq!(base.zombie_count, Config::default().zombie_count);
}

#[tokio::test]
async fn play_and_create_match_are_not_implemented_yet() {
    let (mut client, _dir) = serve().await;
    let err = client
        .create_match(zai_server::grpc::pb::CreateMatchRequest::default())
        .await
        .unwrap_err();
    assert_eq!(err.code(), tonic::Code::Unimplemented);
}
