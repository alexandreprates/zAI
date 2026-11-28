//! gRPC agent API (`proto/zai/v1/agent.proto`, docs/ARCHITECTURE.md section 4.9): external
//! processes play one side of a server match. M7.0 serves `GetInfo`; `CreateMatch` and `Play`
//! come with the remote sides of M7.1.

use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};
use zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{Block, HUMAN_LAYOUT, ZOMBIE_LAYOUT};
use zai_sim::{HUMAN_OBS_DIM, OBS_VERSION, ZOMBIE_OBS_DIM};

use crate::AppState;

/// Generated messages and service traits.
pub mod pb {
    tonic::include_proto!("zai.v1");
}

pub use pb::agent_service_server::AgentServiceServer;

/// Version of the agent protocol; clients send theirs in `Join`.
pub const PROTOCOL_VERSION: u32 = 1;

fn side_spec(obs_dim: usize, layout: &[Block], action_dims: &[usize]) -> pb::SideSpec {
    pb::SideSpec {
        obs_dim: obs_dim as u32,
        blocks: layout
            .iter()
            .map(|b| pb::LayoutBlock {
                name: b.name.to_owned(),
                start: b.start as u32,
                len: b.len() as u32,
                slots: b.slots as u32,
                fields: b.fields.iter().map(|f| (*f).to_owned()).collect(),
            })
            .collect(),
        action_dims: action_dims.iter().map(|&d| d as u32).collect(),
    }
}

/// `GetInfo` of this build, from `OBS_VERSION` and the layout tables of `zai_sim::obs`.
pub fn server_info(state: &AppState) -> pb::ServerInfo {
    pb::ServerInfo {
        version: zai_sim::VERSION.to_owned(),
        protocol: PROTOCOL_VERSION,
        obs_version: OBS_VERSION,
        human: Some(side_spec(
            HUMAN_OBS_DIM,
            &HUMAN_LAYOUT,
            &[MOVE_ACTIONS, INTERACT_ACTIONS],
        )),
        zombie: Some(side_spec(ZOMBIE_OBS_DIM, &ZOMBIE_LAYOUT, &[MOVE_ACTIONS])),
        base_config_json: serde_json::to_string(&state.base_config).unwrap_or_default(),
    }
}

pub struct AgentApi {
    state: Arc<AppState>,
}

type ServerStream = Pin<Box<dyn Stream<Item = Result<pb::ServerMessage, Status>> + Send>>;

#[tonic::async_trait]
impl pb::agent_service_server::AgentService for AgentApi {
    async fn get_info(
        &self,
        _request: Request<pb::GetInfoRequest>,
    ) -> Result<Response<pb::ServerInfo>, Status> {
        Ok(Response::new(server_info(&self.state)))
    }

    async fn create_match(
        &self,
        _request: Request<pb::CreateMatchRequest>,
    ) -> Result<Response<pb::MatchCreated>, Status> {
        Err(Status::unimplemented("remote sides land in M7.1"))
    }

    type PlayStream = ServerStream;

    async fn play(
        &self,
        _request: Request<Streaming<pb::ClientMessage>>,
    ) -> Result<Response<Self::PlayStream>, Status> {
        Err(Status::unimplemented("remote sides land in M7.1"))
    }
}

/// The gRPC service over the shared server state, served next to the HTTP router.
pub fn grpc_service(state: Arc<AppState>) -> AgentServiceServer<AgentApi> {
    AgentServiceServer::new(AgentApi { state })
}
