//! gRPC agent API (`proto/zai/v1/agent.proto`, docs/ARCHITECTURE.md section 4.9): external
//! processes play one side of a server match (lockstep; realtime lands in M7.3).

use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};
use zai_sim::env::{INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{Block, HUMAN_LAYOUT, ZOMBIE_LAYOUT};
use zai_sim::{HUMAN_OBS_DIM, OBS_VERSION, ZOMBIE_OBS_DIM};

use crate::api::{self, CreateError};
use crate::matches::{Agent, MatchRequest};
use crate::remote::{Conn, MatchSide};
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
        request: Request<pb::CreateMatchRequest>,
    ) -> Result<Response<pb::MatchCreated>, Status> {
        let r = request.into_inner();
        let config = if r.config_json.trim().is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_str(&r.config_json)
                .map_err(|e| Status::invalid_argument(format!("config_json: {e}")))?
        };
        let req = MatchRequest {
            config,
            seed: r.seed,
            human: parse_agent(&r.human)?,
            zombies: parse_agent(&r.zombies)?,
            deterministic: r.deterministic,
            speed: r.speed,
            auto_restart: r.auto_restart,
            realtime: r.realtime,
        };
        match api::create(&self.state, req).await {
            Ok(info) => Ok(Response::new(pb::MatchCreated { match_id: info.id })),
            Err(e @ CreateError::BadRequest(_)) => Err(Status::invalid_argument(e.message())),
            Err(e @ CreateError::TooManyRemote(_)) => Err(Status::resource_exhausted(e.message())),
            Err(e @ CreateError::Internal(_)) => Err(Status::internal(e.message())),
        }
    }

    type PlayStream = ServerStream;

    async fn play(
        &self,
        request: Request<Streaming<pb::ClientMessage>>,
    ) -> Result<Response<Self::PlayStream>, Status> {
        let mut inbound = request.into_inner();
        let join = match inbound.message().await? {
            Some(pb::ClientMessage {
                msg: Some(pb::client_message::Msg::Join(j)),
            }) => j,
            _ => return Err(Status::invalid_argument("the first message must be Join")),
        };
        let side = MatchSide::from_proto(join.side)
            .ok_or_else(|| Status::invalid_argument("Join.side must be human or zombies"))?;
        check_versions(&join, side)?;
        let handle = self
            .state
            .get_match(&join.match_id)
            .ok_or_else(|| Status::not_found(format!("match {} not found", join.match_id)))?;
        let sender = handle.join[side.index()]
            .as_ref()
            .ok_or_else(|| {
                Status::already_exists(format!(
                    "the {} side of match {} is not remote",
                    side.name(),
                    join.match_id
                ))
            })?
            .take()
            .ok_or_else(|| {
                Status::already_exists(format!(
                    "the {} side of match {} has already joined",
                    side.name(),
                    join.match_id
                ))
            })?;
        let (out_tx, out_rx) = mpsc::channel(16);
        let (act_tx, act_rx) = mpsc::channel(16);
        // Forwards the client's actions; dropping `act_tx` tells the match it disconnected.
        tokio::spawn(async move {
            while let Ok(Some(m)) = inbound.message().await {
                if let Some(pb::client_message::Msg::Actions(a)) = m.msg {
                    if act_tx.send(a).await.is_err() {
                        break;
                    }
                }
            }
        });
        let conn = Conn {
            tx: out_tx,
            actions: act_rx,
            agent_name: if join.agent_name.is_empty() {
                format!("remote-{}", side.name())
            } else {
                join.agent_name
            },
        };
        sender
            .send(conn)
            .map_err(|_| Status::failed_precondition("the match no longer accepts joins"))?;
        Ok(Response::new(Box::pin(ReceiverStream::new(out_rx))))
    }
}

/// `"remote"`, `"scripted"`, `"random"` or `"policy:<model id>"` (empty means scripted).
fn parse_agent(s: &str) -> Result<Agent, Status> {
    match s {
        "" | "scripted" => Ok(Agent::Scripted),
        "random" => Ok(Agent::Random),
        "remote" => Ok(Agent::Remote),
        _ => match s.strip_prefix("policy:") {
            Some(id) if !id.is_empty() => Ok(Agent::Policy(id.to_owned())),
            _ => Err(Status::invalid_argument(format!("unknown agent {s:?}"))),
        },
    }
}

/// Refuses a client whose protocol or observation layout differs from this build.
fn check_versions(join: &pb::Join, side: MatchSide) -> Result<(), Status> {
    if join.protocol != PROTOCOL_VERSION {
        return Err(Status::failed_precondition(format!(
            "protocol {} (client) != {PROTOCOL_VERSION} (server)",
            join.protocol
        )));
    }
    if join.obs_version != OBS_VERSION {
        return Err(Status::failed_precondition(format!(
            "obs_version {} (client) != {OBS_VERSION} (server)",
            join.obs_version
        )));
    }
    if join.obs_dim as usize != side.obs_dim() {
        return Err(Status::failed_precondition(format!(
            "{} obs_dim {} (client) != {} (server)",
            side.name(),
            join.obs_dim,
            side.obs_dim()
        )));
    }
    Ok(())
}

/// Requires `authorization: Bearer <ZAI_AGENT_TOKEN>` on every call when a token is set.
#[derive(Clone)]
pub struct TokenCheck(Option<Arc<str>>);

impl tonic::service::Interceptor for TokenCheck {
    fn call(&mut self, request: Request<()>) -> Result<Request<()>, Status> {
        let Some(token) = &self.0 else {
            return Ok(request);
        };
        let given = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "));
        match given {
            Some(t) if t.as_bytes() == token.as_bytes() => Ok(request),
            _ => Err(Status::unauthenticated("missing or wrong agent token")),
        }
    }
}

pub type AgentServer =
    tonic::service::interceptor::InterceptedService<AgentServiceServer<AgentApi>, TokenCheck>;

/// The gRPC service over the shared server state, served next to the HTTP router.
pub fn grpc_service(state: Arc<AppState>) -> AgentServer {
    let token = TokenCheck(state.agent_token.as_deref().map(Arc::from));
    AgentServiceServer::with_interceptor(AgentApi { state }, token)
}
