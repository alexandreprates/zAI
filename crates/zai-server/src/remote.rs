//! Remote sides of a live match (docs/ARCHITECTURE.md 4.9.3): a gRPC client plays the side
//! through a `Play` stream. The match task owns the connection; the gRPC handler hands it over
//! through the side's `JoinSlot` when the client joins.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tokio::sync::{mpsc, oneshot};
use tonic::Status;
use zai_sim::env::{INTERACT_ACTIONS, INTERACT_NONE, MOVE_ACTIONS, STAY};
use zai_sim::obs::{write_human_obs, write_zombie_obs};
use zai_sim::{Env, HumanAction, HUMAN_OBS_DIM, ZOMBIE_OBS_DIM};

use crate::grpc::pb;

/// Messages to the client; an `Err` ends its stream with that status.
pub type ToClient = mpsc::Sender<Result<pb::ServerMessage, Status>>;

/// Timeouts and limits of remote sides (env `ZAI_REMOTE_JOIN_TIMEOUT_S`,
/// `ZAI_REMOTE_TIMEOUT_S`, `ZAI_MAX_REMOTE_MATCHES`).
#[derive(Clone, Copy, Debug)]
pub struct RemoteSettings {
    pub join_timeout: Duration,
    pub action_timeout: Duration,
    pub max_matches: usize,
}

impl Default for RemoteSettings {
    fn default() -> Self {
        Self {
            join_timeout: Duration::from_secs(60),
            action_timeout: Duration::from_secs(30),
            max_matches: 8,
        }
    }
}

impl RemoteSettings {
    pub fn from_env() -> Self {
        let d = Self::default();
        let secs = |name: &str, default: Duration| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|s| *s > 0.0)
                .map_or(default, Duration::from_secs_f64)
        };
        Self {
            join_timeout: secs("ZAI_REMOTE_JOIN_TIMEOUT_S", d.join_timeout),
            action_timeout: secs("ZAI_REMOTE_TIMEOUT_S", d.action_timeout),
            max_matches: std::env::var("ZAI_MAX_REMOTE_MATCHES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.max_matches),
        }
    }
}

/// Side of a match, as in the protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatchSide {
    Human,
    Zombies,
}

impl MatchSide {
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            MatchSide::Human => "human",
            MatchSide::Zombies => "zombies",
        }
    }

    pub fn obs_dim(self) -> usize {
        match self {
            MatchSide::Human => HUMAN_OBS_DIM,
            MatchSide::Zombies => ZOMBIE_OBS_DIM,
        }
    }

    pub fn proto(self) -> pb::Side {
        match self {
            MatchSide::Human => pb::Side::Human,
            MatchSide::Zombies => pb::Side::Zombies,
        }
    }

    pub fn from_proto(side: i32) -> Option<Self> {
        match pb::Side::try_from(side).ok()? {
            pb::Side::Human => Some(MatchSide::Human),
            pb::Side::Zombies => Some(MatchSide::Zombies),
            pb::Side::Unspecified => None,
        }
    }
}

/// State of a remote side, shown in `MatchInfo.remote`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RemoteState {
    Waiting,
    Joined,
    Left,
}

/// A joined client: messages to it, and the `Actions` it sends.
pub struct Conn {
    pub tx: ToClient,
    pub actions: mpsc::Receiver<pb::Actions>,
    pub agent_name: String,
}

/// Where a client hands its connection to the match task; taken by the first join.
#[derive(Clone)]
pub struct JoinSlot(Arc<Mutex<Option<oneshot::Sender<Conn>>>>);

impl JoinSlot {
    pub fn new() -> (Self, oneshot::Receiver<Conn>) {
        let (tx, rx) = oneshot::channel();
        (Self(Arc::new(Mutex::new(Some(tx)))), rx)
    }

    /// The sender of the first caller; `None` once the side has been joined.
    pub fn take(&self) -> Option<oneshot::Sender<Conn>> {
        self.0.lock().expect("join slot lock").take()
    }
}

/// Why a remote side ended the match.
pub struct RemoteError {
    pub reason: String,
    /// Status for the offending client's stream (`None`: it is gone).
    pub status: Option<Status>,
}

impl RemoteError {
    fn gone(reason: String) -> Self {
        Self {
            reason,
            status: None,
        }
    }
}

/// One remote side inside the match task.
pub struct RemoteSide {
    pub side: MatchSide,
    pub conn: Conn,
    obs: Vec<f32>,
    /// Last applied actions (realtime fallback): human `[movement, interact]`, zombies `[N]`.
    last: Vec<u8>,
    /// Realtime: actions that arrived after their deadline.
    pub late: u32,
}

impl RemoteSide {
    pub fn new(side: MatchSide, conn: Conn) -> Self {
        Self {
            side,
            conn,
            obs: Vec::new(),
            last: Vec::new(),
            late: 0,
        }
    }

    /// Starts an episode: the realtime fallback becomes "stay, no interaction".
    pub fn start_episode(&mut self, num_zombies: usize) {
        self.last = match self.side {
            MatchSide::Human => vec![STAY, INTERACT_NONE],
            MatchSide::Zombies => vec![STAY; num_zombies],
        };
    }

    /// The fallback of a missed deadline: the last applied movement, no interaction.
    fn fallback(&self) -> Vec<u8> {
        match self.side {
            MatchSide::Human => vec![self.last[0], INTERACT_NONE],
            MatchSide::Zombies => self.last.clone(),
        }
    }

    /// Realtime: the actions of `(episode, tick)` if they arrive before `deadline`, else the
    /// fallback. Actions for an older tick arrived late: they are dropped and counted.
    pub async fn receive_by(
        &mut self,
        episode: u32,
        tick: u32,
        num_zombies: usize,
        deadline: tokio::time::Instant,
    ) -> Result<Vec<u8>, RemoteError> {
        let name = self.side.name();
        loop {
            let a = match tokio::time::timeout_at(deadline, self.conn.actions.recv()).await {
                Err(_) => {
                    let a = self.fallback();
                    self.last.clone_from(&a);
                    return Ok(a);
                }
                Ok(None) => return Err(RemoteError::gone(format!("remote {name} disconnected"))),
                Ok(Some(a)) => a,
            };
            if (a.episode, a.tick) < (episode, tick) {
                self.late += 1;
                continue;
            }
            if (a.episode, a.tick) > (episode, tick) {
                return Err(invalid(format!(
                    "remote {name}: actions for episode {} tick {} before the observation",
                    a.episode, a.tick
                )));
            }
            validate(self.side, &a.actions, num_zombies).map_err(invalid)?;
            self.last.clone_from(&a.actions);
            return Ok(a.actions);
        }
    }

    pub async fn send(&self, msg: pb::server_message::Msg) -> Result<(), RemoteError> {
        self.conn
            .tx
            .send(Ok(pb::ServerMessage { msg: Some(msg) }))
            .await
            .map_err(|_| RemoteError::gone(format!("remote {} disconnected", self.side.name())))
    }

    /// Sends the observation of the current decision; `reward` is per agent, of the step that
    /// led here.
    pub async fn observe(
        &mut self,
        env: &Env,
        episode: u32,
        reward: Vec<f32>,
        deadline_ms: u32,
    ) -> Result<(), RemoteError> {
        let dim = self.side.obs_dim();
        let (n, mask) = match self.side {
            MatchSide::Human => {
                self.obs.resize(dim, 0.0);
                write_human_obs(env, &mut self.obs);
                let mask = env.human_action_mask().iter().map(|&m| m as u8).collect();
                (1, mask)
            }
            MatchSide::Zombies => {
                let n = env.zombies().len();
                self.obs.resize(n * dim, 0.0);
                for (i, chunk) in self.obs.chunks_mut(dim).enumerate() {
                    write_zombie_obs(env, i, chunk);
                }
                (n, Vec::new())
            }
        };
        let bytes = self.obs.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.send(pb::server_message::Msg::Obs(pb::Observation {
            episode,
            tick: env.decisions(),
            num_agents: n as u32,
            obs_dim: dim as u32,
            obs: bytes,
            mask,
            reward,
            state_hash: env.state_hash(),
            time: env.time(),
            deadline_ms,
        }))
        .await
    }

    /// Waits for the client's actions of `(episode, tick)` (lockstep); older ones are dropped.
    pub async fn receive(
        &mut self,
        episode: u32,
        tick: u32,
        num_zombies: usize,
        timeout: Duration,
    ) -> Result<Vec<u8>, RemoteError> {
        let name = self.side.name();
        loop {
            let a = match tokio::time::timeout(timeout, self.conn.actions.recv()).await {
                Err(_) => {
                    return Err(RemoteError {
                        reason: format!(
                            "remote {name}: no actions for tick {tick} within {:.0} s",
                            timeout.as_secs_f32()
                        ),
                        status: Some(Status::deadline_exceeded("no actions within the timeout")),
                    })
                }
                Ok(None) => return Err(RemoteError::gone(format!("remote {name} disconnected"))),
                Ok(Some(a)) => a,
            };
            if (a.episode, a.tick) < (episode, tick) {
                continue;
            }
            if (a.episode, a.tick) > (episode, tick) {
                return Err(invalid(format!(
                    "remote {name}: actions for episode {} tick {} before the observation \
                     (expected episode {episode} tick {tick})",
                    a.episode, a.tick
                )));
            }
            validate(self.side, &a.actions, num_zombies).map_err(invalid)?;
            return Ok(a.actions);
        }
    }
}

fn invalid(reason: String) -> RemoteError {
    RemoteError {
        status: Some(Status::invalid_argument(reason.clone())),
        reason,
    }
}

/// Checks the length and range of one decision's action bytes.
pub fn validate(side: MatchSide, actions: &[u8], num_zombies: usize) -> Result<(), String> {
    let name = side.name();
    let (len, bad) = match side {
        MatchSide::Human => (
            2,
            actions.len() == 2
                && (actions[0] as usize >= MOVE_ACTIONS || actions[1] as usize >= INTERACT_ACTIONS),
        ),
        MatchSide::Zombies => (
            num_zombies,
            actions.iter().any(|&a| a as usize >= MOVE_ACTIONS),
        ),
    };
    if actions.len() != len {
        return Err(format!(
            "remote {name}: {} action bytes, expected {len}",
            actions.len()
        ));
    }
    if bad {
        return Err(format!("remote {name}: action out of range: {actions:?}"));
    }
    Ok(())
}

/// The human action of validated bytes.
pub fn human_action(bytes: &[u8]) -> HumanAction {
    HumanAction {
        movement: bytes[0],
        interact: bytes[1],
    }
}
