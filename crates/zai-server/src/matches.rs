//! Live matches between bots, trained policies (`policy.rs`) and remote gRPC agents
//! (`remote.rs`), run as tokio tasks in real time (speed x0.25-x16, or unthrottled with speed
//! 0). Every finished episode is stored as a replay.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use zai_sim::bots::{BotKind, HumanBot, ZombieBot};
use zai_sim::{Config, Env, HumanAction, Outcome};

use crate::frames;
use crate::grpc::pb::{self, server_message::Msg};
use crate::policy::{LoadedModel, PolicyAgent};
use crate::remote::{
    human_action, Conn, JoinSlot, MatchSide, RemoteError, RemoteSettings, RemoteSide, RemoteState,
};
use crate::replay::{Replay, ReplayStore};

/// Who plays one side of a match: a built-in bot, a trained model from the model store, or a
/// remote gRPC client. JSON: `"scripted"`, `"random"`, `"remote"` or `{"policy": "<model id>"}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Agent {
    Scripted,
    Random,
    Remote,
    Policy(String),
}

impl Agent {
    fn bot(&self) -> Option<BotKind> {
        match self {
            Agent::Scripted => Some(BotKind::Scripted),
            Agent::Random => Some(BotKind::Random),
            Agent::Policy(_) | Agent::Remote => None,
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, Agent::Remote)
    }

    pub fn model_id(&self) -> Option<&str> {
        match self {
            Agent::Policy(id) => Some(id),
            _ => None,
        }
    }
}

/// Models loaded for a match (`None` for bot sides).
#[derive(Clone, Default)]
pub struct MatchModels {
    pub human: Option<LoadedModel>,
    pub zombies: Option<LoadedModel>,
}

enum HumanDriver {
    Bot(HumanBot),
    Policy(PolicyAgent),
}

impl HumanDriver {
    fn act(&mut self, env: &Env) -> Result<HumanAction, String> {
        match self {
            HumanDriver::Bot(b) => Ok(b.act(env)),
            HumanDriver::Policy(p) => p.act_human(env),
        }
    }
}

enum ZombieDriver {
    Bot(ZombieBot),
    Policy(PolicyAgent),
}

impl ZombieDriver {
    fn act(&mut self, env: &Env) -> Result<Vec<u8>, String> {
        match self {
            ZombieDriver::Bot(b) => Ok(b.act(env)),
            ZombieDriver::Policy(p) => p.act_zombies(env),
        }
    }
}

pub const MAX_SPEED: f32 = 16.0;
pub const MIN_SPEED: f32 = 0.25;

/// Body of `POST /api/matches`.
#[derive(Clone, Debug, Deserialize)]
pub struct MatchRequest {
    /// Partial config merged over the server's base config.
    #[serde(default)]
    pub config: serde_json::Value,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default = "scripted")]
    pub human: Agent,
    #[serde(default = "scripted")]
    pub zombies: Agent,
    /// Trained policies take their most likely action instead of sampling.
    #[serde(default)]
    pub deterministic: bool,
    #[serde(default = "one")]
    pub speed: f32,
    /// Start a new episode (next seed) whenever one ends.
    #[serde(default)]
    pub auto_restart: bool,
}

fn scripted() -> Agent {
    Agent::Scripted
}

fn one() -> f32 {
    1.0
}

/// Clamps a requested speed: 0 means unthrottled, otherwise x0.25-x16.
pub fn clamp_speed(speed: f32) -> f32 {
    if speed <= 0.0 || !speed.is_finite() {
        0.0
    } else {
        speed.clamp(MIN_SPEED, MAX_SPEED)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchStatus {
    /// Remote sides have not joined yet.
    Waiting,
    Running,
    Finished,
    Stopped,
    /// A policy failed to run or a remote side failed; `outcome` holds the reason.
    Failed,
}

/// Per side: the state of a remote side (`None` for other sides), or a remote agent's name.
#[derive(Clone, Debug, Serialize)]
pub struct PerSide<T> {
    pub human: Option<T>,
    pub zombies: Option<T>,
}

impl<T> Default for PerSide<T> {
    fn default() -> Self {
        Self {
            human: None,
            zombies: None,
        }
    }
}

impl<T> PerSide<T> {
    pub fn get_mut(&mut self, side: MatchSide) -> &mut Option<T> {
        match side {
            MatchSide::Human => &mut self.human,
            MatchSide::Zombies => &mut self.zombies,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MatchInfo {
    pub id: String,
    pub created_at: u64,
    pub human: Agent,
    pub zombies: Agent,
    pub deterministic: bool,
    pub base_seed: u64,
    pub seed: u64,
    pub episode: u32,
    pub tick: u32,
    pub time: f32,
    pub speed: f32,
    pub paused: bool,
    pub auto_restart: bool,
    pub status: MatchStatus,
    pub outcome: String,
    pub replays: Vec<String>,
    pub remote: PerSide<RemoteState>,
    pub agent_names: PerSide<String>,
    /// Realtime mode: actions that arrived after their deadline (0 in lockstep).
    pub late_actions: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Control {
    Speed(f32),
    Pause(bool),
    Stop,
}

/// Shared handle to a running match.
#[derive(Clone)]
pub struct MatchHandle {
    pub info: Arc<Mutex<MatchInfo>>,
    control: mpsc::UnboundedSender<Control>,
    pub frames: broadcast::Sender<Bytes>,
    /// Latest `(init, snapshot)` frames, sent first to viewers that join mid-episode.
    pub latest: Arc<Mutex<(Bytes, Bytes)>>,
    /// Join slots of the remote sides, indexed by `MatchSide::index`.
    pub join: [Option<JoinSlot>; 2],
    pub config: Arc<Config>,
}

impl MatchHandle {
    pub fn control(&self, c: Control) {
        // The task may have finished; controlling a finished match is a no-op.
        let _ = self.control.send(c);
    }

    /// Whether the match has a remote side and is still waiting or running.
    pub fn is_live_remote(&self) -> bool {
        let info = self.info.lock().expect("match info lock");
        (info.human.is_remote() || info.zombies.is_remote())
            && matches!(info.status, MatchStatus::Waiting | MatchStatus::Running)
    }
}

/// Seed of episode `episode` of a match.
pub fn episode_seed(base: u64, episode: u32) -> u64 {
    base.wrapping_add(episode as u64)
}

/// Creates the match and spawns its task.
pub fn start(
    id: String,
    cfg: Config,
    req: &MatchRequest,
    base_seed: u64,
    replays: ReplayStore,
    models: MatchModels,
    remote: RemoteSettings,
) -> MatchHandle {
    let now = crate::unix_now();
    let env = Env::new(cfg.clone(), episode_seed(base_seed, 0)).expect("validated config");
    let speed = clamp_speed(req.speed);
    let mut join = [None, None];
    let mut join_rx = [None, None];
    let mut remote_state = PerSide::default();
    for (side, agent) in [
        (MatchSide::Human, &req.human),
        (MatchSide::Zombies, &req.zombies),
    ] {
        if agent.is_remote() {
            let (slot, rx) = JoinSlot::new();
            join[side.index()] = Some(slot);
            join_rx[side.index()] = Some(rx);
            *remote_state.get_mut(side) = Some(RemoteState::Waiting);
        }
    }
    let waiting = join_rx.iter().any(Option::is_some);
    let info = Arc::new(Mutex::new(MatchInfo {
        id: id.clone(),
        created_at: now,
        human: req.human.clone(),
        zombies: req.zombies.clone(),
        deterministic: req.deterministic,
        base_seed,
        seed: env.seed(),
        episode: 0,
        tick: 0,
        time: 0.0,
        speed,
        paused: false,
        auto_restart: req.auto_restart,
        status: if waiting {
            MatchStatus::Waiting
        } else {
            MatchStatus::Running
        },
        outcome: if waiting { "waiting" } else { "running" }.into(),
        replays: Vec::new(),
        remote: remote_state,
        agent_names: PerSide::default(),
        late_actions: 0,
    }));
    let (control, rx) = mpsc::unbounded_channel();
    let (frames_tx, _) = broadcast::channel(256);
    let latest = Arc::new(Mutex::new((
        frames::init(&env, "live", &id, 0, None),
        frames::snapshot(&env, speed, false),
    )));
    let handle = MatchHandle {
        info,
        control,
        frames: frames_tx,
        latest,
        join,
        config: Arc::new(cfg.clone()),
    };
    let task = MatchTask {
        id,
        cfg,
        base_seed,
        human: req.human.clone(),
        zombies: req.zombies.clone(),
        deterministic: req.deterministic,
        models,
        speed,
        paused: false,
        stopped: false,
        handle: handle.clone(),
        replays,
        rx,
        join_rx,
        remote,
    };
    tokio::spawn(task.run());
    handle
}

struct MatchTask {
    id: String,
    cfg: Config,
    base_seed: u64,
    human: Agent,
    zombies: Agent,
    deterministic: bool,
    models: MatchModels,
    speed: f32,
    paused: bool,
    stopped: bool,
    handle: MatchHandle,
    replays: ReplayStore,
    rx: mpsc::UnboundedReceiver<Control>,
    join_rx: [Option<tokio::sync::oneshot::Receiver<Conn>>; 2],
    remote: RemoteSettings,
}

/// Remote sides of a running match, indexed by `MatchSide::index`.
type Remotes = [Option<RemoteSide>; 2];

impl MatchTask {
    /// In-process drivers of one episode; `None` for remote sides.
    fn drivers(&self, seed: u64) -> (Option<HumanDriver>, Option<ZombieDriver>) {
        let human_seed = seed ^ 0x4855_4d41_4e00_0000;
        let zombie_seed = seed ^ 0x5a4f_4d42_4945_0000;
        let human = match (self.human.bot(), &self.models.human) {
            _ if self.human.is_remote() => None,
            (Some(kind), _) => Some(HumanDriver::Bot(HumanBot::new(kind, human_seed))),
            (None, Some(m)) => Some(HumanDriver::Policy(PolicyAgent::new(
                m.clone(),
                human_seed,
                self.deterministic,
            ))),
            (None, None) => unreachable!("policy sides are loaded before the match starts"),
        };
        let zombies = match (self.zombies.bot(), &self.models.zombies) {
            _ if self.zombies.is_remote() => None,
            (Some(kind), _) => Some(ZombieDriver::Bot(ZombieBot::new(kind, zombie_seed))),
            (None, Some(m)) => Some(ZombieDriver::Policy(PolicyAgent::new(
                m.clone(),
                zombie_seed,
                self.deterministic,
            ))),
            (None, None) => unreachable!("policy sides are loaded before the match starts"),
        };
        (human, zombies)
    }

    fn set_remote_state(&self, side: MatchSide, state: RemoteState) {
        let mut info = self.handle.info.lock().expect("match info lock");
        *info.remote.get_mut(side) = Some(state);
    }

    /// Sends `MatchEnd` to every joined remote side and marks them as left.
    async fn end_remotes(&self, remotes: &Remotes, status: &str, reason: &str) {
        for r in remotes.iter().flatten() {
            let _ = r
                .send(Msg::MatchEnd(pb::MatchEnd {
                    status: status.into(),
                    reason: reason.into(),
                }))
                .await;
            self.set_remote_state(r.side, RemoteState::Left);
        }
    }

    /// Fails the match: the offending client gets its status, the others `MatchEnd{failed}`.
    async fn fail(&self, remotes: &Remotes, offender: Option<MatchSide>, err: RemoteError) {
        tracing::warn!("match {}: {}", self.id, err.reason);
        {
            let mut info = self.handle.info.lock().expect("match info lock");
            info.status = MatchStatus::Failed;
            info.outcome = err.reason.clone();
        }
        for r in remotes.iter().flatten() {
            if Some(r.side) == offender {
                if let Some(status) = err.status.clone() {
                    let _ = r.conn.tx.send(Err(status)).await;
                }
                self.set_remote_state(r.side, RemoteState::Left);
            }
        }
        for r in remotes.iter().flatten() {
            if Some(r.side) != offender {
                let _ = r
                    .send(Msg::MatchEnd(pb::MatchEnd {
                        status: "failed".into(),
                        reason: err.reason.clone(),
                    }))
                    .await;
                self.set_remote_state(r.side, RemoteState::Left);
            }
        }
    }

    /// Waits for every remote side to join (`None`: the match was stopped or failed).
    async fn join_remotes(&mut self) -> Option<Remotes> {
        let mut remotes: Remotes = [None, None];
        let deadline = tokio::time::Instant::now() + self.remote.join_timeout;
        for side in [MatchSide::Human, MatchSide::Zombies] {
            let Some(mut rx) = self.join_rx[side.index()].take() else {
                continue;
            };
            let conn = loop {
                tokio::select! {
                    r = &mut rx => break r.ok(),
                    _ = tokio::time::sleep_until(deadline) => break None,
                    c = self.rx.recv() => {
                        match c {
                            Some(c) => self.apply(c),
                            None => self.stopped = true,
                        }
                        if self.stopped {
                            self.end_remotes(&remotes, "stopped", "stopped while waiting").await;
                            let mut info = self.handle.info.lock().expect("match info lock");
                            info.status = MatchStatus::Stopped;
                            info.outcome = "stopped".into();
                            return None;
                        }
                    }
                }
            };
            let Some(conn) = conn else {
                let err = RemoteError {
                    reason: format!("remote {} did not join", side.name()),
                    status: None,
                };
                self.fail(&remotes, None, err).await;
                return None;
            };
            {
                let mut info = self.handle.info.lock().expect("match info lock");
                *info.agent_names.get_mut(side) = Some(conn.agent_name.clone());
                *info.remote.get_mut(side) = Some(RemoteState::Joined);
            }
            let r = RemoteSide::new(side, conn);
            let joined = Msg::Joined(pb::Joined {
                match_id: self.id.clone(),
                side: side.proto() as i32,
                realtime: false,
                decision_dt: self.cfg.decision_dt(),
            });
            if let Err(e) = r.send(joined).await {
                remotes[side.index()] = Some(r);
                self.fail(&remotes, Some(side), e).await;
                return None;
            }
            remotes[side.index()] = Some(r);
        }
        let mut info = self.handle.info.lock().expect("match info lock");
        info.status = MatchStatus::Running;
        info.outcome = "running".into();
        Some(remotes)
    }

    fn apply(&mut self, c: Control) {
        match c {
            Control::Speed(s) => self.speed = clamp_speed(s),
            Control::Pause(p) => self.paused = p,
            Control::Stop => self.stopped = true,
        }
        let mut info = self.handle.info.lock().expect("match info lock");
        info.speed = self.speed;
        info.paused = self.paused;
    }

    fn publish(&self, frame: Bytes, is_init: bool) {
        {
            let mut latest = self.handle.latest.lock().expect("latest frames lock");
            if is_init {
                latest.0 = frame.clone();
            } else {
                latest.1 = frame.clone();
            }
        }
        // No receivers is fine: nobody is watching.
        let _ = self.handle.frames.send(frame);
    }

    async fn run(mut self) {
        let mut remotes: Remotes = [None, None];
        if self.join_rx.iter().any(Option::is_some) {
            match self.join_remotes().await {
                Some(r) => remotes = r,
                None => return,
            }
        }
        let config_json = serde_json::to_string(&self.cfg).unwrap_or_default();
        let mut episode = 0u32;
        loop {
            let seed = episode_seed(self.base_seed, episode);
            let mut env = Env::new(self.cfg.clone(), seed).expect("validated config");
            let (mut human, mut zombies) = self.drivers(seed);
            let mut actions = Vec::new();
            {
                let mut info = self.handle.info.lock().expect("match info lock");
                info.episode = episode;
                info.seed = seed;
                info.tick = 0;
                info.time = 0.0;
                info.outcome = "running".into();
            }
            self.publish(frames::init(&env, "live", &self.id, episode, None), true);
            self.publish(frames::snapshot(&env, self.speed, self.paused), false);
            let n = env.zombies().len();
            for r in remotes.iter().flatten() {
                let start = Msg::Start(pb::EpisodeStart {
                    episode,
                    seed,
                    num_agents: if r.side == MatchSide::Human {
                        1
                    } else {
                        n as u32
                    },
                    config_json: config_json.clone(),
                });
                if let Err(e) = r.send(start).await {
                    let side = r.side;
                    self.fail(&remotes, Some(side), e).await;
                    return;
                }
            }
            let (mut human_reward, mut zombie_reward) = (0.0f32, vec![0.0f32; n]);

            let decision_dt = self.cfg.decision_dt();
            let has_remote = remotes.iter().any(Option::is_some);
            while env.outcome() == Outcome::Running && !self.stopped {
                if self.paused {
                    match self.rx.recv().await {
                        Some(c) => {
                            self.apply(c);
                            self.publish(frames::snapshot(&env, self.speed, self.paused), false);
                        }
                        None => self.stopped = true,
                    }
                    continue;
                }
                while let Ok(c) = self.rx.try_recv() {
                    self.apply(c);
                }
                if self.paused || self.stopped {
                    continue;
                }

                // Phase 1: observations to the remote sides, in-process sides act right away.
                let tick = env.decisions();
                for r in remotes.iter_mut().flatten() {
                    let reward = match r.side {
                        MatchSide::Human => vec![human_reward],
                        MatchSide::Zombies => zombie_reward.clone(),
                    };
                    if let Err(e) = r.observe(&env, episode, reward).await {
                        let side = r.side;
                        self.fail(&remotes, Some(side), e).await;
                        return;
                    }
                }
                let local = human
                    .as_mut()
                    .map(|h| h.act(&env))
                    .transpose()
                    .and_then(|a| Ok((a, zombies.as_mut().map(|z| z.act(&env)).transpose()?)));
                let (local_h, local_z) = match local {
                    Ok(a) => a,
                    Err(e) => {
                        let err = RemoteError {
                            reason: format!("policy failed: {e}"),
                            status: None,
                        };
                        self.fail(&remotes, None, err).await;
                        return;
                    }
                };

                // Phase 2: wait for the remote actions and the throttle together.
                let delay =
                    (self.speed > 0.0).then(|| Duration::from_secs_f32(decision_dt / self.speed));
                let timeout = self.remote.action_timeout;
                let received = {
                    let [rh, rz] = &mut remotes;
                    let wait = async {
                        let throttle = async {
                            if let Some(d) = delay {
                                tokio::time::sleep(d).await;
                            }
                        };
                        let recv_h = async {
                            match rh.as_mut() {
                                Some(r) => r.receive(episode, tick, n, timeout).await.map(Some),
                                None => Ok(None),
                            }
                        };
                        let recv_z = async {
                            match rz.as_mut() {
                                Some(r) => r.receive(episode, tick, n, timeout).await.map(Some),
                                None => Ok(None),
                            }
                        };
                        let (_, h, z) = tokio::join!(throttle, recv_h, recv_z);
                        (h, z)
                    };
                    tokio::pin!(wait);
                    loop {
                        tokio::select! {
                            out = &mut wait => break Some(out),
                            c = self.rx.recv() => {
                                match c {
                                    Some(c) => self.apply(c),
                                    None => self.stopped = true,
                                }
                                if self.stopped {
                                    break None;
                                }
                            }
                        }
                    }
                };
                let Some((remote_h, remote_z)) = received else {
                    break;
                };
                let remote_h = match remote_h {
                    Ok(a) => a,
                    Err(e) => {
                        self.fail(&remotes, Some(MatchSide::Human), e).await;
                        return;
                    }
                };
                let remote_z = match remote_z {
                    Ok(a) => a,
                    Err(e) => {
                        self.fail(&remotes, Some(MatchSide::Zombies), e).await;
                        return;
                    }
                };
                if delay.is_none() && !has_remote {
                    tokio::task::yield_now().await;
                }
                let a = local_h.unwrap_or_else(|| human_action(&remote_h.expect("remote human")));
                let za = local_z.unwrap_or_else(|| remote_z.expect("remote zombies"));

                actions.push(Replay::encode_step(a, &za));
                human_reward = env.step(a, &za).human_reward;
                zombie_reward.clear();
                zombie_reward.extend_from_slice(env.zombie_rewards());
                {
                    let mut info = self.handle.info.lock().expect("match info lock");
                    info.tick = env.decisions();
                    info.time = env.time();
                    info.outcome = frames::outcome_name(env.outcome()).into();
                }
                self.publish(frames::snapshot(&env, self.speed, self.paused), false);
            }

            let mut replay_id = String::new();
            if !actions.is_empty() && env.outcome() != Outcome::Running {
                let replay = Replay {
                    id: format!("{}-e{episode}", self.id),
                    match_id: self.id.clone(),
                    episode,
                    created_at: crate::unix_now(),
                    config: self.cfg.clone(),
                    seed,
                    human: (&self.human).into(),
                    zombies: (&self.zombies).into(),
                    human_model: self.side_label(MatchSide::Human),
                    zombie_model: self.side_label(MatchSide::Zombies),
                    actions,
                    outcome: frames::outcome_name(env.outcome()).into(),
                    final_hash: format!("{:016x}", env.state_hash()),
                };
                let store = self.replays.clone();
                let id = replay.id.clone();
                match tokio::task::spawn_blocking(move || store.save(&replay)).await {
                    Ok(Ok(())) => {
                        let mut info = self.handle.info.lock().expect("match info lock");
                        info.replays.push(id.clone());
                        replay_id = id;
                    }
                    Ok(Err(e)) => tracing::error!("failed to save replay {id}: {e}"),
                    Err(e) => tracing::error!("replay save task failed for {id}: {e}"),
                }
            }
            if env.outcome() != Outcome::Running {
                for r in remotes.iter().flatten() {
                    let reward = match r.side {
                        MatchSide::Human => vec![human_reward],
                        MatchSide::Zombies => zombie_reward.clone(),
                    };
                    let end = Msg::End(pb::EpisodeEnd {
                        episode,
                        outcome: frames::outcome_name(env.outcome()).into(),
                        ticks: env.decisions(),
                        reward,
                        state_hash: env.state_hash(),
                        replay_id: replay_id.clone(),
                    });
                    if let Err(e) = r.send(end).await {
                        let side = r.side;
                        self.fail(&remotes, Some(side), e).await;
                        return;
                    }
                }
            }

            let auto_restart = self
                .handle
                .info
                .lock()
                .expect("match info lock")
                .auto_restart;
            if self.stopped || !auto_restart {
                break;
            }
            episode += 1;
        }
        let status = if self.stopped { "stopped" } else { "finished" };
        self.end_remotes(&remotes, status, "").await;
        let mut info = self.handle.info.lock().expect("match info lock");
        info.status = if self.stopped {
            MatchStatus::Stopped
        } else {
            MatchStatus::Finished
        };
    }

    /// Model id of a policy side, agent name of a remote side.
    fn side_label(&self, side: MatchSide) -> Option<String> {
        let agent = match side {
            MatchSide::Human => &self.human,
            MatchSide::Zombies => &self.zombies,
        };
        if agent.is_remote() {
            let info = self.handle.info.lock().expect("match info lock");
            return match side {
                MatchSide::Human => info.agent_names.human.clone(),
                MatchSide::Zombies => info.agent_names.zombies.clone(),
            };
        }
        agent.model_id().map(str::to_owned)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_is_clamped() {
        assert_eq!(clamp_speed(0.0), 0.0);
        assert_eq!(clamp_speed(-3.0), 0.0);
        assert_eq!(clamp_speed(100.0), MAX_SPEED);
        assert_eq!(clamp_speed(0.1), MIN_SPEED);
        assert_eq!(clamp_speed(4.0), 4.0);
    }
}
