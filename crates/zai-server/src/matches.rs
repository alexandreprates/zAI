//! Live matches between bots, run as tokio tasks in real time (speed x0.25-x16, or unthrottled
//! with speed 0). Every finished episode is stored as a replay.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use zai_sim::bots::{BotKind, HumanBot, ZombieBot};
use zai_sim::{Config, Env, Outcome};

use crate::frames;
use crate::replay::{Replay, ReplayStore};

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
    pub human: BotKind,
    #[serde(default = "scripted")]
    pub zombies: BotKind,
    #[serde(default = "one")]
    pub speed: f32,
    /// Start a new episode (next seed) whenever one ends.
    #[serde(default)]
    pub auto_restart: bool,
}

fn scripted() -> BotKind {
    BotKind::Scripted
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
    Running,
    Finished,
    Stopped,
}

#[derive(Clone, Debug, Serialize)]
pub struct MatchInfo {
    pub id: String,
    pub created_at: u64,
    pub human: BotKind,
    pub zombies: BotKind,
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
}

impl MatchHandle {
    pub fn control(&self, c: Control) {
        // The task may have finished; controlling a finished match is a no-op.
        let _ = self.control.send(c);
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
    now: u64,
    replays: ReplayStore,
) -> MatchHandle {
    let env = Env::new(cfg.clone(), episode_seed(base_seed, 0)).expect("validated config");
    let speed = clamp_speed(req.speed);
    let info = Arc::new(Mutex::new(MatchInfo {
        id: id.clone(),
        created_at: now,
        human: req.human,
        zombies: req.zombies,
        base_seed,
        seed: env.seed(),
        episode: 0,
        tick: 0,
        time: 0.0,
        speed,
        paused: false,
        auto_restart: req.auto_restart,
        status: MatchStatus::Running,
        outcome: "running".into(),
        replays: Vec::new(),
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
    };
    let task = MatchTask {
        id,
        cfg,
        base_seed,
        human_kind: req.human,
        zombie_kind: req.zombies,
        speed,
        paused: false,
        stopped: false,
        handle: handle.clone(),
        replays,
        rx,
    };
    tokio::spawn(task.run());
    handle
}

struct MatchTask {
    id: String,
    cfg: Config,
    base_seed: u64,
    human_kind: BotKind,
    zombie_kind: BotKind,
    speed: f32,
    paused: bool,
    stopped: bool,
    handle: MatchHandle,
    replays: ReplayStore,
    rx: mpsc::UnboundedReceiver<Control>,
}

impl MatchTask {
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
        let mut episode = 0u32;
        loop {
            let seed = episode_seed(self.base_seed, episode);
            let mut env = Env::new(self.cfg.clone(), seed).expect("validated config");
            let mut human = HumanBot::new(self.human_kind, seed ^ 0x4855_4d41_4e00_0000);
            let mut zombies = ZombieBot::new(self.zombie_kind, seed ^ 0x5a4f_4d42_4945_0000);
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

            let decision_dt = self.cfg.decision_dt();
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
                if self.speed > 0.0 {
                    let delay = Duration::from_secs_f32(decision_dt / self.speed);
                    tokio::select! {
                        _ = tokio::time::sleep(delay) => {}
                        c = self.rx.recv() => {
                            match c {
                                Some(c) => self.apply(c),
                                None => self.stopped = true,
                            }
                            continue;
                        }
                    }
                } else {
                    while let Ok(c) = self.rx.try_recv() {
                        self.apply(c);
                    }
                    tokio::task::yield_now().await;
                    if self.paused || self.stopped {
                        continue;
                    }
                }

                let a = human.act(&env);
                let za = zombies.act(&env);
                actions.push(Replay::encode_step(a, &za));
                env.step(a, &za);
                {
                    let mut info = self.handle.info.lock().expect("match info lock");
                    info.tick = env.decisions();
                    info.time = env.time();
                    info.outcome = frames::outcome_name(env.outcome()).into();
                }
                self.publish(frames::snapshot(&env, self.speed, self.paused), false);
            }

            if !actions.is_empty() && env.outcome() != Outcome::Running {
                let replay = Replay {
                    id: format!("{}-e{episode}", self.id),
                    match_id: self.id.clone(),
                    episode,
                    created_at: crate::unix_now(),
                    config: self.cfg.clone(),
                    seed,
                    human: self.human_kind.into(),
                    zombies: self.zombie_kind.into(),
                    actions,
                    outcome: frames::outcome_name(env.outcome()).into(),
                    final_hash: format!("{:016x}", env.state_hash()),
                };
                let store = self.replays.clone();
                let id = replay.id.clone();
                match tokio::task::spawn_blocking(move || store.save(&replay)).await {
                    Ok(Ok(())) => {
                        let mut info = self.handle.info.lock().expect("match info lock");
                        info.replays.push(id);
                    }
                    Ok(Err(e)) => tracing::error!("failed to save replay {id}: {e}"),
                    Err(e) => tracing::error!("replay save task failed for {id}: {e}"),
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
        let mut info = self.handle.info.lock().expect("match info lock");
        info.status = if self.stopped {
            MatchStatus::Stopped
        } else {
            MatchStatus::Finished
        };
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
