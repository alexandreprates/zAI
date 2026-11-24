//! Viewer frames (MessagePack maps) describing a match: one `init` frame with the static map,
//! then one `snapshot` frame per decision step. Live matches and replays use the same frames.

use axum::body::Bytes;
use serde::Serialize;
use zai_sim::env::NoiseKind;
use zai_sim::{Config, Env, Outcome};

#[derive(Serialize)]
struct HouseFrame {
    x: f32,
    y: f32,
    angle: f32,
    /// Wall segments `[ax, ay, bx, by]`.
    walls: Vec<[f32; 4]>,
    /// Door segments `[ax, ay, bx, by]`, front then back.
    doors: Vec<[f32; 4]>,
}

#[derive(Serialize)]
struct InitFrame<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    /// `live` or `replay`.
    source: &'static str,
    id: &'a str,
    episode: u32,
    seed: u64,
    /// Number of decision steps of a replay (absent for live matches).
    total_ticks: Option<u32>,
    decision_dt: f32,
    config: &'a Config,
    map_size: f32,
    houses: Vec<HouseFrame>,
}

#[derive(Serialize)]
struct HumanFrame {
    x: f32,
    y: f32,
    hx: f32,
    hy: f32,
    carried: u32,
    hunger: f32,
}

#[derive(Serialize)]
struct SnapshotFrame {
    #[serde(rename = "type")]
    kind: &'static str,
    tick: u32,
    time: f32,
    speed: f32,
    paused: bool,
    human: HumanFrame,
    /// `[x, y, heading_x, heading_y]` per zombie.
    zombies: Vec<[f32; 4]>,
    /// Bit `2 * house + door` is set when that door is closed.
    doors: u32,
    /// Positions of the active food packages.
    food: Vec<[f32; 2]>,
    /// Noise events of the last step: `[kind (0 walk, 1 pickup, 2 door), x, y, radius]`.
    events: Vec<[f32; 4]>,
    outcome: &'static str,
}

pub fn outcome_name(o: Outcome) -> &'static str {
    match o {
        Outcome::Running => "running",
        Outcome::Caught => "caught",
        Outcome::Starved => "starved",
        Outcome::Truncated => "truncated",
    }
}

fn encode<T: Serialize>(frame: &T) -> Bytes {
    Bytes::from(rmp_serde::to_vec_named(frame).expect("frames serialize to MessagePack"))
}

pub fn init(
    env: &Env,
    source: &'static str,
    id: &str,
    episode: u32,
    total_ticks: Option<u32>,
) -> Bytes {
    let seg = |s: &zai_sim::geom::Segment| [s.a.x, s.a.y, s.b.x, s.b.y];
    let houses = env
        .map()
        .houses
        .iter()
        .map(|h| HouseFrame {
            x: h.center.x,
            y: h.center.y,
            angle: h.angle(),
            walls: h.walls.iter().map(seg).collect(),
            doors: h.doors.iter().map(|d| seg(&d.segment)).collect(),
        })
        .collect();
    encode(&InitFrame {
        kind: "init",
        source,
        id,
        episode,
        seed: env.seed(),
        total_ticks,
        decision_dt: env.config().decision_dt(),
        config: env.config(),
        map_size: env.map().size,
        houses,
    })
}

pub fn snapshot(env: &Env, speed: f32, paused: bool) -> Bytes {
    let h = env.human();
    let mut doors = 0u32;
    for (i, house) in env.map().houses.iter().enumerate() {
        for (k, d) in house.doors.iter().enumerate() {
            if d.closed {
                doors |= 1 << (2 * i + k);
            }
        }
    }
    encode(&SnapshotFrame {
        kind: "snapshot",
        tick: env.decisions(),
        time: env.time(),
        speed,
        paused,
        human: HumanFrame {
            x: h.pos.x,
            y: h.pos.y,
            hx: h.heading.x,
            hy: h.heading.y,
            carried: h.carried,
            hunger: h.hunger_s,
        },
        zombies: env
            .zombies()
            .iter()
            .map(|z| [z.pos.x, z.pos.y, z.heading.x, z.heading.y])
            .collect(),
        doors,
        food: env
            .food()
            .iter()
            .filter(|f| f.active)
            .map(|f| [f.pos.x, f.pos.y])
            .collect(),
        events: env
            .events()
            .iter()
            .map(|e| {
                let kind = match e.kind {
                    NoiseKind::Walk => 0.0,
                    NoiseKind::Pickup => 1.0,
                    NoiseKind::Door => 2.0,
                };
                [kind, e.pos.x, e.pos.y, e.radius]
            })
            .collect(),
        outcome: outcome_name(env.outcome()),
    })
}
