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
    /// Indices into `zombies` / `food` of what the human sees (`Env::visible_zombies` /
    /// `visible_food`, the rule behind its observation), nearest first.
    seen_zombies: Vec<u16>,
    seen_food: Vec<u16>,
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

/// `(seen zombies, seen food)` as indices into the snapshot's `zombies` and active `food`.
pub fn seen(env: &Env) -> (Vec<u16>, Vec<u16>) {
    let zombies = env
        .visible_zombies()
        .iter()
        .map(|&(i, _)| i as u16)
        .collect();
    // The snapshot lists active food only: map env indices to positions in that list.
    let mut slot = vec![u16::MAX; env.food().len()];
    let mut n = 0u16;
    for (i, f) in env.food().iter().enumerate() {
        if f.active {
            slot[i] = n;
            n += 1;
        }
    }
    let food = env.visible_food().iter().map(|&(i, _)| slot[i]).collect();
    (zombies, food)
}

pub fn snapshot(env: &Env, speed: f32, paused: bool) -> Bytes {
    let h = env.human();
    let (seen_zombies, seen_food) = seen(env);
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
        seen_zombies,
        seen_food,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use zai_sim::obs::{write_human_obs, HUMAN_LAYOUT};
    use zai_sim::{HumanAction, HUMAN_OBS_DIM};

    /// Valid rows of the human observation's `zombies` block.
    fn observed_zombies(env: &Env) -> usize {
        let mut obs = vec![0.0; HUMAN_OBS_DIM];
        write_human_obs(env, &mut obs);
        let b = HUMAN_LAYOUT.iter().find(|b| b.name == "zombies").unwrap();
        let valid = b.fields.iter().position(|f| *f == "valid").unwrap();
        (0..b.slots)
            .filter(|s| obs[b.start + s * b.fields.len() + valid] > 0.5)
            .count()
    }

    #[test]
    fn seen_lists_follow_the_human_visibility_and_observation() {
        let cfg = Config {
            map_size_m: 75.0,
            house_count: 3,
            zombie_count: 9,
            food_count: 40,
            ..Config::default()
        };
        let mut seen_any = (false, false);
        for seed in 0..10 {
            let mut env = Env::new(cfg.clone(), seed).unwrap();
            for t in 0..40u8 {
                let frame: serde_json::Value =
                    rmp_serde::from_slice(&snapshot(&env, 1.0, false)).unwrap();
                let (zombies, food) = seen(&env);
                let want_z: Vec<usize> = env.visible_zombies().iter().map(|v| v.0).collect();
                let got_z: Vec<usize> = zombies.iter().map(|&i| i as usize).collect();
                assert_eq!(got_z, want_z);
                assert_eq!(
                    frame["seen_zombies"].as_array().unwrap().len(),
                    zombies.len()
                );
                if zombies.len() <= 32 {
                    assert_eq!(zombies.len(), observed_zombies(&env));
                }
                // Seen food indices point at the right packages of the frame's active list.
                let frame_food = frame["food"].as_array().unwrap();
                for (&slot, &(i, _)) in food.iter().zip(env.visible_food().iter()) {
                    let p = env.food()[i].pos;
                    let f = &frame_food[slot as usize];
                    assert_eq!(f[0].as_f64().unwrap() as f32, p.x);
                    assert_eq!(f[1].as_f64().unwrap() as f32, p.y);
                }
                seen_any.0 |= !zombies.is_empty();
                seen_any.1 |= !food.is_empty();
                let za = vec![t % 16; env.zombies().len()];
                env.step(
                    HumanAction {
                        movement: t % 16,
                        interact: 0,
                    },
                    &za,
                );
                if env.outcome() != Outcome::Running {
                    break;
                }
            }
        }
        assert!(
            seen_any.0 && seen_any.1,
            "the episodes saw zombies and food"
        );
    }
}
