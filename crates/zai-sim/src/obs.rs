//! Observation and state encoding shared by the trainer and the server.
//! Layouts are documented in docs/ARCHITECTURE.md section 3.6 (v1) and 4.8.2 (v2).
//!
//! Layout v2 normalizes distances by constants instead of the map size, so the same input
//! means the same distance on every curriculum stage: deltas to static landmarks are
//! `sat(v) = v / max(|v|, NEAR_M)` (linear up to 100 m, then a unit direction) plus a far-scale
//! distance `|v| / FAR_M`, absolute positions are `pos / FAR_M`, and the time left is
//! `(cap - t) / TIME_SCALE_S`.

use std::f32::consts::TAU;

use glam::Vec2;

use crate::env::{Env, NoiseKind};
use crate::map::ObstacleKind;
use crate::{MAX_FOOD, MAX_HOUSES, MAX_ZOMBIES};

/// Version of the observation layouts, saved with checkpoints and exported models.
pub const OBS_VERSION: u32 = 2;

pub const HUMAN_OBS_DIM: usize = 608;
pub const ZOMBIE_OBS_DIM: usize = 276;
pub const STATE_DIM: usize = 1144;

/// Near scale (vision radius, shaping range): deltas are linear up to this distance.
pub const NEAR_M: f32 = 100.0;
/// Far scale (the PRD map side) for absolute positions and far distances.
pub const FAR_M: f32 = 500.0;
/// Time scale (the PRD episode cap) for the time left.
pub const TIME_SCALE_S: f32 = 1200.0;

const MAX_VISIBLE_ZOMBIES: usize = 32;
const MAX_VISIBLE_FOOD: usize = 16;
const HUMAN_RAYS: usize = 64;
const MAX_PEERS: usize = 8;
const PEER_RANGE_M: f32 = 50.0;
const ZOMBIE_RAYS: usize = 16;
const ZOMBIE_RAY_RANGE_M: f32 = 10.0;
/// Normalizer for the carried-food count and size of its one-hot bucket minus one.
const CARRIED_CAP: u32 = 5;

/// Sequential writer over a zero-filled buffer.
struct Writer<'a> {
    buf: &'a mut [f32],
    i: usize,
}

impl<'a> Writer<'a> {
    fn new(buf: &'a mut [f32], expected: usize) -> Self {
        assert_eq!(
            buf.len(),
            expected,
            "observation buffer has the wrong length"
        );
        buf.fill(0.0);
        Self { buf, i: 0 }
    }
    fn push(&mut self, v: f32) {
        self.buf[self.i] = v;
        self.i += 1;
    }
    fn vec(&mut self, v: Vec2) {
        self.push(v.x);
        self.push(v.y);
    }
    fn flag(&mut self, b: bool) {
        self.push(b as u8 as f32);
    }
    /// Advances to the end of a padded block that started at `start`.
    fn end_block(&mut self, start: usize, len: usize) {
        debug_assert!(self.i <= start + len, "block overflow");
        self.i = start + len;
    }
    fn finish(self) {
        assert_eq!(self.i, self.buf.len(), "observation layout mismatch");
    }
}

/// Delta to a landmark: linear up to `NEAR_M`, a unit direction beyond.
fn sat(v: Vec2) -> Vec2 {
    v / v.length().max(NEAR_M)
}

/// Distances to the west, east, south and north map edges, saturated at `NEAR_M`.
fn wall_distances(w: &mut Writer, pos: Vec2, size: f32) {
    for d in [pos.x, size - pos.x, pos.y, size - pos.y] {
        w.push(d.clamp(0.0, NEAR_M) / NEAR_M);
    }
}

fn angle_sin_cos(dir: Vec2) -> Vec2 {
    let d = dir.normalize_or(Vec2::X);
    Vec2::new(d.y, d.x)
}

/// Human actor observation (`HUMAN_OBS_DIM` floats).
pub fn write_human_obs(env: &Env, out: &mut [f32]) {
    let mut w = Writer::new(out, HUMAN_OBS_DIM);
    let cfg = &env.cfg;
    let h = &env.human;
    let size = cfg.map_size_m;
    let vr = cfg.human.vision_radius_m;
    let inside = env.map.house_at(h.pos);

    // Self (20).
    w.vec(h.pos / FAR_M);
    wall_distances(&mut w, h.pos, size);
    w.vec(angle_sin_cos(h.heading));
    w.vec(h.vel);
    let bucket = h.carried.min(CARRIED_CAP);
    w.push(bucket as f32 / CARRIED_CAP as f32);
    for k in 0..=CARRIED_CAP {
        w.flag(k == bucket);
    }
    w.push(h.hunger_s / cfg.human.hunger_interval_s);
    w.flag(inside.is_some());
    w.push(
        h.last_noise_time
            .map_or(1.0, |t| ((env.time - t) / 30.0).min(1.0)),
    );

    // Visible zombies (32 x 6).
    let start = w.i;
    let zspeed = cfg.zombie.speed_mps;
    for (i, d) in env.visible_zombies().into_iter().take(MAX_VISIBLE_ZOMBIES) {
        let z = &env.zombies[i];
        w.vec((z.pos - h.pos) / vr);
        w.push(d / vr);
        w.vec(z.vel / zspeed);
        w.flag(true);
    }
    w.end_block(start, MAX_VISIBLE_ZOMBIES * 6);

    // Visible food (16 x 4).
    let start = w.i;
    for (i, d) in env.visible_food().into_iter().take(MAX_VISIBLE_FOOD) {
        w.vec((env.food[i].pos - h.pos) / vr);
        w.push(d / vr);
        w.flag(true);
    }
    w.end_block(start, MAX_VISIBLE_FOOD * 4);

    // Houses (10 x 14).
    let start = w.i;
    for (hi, house) in env.map.houses.iter().enumerate() {
        let d = house.center - h.pos;
        w.flag(true);
        w.vec(sat(d));
        w.push(d.length() / FAR_M);
        w.vec(angle_sin_cos(house.axis_x));
        w.vec(sat(house.doors[0].center - h.pos));
        w.vec(sat(house.doors[1].center - h.pos));
        w.flag(h.door_memory[hi][0]);
        w.flag(h.door_memory[hi][1]);
        w.flag(h.doors_visible[hi]);
        w.flag(inside == Some(hi));
    }
    w.end_block(start, MAX_HOUSES * 14);

    // Ray cast (64 x 3): world-frame directions over 360 degrees, or spread across the
    // field of view around the heading when it is narrower.
    let fov = cfg.human.vision_fov_deg.to_radians();
    let heading = h.heading.y.atan2(h.heading.x);
    for k in 0..HUMAN_RAYS {
        let a = if cfg.human.vision_fov_deg >= 360.0 {
            k as f32 * TAU / HUMAN_RAYS as f32
        } else {
            heading - fov * 0.5 + fov * (k as f32 + 0.5) / HUMAN_RAYS as f32
        };
        let hit = env.map.ray_cast(h.pos, Vec2::new(a.cos(), a.sin()), vr);
        w.push(hit.distance / vr);
        w.flag(hit.kind == Some(ObstacleKind::Wall));
        w.flag(hit.kind == Some(ObstacleKind::ClosedDoor));
    }
    w.finish();
}

/// Actor observation of zombie `index` (`ZOMBIE_OBS_DIM` floats).
pub fn write_zombie_obs(env: &Env, index: usize, out: &mut [f32]) {
    let mut w = Writer::new(out, ZOMBIE_OBS_DIM);
    let cfg = &env.cfg;
    let z = &env.zombies[index];
    let size = cfg.map_size_m;
    let memory = cfg.noise.memory_s;
    let hearing = cfg.zombie.hearing_radius_m;
    let inside = env.map.house_at(z.pos);

    // Self (16).
    w.vec(z.pos / FAR_M);
    wall_distances(&mut w, z.pos, size);
    w.vec(angle_sin_cos(z.heading));
    w.vec(z.vel / cfg.zombie.speed_mps);
    w.flag(inside.is_some());
    w.flag(z.blocked);
    w.push(
        z.last_heard_time
            .map_or(1.0, |t| ((env.time - t) / memory).min(1.0)),
    );
    w.push(env.zombies.len() as f32 / MAX_ZOMBIES as f32);
    w.push((cfg.episode_cap_s - env.time) / TIME_SCALE_S);
    w.push(1.0);

    // Heard noise buffer (8 x 9), newest first.
    let start = w.i;
    let mut heard: Vec<_> = z.heard.iter().collect();
    heard.sort_by(|a, b| b.time.total_cmp(&a.time));
    for e in heard {
        let d = e.pos - z.pos;
        w.vec(d / hearing);
        w.push(d.length() / hearing);
        w.flag(e.kind == NoiseKind::Walk);
        w.flag(e.kind == NoiseKind::Pickup);
        w.flag(e.kind == NoiseKind::Door);
        w.push((env.time - e.time) / memory);
        w.flag(env.time - e.time < cfg.decision_dt());
        w.flag(true);
    }
    w.end_block(start, crate::env::HEARD_CAPACITY * 9);

    // Peer zombies within 50 m (8 x 5).
    let start = w.i;
    let mut peers: Vec<(usize, f32)> = env
        .zombies
        .iter()
        .enumerate()
        .filter(|&(j, _)| j != index)
        .map(|(j, o)| (j, o.pos.distance(z.pos)))
        .filter(|&(_, d)| d <= PEER_RANGE_M)
        .collect();
    peers.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (j, d) in peers.into_iter().take(MAX_PEERS) {
        let o = &env.zombies[j];
        w.vec((o.pos - z.pos) / PEER_RANGE_M);
        w.push(d / PEER_RANGE_M);
        w.flag(inside.is_some() && env.map.house_at(o.pos) == inside);
        w.flag(true);
    }
    w.end_block(start, MAX_PEERS * 5);

    // Static map: houses (10 x 10), door states not observed.
    let start = w.i;
    for house in &env.map.houses {
        let d = house.center - z.pos;
        w.flag(true);
        w.vec(sat(d));
        w.push(d.length() / FAR_M);
        w.vec(angle_sin_cos(house.axis_x));
        w.vec(sat(house.doors[0].center - z.pos));
        w.vec(sat(house.doors[1].center - z.pos));
    }
    w.end_block(start, MAX_HOUSES * 10);

    // Proximity rays (16 x 3).
    for k in 0..ZOMBIE_RAYS {
        let a = k as f32 * TAU / ZOMBIE_RAYS as f32;
        let hit = env
            .map
            .ray_cast(z.pos, Vec2::new(a.cos(), a.sin()), ZOMBIE_RAY_RANGE_M);
        w.push(hit.distance / ZOMBIE_RAY_RANGE_M);
        w.flag(hit.kind == Some(ObstacleKind::Wall));
        w.flag(hit.kind == Some(ObstacleKind::ClosedDoor));
    }
    w.finish();
}

/// Privileged global state for the critics (`STATE_DIM` floats).
pub fn write_state(env: &Env, out: &mut [f32]) {
    let mut w = Writer::new(out, STATE_DIM);
    let cfg = &env.cfg;
    let h = &env.human;

    // Human (16).
    w.vec(h.pos / FAR_M);
    w.vec(h.vel / cfg.human.base_speed_mps);
    w.push(h.carried.min(CARRIED_CAP) as f32 / CARRIED_CAP as f32);
    w.push(h.hunger_s / cfg.human.hunger_interval_s);
    let inside = env.map.house_at(h.pos);
    for hi in 0..MAX_HOUSES {
        w.flag(inside == Some(hi));
    }

    // Zombies (64 x 8).
    let start = w.i;
    for z in &env.zombies {
        let d = z.pos.distance(h.pos);
        w.vec(z.pos / FAR_M);
        w.vec(z.vel / cfg.zombie.speed_mps);
        w.flag(env.map.house_at(z.pos).is_some());
        w.flag(true);
        w.push(d.min(NEAR_M) / NEAR_M);
        w.flag(env.human_sees(z.pos));
    }
    w.end_block(start, MAX_ZOMBIES * 8);

    // Food (128 x 4).
    let start = w.i;
    for f in &env.food {
        w.vec(f.pos / FAR_M);
        w.flag(f.active);
        w.flag(env.map.house_at(f.pos).is_some());
    }
    w.end_block(start, MAX_FOOD * 4);

    // Houses (10 x 10).
    let start = w.i;
    for house in &env.map.houses {
        w.vec(house.center / FAR_M);
        w.vec(angle_sin_cos(house.axis_x));
        w.vec(house.doors[0].center / FAR_M);
        w.vec(house.doors[1].center / FAR_M);
        w.flag(house.doors[0].closed);
        w.flag(house.doors[1].closed);
    }
    w.end_block(start, MAX_HOUSES * 10);

    // Time features (4).
    w.push(env.time / cfg.episode_cap_s);
    w.push((cfg.episode_cap_s - env.time) / TIME_SCALE_S);
    w.push(env.zombies.len() as f32 / MAX_ZOMBIES as f32);
    w.push(env.food.iter().filter(|f| f.active).count() as f32 / MAX_FOOD as f32);
    w.finish();
}

/// A block of an observation vector: `slots` consecutive groups of `fields`.
#[derive(Clone, Copy, Debug)]
pub struct Block {
    pub name: &'static str,
    pub start: usize,
    pub slots: usize,
    pub fields: &'static [&'static str],
}

impl Block {
    pub const fn len(&self) -> usize {
        self.slots * self.fields.len()
    }
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

const fn block(
    name: &'static str,
    start: usize,
    slots: usize,
    fields: &'static [&'static str],
) -> Block {
    Block {
        name,
        start,
        slots,
        fields,
    }
}

/// Layout of the human observation (see `write_human_obs`).
pub const HUMAN_LAYOUT: [Block; 5] = [
    block(
        "self",
        0,
        1,
        &[
            "x",
            "y",
            "wall_w",
            "wall_e",
            "wall_s",
            "wall_n",
            "heading_sin",
            "heading_cos",
            "vx",
            "vy",
            "carried",
            "carried_0",
            "carried_1",
            "carried_2",
            "carried_3",
            "carried_4",
            "carried_5",
            "hunger",
            "inside",
            "since_noise",
        ],
    ),
    block(
        "zombies",
        20,
        MAX_VISIBLE_ZOMBIES,
        &["dx", "dy", "dist", "vx", "vy", "valid"],
    ),
    block(
        "food",
        212,
        MAX_VISIBLE_FOOD,
        &["dx", "dy", "dist", "valid"],
    ),
    block(
        "houses",
        276,
        MAX_HOUSES,
        &[
            "valid",
            "dx",
            "dy",
            "dist",
            "rot_sin",
            "rot_cos",
            "front_dx",
            "front_dy",
            "back_dx",
            "back_dy",
            "front_closed",
            "back_closed",
            "door_visible",
            "inside",
        ],
    ),
    block("rays", 416, HUMAN_RAYS, &["dist", "wall", "closed_door"]),
];

/// Layout of a zombie observation (see `write_zombie_obs`).
pub const ZOMBIE_LAYOUT: [Block; 5] = [
    block(
        "self",
        0,
        1,
        &[
            "x",
            "y",
            "wall_w",
            "wall_e",
            "wall_s",
            "wall_n",
            "heading_sin",
            "heading_cos",
            "vx",
            "vy",
            "inside",
            "blocked",
            "since_heard",
            "num_zombies",
            "time_left",
            "bias",
        ],
    ),
    block(
        "heard",
        16,
        crate::env::HEARD_CAPACITY,
        &[
            "dx", "dy", "dist", "walk", "pickup", "door", "age", "fresh", "valid",
        ],
    ),
    block(
        "peers",
        88,
        MAX_PEERS,
        &["dx", "dy", "dist", "same_house", "valid"],
    ),
    block(
        "houses",
        128,
        MAX_HOUSES,
        &[
            "valid", "dx", "dy", "dist", "rot_sin", "rot_cos", "front_dx", "front_dy", "back_dx",
            "back_dy",
        ],
    ),
    block("rays", 228, ZOMBIE_RAYS, &["dist", "wall", "closed_door"]),
];

/// Layout of the global critic state (see `write_state`).
pub const STATE_LAYOUT: [Block; 5] = [
    block(
        "human",
        0,
        1,
        &[
            "x", "y", "vx", "vy", "carried", "hunger", "house_0", "house_1", "house_2", "house_3",
            "house_4", "house_5", "house_6", "house_7", "house_8", "house_9",
        ],
    ),
    block(
        "zombies",
        16,
        MAX_ZOMBIES,
        &[
            "x",
            "y",
            "vx",
            "vy",
            "inside",
            "valid",
            "dist_to_human",
            "sees_human",
        ],
    ),
    block("food", 528, MAX_FOOD, &["x", "y", "active", "inside"]),
    block(
        "houses",
        1040,
        MAX_HOUSES,
        &[
            "x",
            "y",
            "rot_sin",
            "rot_cos",
            "front_x",
            "front_y",
            "back_x",
            "back_y",
            "front_closed",
            "back_closed",
        ],
    ),
    block(
        "time",
        1140,
        1,
        &["time", "time_left", "num_zombies", "active_food"],
    ),
];

#[cfg(test)]
mod layout_tests {
    use glam::Vec2;

    use super::*;
    use crate::env::{HumanAction, INTERACT_NONE};
    use crate::Config;

    fn check(layout: &[Block], dim: usize) {
        let mut next = 0;
        for b in layout {
            assert_eq!(b.start, next, "block {} is not contiguous", b.name);
            next += b.len();
        }
        assert_eq!(next, dim);
    }

    fn field(layout: &[Block], obs: &[f32], block: &str, slot: usize, name: &str) -> f32 {
        let b = layout.iter().find(|b| b.name == block).unwrap();
        let i = b.fields.iter().position(|f| *f == name).unwrap();
        obs[b.start + slot * b.fields.len() + i]
    }

    #[test]
    fn layouts_are_contiguous_and_match_dims() {
        check(&HUMAN_LAYOUT, HUMAN_OBS_DIM);
        check(&ZOMBIE_LAYOUT, ZOMBIE_OBS_DIM);
        check(&STATE_LAYOUT, STATE_DIM);
    }

    #[test]
    fn v2_distances_do_not_depend_on_the_map_size() {
        // The same arrangement relative to a house, on a small and on the PRD map.
        let mut values = Vec::new();
        for size in [75.0, 500.0] {
            let cfg = Config {
                map_size_m: size,
                house_count: 1,
                zombie_count: 1,
                food_count: 1,
                ..Config::default()
            };
            let mut env = Env::new(cfg, 5).unwrap();
            let house = env.map.houses[0].center;
            env.human.pos = house + Vec2::new(0.0, 30.0);
            env.zombies[0].pos = env.human.pos + Vec2::new(40.0, 0.0);
            let mut h = vec![0.0; HUMAN_OBS_DIM];
            write_human_obs(&env, &mut h);
            let f = |b: &str, s: usize, n: &str| field(&HUMAN_LAYOUT, &h, b, s, n);
            assert_eq!(f("houses", 0, "valid"), 1.0);
            assert_eq!(f("houses", 1, "valid"), 0.0);
            assert!(
                (f("houses", 0, "dy") + 0.3).abs() < 1e-5,
                "linear below 100 m"
            );
            assert!((f("houses", 0, "dist") - 30.0 / 500.0).abs() < 1e-5);
            assert!((f("self", 0, "x") - env.human.pos.x / 500.0).abs() < 1e-6);
            let west = env.human.pos.x.min(100.0) / 100.0;
            assert!((f("self", 0, "wall_w") - west).abs() < 1e-6);
            let mut s = vec![0.0; STATE_DIM];
            write_state(&env, &mut s);
            let k = |b: &str, sl: usize, n: &str| field(&STATE_LAYOUT, &s, b, sl, n);
            assert!((k("zombies", 0, "dist_to_human") - 0.4).abs() < 1e-5);
            values.push((
                f("houses", 0, "dx"),
                f("houses", 0, "dy"),
                f("houses", 0, "dist"),
            ));
        }
        let (a, b) = (values[0], values[1]);
        assert!((a.0 - b.0).abs() + (a.1 - b.1).abs() + (a.2 - b.2).abs() < 1e-5);
    }

    #[test]
    fn v2_far_landmarks_keep_direction_and_distance() {
        let cfg = Config {
            house_count: 1,
            zombie_count: 1,
            food_count: 1,
            ..Config::default()
        };
        let mut env = Env::new(cfg, 5).unwrap();
        let house = env.map.houses[0].center;
        let offset = if house.x > 250.0 { -300.0 } else { 300.0 };
        env.human.pos = house + Vec2::new(offset, 0.0);
        let mut z = vec![0.0; ZOMBIE_OBS_DIM];
        env.zombies[0].pos = env.human.pos;
        write_zombie_obs(&env, 0, &mut z);
        let g = |b: &str, s: usize, n: &str| field(&ZOMBIE_LAYOUT, &z, b, s, n);
        assert!(
            (g("houses", 0, "dx") + offset.signum()).abs() < 1e-5,
            "unit direction"
        );
        assert!(g("houses", 0, "dy").abs() < 1e-5);
        assert!((g("houses", 0, "dist") - 300.0 / 500.0).abs() < 1e-5);
        let left = (env.cfg.episode_cap_s - env.time) / 1200.0;
        assert!((g("self", 0, "time_left") - left).abs() < 1e-6);
    }

    #[test]
    fn layout_fields_match_the_writers() {
        let mut cfg = Config {
            house_count: 0,
            zombie_count: 2,
            food_count: 1,
            ..Config::default()
        };
        cfg.human.vision_fov_deg = 360.0;
        let mut env = Env::new(cfg, 3).unwrap();
        env.human.pos = Vec2::new(100.0, 100.0);
        env.human.carried = 2;
        env.zombies[0].pos = Vec2::new(120.0, 100.0);
        env.zombies[1].pos = Vec2::new(103.0, 100.0);
        env.food[0].pos = Vec2::new(100.0, 130.0);
        env.step(
            HumanAction {
                movement: 0,
                interact: INTERACT_NONE,
            },
            &[16, 16],
        );

        let mut h = vec![0.0; HUMAN_OBS_DIM];
        write_human_obs(&env, &mut h);
        let f = |b: &str, s: usize, n: &str| field(&HUMAN_LAYOUT, &h, b, s, n);
        assert_eq!(f("self", 0, "carried_2"), 1.0);
        assert!((f("self", 0, "x") - env.human.pos.x / 500.0).abs() < 1e-6);
        assert_eq!(f("zombies", 0, "valid"), 1.0);
        assert_eq!(f("zombies", 1, "valid"), 1.0);
        assert_eq!(f("zombies", 2, "valid"), 0.0);
        assert!(
            f("zombies", 0, "dist") < f("zombies", 1, "dist"),
            "nearest first"
        );
        assert_eq!(f("food", 0, "valid"), 1.0);
        assert!((f("food", 0, "dy") - 0.3).abs() < 1e-3);

        let mut z = vec![0.0; ZOMBIE_OBS_DIM];
        write_zombie_obs(&env, 1, &mut z);
        let g = |b: &str, s: usize, n: &str| field(&ZOMBIE_LAYOUT, &z, b, s, n);
        assert_eq!(g("heard", 0, "valid"), 1.0, "walking noise within 10 m");
        assert_eq!(g("heard", 0, "walk"), 1.0);
        assert_eq!(g("heard", 0, "fresh"), 1.0);
        assert_eq!(g("heard", 1, "valid"), 0.0);
        assert_eq!(g("peers", 0, "valid"), 1.0);
        assert_eq!(g("self", 0, "bias"), 1.0);

        let mut s = vec![0.0; STATE_DIM];
        write_state(&env, &mut s);
        let k = |b: &str, sl: usize, n: &str| field(&STATE_LAYOUT, &s, b, sl, n);
        assert_eq!(k("zombies", 1, "valid"), 1.0);
        assert_eq!(k("zombies", 2, "valid"), 0.0);
        assert_eq!(k("zombies", 1, "sees_human"), 1.0);
        assert_eq!(k("food", 0, "active"), 1.0);
        assert!(
            (k("time", 0, "time_left") - (env.cfg.episode_cap_s - env.time) / 1200.0).abs() < 1e-6
        );
    }
}
