//! Crate-internal tests that need to arrange exact world states.

mod props;
mod rules;

use glam::Vec2;

use crate::env::{HumanAction, INTERACT_NONE, STAY};
use crate::map::House;
use crate::{Config, Env};

/// Open 500 x 500 map with no houses and no food, and omnidirectional vision so rule tests
/// do not depend on the human's heading (the field of view has its own test).
pub(super) fn open_field(zombies: usize) -> Config {
    let mut cfg = Config {
        house_count: 0,
        zombie_count: zombies,
        food_count: 0,
        food_respawn_interval_s: None,
        ..Config::default()
    };
    cfg.human.vision_fov_deg = 360.0;
    cfg
}

pub(super) fn make_env(cfg: Config) -> Env {
    Env::new(cfg, 7).expect("valid config")
}

/// Replaces the generated map with one axis-aligned house centered at (250, 250).
/// Front door center: (250, 254); back door center: (250, 246); walls at x = 244 and 256.
pub(super) fn with_single_house(env: &mut Env) {
    env.map.houses = vec![House::new(Vec2::new(250.0, 250.0), 0, &env.cfg)];
    env.human.door_memory = vec![[false; 2]];
    env.human.doors_visible = vec![false];
}

/// Moves every zombie to a corner far from the center so it does not interfere.
pub(super) fn park_zombies(env: &mut Env) {
    for z in &mut env.zombies {
        z.pos = Vec2::new(5.0, 5.0);
        z.heard.clear();
        z.last_heard_time = None;
    }
}

pub(super) fn stay(env: &Env) -> Vec<u8> {
    vec![STAY; env.zombies.len()]
}

pub(super) fn human(movement: u8, interact: u8) -> HumanAction {
    HumanAction { movement, interact }
}

pub(super) const EAST: u8 = 0;
pub(super) const NORTH: u8 = 4;
pub(super) const IDLE: HumanAction = HumanAction {
    movement: STAY,
    interact: INTERACT_NONE,
};
