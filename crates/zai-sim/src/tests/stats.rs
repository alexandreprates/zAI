//! One test per tactic counter (`crate::stats`, docs/ARCHITECTURE.md section 4.8.6).

use glam::Vec2;

use super::*;
use crate::env::{INTERACT_CLOSE_DOOR, INTERACT_PICKUP};
use crate::stats::{EpisodeStats, STATS_DIM};
use crate::Outcome;

#[test]
fn pickups_count_as_risky_when_a_zombie_can_hear_them() {
    let mut cfg = open_field(1);
    cfg.food_count = 2;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(201.0, 200.0);
    env.food[1].pos = Vec2::new(100.0, 100.0);
    env.zombies[0].pos = Vec2::new(201.0, 299.0); // 99 m from the package
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    env.zombies[0].pos = Vec2::new(450.0, 450.0);
    env.human.pos = Vec2::new(100.5, 100.0);
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    let st = env.episode_stats();
    assert_eq!((st.pickups, st.risky_pickups), (2, 1));
    assert_eq!(st.decisions, 2);
    assert_eq!(
        st.carried_sum, 3.0,
        "1 after the first decision, 2 after the second"
    );
}

#[test]
fn eating_load_and_walking_are_counted_per_decision() {
    let mut env = make_env(open_field(1));
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.human.carried = 3;
    env.human.hunger_s = 0.25;
    env.step(human(EAST, crate::env::INTERACT_NONE), &stay(&env));
    env.step(IDLE, &stay(&env));
    let st = env.episode_stats();
    assert_eq!(st.eaten, 1);
    assert_eq!(st.heavy_decisions, 0, "2 packages left after eating one");
    assert_eq!(st.walk_noises, 1, "standing still is silent");
    assert_eq!(st.inside_decisions, 0);
}

#[test]
fn defensive_door_closes_need_a_zombie_near_the_door() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    let front = env.map.houses[0].doors[0].center;
    env.zombies[0].pos = front + Vec2::new(0.0, 25.0);
    env.human.pos = Vec2::new(250.0, 252.5); // inside, next to the front door
    env.step(
        human(crate::env::STAY, INTERACT_CLOSE_DOOR),
        &[crate::env::STAY],
    );
    env.zombies[0].pos = front + Vec2::new(0.0, 40.0);
    env.human.pos = Vec2::new(250.0, 247.5); // inside, next to the back door
    env.step(
        human(crate::env::STAY, INTERACT_CLOSE_DOOR),
        &[crate::env::STAY],
    );
    let st = env.episode_stats();
    assert_eq!((st.door_closes, st.defensive_closes), (2, 1));
    assert_eq!(st.inside_decisions, 2);
}

#[test]
fn catches_record_encirclement_and_flanking() {
    let mut env = make_env(open_field(4));
    let h = Vec2::new(200.0, 200.0);
    env.human.pos = h;
    // One zombie touching the human from the east, three around it within 50 m.
    env.zombies[0].pos = h + Vec2::new(0.5, 0.0);
    env.zombies[1].pos = h + Vec2::new(0.0, 20.0);
    env.zombies[2].pos = h + Vec2::new(-20.0, 0.0);
    env.zombies[3].pos = h + Vec2::new(0.0, -60.0); // outside 50 m
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Caught);
    let st = env.episode_stats();
    assert_eq!(st.catch_close_zombies, 3);
    assert!(
        (st.catch_gap_deg - 180.0).abs() < 1.0,
        "east, north, west: {}",
        st.catch_gap_deg
    );
    assert_eq!(st.catch_sectors, 3);
}

#[test]
fn dispersion_is_measured_only_while_no_zombie_heard_anything() {
    let mut env = make_env(open_field(2));
    park_zombies(&mut env);
    env.zombies[1].pos = Vec2::new(5.0, 35.0);
    env.human.pos = Vec2::new(400.0, 400.0);
    env.step(IDLE, &stay(&env));
    assert_eq!(env.episode_stats().quiet_decisions, 1);
    assert!((env.episode_stats().quiet_dispersion_sum - 30.0).abs() < 1e-3);
    // A zombie hears the human walk: not quiet for the next 30 s.
    env.human.pos = Vec2::new(5.0, 10.0);
    env.step(human(EAST, crate::env::INTERACT_NONE), &stay(&env));
    assert_eq!(env.episode_stats().quiet_decisions, 1);
}

#[test]
fn counters_reset_with_the_episode_and_fill_the_layout() {
    let mut cfg = open_field(1);
    cfg.food_count = 1;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(201.0, 200.0);
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    let mut v = [0.0; STATS_DIM];
    env.episode_stats().write(&mut v);
    assert_eq!(v[0], 1.0, "decisions first");
    assert_eq!(v[1], 1.0, "then pickups");
    env.reset(3);
    assert_eq!(env.episode_stats(), &EpisodeStats::default());
}
