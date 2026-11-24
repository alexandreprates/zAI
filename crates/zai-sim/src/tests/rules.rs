//! One test per game rule from ZAI_PRD.md / docs/ARCHITECTURE.md section 3.

use glam::Vec2;

use super::*;
use crate::env::{NoiseKind, INTERACT_CLOSE_DOOR, INTERACT_PICKUP, MOVE_ACTIONS};
use crate::obs::{write_human_obs, write_state, write_zombie_obs};
use crate::{Outcome, HUMAN_OBS_DIM, STATE_DIM, ZOMBIE_OBS_DIM};

#[test]
fn speed_is_linear_on_base_speed_and_zero_at_five_packages() {
    for (carried, expected) in [(0, 0.5), (1, 0.4), (2, 0.3), (4, 0.1), (5, 0.0), (7, 0.0)] {
        let mut env = make_env(open_field(1));
        park_zombies(&mut env);
        env.human.pos = Vec2::new(100.0, 100.0);
        env.human.carried = carried;
        env.step(human(EAST, INTERACT_NONE), &stay(&env));
        let moved = env.human.pos.x - 100.0;
        assert!(
            (moved - expected).abs() < 1e-4,
            "carried {carried}: moved {moved}, expected {expected}"
        );
    }
}

#[test]
fn zombie_speed_is_point_seven_meters_per_second() {
    let mut env = make_env(open_field(1));
    env.human.pos = Vec2::new(400.0, 400.0);
    env.zombies[0].pos = Vec2::new(100.0, 100.0);
    env.step(IDLE, &[EAST]);
    assert!((env.zombies[0].pos.x - 100.35).abs() < 1e-4);
}

#[test]
fn walking_noise_is_heard_within_ten_meters_only() {
    let mut env = make_env(open_field(2));
    env.human.pos = Vec2::new(100.0, 100.0);
    // The walk noise is emitted at the end position (100.5, 100).
    env.zombies[0].pos = Vec2::new(100.5, 109.0);
    env.zombies[1].pos = Vec2::new(100.5, 111.0);
    for z in &mut env.zombies {
        z.heard.clear();
    }
    env.step(human(EAST, INTERACT_NONE), &stay(&env));
    assert_eq!(env.events().len(), 1);
    assert_eq!(env.events()[0].kind, NoiseKind::Walk);
    assert_eq!(env.zombies[0].heard.len(), 1);
    assert!(env.zombies[1].heard.is_empty());

    // Standing still is silent.
    env.step(IDLE, &stay(&env));
    assert!(env.events().is_empty());
}

#[test]
fn consecutive_walk_noises_share_one_slot() {
    let mut env = make_env(open_field(1));
    env.human.pos = Vec2::new(100.0, 100.0);
    env.zombies[0].pos = Vec2::new(103.0, 105.0);
    env.zombies[0].heard.clear();
    for _ in 0..4 {
        env.step(human(EAST, INTERACT_NONE), &stay(&env));
    }
    assert_eq!(env.zombies[0].heard.len(), 1);
    assert_eq!(env.zombies[0].heard[0].kind, NoiseKind::Walk);
}

#[test]
fn pickup_collects_food_and_makes_hundred_meter_noise() {
    let mut cfg = open_field(2);
    cfg.food_count = 1;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(201.0, 200.0);
    env.zombies[0].pos = Vec2::new(201.0, 299.0);
    env.zombies[1].pos = Vec2::new(201.0, 302.0);
    for z in &mut env.zombies {
        z.heard.clear();
    }
    assert!(env.human_action_mask()[MOVE_ACTIONS + INTERACT_PICKUP as usize]);

    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    assert_eq!(env.human.carried, 1);
    assert!(!env.food[0].active);
    assert_eq!(env.zombies[0].heard.len(), 1);
    assert_eq!(env.zombies[0].heard[0].kind, NoiseKind::Pickup);
    assert!(env.zombies[1].heard.is_empty());
    assert!(!env.human_action_mask()[MOVE_ACTIONS + INTERACT_PICKUP as usize]);
}

#[test]
fn pickup_out_of_range_is_a_no_op() {
    let mut cfg = open_field(1);
    cfg.food_count = 1;
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(202.0, 200.0);
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    assert_eq!(env.human.carried, 0);
    assert!(env.food[0].active);
    assert!(env.events().is_empty());
}

#[test]
fn closing_a_door_opens_its_twin_and_makes_two_hundred_meter_noise() {
    let mut cfg = open_field(2);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    let front = env.map.houses[0].doors[0].center;
    env.zombies[0].pos = front - Vec2::new(0.0, 199.0);
    env.zombies[1].pos = front + Vec2::new(201.0, 0.0);
    for z in &mut env.zombies {
        z.heard.clear();
    }

    env.human.pos = Vec2::new(250.0, 255.5);
    env.step(human(crate::env::STAY, INTERACT_CLOSE_DOOR), &stay(&env));
    let doors = &env.map.houses[0].doors;
    assert!(doors[0].closed && !doors[1].closed);
    assert_eq!(env.events()[0].kind, NoiseKind::Door);
    assert_eq!(env.zombies[0].heard.len(), 1);
    assert!(env.zombies[1].heard.is_empty());
    assert_eq!(env.human.door_memory[0], [true, false]);

    env.human.pos = Vec2::new(250.0, 244.5);
    env.step(human(crate::env::STAY, INTERACT_CLOSE_DOOR), &stay(&env));
    let doors = &env.map.houses[0].doors;
    assert!(!doors[0].closed && doors[1].closed);
}

#[test]
fn hearing_is_capped_at_three_hundred_meters() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    cfg.noise.door_close_radius_m = 400.0;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    env.zombies[0].pos = Vec2::new(250.0, 254.0 - 240.0);
    env.zombies[0].heard.clear();
    env.human.pos = Vec2::new(250.0, 255.5);
    env.step(human(crate::env::STAY, INTERACT_CLOSE_DOOR), &stay(&env));
    assert_eq!(env.zombies[0].heard.len(), 1, "240 m is within hearing");

    let mut env2 = make_env(env.cfg.clone());
    with_single_house(&mut env2);
    env2.zombies[0].pos = Vec2::new(250.0 + 320.0_f32.min(240.0), 254.0 + 220.0);
    let dist = env2.zombies[0].pos.distance(Vec2::new(250.0, 254.0));
    assert!(dist > 300.0 && dist < 400.0);
    env2.zombies[0].heard.clear();
    env2.human.pos = Vec2::new(250.0, 255.5);
    env2.step(human(crate::env::STAY, INTERACT_CLOSE_DOOR), &stay(&env2));
    assert!(
        env2.zombies[0].heard.is_empty(),
        "{dist} m exceeds hearing radius"
    );
}

#[test]
fn door_cannot_close_while_doorway_is_occupied() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    env.zombies[0].pos = env.map.houses[0].doors[0].center;
    env.human.pos = Vec2::new(250.0, 255.8);
    assert!(env.door_target().is_none());
    env.step(human(crate::env::STAY, INTERACT_CLOSE_DOOR), &stay(&env));
    assert!(!env.map.houses[0].doors[0].closed);
}

#[test]
fn walls_block_movement_and_doors_let_entities_through() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(250.0, 250.0);
    for _ in 0..30 {
        env.step(human(EAST, INTERACT_NONE), &stay(&env));
    }
    assert!(env.human.pos.x <= 256.0 - 0.3 + 1e-3, "{:?}", env.human.pos);
    assert_eq!(env.map.house_at(env.human.pos), Some(0));

    env.human.pos = Vec2::new(250.0, 250.0);
    for _ in 0..30 {
        env.step(human(NORTH, INTERACT_NONE), &stay(&env));
    }
    assert!(env.human.pos.y > 260.0, "left through the open front door");

    // A closed door blocks like a wall.
    env.map.houses[0].doors[0].closed = true;
    env.map.houses[0].doors[1].closed = false;
    env.human.pos = Vec2::new(250.0, 250.0);
    for _ in 0..30 {
        env.step(human(NORTH, INTERACT_NONE), &stay(&env));
    }
    assert!(env.human.pos.y <= 254.0 - 0.3 + 1e-3);
}

#[test]
fn zombies_cannot_open_doors() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    env.map.houses[0].doors[0].closed = true;
    env.human.pos = Vec2::new(400.0, 400.0);
    env.zombies[0].pos = Vec2::new(250.0, 260.0);
    for _ in 0..40 {
        env.step(IDLE, &[12]); // south, into the closed front door
    }
    assert!(env.zombies[0].pos.y >= 254.0 + 0.3 - 1e-3);
    assert!(env.zombies[0].blocked);
}

#[test]
fn human_sees_only_within_line_of_sight() {
    let mut cfg = open_field(2);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    env.human.pos = Vec2::new(250.0, 250.0);
    env.zombies[0].pos = Vec2::new(265.0, 250.0); // behind the east wall
    env.zombies[1].pos = Vec2::new(250.0, 265.0); // through the open front door
    let visible: Vec<usize> = env.visible_zombies().iter().map(|v| v.0).collect();
    assert_eq!(visible, vec![1]);

    env.zombies[1].pos = Vec2::new(250.0, 360.0); // beyond 100 m
    assert!(env.visible_zombies().is_empty());
}

#[test]
fn narrow_field_of_view_hides_targets_outside_the_cone() {
    let mut cfg = open_field(4);
    cfg.human.vision_fov_deg = 45.0;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.human.heading = Vec2::X;
    let at = |deg: f32| {
        let a = deg.to_radians();
        Vec2::new(200.0, 200.0) + Vec2::new(a.cos(), a.sin()) * 20.0
    };
    env.zombies[0].pos = at(0.0); // straight ahead
    env.zombies[1].pos = at(20.0); // inside the 22.5 degree half-angle
    env.zombies[2].pos = at(30.0); // outside
    env.zombies[3].pos = at(180.0); // behind
    let mut visible: Vec<usize> = env.visible_zombies().iter().map(|v| v.0).collect();
    visible.sort();
    assert_eq!(visible, vec![0, 1]);

    // Turning (by moving) changes what is seen.
    env.human.heading = -Vec2::X;
    let visible: Vec<usize> = env.visible_zombies().iter().map(|v| v.0).collect();
    assert_eq!(visible, vec![3]);
}

#[test]
fn starving_with_no_food_ends_the_episode() {
    let mut env = make_env(open_field(1));
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.human.hunger_s = 0.25;
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Starved);
    assert!(r.terminated && !r.truncated);
    assert_eq!(r.human_reward, -1.0);
    assert!(
        (r.zombie_reward - -0.002).abs() < 1e-6,
        "starvation is not a zombie win"
    );
}

#[test]
fn carried_food_is_eaten_automatically() {
    let mut env = make_env(open_field(1));
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.human.carried = 1;
    env.human.hunger_s = 0.25;
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Running);
    assert_eq!(env.human.carried, 0);
    assert!((env.human.hunger_s - (90.0 + 0.25 - 0.5)).abs() < 1e-3);
}

#[test]
fn human_must_eat_every_ninety_seconds() {
    let mut env = make_env(open_field(1));
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    let mut steps = 0;
    while env.outcome == Outcome::Running {
        env.step(IDLE, &stay(&env));
        steps += 1;
    }
    assert_eq!(env.outcome, Outcome::Starved);
    assert_eq!(steps, 180, "90 s at 2 decisions per second");
}

#[test]
fn zombie_contact_kills_the_human() {
    let mut env = make_env(open_field(1));
    env.human.pos = Vec2::new(200.0, 200.0);
    env.zombies[0].pos = Vec2::new(200.8, 200.0);
    let r = env.step(human(EAST, INTERACT_NONE), &stay(&env));
    assert_eq!(r.outcome, Outcome::Caught);
    assert!(r.terminated);
    assert_eq!(r.human_reward, -1.0);
    assert!((r.zombie_reward - (1.0 - 0.002)).abs() < 1e-6);
}

#[test]
fn contact_through_a_wall_does_not_kill() {
    let mut cfg = open_field(1);
    cfg.house_count = 1;
    let mut env = make_env(cfg);
    with_single_house(&mut env);
    env.human.pos = Vec2::new(243.7, 250.0);
    env.zombies[0].pos = Vec2::new(244.3, 250.0);
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Running);
}

#[test]
fn episode_is_truncated_at_the_cap() {
    let mut cfg = open_field(1);
    cfg.episode_cap_s = 5.0;
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    for i in 1..=10 {
        let r = env.step(IDLE, &stay(&env));
        if i < 10 {
            assert_eq!(r.outcome, Outcome::Running);
        } else {
            assert_eq!(r.outcome, Outcome::Truncated);
            assert!(r.truncated && !r.terminated);
            assert!(r.human_reward > 0.0);
        }
    }
}

#[test]
fn food_respawns_after_the_interval() {
    let mut cfg = open_field(1);
    cfg.food_count = 3;
    cfg.food_respawn_interval_s = Some(1.0);
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(200.5, 200.0);
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    assert!(!env.food[0].active);
    for _ in 0..2 {
        env.step(IDLE, &stay(&env));
    }
    assert!(env.food[0].active);
    assert_eq!(env.food.iter().filter(|f| f.active).count(), 3);
}

#[test]
fn food_does_not_respawn_when_disabled() {
    let mut cfg = open_field(1);
    cfg.food_count = 1;
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(200.5, 200.0);
    env.step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env));
    for _ in 0..200 {
        env.human.hunger_s = 90.0;
        env.step(IDLE, &stay(&env));
    }
    assert!(!env.food[0].active);
}

#[test]
fn heard_noises_are_forgotten_after_thirty_seconds() {
    let mut env = make_env(open_field(1));
    env.human.pos = Vec2::new(100.0, 100.0);
    env.zombies[0].pos = Vec2::new(103.0, 103.0);
    env.zombies[0].heard.clear();
    env.step(human(EAST, INTERACT_NONE), &[crate::env::STAY]);
    assert_eq!(env.zombies[0].heard.len(), 1);
    for _ in 0..61 {
        env.step(IDLE, &[crate::env::STAY]);
    }
    assert!(env.zombies[0].heard.is_empty());
}

#[test]
fn default_map_matches_the_prd() {
    let env = make_env(Config::default());
    assert_eq!(env.map.houses.len(), 10);
    assert_eq!(env.food.len(), 120);
    assert_eq!(env.zombies.len(), 20);
    let inside = env
        .food
        .iter()
        .filter(|f| env.map.house_at(f.pos).is_some())
        .count();
    assert!(inside >= 6, "at least 5% of the food starts inside houses");
    for z in &env.zombies {
        assert!(z.pos.distance(env.human.pos) >= 150.0);
    }
    assert!(env.map.house_at(env.human.pos).is_none());
}

#[test]
fn observations_have_the_documented_sizes_and_are_finite() {
    let mut env = make_env(Config::default());
    let mut h = vec![0.0; HUMAN_OBS_DIM];
    let mut z = vec![0.0; ZOMBIE_OBS_DIM];
    let mut s = vec![0.0; STATE_DIM];
    for i in 0..300u32 {
        let acts: Vec<u8> = (0..env.zombies.len())
            .map(|k| ((i + k as u32) % 17) as u8)
            .collect();
        let r = env.step(human((i % 17) as u8, (i % 3) as u8), &acts);
        write_human_obs(&env, &mut h);
        write_state(&env, &mut s);
        for k in 0..env.zombies.len() {
            write_zombie_obs(&env, k, &mut z);
            assert!(z.iter().all(|v| v.is_finite()));
        }
        assert!(h.iter().all(|v| v.is_finite()));
        assert!(s.iter().all(|v| v.is_finite()));
        if r.outcome != Outcome::Running {
            env.reset(i as u64);
        }
    }
}

#[test]
fn food_shaping_rewards_pickup_and_never_pays_for_dying() {
    let mut cfg = open_field(1);
    cfg.food_count = 1;
    cfg.reward.human_food_shaping = 1.0;
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.food[0].pos = Vec2::new(200.5, 200.0);
    let idle = env.step(IDLE, &stay(&env)).human_reward;
    let pickup = env
        .step(human(crate::env::STAY, INTERACT_PICKUP), &stay(&env))
        .human_reward;
    assert!(pickup > idle + 0.2, "pickup {pickup} vs idle {idle}");

    let mut cfg = open_field(1);
    cfg.reward.human_food_shaping = 1.0;
    let mut env = make_env(cfg);
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.human.hunger_s = 0.25;
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Starved);
    assert!(r.human_reward <= -1.0, "death reward {}", r.human_reward);
}

#[test]
fn shaping_is_zero_by_default_and_potential_based_when_enabled() {
    let mut env = make_env(open_field(1));
    park_zombies(&mut env);
    env.human.pos = Vec2::new(200.0, 200.0);
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.human_reward, 0.005);

    let mut cfg = open_field(1);
    cfg.reward.zombie_distance_shaping = 0.5;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env.zombies[0].pos = Vec2::new(250.0, 200.0);
    let r = env.step(IDLE, &[8]); // west, toward the human
    assert_eq!(r.zombie_reward, -0.002, "the team part has no shaping");
    assert!(
        env.zombie_rewards()[0] > -0.002,
        "approaching the human is rewarded"
    );
}

fn shaped(zombies: usize) -> Env {
    let mut cfg = open_field(zombies);
    cfg.reward.zombie_distance_shaping = 0.5;
    let mut env = make_env(cfg);
    env.human.pos = Vec2::new(200.0, 200.0);
    env
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-5
}

#[test]
fn zombie_distance_potential_is_per_agent_and_saturates_at_the_range() {
    let mut env = shaped(3);
    env.zombies[0].pos = Vec2::new(220.0, 200.0); // 20 m: Phi = 0.5 * 0.8 = 0.4
    env.zombies[1].pos = Vec2::new(260.0, 200.0); // 60 m: Phi = 0.2
    env.zombies[2].pos = Vec2::new(200.0, 340.0); // 140 m: beyond the range, Phi = 0
    env.step(IDLE, &stay(&env));
    let g = env.cfg.reward.gamma;
    let r = env.zombie_rewards();
    assert!(close(r[0], -0.002 + (g - 1.0) * 0.4), "{r:?}");
    assert!(close(r[1], -0.002 + (g - 1.0) * 0.2), "{r:?}");
    assert!(close(r[2], -0.002), "{r:?}");
}

#[test]
fn team_min_shaping_shares_the_nearest_zombie_potential() {
    let mut env = shaped(2);
    env.cfg.reward.zombie_distance_shaping_mode = crate::ZombieShapingMode::TeamMin;
    env.zombies[0].pos = Vec2::new(220.0, 200.0);
    env.zombies[1].pos = Vec2::new(260.0, 200.0);
    env.step(IDLE, &stay(&env));
    let g = env.cfg.reward.gamma;
    for &r in env.zombie_rewards() {
        assert!(close(r, -0.002 + (g - 1.0) * 0.4), "{r}");
    }
}

#[test]
fn starvation_pays_zombies_nothing_beyond_the_lost_potential() {
    let mut env = shaped(1);
    env.zombies[0].pos = Vec2::new(250.0, 200.0); // 50 m: Phi = 0.25
    env.human.hunger_s = 0.25;
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(r.outcome, Outcome::Starved);
    assert!(close(env.zombie_rewards()[0], -0.002 - 0.25));
}

#[test]
fn a_catch_pays_the_catch_reward_plus_the_maximum_potential() {
    let mut env = shaped(1);
    env.zombies[0].pos = Vec2::new(200.8, 200.0); // Phi = 0.5 * (1 - 0.8 / 100)
    let r = env.step(human(EAST, INTERACT_NONE), &stay(&env));
    assert_eq!(r.outcome, Outcome::Caught);
    let g = env.cfg.reward.gamma;
    let expected = -0.002 + 1.0 + g * 0.5 - 0.5 * (1.0 - 0.8 / 100.0);
    assert!(
        close(env.zombie_rewards()[0], expected),
        "{:?}",
        env.zombie_rewards()
    );
}

#[test]
fn zombie_rewards_equal_the_team_reward_without_shaping() {
    let mut env = make_env(open_field(3));
    env.human.pos = Vec2::new(200.0, 200.0);
    let r = env.step(IDLE, &stay(&env));
    assert_eq!(env.zombie_rewards(), &[r.zombie_reward; 3]);
}
