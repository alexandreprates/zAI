//! Property tests over random seeds and random actions.

use proptest::prelude::*;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::env::{HumanAction, INTERACT_ACTIONS, MOVE_ACTIONS};
use crate::{Config, Env, Outcome};

fn small_config() -> Config {
    Config {
        map_size_m: 150.0,
        house_count: 3,
        zombie_count: 6,
        food_count: 20,
        food_respawn_interval_s: Some(5.0),
        min_spawn_distance_m: 20.0,
        ..Config::default()
    }
}

fn random_episode(env: &mut Env, rng: &mut ChaCha8Rng, steps: usize, mut check: impl FnMut(&Env)) {
    for _ in 0..steps {
        let action = HumanAction {
            movement: rng.random_range(0..MOVE_ACTIONS as u8),
            interact: rng.random_range(0..INTERACT_ACTIONS as u8),
        };
        let zs: Vec<u8> = (0..env.zombies.len())
            .map(|_| rng.random_range(0..MOVE_ACTIONS as u8))
            .collect();
        let before = positions(env);
        env.step(action, &zs);
        let after = positions(env);
        // No entity crosses a wall or closed door within a step.
        for (a, b) in before.iter().zip(&after) {
            assert!(
                env.map.line_of_sight(*a, *b),
                "tunneled from {a:?} to {b:?}"
            );
        }
        check(env);
        if env.outcome != Outcome::Running {
            let seed = rng.random();
            env.reset(seed);
        }
    }
}

fn positions(env: &Env) -> Vec<glam::Vec2> {
    std::iter::once(env.human.pos)
        .chain(env.zombies.iter().map(|z| z.pos))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    #[test]
    fn entities_never_penetrate_walls_or_leave_the_map(seed in any::<u64>()) {
        let mut env = Env::new(small_config(), seed).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5eed);
        let r = env.cfg.entity_radius_m;
        let size = env.cfg.map_size_m;
        random_episode(&mut env, &mut rng, 400, |env| {
            for p in positions(env) {
                assert!(p.min_element() >= r - 1e-4 && p.max_element() <= size - r + 1e-4);
                assert!(env.map.clearance(p, r) >= r - 1e-3, "inside an obstacle at {p:?}");
            }
            assert!(env.human.hunger_s <= env.cfg.human.hunger_interval_s + 1e-3);
        });
    }

    #[test]
    fn same_seed_and_actions_give_the_same_trajectory(seed in any::<u64>()) {
        let run = || {
            let mut env = Env::new(small_config(), seed).unwrap();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut hashes = Vec::new();
            random_episode(&mut env, &mut rng, 200, |env| hashes.push(env.state_hash()));
            hashes
        };
        prop_assert_eq!(run(), run());
    }
}

#[test]
fn different_seeds_give_different_worlds() {
    let a = Env::new(Config::default(), 1).unwrap();
    let b = Env::new(Config::default(), 2).unwrap();
    assert_ne!(a.state_hash(), b.state_hash());
    let c = Env::new(Config::default(), 1).unwrap();
    assert_eq!(a.state_hash(), c.state_hash());
}
