//! Headless runner: plays episodes with uniformly random (masked) actions and prints
//! outcome statistics and throughput.
//!
//! Usage: zai-random [--episodes N] [--seed S]
//! `ZAI_ZOMBIE_COUNT` and `ZAI_FOOD_COUNT` override the default counts.

use std::time::Instant;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use zai_sim::env::{HumanAction, INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::{Config, Env, Outcome};

fn main() {
    let mut episodes: u64 = 20;
    let mut seed: u64 = 0;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next();
        let parsed = value.as_deref().and_then(|v| v.parse().ok());
        match (arg.as_str(), parsed) {
            ("--episodes", Some(v)) => episodes = v,
            ("--seed", Some(v)) => seed = v,
            _ => {
                eprintln!("usage: zai-random [--episodes N] [--seed S]");
                std::process::exit(2);
            }
        }
    }

    let cfg = match Config::default().with_env_overrides() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let mut env = match Env::new(cfg, seed) {
        Ok(env) => env,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let (mut caught, mut starved, mut truncated) = (0u64, 0u64, 0u64);
    let mut steps = 0u64;
    let mut sim_time = 0.0f64;
    let mut zs = Vec::new();
    let start = Instant::now();

    for ep in 0..episodes {
        env.reset(seed.wrapping_add(ep));
        loop {
            let mask = env.human_action_mask();
            let valid: Vec<u8> = (0..INTERACT_ACTIONS as u8)
                .filter(|&i| mask[MOVE_ACTIONS + i as usize])
                .collect();
            let action = HumanAction {
                movement: rng.random_range(0..MOVE_ACTIONS as u8),
                interact: valid[rng.random_range(0..valid.len())],
            };
            zs.clear();
            zs.extend((0..env.zombies().len()).map(|_| rng.random_range(0..MOVE_ACTIONS as u8)));
            let r = env.step(action, &zs);
            steps += 1;
            match r.outcome {
                Outcome::Running => continue,
                Outcome::Caught => caught += 1,
                Outcome::Starved => starved += 1,
                Outcome::Truncated => truncated += 1,
            }
            sim_time += env.time() as f64;
            break;
        }
    }

    let secs = start.elapsed().as_secs_f64();
    println!(
        "episodes {episodes}: caught {caught}, starved {starved}, truncated {truncated}; \
         mean length {:.1} s; {steps} decision steps in {secs:.2} s ({:.0} steps/s, 1 thread)",
        sim_time / episodes.max(1) as f64,
        steps as f64 / secs
    );
}
