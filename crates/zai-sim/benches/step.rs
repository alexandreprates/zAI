//! Decision-step throughput of one environment, with and without observation encoding.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use zai_sim::env::{HumanAction, INTERACT_ACTIONS, MOVE_ACTIONS};
use zai_sim::obs::{write_human_obs, write_state, write_zombie_obs};
use zai_sim::{Config, Env, Outcome, HUMAN_OBS_DIM, STATE_DIM, ZOMBIE_OBS_DIM};

fn random_step(env: &mut Env, rng: &mut ChaCha8Rng, zs: &mut Vec<u8>) {
    zs.clear();
    zs.extend((0..env.zombies().len()).map(|_| rng.random_range(0..MOVE_ACTIONS as u8)));
    let action = HumanAction {
        movement: rng.random_range(0..MOVE_ACTIONS as u8),
        interact: rng.random_range(0..INTERACT_ACTIONS as u8),
    };
    if env.step(action, zs).outcome != Outcome::Running {
        let seed = rng.random();
        env.reset(seed);
    }
}

fn bench_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("decision_step");
    group.throughput(Throughput::Elements(1));

    let mut env = Env::new(Config::default(), 0).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let mut zs = Vec::new();
    group.bench_function("physics_only", |b| {
        b.iter(|| random_step(&mut env, &mut rng, &mut zs))
    });

    let mut h = vec![0.0; HUMAN_OBS_DIM];
    let mut z = vec![0.0; ZOMBIE_OBS_DIM];
    let mut s = vec![0.0; STATE_DIM];
    group.bench_function("with_observations", |b| {
        b.iter(|| {
            random_step(&mut env, &mut rng, &mut zs);
            write_human_obs(&env, &mut h);
            write_state(&env, &mut s);
            for i in 0..env.zombies().len() {
                write_zombie_obs(&env, i, &mut z);
            }
            let _ = env.human_action_mask();
        })
    });
    group.finish();
}

criterion_group!(benches, bench_step);
criterion_main!(benches);
