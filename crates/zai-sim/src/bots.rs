//! Fixed policies for live server matches and demos.
//!
//! These mirror the Python bots in `python/zai_rl/bots` (used for the balance sweep and as
//! training opponents) but read the same information from the `Env` directly instead of
//! from observation vectors. Their behavior only needs to be similar, not bit-identical.

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::env::{
    Env, HumanAction, INTERACT_ACTIONS, INTERACT_NONE, INTERACT_PICKUP, MOVE_ACTIONS, STAY,
};

/// Nearest of the 16 movement actions for a direction (`STAY` for a zero vector).
pub fn direction_to_action(d: Vec2) -> u8 {
    if d == Vec2::ZERO {
        return STAY;
    }
    let sector = std::f32::consts::PI / 8.0;
    ((d.y.atan2(d.x) / sector).round() as i32).rem_euclid(16) as u8
}

/// Policy kinds selectable for server matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BotKind {
    Scripted,
    Random,
}

/// Human policy: flee visible zombies closer than `flee_m`; while carrying fewer than
/// `reserve` packages go to the nearest visible food and pick it up; otherwise walk to a
/// random house (random waypoints on maps without houses).
#[derive(Clone, Debug)]
pub struct ScriptedHuman {
    rng: ChaCha8Rng,
    target_house: usize,
    waypoint: Vec2,
    pub flee_m: f32,
    pub reserve: u32,
    pub arrive_m: f32,
}

impl ScriptedHuman {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            target_house: usize::MAX,
            waypoint: Vec2::NAN,
            flee_m: 30.0,
            reserve: 2,
            arrive_m: 3.0,
        }
    }

    pub fn act(&mut self, env: &Env) -> HumanAction {
        let h = env.human();
        let p = h.pos;

        let mut away = Vec2::ZERO;
        let mut flee = false;
        for (i, d) in env.visible_zombies() {
            if d < self.flee_m {
                flee = true;
                away -= (env.zombies()[i].pos - p) / d.max(1e-3).powi(2);
            }
        }
        if flee {
            return HumanAction {
                movement: direction_to_action(away),
                interact: INTERACT_NONE,
            };
        }

        if h.carried < self.reserve {
            if let Some(&(i, _)) = env.visible_food().first() {
                let interact = if env.pickup_target().is_some() {
                    INTERACT_PICKUP
                } else {
                    INTERACT_NONE
                };
                return HumanAction {
                    movement: direction_to_action(env.food()[i].pos - p),
                    interact,
                };
            }
        }

        let houses = &env.map().houses;
        let target = if houses.is_empty() {
            let size = env.map().size;
            if self.waypoint.is_nan() || self.waypoint.distance(p) < self.arrive_m {
                self.waypoint = Vec2::new(
                    self.rng.random_range(0.0..size),
                    self.rng.random_range(0.0..size),
                );
            }
            self.waypoint
        } else {
            if self.target_house >= houses.len()
                || houses[self.target_house].center.distance(p) < self.arrive_m
            {
                self.target_house = self.rng.random_range(0..houses.len());
            }
            houses[self.target_house].center
        };
        HumanAction {
            movement: direction_to_action(target - p),
            interact: INTERACT_NONE,
        }
    }
}

/// Zombie policy: walk to the most recent heard noise until within `arrive_m`, otherwise
/// wander between random waypoints; a blocked zombie takes a random step.
#[derive(Clone, Debug)]
pub struct ScriptedZombies {
    rng: ChaCha8Rng,
    waypoints: Vec<Vec2>,
    pub arrive_m: f32,
}

impl ScriptedZombies {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            waypoints: Vec::new(),
            arrive_m: 3.0,
        }
    }

    pub fn act(&mut self, env: &Env) -> Vec<u8> {
        let size = env.map().size;
        let zombies = env.zombies();
        if self.waypoints.len() != zombies.len() {
            self.waypoints = (0..zombies.len())
                .map(|_| random_point(&mut self.rng, size))
                .collect();
        }
        zombies
            .iter()
            .zip(self.waypoints.iter_mut())
            .map(|(z, wp)| {
                if z.blocked {
                    *wp = random_point(&mut self.rng, size);
                    return self.rng.random_range(0..STAY);
                }
                let newest = z.heard.iter().max_by(|a, b| a.time.total_cmp(&b.time));
                if let Some(e) = newest.filter(|e| e.pos.distance(z.pos) > self.arrive_m) {
                    return direction_to_action(e.pos - z.pos);
                }
                if wp.distance(z.pos) < self.arrive_m {
                    *wp = random_point(&mut self.rng, size);
                }
                direction_to_action(*wp - z.pos)
            })
            .collect()
    }
}

/// Uniform random movement; interaction uniform among the valid ones.
#[derive(Clone, Debug)]
pub struct RandomHuman {
    rng: ChaCha8Rng,
}

impl RandomHuman {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn act(&mut self, env: &Env) -> HumanAction {
        let mask = env.human_action_mask();
        let valid: Vec<u8> = (0..INTERACT_ACTIONS as u8)
            .filter(|&i| mask[MOVE_ACTIONS + i as usize])
            .collect();
        HumanAction {
            movement: self.rng.random_range(0..MOVE_ACTIONS as u8),
            interact: valid[self.rng.random_range(0..valid.len())],
        }
    }
}

/// Uniform random movement for every zombie.
#[derive(Clone, Debug)]
pub struct RandomZombies {
    rng: ChaCha8Rng,
}

impl RandomZombies {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn act(&mut self, env: &Env) -> Vec<u8> {
        (0..env.zombies().len())
            .map(|_| self.rng.random_range(0..MOVE_ACTIONS as u8))
            .collect()
    }
}

/// A human policy chosen at runtime.
#[derive(Clone, Debug)]
pub enum HumanBot {
    Scripted(ScriptedHuman),
    Random(RandomHuman),
}

impl HumanBot {
    pub fn new(kind: BotKind, seed: u64) -> Self {
        match kind {
            BotKind::Scripted => Self::Scripted(ScriptedHuman::new(seed)),
            BotKind::Random => Self::Random(RandomHuman::new(seed)),
        }
    }

    pub fn act(&mut self, env: &Env) -> HumanAction {
        match self {
            Self::Scripted(b) => b.act(env),
            Self::Random(b) => b.act(env),
        }
    }
}

/// A zombie team policy chosen at runtime.
#[derive(Clone, Debug)]
pub enum ZombieBot {
    Scripted(ScriptedZombies),
    Random(RandomZombies),
}

impl ZombieBot {
    pub fn new(kind: BotKind, seed: u64) -> Self {
        match kind {
            BotKind::Scripted => Self::Scripted(ScriptedZombies::new(seed)),
            BotKind::Random => Self::Random(RandomZombies::new(seed)),
        }
    }

    pub fn act(&mut self, env: &Env) -> Vec<u8> {
        match self {
            Self::Scripted(b) => b.act(env),
            Self::Random(b) => b.act(env),
        }
    }
}

fn random_point(rng: &mut ChaCha8Rng, size: f32) -> Vec2 {
    Vec2::new(rng.random_range(0.0..size), rng.random_range(0.0..size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Config, Outcome};

    #[test]
    fn direction_to_action_maps_the_compass() {
        assert_eq!(direction_to_action(Vec2::X), 0);
        assert_eq!(direction_to_action(Vec2::Y), 4);
        assert_eq!(direction_to_action(-Vec2::X), 8);
        assert_eq!(direction_to_action(-Vec2::Y), 12);
        assert_eq!(direction_to_action(Vec2::ZERO), STAY);
    }

    #[test]
    fn bots_play_a_full_episode_deterministically() {
        let run = || {
            let cfg = Config {
                episode_cap_s: 120.0,
                ..Config::default()
            };
            let mut env = Env::new(cfg, 11).unwrap();
            let mut h = ScriptedHuman::new(1);
            let mut z = ScriptedZombies::new(2);
            while env.outcome() == Outcome::Running {
                let a = h.act(&env);
                let za = z.act(&env);
                env.step(a, &za);
            }
            env.state_hash()
        };
        assert_eq!(run(), run());
    }
}
