//! Core simulation of the ZAI humans-vs-zombies environment.
//!
//! This crate holds the game rules only (no I/O) and is shared by the Python
//! training bindings (`zai-py`) and the game server (`zai-server`).

pub mod bots;
pub mod config;
pub mod env;
pub mod geom;
pub mod map;
pub mod obs;
pub mod stats;

#[cfg(test)]
mod tests;

pub use config::{Config, ConfigError, ZombieShapingMode};
pub use env::{Env, HumanAction, NoiseKind, Outcome, StepResult};
pub use obs::{HUMAN_OBS_DIM, OBS_VERSION, STATE_DIM, ZOMBIE_OBS_DIM};
pub use stats::{EpisodeStats, STATS_DIM, STATS_LAYOUT};

/// Version of the simulation crate, exposed to the bindings and the server.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Observation slot limits; configurations above them are rejected.
pub const MAX_ZOMBIES: usize = 64;
pub const MAX_FOOD: usize = 128;
pub const MAX_HOUSES: usize = 10;
