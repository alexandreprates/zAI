//! Environment configuration. Defaults are the PRD values (curriculum stage S2).

use serde::{Deserialize, Serialize};

use crate::{MAX_FOOD, MAX_HOUSES, MAX_ZOMBIES};

/// Full environment configuration. All distances are in meters, times in seconds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// Side of the square map.
    pub map_size_m: f32,
    pub house_count: usize,
    pub zombie_count: usize,
    pub food_count: usize,
    /// Delay before a picked-up food package reappears elsewhere; `None` disables respawn.
    pub food_respawn_interval_s: Option<f32>,
    /// Episode length cap (truncation).
    pub episode_cap_s: f32,
    /// Physics tick length.
    pub dt_s: f32,
    /// Physics ticks per decision step.
    pub action_repeat: u32,
    /// Minimum distance between the human and any zombie at spawn (best effort on small maps).
    pub min_spawn_distance_m: f32,
    pub entity_radius_m: f32,
    pub pickup_range_m: f32,
    pub door_range_m: f32,
    /// Minimum fraction of the initial food placed inside houses.
    pub food_inside_house_fraction: f32,
    pub human: HumanConfig,
    pub zombie: ZombieConfig,
    pub noise: NoiseConfig,
    pub house: HouseConfig,
    pub reward: RewardConfig,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HumanConfig {
    pub vision_radius_m: f32,
    /// Field of view centered on the heading (360 = omnidirectional).
    pub vision_fov_deg: f32,
    pub base_speed_mps: f32,
    /// Speed reduction per carried package, as a fraction of the base speed (linear).
    pub speed_penalty_per_food: f32,
    /// A package must be eaten every `hunger_interval_s`.
    pub hunger_interval_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZombieConfig {
    pub hearing_radius_m: f32,
    pub speed_mps: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NoiseConfig {
    pub walk_radius_m: f32,
    pub food_pickup_radius_m: f32,
    pub door_close_radius_m: f32,
    /// Std-dev of the heard source position, as a fraction of the distance (0 = exact).
    pub localization_sigma: f32,
    /// Heard events older than this are forgotten.
    pub memory_s: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct HouseConfig {
    pub width_m: f32,
    pub depth_m: f32,
    pub door_width_m: f32,
    /// Maximum jitter of a house center around its grid cell center.
    pub jitter_m: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RewardConfig {
    pub gamma: f32,
    pub human_survival_per_step: f32,
    pub human_death: f32,
    pub zombie_catch: f32,
    pub zombie_step_penalty: f32,
    /// Scale of the potential-based food shaping for the human (0 disables it).
    pub human_food_shaping: f32,
    /// Food reserve (seconds until starvation) at which the food shaping saturates.
    pub human_food_shaping_reserve_s: f32,
    /// Scale of the potential-based distance shaping for zombies (0 disables it).
    pub zombie_distance_shaping: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            map_size_m: 500.0,
            house_count: 10,
            zombie_count: 20,
            food_count: 120,
            food_respawn_interval_s: Some(60.0),
            episode_cap_s: 1200.0,
            dt_s: 0.1,
            action_repeat: 5,
            min_spawn_distance_m: 150.0,
            entity_radius_m: 0.3,
            pickup_range_m: 1.5,
            door_range_m: 2.0,
            food_inside_house_fraction: 0.05,
            human: HumanConfig::default(),
            zombie: ZombieConfig::default(),
            noise: NoiseConfig::default(),
            house: HouseConfig::default(),
            reward: RewardConfig::default(),
        }
    }
}

impl Default for HumanConfig {
    fn default() -> Self {
        Self {
            vision_radius_m: 100.0,
            vision_fov_deg: 120.0,
            base_speed_mps: 1.0,
            speed_penalty_per_food: 0.2,
            hunger_interval_s: 90.0,
        }
    }
}

impl Default for ZombieConfig {
    fn default() -> Self {
        Self {
            hearing_radius_m: 300.0,
            speed_mps: 0.7,
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            walk_radius_m: 10.0,
            food_pickup_radius_m: 100.0,
            door_close_radius_m: 200.0,
            localization_sigma: 0.0,
            memory_s: 30.0,
        }
    }
}

impl Default for HouseConfig {
    fn default() -> Self {
        Self {
            width_m: 12.0,
            depth_m: 8.0,
            door_width_m: 1.5,
            jitter_m: 30.0,
        }
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            gamma: 0.995,
            human_survival_per_step: 0.005,
            human_death: -1.0,
            zombie_catch: 1.0,
            zombie_step_penalty: -0.002,
            human_food_shaping: 0.0,
            human_food_shaping_reserve_s: 180.0,
            zombie_distance_shaping: 0.0,
        }
    }
}

/// Error returned when a configuration is outside the supported ranges.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfigError(pub String);

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid config: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

impl Config {
    /// Apply the PRD environment variables `ZAI_ZOMBIE_COUNT` and `ZAI_FOOD_COUNT`.
    pub fn with_env_overrides(mut self) -> Result<Self, ConfigError> {
        if let Some(v) = read_env_usize("ZAI_ZOMBIE_COUNT")? {
            self.zombie_count = v;
        }
        if let Some(v) = read_env_usize("ZAI_FOOD_COUNT")? {
            self.food_count = v;
        }
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let err = |msg: String| Err(ConfigError(msg));
        if !(1..=MAX_ZOMBIES).contains(&self.zombie_count) {
            return err(format!("zombie_count must be in 1..={MAX_ZOMBIES}"));
        }
        if self.food_count > MAX_FOOD {
            return err(format!("food_count must be <= {MAX_FOOD}"));
        }
        if self.house_count > MAX_HOUSES {
            return err(format!("house_count must be <= {MAX_HOUSES}"));
        }
        if self.map_size_m < 50.0 {
            return err("map_size_m must be >= 50".into());
        }
        if self.dt_s <= 0.0 || self.action_repeat == 0 || self.episode_cap_s <= 0.0 {
            return err("dt_s, action_repeat and episode_cap_s must be positive".into());
        }
        let step = self.dt_s * self.human.base_speed_mps.max(self.zombie.speed_mps);
        if step >= self.entity_radius_m {
            return err("per-tick displacement must be smaller than entity_radius_m".into());
        }
        if self.house.door_width_m <= 2.0 * self.entity_radius_m
            || self.house.door_width_m >= self.house.width_m
        {
            return err("door_width_m must fit an entity and be narrower than the house".into());
        }
        if !(self.human.vision_fov_deg > 0.0 && self.human.vision_fov_deg <= 360.0) {
            return err("human.vision_fov_deg must be in (0, 360]".into());
        }
        if let Some(i) = self.food_respawn_interval_s {
            if i < 0.0 {
                return err("food_respawn_interval_s must be >= 0".into());
            }
        }
        Ok(())
    }

    /// Duration of one decision step.
    pub fn decision_dt(&self) -> f32 {
        self.dt_s * self.action_repeat as f32
    }
}

fn read_env_usize(name: &str) -> Result<Option<usize>, ConfigError> {
    match std::env::var(name) {
        Ok(v) => {
            v.trim().parse().map(Some).map_err(|_| {
                ConfigError(format!("{name} must be a non-negative integer, got {v:?}"))
            })
        }
        Err(_) => Ok(None),
    }
}
