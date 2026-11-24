//! Environment state and step logic.

use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};

use glam::Vec2;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::config::{Config, ConfigError, ZombieShapingMode};
use crate::map::Map;

/// Movement action meaning "do not move". Actions `0..16` move in direction `i * 22.5` degrees.
pub const STAY: u8 = 16;
/// Number of movement actions (16 directions + stay).
pub const MOVE_ACTIONS: usize = 17;
/// Human interaction actions.
pub const INTERACT_NONE: u8 = 0;
pub const INTERACT_PICKUP: u8 = 1;
pub const INTERACT_CLOSE_DOOR: u8 = 2;
pub const INTERACT_ACTIONS: usize = 3;
/// Capacity of each zombie's heard-noise buffer.
pub const HEARD_CAPACITY: usize = 8;
/// A new walking noise replaces the newest heard walking noise if that one is younger than this.
const WALK_MERGE_S: f32 = 2.0;
/// Weight of the food proximity term in the human food potential. Below the 0.5 reserve gain of
/// a pickup, so the pickup itself stays rewarded although its target disappears.
const FOOD_PROXIMITY_WEIGHT: f32 = 0.25;
/// Minimum human displacement in a decision step that produces a walking noise.
const WALK_NOISE_MIN_M: f32 = 0.05;

/// Unit vector for a movement action (zero for `STAY` or out-of-range values).
pub fn move_direction(action: u8) -> Vec2 {
    if action < STAY {
        let a = action as f32 * (PI / 8.0);
        Vec2::new(a.cos(), a.sin())
    } else {
        Vec2::ZERO
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoiseKind {
    Walk,
    Pickup,
    Door,
}

/// A noise emitted during the last step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseEvent {
    pub pos: Vec2,
    pub kind: NoiseKind,
    pub radius: f32,
    pub time: f32,
}

/// A noise as remembered by a zombie (position may be perturbed by localization noise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeardEvent {
    pub pos: Vec2,
    pub kind: NoiseKind,
    pub time: f32,
}

#[derive(Clone, Debug)]
pub struct Human {
    pub pos: Vec2,
    pub vel: Vec2,
    pub heading: Vec2,
    pub carried: u32,
    pub hunger_s: f32,
    pub last_noise_time: Option<f32>,
    /// Last observed closed state of each house's doors (`[front, back]`).
    pub door_memory: Vec<[bool; 2]>,
    /// Whether any door of each house is currently visible.
    pub doors_visible: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct Zombie {
    pub pos: Vec2,
    pub vel: Vec2,
    pub heading: Vec2,
    /// True if the last tick's movement was mostly blocked by an obstacle.
    pub blocked: bool,
    /// Heard noises, oldest first.
    pub heard: VecDeque<HeardEvent>,
    pub last_heard_time: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct Food {
    pub pos: Vec2,
    pub active: bool,
    /// Simulation time at which an inactive package reappears.
    pub respawn_at: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Running,
    /// A zombie touched the human.
    Caught,
    /// The hunger timer expired with no food carried.
    Starved,
    /// The episode cap was reached.
    Truncated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HumanAction {
    pub movement: u8,
    pub interact: u8,
}

impl HumanAction {
    pub const IDLE: Self = Self {
        movement: STAY,
        interact: INTERACT_NONE,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepResult {
    pub human_reward: f32,
    /// Team part of the zombie reward (catch bonus and step penalty), identical for every
    /// zombie. The full per-zombie reward, including distance shaping, is `Env::zombie_rewards`.
    pub zombie_reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub outcome: Outcome,
}

/// One match. Deterministic given `(config, seed, actions)`.
#[derive(Clone, Debug)]
pub struct Env {
    pub(crate) cfg: Config,
    pub(crate) rng: ChaCha8Rng,
    pub(crate) seed: u64,
    pub(crate) map: Map,
    pub(crate) human: Human,
    pub(crate) zombies: Vec<Zombie>,
    pub(crate) food: Vec<Food>,
    pub(crate) time: f32,
    pub(crate) decisions: u32,
    pub(crate) outcome: Outcome,
    pub(crate) events: Vec<NoiseEvent>,
    /// Per-zombie reward of the last step (team reward plus distance shaping).
    pub(crate) zombie_rewards: Vec<f32>,
}

impl Env {
    pub fn new(cfg: Config, seed: u64) -> Result<Self, ConfigError> {
        cfg.validate()?;
        let mut env = Self {
            map: Map {
                size: cfg.map_size_m,
                houses: Vec::new(),
            },
            cfg,
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
            human: Human {
                pos: Vec2::ZERO,
                vel: Vec2::ZERO,
                heading: Vec2::X,
                carried: 0,
                hunger_s: 0.0,
                last_noise_time: None,
                door_memory: Vec::new(),
                doors_visible: Vec::new(),
            },
            zombies: Vec::new(),
            food: Vec::new(),
            time: 0.0,
            decisions: 0,
            outcome: Outcome::Running,
            events: Vec::new(),
            zombie_rewards: Vec::new(),
        };
        env.reset(seed);
        Ok(env)
    }

    /// Starts a new episode with a freshly generated map.
    pub fn reset(&mut self, seed: u64) {
        let cfg = &self.cfg;
        let r = cfg.entity_radius_m;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let map = Map::generate(cfg, &mut rng);

        let human_pos = open_position(&map, &mut rng, r + 0.5);
        let zombies = (0..cfg.zombie_count)
            .map(|_| {
                let mut best = open_position(&map, &mut rng, r + 0.5);
                for _ in 0..200 {
                    if best.distance(human_pos) >= cfg.min_spawn_distance_m {
                        break;
                    }
                    let p = open_position(&map, &mut rng, r + 0.5);
                    if p.distance(human_pos) > best.distance(human_pos) {
                        best = p;
                    }
                }
                Zombie {
                    pos: best,
                    vel: Vec2::ZERO,
                    heading: Vec2::X,
                    blocked: false,
                    heard: VecDeque::with_capacity(HEARD_CAPACITY),
                    last_heard_time: None,
                }
            })
            .collect();

        let inside = if map.houses.is_empty() {
            0
        } else {
            (cfg.food_inside_house_fraction * cfg.food_count as f32).ceil() as usize
        };
        let food = (0..cfg.food_count)
            .map(|i| Food {
                pos: if i < inside {
                    position_inside_house(&map, &mut rng, r)
                } else {
                    food_position(&map, &mut rng, r)
                },
                active: true,
                respawn_at: f32::INFINITY,
            })
            .collect();

        let houses = map.houses.len();
        self.human = Human {
            pos: human_pos,
            vel: Vec2::ZERO,
            heading: Vec2::X,
            carried: 0,
            hunger_s: cfg.human.hunger_interval_s,
            last_noise_time: None,
            door_memory: vec![[false; 2]; houses],
            doors_visible: vec![false; houses],
        };
        self.map = map;
        self.zombies = zombies;
        self.food = food;
        self.rng = rng;
        self.seed = seed;
        self.time = 0.0;
        self.decisions = 0;
        self.outcome = Outcome::Running;
        self.events.clear();
        self.zombie_rewards.clear();
        self.zombie_rewards.resize(self.zombies.len(), 0.0);
        self.update_perception();
    }

    /// Advances one decision step (`action_repeat` physics ticks).
    ///
    /// Panics if the episode has ended or `zombie_actions.len() != zombie_count`.
    pub fn step(&mut self, human: HumanAction, zombie_actions: &[u8]) -> StepResult {
        assert_eq!(
            self.outcome,
            Outcome::Running,
            "step() called on a finished episode; call reset()"
        );
        assert_eq!(
            zombie_actions.len(),
            self.zombies.len(),
            "one action per zombie is required"
        );
        let phi_h0 = self.human_potential();
        let mut phi_z0 = std::mem::take(&mut self.zombie_rewards);
        self.zombie_potentials(&mut phi_z0);
        self.events.clear();

        // Interaction happens before movement; invalid (masked) interactions are no-ops.
        match human.interact {
            INTERACT_PICKUP => {
                if let Some(i) = self.pickup_target() {
                    self.pickup(i);
                }
            }
            INTERACT_CLOSE_DOOR => {
                if let Some((h, d)) = self.door_target() {
                    self.close_door(h, d);
                }
            }
            _ => {}
        }

        let human_dir = move_direction(human.movement);
        let zombie_dirs: Vec<Vec2> = zombie_actions.iter().map(|&a| move_direction(a)).collect();
        let start = self.human.pos;
        for _ in 0..self.cfg.action_repeat {
            if let Some(end) = self.tick(human_dir, &zombie_dirs) {
                self.outcome = end;
                break;
            }
        }
        if self.human.pos.distance(start) > WALK_NOISE_MIN_M {
            self.emit(self.human.pos, NoiseKind::Walk);
        }
        self.route_noise();
        self.decisions += 1;
        if self.outcome == Outcome::Running && self.decisions >= self.decision_cap() {
            self.outcome = Outcome::Truncated;
        }
        self.update_perception();
        self.rewards(phi_h0, phi_z0)
    }

    /// Computes the step rewards; `phi_z` holds the zombie potentials before the step and is
    /// turned into the per-zombie rewards (`zombie_rewards`).
    fn rewards(&mut self, phi_h0: f32, mut phi_z: Vec<f32>) -> StepResult {
        let terminated = matches!(self.outcome, Outcome::Caught | Outcome::Starved);
        let phi_h1 = if terminated {
            0.0
        } else {
            self.human_potential()
        };
        let mut phi_z1 = vec![0.0; phi_z.len()];
        self.zombie_potentials(&mut phi_z1);
        let rc = &self.cfg.reward;
        let mut human_reward = if terminated {
            rc.human_death
        } else {
            rc.human_survival_per_step
        };
        human_reward += rc.gamma * phi_h1 - phi_h0;
        let mut zombie_reward = rc.zombie_step_penalty;
        if self.outcome == Outcome::Caught {
            zombie_reward += rc.zombie_catch;
        }
        for (r, p1) in phi_z.iter_mut().zip(phi_z1) {
            *r = zombie_reward + rc.gamma * p1 - *r;
        }
        self.zombie_rewards = phi_z;
        StepResult {
            human_reward,
            zombie_reward,
            terminated,
            truncated: self.outcome == Outcome::Truncated,
            outcome: self.outcome,
        }
    }

    /// Potential for the human food shaping (0 when disabled): food reserve (hunger timer plus
    /// one interval per carried package, capped) plus `FOOD_PROXIMITY_WEIGHT` times the
    /// proximity to the nearest visible food. It is non-negative, so ending the episode (potential 0) never pays a
    /// bonus; picking up food raises it, eating from the inventory leaves it unchanged.
    fn human_potential(&self) -> f32 {
        let rc = &self.cfg.reward;
        if rc.human_food_shaping == 0.0 {
            return 0.0;
        }
        let h = &self.human;
        let reserve_s = h.hunger_s + self.cfg.human.hunger_interval_s * h.carried as f32;
        let reserve =
            reserve_s.min(rc.human_food_shaping_reserve_s) / rc.human_food_shaping_reserve_s;
        let range = self.cfg.human.vision_radius_m;
        let proximity = self
            .visible_food()
            .first()
            .map_or(0.0, |&(_, d)| 1.0 - d.min(range) / range);
        rc.human_food_shaping * (reserve + FOOD_PROXIMITY_WEIGHT * proximity)
    }

    /// Per-zombie rewards of the last step: team reward plus potential-based distance shaping.
    pub fn zombie_rewards(&self) -> &[f32] {
        &self.zombie_rewards
    }

    /// Potentials of the zombie distance shaping (all 0 when disabled), written into `out`.
    /// Uses privileged state. `Phi = c * (1 - min(d, range) / range)` is non-negative and
    /// takes its maximum `c` when the human is caught and 0 when it starves, so neither
    /// ending pays more than its own reward.
    pub(crate) fn zombie_potentials(&self, out: &mut Vec<f32>) {
        let rc = &self.cfg.reward;
        let c = rc.zombie_distance_shaping;
        out.clear();
        let value = match self.outcome {
            _ if c == 0.0 => Some(0.0),
            Outcome::Caught => Some(c),
            Outcome::Starved => Some(0.0),
            Outcome::Running | Outcome::Truncated => None,
        };
        if let Some(v) = value {
            out.resize(self.zombies.len(), v);
            return;
        }
        let range = rc.zombie_distance_shaping_range_m;
        let phi = |d: f32| c * (1.0 - d.min(range) / range);
        let dist = self.zombies.iter().map(|z| z.pos.distance(self.human.pos));
        match rc.zombie_distance_shaping_mode {
            ZombieShapingMode::PerAgent => out.extend(dist.map(phi)),
            ZombieShapingMode::TeamMin => {
                let v = phi(dist.fold(f32::INFINITY, f32::min));
                out.resize(self.zombies.len(), v);
            }
        }
    }

    /// One physics tick. Returns the outcome if the episode ended.
    fn tick(&mut self, human_dir: Vec2, zombie_dirs: &[Vec2]) -> Option<Outcome> {
        let dt = self.cfg.dt_s;
        let r = self.cfg.entity_radius_m;
        self.time += dt;

        let speed = self.human_speed();
        let h = &mut self.human;
        let old = h.pos;
        h.pos = resolve(&self.map, old, human_dir * speed * dt, r);
        h.vel = (h.pos - old) / dt;
        if human_dir != Vec2::ZERO {
            h.heading = human_dir;
        }

        let zspeed = self.cfg.zombie.speed_mps;
        for (z, &dir) in self.zombies.iter_mut().zip(zombie_dirs) {
            let old = z.pos;
            let disp = dir * zspeed * dt;
            z.pos = resolve(&self.map, old, disp, r);
            z.vel = (z.pos - old) / dt;
            if dir != Vec2::ZERO {
                z.heading = dir;
            }
            z.blocked = dir != Vec2::ZERO && z.pos.distance(old) < 0.5 * disp.length();
        }
        self.separate_zombies();

        let contact = 2.0 * r;
        let caught = self.zombies.iter().any(|z| {
            z.pos.distance_squared(self.human.pos) <= contact * contact
                && self.map.line_of_sight(z.pos, self.human.pos)
        });
        if caught {
            return Some(Outcome::Caught);
        }

        self.human.hunger_s -= dt;
        // Half-tick tolerance absorbs f32 accumulation error, so 90 s is exactly 900 ticks.
        if self.human.hunger_s <= 0.5 * dt {
            if self.human.carried > 0 {
                self.human.carried -= 1;
                self.human.hunger_s += self.cfg.human.hunger_interval_s;
            } else {
                return Some(Outcome::Starved);
            }
        }

        if self.cfg.food_respawn_interval_s.is_some() {
            for i in 0..self.food.len() {
                if !self.food[i].active && self.food[i].respawn_at <= self.time {
                    self.food[i].pos = food_position(&self.map, &mut self.rng, r);
                    self.food[i].active = true;
                    self.food[i].respawn_at = f32::INFINITY;
                }
            }
        }
        None
    }

    /// Soft push between overlapping zombies, then wall resolution.
    fn separate_zombies(&mut self) {
        let r = self.cfg.entity_radius_m;
        let min = 2.0 * r;
        let n = self.zombies.len();
        for i in 0..n {
            for j in i + 1..n {
                let d = self.zombies[j].pos - self.zombies[i].pos;
                let dist2 = d.length_squared();
                if dist2 < min * min {
                    let dist = dist2.sqrt();
                    let normal = if dist > 1e-6 { d / dist } else { Vec2::X };
                    let push = normal * ((min - dist) * 0.25);
                    self.zombies[i].pos -= push;
                    self.zombies[j].pos += push;
                }
            }
        }
        for z in &mut self.zombies {
            z.pos = resolve(&self.map, z.pos, Vec2::ZERO, r);
        }
    }

    pub fn human_speed(&self) -> f32 {
        let h = &self.cfg.human;
        h.base_speed_mps * (1.0 - h.speed_penalty_per_food * self.human.carried as f32).max(0.0)
    }

    /// Nearest active food within pickup range and line of sight.
    pub fn pickup_target(&self) -> Option<usize> {
        let p = self.human.pos;
        let range2 = self.cfg.pickup_range_m * self.cfg.pickup_range_m;
        self.food
            .iter()
            .enumerate()
            .filter(|(_, f)| f.active && f.pos.distance_squared(p) <= range2)
            .filter(|(_, f)| self.map.line_of_sight(p, f.pos))
            .min_by(|a, b| {
                a.1.pos
                    .distance_squared(p)
                    .total_cmp(&b.1.pos.distance_squared(p))
            })
            .map(|(i, _)| i)
    }

    /// Nearest open door within range and line of sight whose doorway is not occupied.
    pub fn door_target(&self) -> Option<(usize, usize)> {
        let p = self.human.pos;
        let range = self.cfg.door_range_m;
        let r = self.cfg.entity_radius_m;
        let mut best: Option<((usize, usize), f32)> = None;
        for (hi, house) in self.map.houses.iter().enumerate() {
            if house.center.distance(p) > house.bounding_radius + range {
                continue;
            }
            for (di, door) in house.doors.iter().enumerate() {
                let dist = door.segment.distance(p);
                if door.closed || dist > range || best.is_some_and(|(_, d)| d <= dist) {
                    continue;
                }
                let occupied = std::iter::once(p)
                    .chain(self.zombies.iter().map(|z| z.pos))
                    .any(|e| door.segment.distance(e) < r);
                if !occupied && self.map.line_of_sight(p, door.center) {
                    best = Some(((hi, di), dist));
                }
            }
        }
        best.map(|(t, _)| t)
    }

    /// Validity mask: 17 movement actions followed by 3 interaction actions.
    pub fn human_action_mask(&self) -> [bool; MOVE_ACTIONS + INTERACT_ACTIONS] {
        let mut m = [true; MOVE_ACTIONS + INTERACT_ACTIONS];
        m[MOVE_ACTIONS + INTERACT_PICKUP as usize] = self.pickup_target().is_some();
        m[MOVE_ACTIONS + INTERACT_CLOSE_DOOR as usize] = self.door_target().is_some();
        m
    }

    fn pickup(&mut self, i: usize) {
        let f = &mut self.food[i];
        f.active = false;
        f.respawn_at = self
            .cfg
            .food_respawn_interval_s
            .map_or(f32::INFINITY, |s| self.time + s);
        let pos = f.pos;
        self.human.carried += 1;
        self.emit(pos, NoiseKind::Pickup);
    }

    fn close_door(&mut self, house: usize, door: usize) {
        let h = &mut self.map.houses[house];
        h.doors[door].closed = true;
        h.doors[1 - door].closed = false;
        let pos = h.doors[door].center;
        self.human.door_memory[house] = [h.doors[0].closed, h.doors[1].closed];
        self.emit(pos, NoiseKind::Door);
    }

    fn emit(&mut self, pos: Vec2, kind: NoiseKind) {
        let n = &self.cfg.noise;
        let radius = match kind {
            NoiseKind::Walk => n.walk_radius_m,
            NoiseKind::Pickup => n.food_pickup_radius_m,
            NoiseKind::Door => n.door_close_radius_m,
        };
        self.events.push(NoiseEvent {
            pos,
            kind,
            radius,
            time: self.time,
        });
        self.human.last_noise_time = Some(self.time);
    }

    /// Delivers this step's noise events to zombies in range and forgets old ones.
    fn route_noise(&mut self) {
        let Self {
            cfg,
            rng,
            zombies,
            events,
            time,
            ..
        } = self;
        let hearing = cfg.zombie.hearing_radius_m;
        let sigma = cfg.noise.localization_sigma;
        for ev in events.iter() {
            let range = ev.radius.min(hearing);
            for z in zombies.iter_mut() {
                let dist = z.pos.distance(ev.pos);
                if dist > range {
                    continue;
                }
                let mut pos = ev.pos;
                if sigma > 0.0 {
                    pos += gaussian2(rng) * (sigma * dist);
                }
                push_heard(
                    &mut z.heard,
                    HeardEvent {
                        pos,
                        kind: ev.kind,
                        time: ev.time,
                    },
                );
                z.last_heard_time = Some(ev.time);
            }
        }
        let memory = cfg.noise.memory_s;
        for z in zombies.iter_mut() {
            z.heard.retain(|e| *time - e.time <= memory);
        }
    }

    /// Updates what the human remembers about door states.
    fn update_perception(&mut self) {
        for hi in 0..self.map.houses.len() {
            let mut visible = false;
            for di in 0..2 {
                let door = &self.map.houses[hi].doors[di];
                if self.human_sees(door.center) {
                    self.human.door_memory[hi][di] = door.closed;
                    visible = true;
                }
            }
            self.human.doors_visible[hi] = visible;
        }
    }

    /// True if `target` is within the human's field of view (around the heading).
    pub fn in_human_fov(&self, target: Vec2) -> bool {
        let fov = self.cfg.human.vision_fov_deg;
        if fov >= 360.0 {
            return true;
        }
        let d = target - self.human.pos;
        let len = d.length();
        if len < 1e-6 {
            return true;
        }
        let cos = (d / len).dot(self.human.heading.normalize_or(Vec2::X));
        cos >= (fov * 0.5).to_radians().cos()
    }

    /// True if the human sees `target`: within vision radius, field of view and line of sight.
    pub fn human_sees(&self, target: Vec2) -> bool {
        let p = self.human.pos;
        target.distance(p) <= self.cfg.human.vision_radius_m
            && self.in_human_fov(target)
            && self.map.line_of_sight(p, target)
    }

    /// Visible active food, nearest first.
    pub fn visible_food(&self) -> Vec<(usize, f32)> {
        let p = self.human.pos;
        let mut v: Vec<(usize, f32)> = self
            .food
            .iter()
            .enumerate()
            .filter(|(_, f)| f.active && self.human_sees(f.pos))
            .map(|(i, f)| (i, f.pos.distance(p)))
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v
    }

    /// Zombies visible to the human, nearest first.
    pub fn visible_zombies(&self) -> Vec<(usize, f32)> {
        let p = self.human.pos;
        let mut v: Vec<(usize, f32)> = self
            .zombies
            .iter()
            .enumerate()
            .filter(|(_, z)| self.human_sees(z.pos))
            .map(|(i, z)| (i, z.pos.distance(p)))
            .collect();
        v.sort_by(|a, b| a.1.total_cmp(&b.1));
        v
    }

    pub fn decision_cap(&self) -> u32 {
        (self.cfg.episode_cap_s / self.cfg.decision_dt()).round() as u32
    }

    /// Hash of the full dynamic state, for determinism tests and replay checks.
    pub fn state_hash(&self) -> u64 {
        let mut h = Fnv::new();
        h.f32(self.time);
        h.u64(self.decisions as u64);
        h.vec(self.human.pos);
        h.u64(self.human.carried as u64);
        h.f32(self.human.hunger_s);
        for z in &self.zombies {
            h.vec(z.pos);
            h.u64(z.heard.len() as u64);
        }
        for f in &self.food {
            h.vec(f.pos);
            h.u64(f.active as u64);
        }
        for house in &self.map.houses {
            h.vec(house.center);
            h.u64(house.doors[0].closed as u64 | (house.doors[1].closed as u64) << 1);
        }
        h.0
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn map(&self) -> &Map {
        &self.map
    }
    pub fn human(&self) -> &Human {
        &self.human
    }
    pub fn zombies(&self) -> &[Zombie] {
        &self.zombies
    }
    pub fn food(&self) -> &[Food] {
        &self.food
    }
    /// Simulation time in seconds.
    pub fn time(&self) -> f32 {
        self.time
    }
    pub fn decisions(&self) -> u32 {
        self.decisions
    }
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }
    /// Noise events emitted during the last step.
    pub fn events(&self) -> &[NoiseEvent] {
        &self.events
    }
}

/// Moves a circle of radius `r` by `disp`, sliding along walls and closed doors.
pub(crate) fn resolve(map: &Map, from: Vec2, disp: Vec2, r: f32) -> Vec2 {
    let mut p = from + disp;
    for _ in 0..4 {
        let mut corrected = false;
        map.for_each_obstacle_near(p, r + 1.0, |s, _| {
            let c = s.closest_point(p);
            let d = p - c;
            let dist = d.length();
            if dist < r {
                let normal = if dist > 1e-6 {
                    d / dist
                } else {
                    let n = (s.b - s.a).perp().normalize_or_zero();
                    if (from - c).dot(n) >= 0.0 {
                        n
                    } else {
                        -n
                    }
                };
                p = c + normal * r;
                corrected = true;
            }
        });
        if !corrected {
            break;
        }
    }
    p.clamp(Vec2::splat(r), Vec2::splat(map.size - r))
}

fn push_heard(buf: &mut VecDeque<HeardEvent>, ev: HeardEvent) {
    if ev.kind == NoiseKind::Walk {
        if let Some(i) = buf.iter().rposition(|e| e.kind == NoiseKind::Walk) {
            if ev.time - buf[i].time < WALK_MERGE_S {
                buf.remove(i);
            }
        }
    }
    if buf.len() == HEARD_CAPACITY {
        buf.pop_front();
    }
    buf.push_back(ev);
}

/// Uniform position outside houses with at least `margin` clearance from obstacles.
fn open_position(map: &Map, rng: &mut impl Rng, margin: f32) -> Vec2 {
    let mut p = Vec2::splat(map.size * 0.5);
    for _ in 0..1000 {
        p = Vec2::new(
            rng.random_range(margin..map.size - margin),
            rng.random_range(margin..map.size - margin),
        );
        if map.house_at(p).is_none() && map.clearance(p, margin) >= margin {
            break;
        }
    }
    p
}

/// Uniform position anywhere (inside or outside houses) away from walls.
fn food_position(map: &Map, rng: &mut impl Rng, margin: f32) -> Vec2 {
    let mut p = Vec2::splat(map.size * 0.5);
    for _ in 0..1000 {
        p = Vec2::new(
            rng.random_range(margin..map.size - margin),
            rng.random_range(margin..map.size - margin),
        );
        if map.clearance(p, margin) >= margin {
            break;
        }
    }
    p
}

fn position_inside_house(map: &Map, rng: &mut impl Rng, margin: f32) -> Vec2 {
    let h = &map.houses[rng.random_range(0..map.houses.len())];
    let ext = h.half_extents - Vec2::splat(margin + 0.2);
    h.to_world(Vec2::new(
        rng.random_range(-ext.x..ext.x),
        rng.random_range(-ext.y..ext.y),
    ))
}

/// Standard 2D Gaussian sample (Box-Muller).
fn gaussian2(rng: &mut impl Rng) -> Vec2 {
    let u1: f32 = rng.random::<f32>().max(1e-7);
    let u2: f32 = rng.random();
    let r = (-2.0 * u1.ln()).sqrt();
    Vec2::new(r * (TAU * u2).cos(), r * (TAU * u2).sin())
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn u64(&mut self, v: u64) {
        for b in v.to_le_bytes() {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn f32(&mut self, v: f32) {
        self.u64(v.to_bits() as u64);
    }
    fn vec(&mut self, v: Vec2) {
        self.f32(v.x);
        self.f32(v.y);
    }
}
