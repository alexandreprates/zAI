//! Per-episode tactic counters (M6.2, docs/ARCHITECTURE.md section 4.8.6).
//!
//! `Env` accumulates them during `step`, so they are available for every training episode and
//! in the server at no extra simulation cost. Python reads them as a fixed-size vector in the
//! order of `STATS_LAYOUT` (`VecEnv` `info["final_stats"]` for finished episodes) and turns
//! them into rates and intervals.

use std::f32::consts::TAU;

use glam::Vec2;

/// Zombies within this distance of the human at the catching tick count for encirclement.
pub const ENCIRCLE_RANGE_M: f32 = 50.0;
/// A catch is an encirclement when the largest angular gap between those zombies is below this.
pub const ENCIRCLE_GAP_DEG: f32 = 120.0;
/// A door close is defensive when a zombie is within this distance of the door.
pub const DEFENSIVE_CLOSE_M: f32 = 30.0;
/// A decision is quiet when no zombie heard anything during the last `QUIET_S` seconds.
pub const QUIET_S: f32 = 30.0;
/// Carried packages from which the human counts as heavily loaded.
pub const HEAVY_LOAD: u32 = 3;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EpisodeStats {
    pub decisions: u32,
    /// Food packages picked up, and those picked while a zombie could hear the pickup.
    pub pickups: u32,
    pub risky_pickups: u32,
    /// Packages eaten from the inventory (each resets the hunger timer).
    pub eaten: u32,
    /// Sum over decisions of the carried count, and decisions with `carried >= HEAVY_LOAD`.
    pub carried_sum: f32,
    pub heavy_decisions: u32,
    pub walk_noises: u32,
    pub door_closes: u32,
    pub defensive_closes: u32,
    /// Decisions that ended with the human inside a house.
    pub inside_decisions: u32,
    /// At the catching tick (only meaningful for caught episodes): zombies within
    /// `ENCIRCLE_RANGE_M`, the largest angular gap between them around the human (360 for
    /// fewer than two), and the number of distinct 45-degree sectors of the three nearest
    /// zombies.
    pub catch_close_zombies: u32,
    pub catch_gap_deg: f32,
    pub catch_sectors: u32,
    /// Sum of the mean pairwise zombie distance over quiet decisions, and their count.
    pub quiet_dispersion_sum: f32,
    pub quiet_decisions: u32,
}

/// Names of the values of `EpisodeStats::to_vec`, in order.
pub const STATS_LAYOUT: [&str; 15] = [
    "decisions",
    "pickups",
    "risky_pickups",
    "eaten",
    "carried_sum",
    "heavy_decisions",
    "walk_noises",
    "door_closes",
    "defensive_closes",
    "inside_decisions",
    "catch_close_zombies",
    "catch_gap_deg",
    "catch_sectors",
    "quiet_dispersion_sum",
    "quiet_decisions",
];
pub const STATS_DIM: usize = STATS_LAYOUT.len();

impl EpisodeStats {
    pub fn write(&self, out: &mut [f32]) {
        let v = [
            self.decisions as f32,
            self.pickups as f32,
            self.risky_pickups as f32,
            self.eaten as f32,
            self.carried_sum,
            self.heavy_decisions as f32,
            self.walk_noises as f32,
            self.door_closes as f32,
            self.defensive_closes as f32,
            self.inside_decisions as f32,
            self.catch_close_zombies as f32,
            self.catch_gap_deg,
            self.catch_sectors as f32,
            self.quiet_dispersion_sum,
            self.quiet_decisions as f32,
        ];
        out.copy_from_slice(&v);
    }
}

/// Largest angular gap (degrees) between the directions from `center` to `points`; 360 when
/// there are fewer than two points.
pub fn max_angular_gap_deg(center: Vec2, points: &[Vec2]) -> f32 {
    if points.len() < 2 {
        return 360.0;
    }
    let mut a: Vec<f32> = points
        .iter()
        .map(|p| {
            let d = *p - center;
            d.y.atan2(d.x).rem_euclid(TAU)
        })
        .collect();
    a.sort_by(f32::total_cmp);
    let mut gap = a[0] + TAU - a[a.len() - 1];
    for w in a.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    gap.to_degrees()
}

/// Number of distinct 45-degree sectors (around `center`) occupied by `points`.
pub fn distinct_sectors(center: Vec2, points: &[Vec2]) -> u32 {
    let mut mask = 0u8;
    for p in points {
        let d = *p - center;
        let s = ((d.y.atan2(d.x).rem_euclid(TAU)) / (TAU / 8.0)) as u32 % 8;
        mask |= 1 << s;
    }
    mask.count_ones()
}

/// Mean pairwise distance between `points` (0 for fewer than two).
pub fn mean_pairwise_distance(points: &[Vec2]) -> f32 {
    let n = points.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        for j in i + 1..n {
            sum += points[i].distance(points[j]);
        }
    }
    sum / (n * (n - 1) / 2) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_and_sectors() {
        let c = Vec2::new(10.0, 10.0);
        let ring: Vec<Vec2> = (0..4)
            .map(|k| c + Vec2::from_angle(k as f32 * TAU / 4.0) * 5.0)
            .collect();
        assert!((max_angular_gap_deg(c, &ring) - 90.0).abs() < 1e-3);
        assert_eq!(distinct_sectors(c, &ring[..3]), 3);
        let same_side = [c + Vec2::new(5.0, 0.5), c + Vec2::new(5.0, -0.5)];
        assert!(max_angular_gap_deg(c, &same_side) > 340.0);
        assert_eq!(max_angular_gap_deg(c, &ring[..1]), 360.0);
        assert!((mean_pairwise_distance(&[Vec2::ZERO, Vec2::new(3.0, 4.0)]) - 5.0).abs() < 1e-6);
    }
}
