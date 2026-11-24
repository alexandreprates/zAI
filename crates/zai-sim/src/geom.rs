//! 2D geometry helpers.

use glam::Vec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Segment {
    pub a: Vec2,
    pub b: Vec2,
}

impl Segment {
    pub fn new(a: Vec2, b: Vec2) -> Self {
        Self { a, b }
    }

    pub fn closest_point(&self, p: Vec2) -> Vec2 {
        let ab = self.b - self.a;
        let len2 = ab.length_squared();
        if len2 <= f32::EPSILON {
            return self.a;
        }
        let t = ((p - self.a).dot(ab) / len2).clamp(0.0, 1.0);
        self.a + ab * t
    }

    pub fn distance(&self, p: Vec2) -> f32 {
        p.distance(self.closest_point(p))
    }

    pub fn midpoint(&self) -> Vec2 {
        (self.a + self.b) * 0.5
    }
}

/// Distance along the ray `origin + t * dir` (t >= 0) at which it hits the segment.
/// `dir` need not be normalized; `t` is in units of `dir`.
pub fn ray_segment(origin: Vec2, dir: Vec2, s: &Segment) -> Option<f32> {
    let e = s.b - s.a;
    let denom = dir.perp_dot(e);
    if denom.abs() < 1e-9 {
        return None;
    }
    let w = s.a - origin;
    let t = w.perp_dot(e) / denom;
    let u = w.perp_dot(dir) / denom;
    (t >= 0.0 && (0.0..=1.0).contains(&u)).then_some(t)
}

/// True if the open segment `p -> q` crosses `s`. Touching at `q` itself does not count,
/// so a target lying on a segment is still considered reachable by line of sight.
pub fn segment_blocks(p: Vec2, q: Vec2, s: &Segment) -> bool {
    matches!(ray_segment(p, q - p, s), Some(t) if t < 1.0 - 1e-4)
}

/// Distance from `origin` along unit `dir` to the boundary of the square `[0, size]^2`.
/// `origin` must be inside the square.
pub fn ray_box_exit(origin: Vec2, dir: Vec2, size: f32) -> f32 {
    let axis = |o: f32, d: f32| {
        if d > 1e-9 {
            (size - o) / d
        } else if d < -1e-9 {
            -o / d
        } else {
            f32::INFINITY
        }
    };
    axis(origin.x, dir.x).min(axis(origin.y, dir.y)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_segment_at_expected_distance() {
        let s = Segment::new(Vec2::new(5.0, -1.0), Vec2::new(5.0, 1.0));
        let t = ray_segment(Vec2::ZERO, Vec2::X, &s).unwrap();
        assert!((t - 5.0).abs() < 1e-5);
        assert!(ray_segment(Vec2::ZERO, -Vec2::X, &s).is_none());
        assert!(ray_segment(Vec2::ZERO, Vec2::Y, &s).is_none());
    }

    #[test]
    fn segment_blocks_excludes_target_endpoint() {
        let wall = Segment::new(Vec2::new(5.0, -1.0), Vec2::new(5.0, 1.0));
        assert!(segment_blocks(Vec2::ZERO, Vec2::new(10.0, 0.0), &wall));
        assert!(!segment_blocks(Vec2::ZERO, Vec2::new(5.0, 0.0), &wall));
        assert!(!segment_blocks(Vec2::ZERO, Vec2::new(4.0, 0.0), &wall));
    }

    #[test]
    fn closest_point_clamps_to_endpoints() {
        let s = Segment::new(Vec2::ZERO, Vec2::new(10.0, 0.0));
        assert_eq!(s.closest_point(Vec2::new(-3.0, 4.0)), Vec2::ZERO);
        assert!((s.distance(Vec2::new(5.0, 3.0)) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ray_box_exit_reaches_boundary() {
        let d = ray_box_exit(Vec2::new(10.0, 50.0), Vec2::X, 100.0);
        assert!((d - 90.0).abs() < 1e-5);
    }
}
