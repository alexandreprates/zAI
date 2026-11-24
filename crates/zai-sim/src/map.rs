//! Map: houses with walls and two doors each, generated on a jittered grid.

use glam::Vec2;
use rand::Rng;

use crate::config::Config;
use crate::geom::{ray_box_exit, ray_segment, segment_blocks, Segment};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObstacleKind {
    Wall,
    ClosedDoor,
}

#[derive(Clone, Debug)]
pub struct Door {
    pub segment: Segment,
    pub center: Vec2,
    pub closed: bool,
}

/// A rectangular house. Local frame: x along the width, y along the depth; the front door
/// is centered on the `+y` wall and the back door on the `-y` wall.
#[derive(Clone, Debug)]
pub struct House {
    pub center: Vec2,
    /// Rotation in quarter turns (0..4).
    pub quarter_turns: u8,
    pub axis_x: Vec2,
    pub axis_y: Vec2,
    pub half_extents: Vec2,
    /// Wall pieces: front-left, front-right, back-left, back-right, left side, right side.
    pub walls: [Segment; 6],
    /// `doors[0]` is the front door, `doors[1]` the back door.
    pub doors: [Door; 2],
    /// Radius of the bounding circle, used to cull geometric queries.
    pub bounding_radius: f32,
}

impl House {
    pub fn new(center: Vec2, quarter_turns: u8, cfg: &Config) -> Self {
        let (axis_x, axis_y) = match quarter_turns % 4 {
            0 => (Vec2::X, Vec2::Y),
            1 => (Vec2::Y, -Vec2::X),
            2 => (-Vec2::X, -Vec2::Y),
            _ => (-Vec2::Y, Vec2::X),
        };
        let hw = cfg.house.width_m * 0.5;
        let hd = cfg.house.depth_m * 0.5;
        let g = cfg.house.door_width_m * 0.5;
        let w = |x: f32, y: f32| center + axis_x * x + axis_y * y;
        let seg = |a: (f32, f32), b: (f32, f32)| Segment::new(w(a.0, a.1), w(b.0, b.1));
        let walls = [
            seg((-hw, hd), (-g, hd)),
            seg((g, hd), (hw, hd)),
            seg((-hw, -hd), (-g, -hd)),
            seg((g, -hd), (hw, -hd)),
            seg((-hw, -hd), (-hw, hd)),
            seg((hw, -hd), (hw, hd)),
        ];
        let door = |y: f32| {
            let segment = seg((-g, y), (g, y));
            Door {
                center: segment.midpoint(),
                segment,
                closed: false,
            }
        };
        Self {
            center,
            quarter_turns: quarter_turns % 4,
            axis_x,
            axis_y,
            half_extents: Vec2::new(hw, hd),
            walls,
            doors: [door(hd), door(-hd)],
            bounding_radius: Vec2::new(hw, hd).length(),
        }
    }

    pub fn to_local(&self, p: Vec2) -> Vec2 {
        let d = p - self.center;
        Vec2::new(d.dot(self.axis_x), d.dot(self.axis_y))
    }

    pub fn to_world(&self, local: Vec2) -> Vec2 {
        self.center + self.axis_x * local.x + self.axis_y * local.y
    }

    /// True if `p` is strictly inside the house footprint.
    pub fn contains(&self, p: Vec2) -> bool {
        let l = self.to_local(p);
        l.x.abs() < self.half_extents.x && l.y.abs() < self.half_extents.y
    }

    /// Rotation angle in radians.
    pub fn angle(&self) -> f32 {
        self.axis_x.y.atan2(self.axis_x.x)
    }

    fn for_each_obstacle(&self, f: &mut impl FnMut(&Segment, ObstacleKind)) {
        for w in &self.walls {
            f(w, ObstacleKind::Wall);
        }
        for d in &self.doors {
            if d.closed {
                f(&d.segment, ObstacleKind::ClosedDoor);
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Map {
    pub size: f32,
    pub houses: Vec<House>,
}

/// Result of a ray cast.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub distance: f32,
    /// `None` when nothing is hit within range; map boundaries count as walls.
    pub kind: Option<ObstacleKind>,
}

impl Map {
    pub fn generate(cfg: &Config, rng: &mut impl Rng) -> Self {
        let size = cfg.map_size_m;
        let n = cfg.house_count;
        let mut houses = Vec::with_capacity(n);
        if n > 0 {
            let (cols, rows) = grid_shape(n);
            let cell = Vec2::new(size / cols as f32, size / rows as f32);
            let mut cells: Vec<(usize, usize)> = (0..rows)
                .flat_map(|r| (0..cols).map(move |c| (c, r)))
                .collect();
            // Drop random cells when the grid has more cells than houses.
            while cells.len() > n {
                let i = rng.random_range(0..cells.len());
                cells.remove(i);
            }
            let probe = House::new(Vec2::ZERO, 0, cfg);
            let margin = probe.bounding_radius + 2.0;
            let jitter = Vec2::new(
                cfg.house.jitter_m.min(cell.x * 0.5 - margin).max(0.0),
                cfg.house.jitter_m.min(cell.y * 0.5 - margin).max(0.0),
            );
            for (c, r) in cells {
                let base = Vec2::new((c as f32 + 0.5) * cell.x, (r as f32 + 0.5) * cell.y);
                let offset = Vec2::new(
                    rng.random_range(-1.0..=1.0) * jitter.x,
                    rng.random_range(-1.0..=1.0) * jitter.y,
                );
                let turns = rng.random_range(0..4u8);
                houses.push(House::new(base + offset, turns, cfg));
            }
        }
        Self { size, houses }
    }

    /// Calls `f` for every obstacle segment (walls and closed doors) of houses whose
    /// bounding circle is within `radius` of `p`.
    pub fn for_each_obstacle_near(
        &self,
        p: Vec2,
        radius: f32,
        mut f: impl FnMut(&Segment, ObstacleKind),
    ) {
        for h in &self.houses {
            let reach = h.bounding_radius + radius;
            if h.center.distance_squared(p) <= reach * reach {
                h.for_each_obstacle(&mut f);
            }
        }
    }

    /// Index of the house containing `p`, if any.
    pub fn house_at(&self, p: Vec2) -> Option<usize> {
        self.houses.iter().position(|h| h.contains(p))
    }

    /// Distance from `p` to the nearest obstacle segment within `radius` (or `radius`).
    pub fn clearance(&self, p: Vec2, radius: f32) -> f32 {
        let mut best = radius;
        self.for_each_obstacle_near(p, radius, |s, _| best = best.min(s.distance(p)));
        best
    }

    /// True if no wall or closed door crosses the segment from `a` to `b`.
    pub fn line_of_sight(&self, a: Vec2, b: Vec2) -> bool {
        let mid = (a + b) * 0.5;
        let half = a.distance(b) * 0.5;
        for h in &self.houses {
            let reach = h.bounding_radius + half;
            if h.center.distance_squared(mid) > reach * reach {
                continue;
            }
            let mut blocked = false;
            h.for_each_obstacle(&mut |s, _| blocked |= segment_blocks(a, b, s));
            if blocked {
                return false;
            }
        }
        true
    }

    /// Casts a ray from `origin` along unit `dir` up to `range`.
    pub fn ray_cast(&self, origin: Vec2, dir: Vec2, range: f32) -> RayHit {
        let mut hit = RayHit {
            distance: range,
            kind: None,
        };
        let boundary = ray_box_exit(origin, dir, self.size);
        if boundary < hit.distance {
            hit = RayHit {
                distance: boundary,
                kind: Some(ObstacleKind::Wall),
            };
        }
        for h in &self.houses {
            // Cull houses whose bounding circle the ray cannot reach.
            let to_c = h.center - origin;
            let along = to_c.dot(dir);
            if along < -h.bounding_radius || along - h.bounding_radius > hit.distance {
                continue;
            }
            let perp = to_c.perp_dot(dir).abs();
            if perp > h.bounding_radius {
                continue;
            }
            h.for_each_obstacle(&mut |s, kind| {
                if let Some(t) = ray_segment(origin, dir, s) {
                    if t < hit.distance {
                        hit = RayHit {
                            distance: t,
                            kind: Some(kind),
                        };
                    }
                }
            });
        }
        hit
    }
}

/// Grid (cols, rows) with the fewest unused cells, then the most square shape.
fn grid_shape(n: usize) -> (usize, usize) {
    let mut best = (n, 1);
    let mut best_key = (usize::MAX, usize::MAX);
    for rows in 1..=n {
        let cols = n.div_ceil(rows);
        if cols < rows {
            break;
        }
        let key = (cols * rows - n, cols - rows);
        if key < best_key {
            best_key = key;
            best = (cols, rows);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn grid_shape_matches_architecture() {
        assert_eq!(grid_shape(10), (5, 2));
        assert_eq!(grid_shape(3), (3, 1));
        assert_eq!(grid_shape(6), (3, 2));
        assert_eq!(grid_shape(1), (1, 1));
    }

    #[test]
    fn houses_are_inside_map_and_do_not_overlap() {
        let cfg = Config::default();
        for seed in 0..50 {
            let map = Map::generate(&cfg, &mut ChaCha8Rng::seed_from_u64(seed));
            assert_eq!(map.houses.len(), 10);
            for (i, h) in map.houses.iter().enumerate() {
                assert!(h.center.min_element() > h.bounding_radius);
                assert!(h.center.max_element() < cfg.map_size_m - h.bounding_radius);
                for o in &map.houses[i + 1..] {
                    let min_gap = h.bounding_radius + o.bounding_radius + 2.0;
                    assert!(h.center.distance(o.center) > min_gap);
                }
            }
        }
    }

    #[test]
    fn doors_lie_on_opposite_walls() {
        let cfg = Config::default();
        for turns in 0..4 {
            let h = House::new(Vec2::new(50.0, 50.0), turns, &cfg);
            let front = h.to_local(h.doors[0].center);
            let back = h.to_local(h.doors[1].center);
            assert!((front - Vec2::new(0.0, 4.0)).length() < 1e-4);
            assert!((back - Vec2::new(0.0, -4.0)).length() < 1e-4);
        }
    }

    #[test]
    fn walls_block_line_of_sight_but_open_doors_do_not() {
        let cfg = Config::default();
        let mut map = Map {
            size: 100.0,
            houses: vec![House::new(Vec2::new(50.0, 50.0), 0, &cfg)],
        };
        let inside = Vec2::new(50.0, 50.0);
        assert!(
            !map.line_of_sight(inside, Vec2::new(70.0, 50.0)),
            "side wall"
        );
        assert!(
            map.line_of_sight(inside, Vec2::new(50.0, 70.0)),
            "open front door"
        );
        map.houses[0].doors[0].closed = true;
        assert!(
            !map.line_of_sight(inside, Vec2::new(50.0, 70.0)),
            "closed front door"
        );
    }

    #[test]
    fn ray_cast_reports_obstacle_kind() {
        let cfg = Config::default();
        let mut map = Map {
            size: 100.0,
            houses: vec![House::new(Vec2::new(50.0, 50.0), 0, &cfg)],
        };
        map.houses[0].doors[0].closed = true;
        let up = map.ray_cast(Vec2::new(45.0, 30.0), Vec2::Y, 100.0);
        assert_eq!(up.kind, Some(ObstacleKind::Wall));
        assert!((up.distance - 16.0).abs() < 1e-4, "back wall piece: {up:?}");
        let from_inside = map.ray_cast(Vec2::new(50.0, 50.0), Vec2::Y, 100.0);
        assert_eq!(from_inside.kind, Some(ObstacleKind::ClosedDoor));
        assert!((from_inside.distance - 4.0).abs() < 1e-4);
        let open = map.ray_cast(Vec2::new(10.0, 10.0), -Vec2::X, 5.0);
        assert_eq!(open.kind, None);
        assert_eq!(open.distance, 5.0);
    }
}
