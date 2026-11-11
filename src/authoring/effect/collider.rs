/// One edge of a [`Collider2d::Edges`] collider, in plane coordinates. The same
/// shape as viewport-lib's `OccluderEdge`; collision treats both sides alike.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ColliderEdge {
    /// First end.
    pub a: [f32; 2],
    /// Second end.
    pub b: [f32; 2],
}

/// A shape particles collide with, in plane coordinates (plane x and y of the
/// colliding effect's [`PlaneLock`](crate::PlaneLock)). The variants match
/// viewport-lib's `Occluder2d`, so the geometry a game builds for its 2D
/// lighting, including what `occluders_from_tilemap` returns, converts one to
/// one.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Collider2d {
    /// An axis-aligned rectangle's outline.
    Rect {
        /// Lower-left corner.
        min: [f32; 2],
        /// Upper-right corner.
        max: [f32; 2],
    },
    /// A closed outline, in either winding.
    Polygon(Vec<[f32; 2]>),
    /// A single segment, such as a ledge or a fence.
    Segment {
        /// One end.
        a: [f32; 2],
        /// The other end.
        b: [f32; 2],
    },
    /// Loose edges, such as the boundary of a tile layer's solid cells.
    Edges(Vec<ColliderEdge>),
}

impl Collider2d {
    /// Append this shape's edges to `out`.
    pub fn edges(&self, out: &mut Vec<ColliderEdge>) {
        let mut ring = |points: &[[f32; 2]]| {
            if points.len() < 2 {
                return;
            }
            for i in 0..points.len() {
                out.push(ColliderEdge {
                    a: points[i],
                    b: points[(i + 1) % points.len()],
                });
            }
        };
        match self {
            Self::Rect { min, max } => ring(&[
                [min[0], min[1]],
                [max[0], min[1]],
                [max[0], max[1]],
                [min[0], max[1]],
            ]),
            Self::Polygon(points) => ring(points),
            Self::Segment { a, b } => out.push(ColliderEdge { a: *a, b: *b }),
            Self::Edges(edges) => out.extend_from_slice(edges),
        }
    }
}

/// The level geometry a set of effects collides with: plain shapes in plane
/// coordinates that the host builds and owns. Register one with
/// `ParticlePlugin::add_collider_set` and replace it when the level changes.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Collider2dSet {
    /// The shapes.
    pub shapes: Vec<Collider2d>,
}

impl Collider2dSet {
    /// A set of `shapes`.
    pub fn new(shapes: Vec<Collider2d>) -> Self {
        Self { shapes }
    }

    /// Every shape's edges.
    pub fn edges(&self) -> Vec<ColliderEdge> {
        let mut out = Vec::new();
        for shape in &self.shapes {
            shape.edges(&mut out);
        }
        out
    }
}

/// Handle to a collider set registered with a `ParticlePlugin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Collider2dSetId(pub u32);

/// What a particle does when it hits a collider.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CollisionResponse {
    /// Reflect off the edge. `restitution` keeps that fraction of the speed
    /// along the edge normal (1 is a perfect bounce, 0 stops it there);
    /// `friction` sheds that fraction of the speed along the edge.
    Bounce {
        /// Fraction of the normal speed kept.
        restitution: f32,
        /// Fraction of the tangential speed shed.
        friction: f32,
    },
    /// Die at the contact. A sub-emitter with `EventCondition::OnDeath` then
    /// spawns there, for a splash.
    Kill,
    /// Stop at the contact and stay there until the lifetime runs out.
    Stick,
}

/// Collision of an effect's particles with a collider set, as each step's
/// path from the old position to the new one, so a fast particle does not pass
/// through a thin wall. Needs a [`PlaneLock`](crate::PlaneLock): collision runs
/// in the lock's plane coordinates, and an effect without one ignores it.
/// Fixed-function only.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Collide2d {
    /// The collider set.
    pub set: Collider2dSetId,
    /// What a particle does on contact.
    pub response: CollisionResponse,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_turn_into_closed_edge_rings() {
        let set = Collider2dSet::new(vec![
            Collider2d::Rect {
                min: [0.0, 0.0],
                max: [2.0, 1.0],
            },
            Collider2d::Segment {
                a: [5.0, 0.0],
                b: [6.0, 0.0],
            },
            Collider2d::Polygon(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        ]);
        let edges = set.edges();
        assert_eq!(edges.len(), 4 + 1 + 3);
        // The rectangle closes on its first corner.
        assert_eq!(edges[3].b, [0.0, 0.0]);
        assert_eq!(edges[7].b, [0.0, 0.0]);
    }
}
