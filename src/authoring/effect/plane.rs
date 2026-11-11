use viewport_lib_types::overlay::world::WorldPlane;

/// A world plane a 2D effect lives on. The same two planes, with the same axes
/// and elevation, as viewport-lib's `WorldPlane`, `Plane2d` and `TilemapPlane`,
/// so an effect locked to a plane lines up with a `Camera2d` and a tilemap on it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParticlePlane {
    /// The world XY plane seen from above, looking along -Z. Plane x is world
    /// +X and plane y is world +Y; elevation is world +Z.
    #[default]
    Xy,
    /// The world XZ plane seen from the front, looking along +Y. Plane x is
    /// world +X and plane y is world +Z; elevation is world -Y, towards the
    /// camera.
    Xz,
}

impl ParticlePlane {
    /// World direction of plane +x.
    pub fn x_axis(self) -> [f32; 3] {
        [1.0, 0.0, 0.0]
    }

    /// World direction of plane +y.
    pub fn y_axis(self) -> [f32; 3] {
        match self {
            Self::Xy => [0.0, 1.0, 0.0],
            Self::Xz => [0.0, 0.0, 1.0],
        }
    }

    /// World direction of increasing elevation.
    pub fn normal(self) -> [f32; 3] {
        match self {
            Self::Xy => [0.0, 0.0, 1.0],
            Self::Xz => [0.0, -1.0, 0.0],
        }
    }
}

impl From<WorldPlane> for ParticlePlane {
    fn from(plane: WorldPlane) -> Self {
        match plane {
            WorldPlane::Xz => Self::Xz,
            _ => Self::Xy,
        }
    }
}

impl From<ParticlePlane> for WorldPlane {
    fn from(plane: ParticlePlane) -> Self {
        match plane {
            ParticlePlane::Xy => Self::Xy,
            ParticlePlane::Xz => Self::Xz,
        }
    }
}

/// Keeps an effect's particles on a world plane: spawn positions are projected
/// onto it, velocity and acceleration lose their component along its normal,
/// and every step puts the particle back at `depth`, so turbulence and
/// attractors cannot drift it off the plane.
///
/// `depth` is the elevation along [`ParticlePlane::normal`] in world units, so
/// an effect locked at a tilemap's elevation lies on the tilemap. It is a world
/// value under both simulation spaces: under [`SimulationSpace::Local`] the
/// particles follow the instance across the plane, not off it.
///
/// [`SimulationSpace::Local`]: crate::SimulationSpace::Local
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaneLock {
    /// The plane particles stay on.
    pub plane: ParticlePlane,
    /// Elevation along the plane normal.
    pub depth: f32,
}

impl PlaneLock {
    /// A lock to `plane` at elevation `depth`.
    pub fn new(plane: ParticlePlane, depth: f32) -> Self {
        Self { plane, depth }
    }
}

/// A sub-emitter child whose own plane lock names a different plane from its
/// parent's. Returned by [`check_sub_emitter_lock`] and by
/// `ParticlePlugin::try_add_effect`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneLockMismatch {
    /// The parent's plane.
    pub parent: ParticlePlane,
    /// The child's plane.
    pub child: ParticlePlane,
}

impl std::fmt::Display for PlaneLockMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sub-emitter child is locked to the {:?} plane but its parent is locked to {:?}",
            self.child, self.parent
        )
    }
}

impl std::error::Error for PlaneLockMismatch {}

/// The lock a sub-emitter child runs under: its own when it sets one, its
/// parent's otherwise.
pub fn child_plane_lock(parent: Option<PlaneLock>, child: Option<PlaneLock>) -> Option<PlaneLock> {
    child.or(parent)
}

/// Check that a sub-emitter child's own lock, if any, is on its parent's plane.
/// A different depth on the same plane is allowed, so embers can sit just in
/// front of the character that sheds them.
pub fn check_sub_emitter_lock(
    parent: Option<PlaneLock>,
    child: Option<PlaneLock>,
) -> Result<(), PlaneLockMismatch> {
    match (parent, child) {
        (Some(p), Some(c)) if p.plane != c.plane => Err(PlaneLockMismatch {
            parent: p.plane,
            child: c.plane,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn child_lock_wins_and_falls_back_to_parent() {
        let parent = PlaneLock::new(ParticlePlane::Xy, 1.0);
        let child = PlaneLock::new(ParticlePlane::Xy, 2.0);
        assert_eq!(child_plane_lock(Some(parent), None), Some(parent));
        assert_eq!(child_plane_lock(Some(parent), Some(child)), Some(child));
        assert_eq!(child_plane_lock(None, None), None);
    }

    #[test]
    fn mismatched_planes_are_rejected() {
        let xy = PlaneLock::new(ParticlePlane::Xy, 0.0);
        let xz = PlaneLock::new(ParticlePlane::Xz, 0.0);
        assert!(check_sub_emitter_lock(Some(xy), Some(xy)).is_ok());
        assert!(check_sub_emitter_lock(Some(xy), None).is_ok());
        assert!(check_sub_emitter_lock(None, Some(xz)).is_ok());
        assert_eq!(
            check_sub_emitter_lock(Some(xz), Some(xy)),
            Err(PlaneLockMismatch {
                parent: ParticlePlane::Xz,
                child: ParticlePlane::Xy,
            })
        );
    }

    #[test]
    fn axes_match_world_plane() {
        for plane in [ParticlePlane::Xy, ParticlePlane::Xz] {
            let w: WorldPlane = plane.into();
            assert_eq!(w.normal().to_array(), plane.normal());
            assert_eq!(w.y_axis().to_array(), plane.y_axis());
            assert_eq!(ParticlePlane::from(w), plane);
        }
    }
}
