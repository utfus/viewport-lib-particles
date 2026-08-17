
/// How many particles an effect spawns per second.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpawnRate {
    /// Constant rate in particles per second. Fractional values accumulate
    /// across frames.
    PerSecond(f32),
    /// A one-shot burst of `count` particles, emitted once when the effect
    /// first simulates, then nothing.
    Burst {
        /// Particles emitted in the single burst.
        count: u32,
    },
}

impl Default for SpawnRate {
    fn default() -> Self {
        SpawnRate::PerSecond(100.0)
    }
}

/// Shape new particles spawn from, offset by the emitting instance's position.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpawnShape {
    /// All particles spawn at the instance position.
    Point,
    /// Spawn uniformly inside an axis-aligned box (instance-relative corners).
    Box {
        /// Minimum corner.
        min: [f32; 3],
        /// Maximum corner.
        max: [f32; 3],
    },
    /// Spawn on or inside a sphere centered on the instance position.
    Sphere {
        /// Sphere radius.
        radius: f32,
        /// `true` scatters through the volume; `false` places on the surface.
        volume: bool,
    },
    /// Spawn on or inside a circle (disc) in the XY plane, centered on the
    /// instance position. Z-up, so the disc lies flat on the ground plane.
    Circle {
        /// Circle radius.
        radius: f32,
        /// `true` fills the disc; `false` places on the perimeter ring.
        volume: bool,
    },
    /// Spawn on or inside a cone around the +Z axis: the base sits at the
    /// instance position and the apex is offset by `height` along +Z. The radius
    /// tapers linearly from `base_radius` at the base to `top_radius` at the top.
    Cone {
        /// Radius at the base (z = 0).
        base_radius: f32,
        /// Radius at the top (z = height); `0` for a sharp apex.
        top_radius: f32,
        /// Cone height along +Z.
        height: f32,
        /// `true` fills the solid cone; `false` places on the lateral surface.
        volume: bool,
    },
}

impl Default for SpawnShape {
    fn default() -> Self {
        SpawnShape::Point
    }
}

/// Distribution used to assign an initial velocity to a new particle.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VelocityDist {
    /// Every particle gets the same velocity.
    Fixed([f32; 3]),
    /// Velocity sampled uniformly inside an axis-aligned box.
    UniformBox {
        /// Minimum corner of the velocity box.
        min: [f32; 3],
        /// Maximum corner of the velocity box.
        max: [f32; 3],
    },
    /// Direction sampled uniformly inside a cone around `axis`, magnitude in
    /// `[min_speed, max_speed]`.
    UniformCone {
        /// Cone axis direction.
        axis: [f32; 3],
        /// Half-angle of the cone in radians.
        half_angle: f32,
        /// Lower bound on sampled speed.
        min_speed: f32,
        /// Upper bound on sampled speed.
        max_speed: f32,
    },
}

impl Default for VelocityDist {
    fn default() -> Self {
        VelocityDist::Fixed([0.0, 0.0, 1.0])
    }
}

/// A force integrated into every live particle each simulation step.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForceModifier {
    /// Constant acceleration, world units per second squared.
    Accel([f32; 3]),
    /// Velocity-proportional drag; the coefficient is the fraction of velocity
    /// shed per second.
    Drag(f32),
    /// Acceleration toward (or, with negative strength, away from) a point.
    PointAttractor {
        /// World-space attractor position.
        position: [f32; 3],
        /// Acceleration coefficient; negative repels.
        strength: f32,
        /// Softening distance that tames the singularity at the center.
        falloff: f32,
    },
    /// Curl-noise turbulence: a divergence-free vector field sampled at the
    /// particle's position and added as acceleration, giving smooth swirling
    /// flow (smoke, fire). Divergence-free means particles never converge to
    /// sinks or spread from sources, so the motion looks like a fluid.
    CurlNoise {
        /// Spatial frequency of the field. Higher is finer and more chaotic.
        scale: f32,
        /// Acceleration magnitude applied along the curl field.
        strength: f32,
        /// How fast the field scrolls over time (0 is a static field).
        speed: f32,
    },
}

/// Fixed-function emitter configuration.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Emitter {
    /// Emission schedule.
    pub rate: SpawnRate,
    /// Per-particle lifetime range in seconds, sampled uniformly.
    pub lifetime: (f32, f32),
    /// Spawn shape, offset by the emitting instance's position.
    pub spawn: SpawnShape,
    /// Initial velocity distribution.
    pub velocity: VelocityDist,
    /// Per-particle RGBA tint.
    pub colour: [f32; 4],
    /// Per-particle world-space size (billboard edge length).
    pub size: f32,
}

impl Default for Emitter {
    fn default() -> Self {
        Self {
            rate: SpawnRate::default(),
            lifetime: (1.0, 2.0),
            spawn: SpawnShape::default(),
            velocity: VelocityDist::default(),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.25,
        }
    }
}
