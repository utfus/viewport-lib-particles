use super::*;

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

/// A one-shot burst that fires at a set time after an instance starts, layered on
/// top of the emitter's continuous [`SpawnRate`]. A list of these on
/// [`EffectAsset`](crate::EffectAsset) lets one effect hold a steady rate and fire
/// staged bursts (a muzzle flash that also puffs smoke, a multi-stage explosion).
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TimedBurst {
    /// Particles emitted when the burst fires (bounded by the effect capacity and
    /// the frame's free slots).
    pub count: u32,
    /// Seconds after the instance starts simulating at which the burst fires. `0`
    /// fires on the first simulated frame. Each burst fires once.
    pub time: f32,
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
    /// Spawn over the surface of an uploaded mesh (Phase 22), area-weighted so
    /// large triangles receive proportionally more particles. Positions are
    /// instance-relative (offset by the emitting instance's position). Sampled
    /// from a fixed pool baked at registration time from the mesh's triangles;
    /// the fixed-function path only. Useful for dissolve / assembly effects.
    MeshSurface {
        /// The uploaded mesh (`ParticlePlugin::upload_mesh`) whose triangles are
        /// sampled. Must be uploaded before the effect is registered.
        mesh: ParticleMeshId,
        /// Speed added along each sample's surface normal at spawn (Phase 22), so
        /// particles lift off the surface. `0` spawns on the surface with only
        /// the emitter's own velocity distribution.
        normal_velocity: f32,
    },
    /// Spawn from the vertices of an uploaded point set (Phase 22): the mesh's
    /// vertex positions are treated as points and its indices ignored. Positions
    /// are instance-relative. Fixed-function only.
    PointCloud {
        /// The uploaded mesh whose *vertices* are the point set (upload with an
        /// empty index slice for a pure point cloud).
        mesh: ParticleMeshId,
        /// Speed added along each point's outward (radial-from-origin) direction
        /// at spawn. `0` uses only the emitter's velocity distribution.
        normal_velocity: f32,
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

/// A modifier applied to every live particle each simulation step: a force added
/// to its acceleration, a drag, or a rule that kills it inside or outside a
/// region.
///
/// This is a closed, kind-dispatched set. Each variant packs into a fixed GPU
/// slot and is handled by one arm of the shared `eval_modifier` function in the
/// simulate shaders, so adding a new modifier is three local edits: a variant
/// here, a packing arm in `build_sim_params`, and a `case` in `eval_modifier`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Modifier {
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
    /// Acceleration straight out from (or, with negative strength, in toward) a
    /// center. A radial push / implosion.
    Radial {
        /// World-space center the acceleration points away from.
        center: [f32; 3],
        /// Acceleration magnitude; negative pulls inward.
        strength: f32,
        /// When `> 0`, the magnitude fades with distance as
        /// `strength / (1 + dist / falloff)`; `0` keeps it constant.
        falloff: f32,
    },
    /// Tangential swirl around an axis through a center: a deterministic vortex
    /// (whirlwind, magic swirl), distinct from the fluid chaos of `CurlNoise`.
    Vortex {
        /// World-space center the axis passes through.
        center: [f32; 3],
        /// Axis the swirl rotates about.
        axis: [f32; 3],
        /// Tangential acceleration magnitude; negative reverses the spin.
        strength: f32,
        /// When `> 0`, the magnitude fades with distance from the axis as
        /// `strength / (1 + dist / falloff)`; `0` keeps it constant.
        falloff: f32,
    },
    /// Pull particles onto a sphere shell: within `influence` of the shell, they
    /// accelerate toward it at `attraction`, forming a swarm on the surface.
    Conform {
        /// Sphere center.
        center: [f32; 3],
        /// Sphere radius the particles conform to.
        radius: f32,
        /// Acceleration toward the shell.
        attraction: f32,
        /// Only particles within this distance of the shell are affected.
        influence: f32,
        /// Extra drag applied within `influence`, so particles settle rather than
        /// orbit. `0` for none.
        sticky: f32,
    },
    /// Kill particles by position relative to a sphere: inside it when
    /// `kill_inside`, otherwise outside it. Killed particles are recycled by the
    /// emit pass.
    KillSphere {
        /// Sphere center.
        center: [f32; 3],
        /// Sphere radius.
        radius: f32,
        /// `true` kills particles inside the sphere; `false` kills those outside.
        kill_inside: bool,
    },
    /// Kill particles by position relative to an axis-aligned box: inside it when
    /// `kill_inside`, otherwise outside it.
    KillAabb {
        /// Minimum corner.
        min: [f32; 3],
        /// Maximum corner.
        max: [f32; 3],
        /// `true` kills particles inside the box; `false` kills those outside.
        kill_inside: bool,
    },
}

/// Former name of [`Modifier`], kept so existing `ForceModifier::…` code and
/// `.force(…)` calls keep working. The modifier set now includes non-force kinds
/// (kill regions), so `Modifier` is the current name.
pub type ForceModifier = Modifier;

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
