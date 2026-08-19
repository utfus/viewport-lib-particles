use super::*;

/// When a parent effect produces spawn events for its child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EventCondition {
    /// One trigger per live parent particle each simulation step. Suited to
    /// continuous streams: a rocket laying down a smoke trail.
    EveryStep,
    /// One trigger when a parent particle dies (its lifetime crosses zero).
    /// Suited to bursts: a firework shell breaking into sparks at its apex.
    OnDeath,
}

/// Links a parent effect to a child effect so parent particles spawn child
/// particles on the GPU.
///
/// The parent's simulate pass appends spawn events (the parent particle's
/// position and velocity) into a shared buffer; the child's emit pass consumes
/// them, seeding new particles at the event position. The child keeps its own
/// spawn shape, velocity distribution, lifetime, colour, and size, so it shapes
/// the burst; set the child emitter's rate to zero for a purely event-driven
/// child.
///
/// Both effects must be fixed-function (no [`EffectProgram`]). Register the child
/// before the parent so the parent can name it. The plugin orders the passes
/// (parent simulate before child emit) automatically.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubEmitter {
    /// The child effect that consumes this parent's events.
    pub child: EffectId,
    /// When the parent produces events.
    pub condition: EventCondition,
    /// Child particles spawned per trigger. Bounded by the child's capacity.
    pub count: u32,
    /// Fraction of the parent particle's velocity handed to each child particle,
    /// added on top of the child's own velocity distribution. `0` ignores the
    /// parent velocity; `1` matches it.
    pub inherit_velocity: f32,
}

impl SubEmitter {
    /// A sub-emitter that spawns `count` child particles per trigger under
    /// `condition`, inheriting none of the parent velocity.
    pub fn new(child: EffectId, condition: EventCondition, count: u32) -> Self {
        Self {
            child,
            condition,
            count,
            inherit_velocity: 0.0,
        }
    }

    /// Hand each child particle `fraction` of the parent particle's velocity.
    pub fn with_inherit_velocity(mut self, fraction: f32) -> Self {
        self.inherit_velocity = fraction;
        self
    }
}

/// The authored description of one effect.
///
/// Carries no GPU state. Register it with a
/// [`ParticlePlugin`](crate::ParticlePlugin) to get an [`EffectId`].
#[derive(Clone, Debug)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EffectAsset {
    /// Human-readable name, surfaced in debug labels.
    pub name: String,
    /// Maximum simultaneously-live particles. Memory scales linearly.
    pub capacity: u32,
    /// Blend mode for the draw.
    pub blend: ParticleBlend,
    /// Fixed-function emitter. Ignored when [`program`](Self::program) is set.
    pub emitter: Emitter,
    /// Modifiers applied each simulation step: forces summed into acceleration,
    /// plus drag and kill-region rules. Ignored when [`program`](Self::program)
    /// is set. (Was named `forces`; the serde alias keeps old `.effect` files
    /// loading.)
    #[cfg_attr(feature = "serde", serde(alias = "forces"))]
    pub modifiers: Vec<Modifier>,
    /// Optional codegen program. When present, the effect compiles to per-effect
    /// emit/simulate WGSL from the expression graph instead of using the
    /// fixed-function emitter and forces above.
    pub program: Option<EffectProgram>,
    /// Optional colour/size ramp over lifetime, applied at draw. `None` uses the
    /// shared identity ramp (no change). Works for both the fixed and codegen
    /// paths.
    pub gradient: Option<Gradient>,
    /// How live particles are drawn. Defaults to a round camera-facing
    /// billboard.
    pub render: ParticleRender,
    /// Optional texture sampled by the billboard fragment. `None` draws the
    /// procedural soft round dot. Uploaded via
    /// `ParticlePlugin::upload_texture`. Applies to the billboard routes.
    pub texture: Option<ParticleTextureId>,
    /// How [`texture`](Self::texture) modulates the particle colour. Ignored
    /// when no texture is set.
    pub texture_mode: TextureMode,
    /// Optional flipbook animation over [`texture`](Self::texture) treated as an
    /// atlas. `None` samples the whole texture.
    pub flipbook: Option<Flipbook>,
    /// Optional sub-emitter: this effect's particles spawn particles in a child
    /// effect on the GPU. `None` for a leaf effect. Fixed-function only.
    pub sub_emitter: Option<SubEmitter>,
    /// The frame particles simulate in. [`SimulationSpace::Global`] (world space)
    /// is the default; [`SimulationSpace::Local`] makes live particles follow the
    /// emitting instance's translation.
    pub sim_space: SimulationSpace,
    /// Billboard orientation mode (Phase 15). Applies to the
    /// [`ParticleRender::Billboard`] route; ignored by the mesh and trail routes.
    #[cfg_attr(feature = "serde", serde(default))]
    pub orient: Orient,
    /// Size the billboard in screen space (Phase 15): when `true`, a particle
    /// keeps a constant apparent size regardless of distance (the `size` becomes
    /// its world size at unit camera distance). `false` is world-space size.
    /// Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub screen_space_size: bool,
    /// Soft-particle fade distance in world units (Phase 9). `Some(d)` fades the
    /// billboard's alpha to zero as it comes within `d` of the opaque scene
    /// surface, removing the hard clip line where a particle intersects geometry.
    /// Soft effects draw in the renderer's read-only-depth pass (via the
    /// `ItemTypePlugin::paint_depth_read` seam), composited over the scene with
    /// alpha blending — so a soft effect is over-blended regardless of its
    /// [`blend`](Self::blend). Billboard route only; `None` disables it.
    #[cfg_attr(feature = "serde", serde(default))]
    pub soft: Option<f32>,
    /// Per-particle billboard roll at spawn, in radians, plus a symmetric random
    /// spread. Each particle spawns at `rotation` plus a uniform value in
    /// `[-rotation_variance, rotation_variance]`. Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation: f32,
    /// Symmetric random spread on the spawn roll, in radians. See
    /// [`rotation`](Self::rotation).
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation_variance: f32,
    /// Constant billboard spin, in radians per second, advanced by particle age.
    /// Billboard route only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rotation_rate: f32,
    /// Timed bursts layered on top of the emitter's continuous rate. Each fires
    /// once, `count` particles, at its `time` after the instance starts. Empty by
    /// default. Fixed-function only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub spawn_schedule: Vec<TimedBurst>,
    /// World-space offset added to every spawn position, on top of the spawn
    /// shape and the emitting instance's position. Lets a spawn region sit off the
    /// anchor. Fixed-function only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub spawn_offset: [f32; 3],
    /// Symmetric random spread on each particle's spawn size, in the same world
    /// units as [`Emitter::size`]. Each particle spawns at `size` plus a uniform
    /// value in `[-size_variance, size_variance]`, clamped at zero. Fixed-function
    /// only.
    #[cfg_attr(feature = "serde", serde(default))]
    pub size_variance: f32,
}

impl Default for EffectAsset {
    fn default() -> Self {
        Self {
            name: "effect".to_string(),
            capacity: 10_000,
            blend: ParticleBlend::default(),
            emitter: Emitter::default(),
            modifiers: Vec::new(),
            program: None,
            gradient: None,
            render: ParticleRender::default(),
            texture: None,
            texture_mode: TextureMode::default(),
            flipbook: None,
            sub_emitter: None,
            sim_space: SimulationSpace::Global,
            orient: Orient::ParallelCameraPlane,
            screen_space_size: false,
            soft: None,
            rotation: 0.0,
            rotation_variance: 0.0,
            rotation_rate: 0.0,
            spawn_schedule: Vec::new(),
            spawn_offset: [0.0, 0.0, 0.0],
            size_variance: 0.0,
        }
    }
}

impl EffectAsset {
    /// A named, empty effect with default capacity and emitter.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Self::default()
        }
    }

    /// Set the live-particle capacity.
    pub fn with_capacity(mut self, capacity: u32) -> Self {
        self.capacity = capacity;
        self
    }

    /// Set the blend mode.
    pub fn with_blend(mut self, blend: ParticleBlend) -> Self {
        self.blend = blend;
        self
    }

    /// Replace the emitter.
    pub fn with_emitter(mut self, emitter: Emitter) -> Self {
        self.emitter = emitter;
        self
    }

    /// Append a modifier (a force, drag, or kill region).
    pub fn modifier(mut self, modifier: Modifier) -> Self {
        self.modifiers.push(modifier);
        self
    }

    /// Append a modifier. Kept for the pre-existing name; identical to
    /// [`modifier`](Self::modifier).
    pub fn force(self, force: Modifier) -> Self {
        self.modifier(force)
    }

    /// Replace the whole modifier list.
    pub fn with_modifiers(mut self, modifiers: Vec<Modifier>) -> Self {
        self.modifiers = modifiers;
        self
    }

    /// Attach a codegen program. The effect then compiles its emit/simulate
    /// kernels from the expression graph, ignoring the fixed-function emitter
    /// and forces.
    pub fn with_program(mut self, program: EffectProgram) -> Self {
        self.program = Some(program);
        self
    }

    /// Attach a colour/size ramp over lifetime.
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set the render route (billboard with optional stretch, or mesh instances).
    pub fn with_render(mut self, render: ParticleRender) -> Self {
        self.render = render;
        self
    }

    /// Sample `texture` in the billboard fragment, modulating colour per
    /// [`TextureMode::Modulate`]. Combine with [`with_texture_mode`](Self::with_texture_mode)
    /// or [`with_flipbook`](Self::with_flipbook).
    pub fn with_texture(mut self, texture: ParticleTextureId) -> Self {
        self.texture = Some(texture);
        self
    }

    /// Set how the texture modulates the particle colour.
    pub fn with_texture_mode(mut self, mode: TextureMode) -> Self {
        self.texture_mode = mode;
        self
    }

    /// Animate the texture as a flipbook atlas.
    pub fn with_flipbook(mut self, flipbook: Flipbook) -> Self {
        self.flipbook = Some(flipbook);
        self
    }

    /// Spawn particles in a child effect from this effect's particles. See
    /// [`SubEmitter`]. Fixed-function only; ignored if either effect carries a
    /// program.
    pub fn with_sub_emitter(mut self, sub_emitter: SubEmitter) -> Self {
        self.sub_emitter = Some(sub_emitter);
        self
    }

    /// Set the simulation space. [`SimulationSpace::Local`] makes live particles
    /// follow the emitting instance's translation each frame.
    pub fn with_simulation_space(mut self, space: SimulationSpace) -> Self {
        self.sim_space = space;
        self
    }

    /// Set the billboard orientation mode (Phase 15). Billboard route only.
    pub fn with_orient(mut self, orient: Orient) -> Self {
        self.orient = orient;
        self
    }

    /// Size billboards in screen space so distant particles keep a constant
    /// apparent size (Phase 15). Billboard route only.
    pub fn with_screen_size(mut self, enable: bool) -> Self {
        self.screen_space_size = enable;
        self
    }

    /// Make this a soft-particle effect (Phase 9): fade the billboard alpha to
    /// zero within `distance` world units of the opaque scene surface, so it
    /// dissolves into geometry instead of showing a hard intersection edge.
    /// Draws through the read-only-depth pass with alpha blending. Billboard
    /// route only.
    pub fn with_soft(mut self, distance: f32) -> Self {
        self.soft = Some(distance);
        self
    }

    /// Set per-particle billboard rotation: a spawn roll `angle` (radians) with a
    /// symmetric random `variance`, and a constant spin `rate` (radians per
    /// second) advanced by age. Billboard route only.
    pub fn with_rotation(mut self, angle: f32, variance: f32, rate: f32) -> Self {
        self.rotation = angle;
        self.rotation_variance = variance;
        self.rotation_rate = rate;
        self
    }

    /// Append a timed burst: `count` particles firing once at `time` seconds after
    /// an instance starts, on top of the emitter's continuous rate. Fixed-function
    /// only.
    pub fn add_burst(mut self, count: u32, time: f32) -> Self {
        self.spawn_schedule.push(TimedBurst { count, time });
        self
    }

    /// Replace the timed-burst schedule. See [`add_burst`](Self::add_burst).
    pub fn with_spawn_schedule(mut self, schedule: Vec<TimedBurst>) -> Self {
        self.spawn_schedule = schedule;
        self
    }

    /// Offset every spawn position by `offset` world units, on top of the spawn
    /// shape and the instance position. Fixed-function only.
    pub fn with_spawn_offset(mut self, offset: [f32; 3]) -> Self {
        self.spawn_offset = offset;
        self
    }

    /// Set a symmetric random spread on each particle's spawn size, in the same
    /// units as [`Emitter::size`]. Fixed-function only.
    pub fn with_size_variance(mut self, variance: f32) -> Self {
        self.size_variance = variance;
        self
    }
}

/// RON `.effect` serialization, behind the `serde` feature. Round-trips the whole
/// authored asset model (emitter, forces, expression [`Module`], gradient, render
/// route, textures, sub-emitter, simulation space).
///
/// Note the GPU-handle fields ([`EffectAsset::texture`],
/// [`ParticleRender::Mesh`], [`SubEmitter::child`]) serialize as their raw ids.
/// Those ids are session-scoped registration handles: a loaded asset only refers
/// to the right resource if the host re-registers meshes/textures/child effects
/// in the same order (or remaps the ids after loading).
#[cfg(feature = "serde")]
impl EffectAsset {
    /// Serialize this asset to a pretty RON string.
    pub fn to_ron(&self) -> Result<String, ron::Error> {
        ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default())
    }

    /// Parse an asset from a RON string.
    pub fn from_ron(s: &str) -> Result<Self, ron::error::SpannedError> {
        ron::from_str(s)
    }

    /// Write this asset to a `.effect` (RON) file.
    pub fn save_ron(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        let s = self.to_ron().map_err(std::io::Error::other)?;
        std::fs::write(path, s)
    }

    /// Read an asset from a `.effect` (RON) file.
    pub fn load_ron(path: impl AsRef<std::path::Path>) -> std::io::Result<Self> {
        let s = std::fs::read_to_string(path)?;
        Self::from_ron(&s).map_err(std::io::Error::other)
    }
}
