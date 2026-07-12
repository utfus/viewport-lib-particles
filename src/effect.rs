//! Effect assets: the authored description of one kind of effect.
//!
//! An [`EffectAsset`] can be fixed-function: a concrete [`Emitter`] (spawn rate,
//! shape, velocity, lifetime, colour, size) and a list of [`ForceModifier`]s,
//! which the plugin runs directly as emit + simulate compute passes. For richer
//! behaviour it can instead carry an [`EffectProgram`] built from the expression
//! graph ([`Module`](crate::Module) / [`Expr`](crate::Expr)), which the modifier
//! codegen lowers to per-effect kernels in place of the fixed set.
//!
//! An asset carries no GPU state. Register it with a
//! [`ParticlePlugin`](crate::ParticlePlugin) to get an [`EffectId`]; the plugin
//! allocates the particle buffer and pipelines.

use crate::expr::{ExprHandle, Module};

/// Identifies an [`EffectAsset`] registered with a
/// [`ParticlePlugin`](crate::ParticlePlugin).
///
/// Returned by `ParticlePlugin::add_effect`. Stable for the plugin's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EffectId(pub(crate) u32);

/// Handle to a mesh uploaded to a [`ParticlePlugin`](crate::ParticlePlugin) for
/// the mesh render route. Returned by `ParticlePlugin::upload_mesh`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParticleMeshId(pub(crate) u32);

/// Handle to a texture uploaded to a [`ParticlePlugin`](crate::ParticlePlugin)
/// for the billboard texture / flipbook routes. Returned by
/// `ParticlePlugin::upload_texture`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParticleTextureId(pub(crate) u32);

/// How an effect's texture modulates the per-particle colour in the billboard
/// fragment shader. Mirrors Hanabi's `ImageSampleMapping`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextureMode {
    /// Multiply both colour (rgb) and alpha by the sampled texel.
    #[default]
    Modulate,
    /// Multiply only the colour (rgb); keep the particle's own alpha.
    ModulateRgb,
    /// Take alpha from the texel's red channel (a mask); keep the colour. The
    /// masked-spark idiom.
    ModulateAlphaFromR,
}

/// Flipbook (sprite-sheet) animation over a texture atlas. The texture is a grid
/// of `columns` x `rows` animation frames; the billboard fragment picks a cell
/// per particle and offsets its UVs into it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flipbook {
    /// Atlas columns.
    pub columns: u32,
    /// Atlas rows.
    pub rows: u32,
    /// Playback rate. `0` stretches the whole sheet once across each particle's
    /// life (first frame at spawn, last near death); `> 0` loops the sheet at
    /// this many frames per second driven by the particle's elapsed age.
    pub fps: f32,
}

impl Flipbook {
    /// A `columns` x `rows` sheet stretched once across particle life.
    pub fn new(columns: u32, rows: u32) -> Self {
        Self {
            columns,
            rows,
            fps: 0.0,
        }
    }

    /// Loop the sheet at `fps` frames per second instead of stretching it across
    /// life.
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }
}

/// How a mesh-route particle is oriented each frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshAlign {
    /// Align the mesh's local +Z to the particle's velocity direction.
    Velocity,
    /// A stable per-particle random tumble that advances with age.
    Random,
}

/// How an effect's live particles are drawn.
#[derive(Clone, Copy, Debug)]
pub enum ParticleRender {
    /// Camera-facing billboard. `stretch` (> 0) elongates it along the
    /// screen-projected velocity, scaled by speed, for motion streaks.
    Billboard {
        /// Stretch factor. `0` is a round billboard.
        stretch: f32,
    },
    /// One instance of an uploaded mesh per particle.
    Mesh {
        /// The uploaded mesh to instance.
        mesh: ParticleMeshId,
        /// Per-particle orientation.
        align: MeshAlign,
    },
    /// A camera-facing ribbon swept through each particle's recent position
    /// history, tapering to a point at the tail. A comet / streak that follows
    /// the actual simulated path, not just the instantaneous velocity.
    Trail {
        /// Ribbon half-width at the head, in world units. Tapers to zero.
        width: f32,
        /// History segments to sweep. Clamped to the trail history length.
        segments: u32,
    },
}

impl Default for ParticleRender {
    fn default() -> Self {
        ParticleRender::Billboard { stretch: 0.0 }
    }
}

/// How many particles an effect spawns per second.
#[derive(Clone, Copy, Debug)]
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
}

impl Default for SpawnShape {
    fn default() -> Self {
        SpawnShape::Point
    }
}

/// Distribution used to assign an initial velocity to a new particle.
#[derive(Clone, Copy, Debug)]
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

/// Colour and size ramps applied over a particle's lifetime.
///
/// Baked into a small 1D lookup texture the draw shader samples by normalized
/// age (0 at spawn, 1 at death). The sampled RGB multiplies the particle colour
/// and the sampled scale multiplies its size, so an identity gradient (the
/// default) leaves both unchanged. Keys are `(t in 0..1, value)` pairs in
/// ascending `t`; values are linearly interpolated.
///
/// When using a colour ramp, set the emitter/program colour to white so the
/// multiply yields the ramp colour directly.
#[derive(Clone, Debug)]
pub struct Gradient {
    /// Colour keys `(t, rgb)`. Multiplies the particle colour.
    pub colour: Vec<(f32, [f32; 3])>,
    /// Size-scale keys `(t, scale)`. Multiplies the particle size.
    pub size: Vec<(f32, f32)>,
}

impl Default for Gradient {
    fn default() -> Self {
        // Identity: no colour or size change over life.
        Self {
            colour: vec![(0.0, [1.0, 1.0, 1.0]), (1.0, [1.0, 1.0, 1.0])],
            size: vec![(0.0, 1.0), (1.0, 1.0)],
        }
    }
}

impl Gradient {
    /// An identity gradient (no change over life).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the colour ramp keys (ascending `t`).
    pub fn with_colour(mut self, keys: Vec<(f32, [f32; 3])>) -> Self {
        if !keys.is_empty() {
            self.colour = keys;
        }
        self
    }

    /// Set the size-scale ramp keys (ascending `t`).
    pub fn with_size(mut self, keys: Vec<(f32, f32)>) -> Self {
        if !keys.is_empty() {
            self.size = keys;
        }
        self
    }

    /// Sample the colour ramp at normalized age `t`.
    pub(crate) fn sample_colour(&self, t: f32) -> [f32; 3] {
        sample_keys3(&self.colour, t)
    }

    /// Sample the size ramp at normalized age `t`.
    pub(crate) fn sample_size(&self, t: f32) -> f32 {
        sample_keys1(&self.size, t)
    }
}

/// Piecewise-linear sample of ascending `(t, vec3)` keys.
fn sample_keys3(keys: &[(f32, [f32; 3])], t: f32) -> [f32; 3] {
    if keys.is_empty() {
        return [1.0, 1.0, 1.0];
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return [
                a[0] + (b[0] - a[0]) * f,
                a[1] + (b[1] - a[1]) * f,
                a[2] + (b[2] - a[2]) * f,
            ];
        }
    }
    keys[keys.len() - 1].1
}

/// Piecewise-linear sample of ascending `(t, f32)` keys.
fn sample_keys1(keys: &[(f32, f32)], t: f32) -> f32 {
    if keys.is_empty() {
        return 1.0;
    }
    if t <= keys[0].0 {
        return keys[0].1;
    }
    for w in keys.windows(2) {
        let (t0, a) = w[0];
        let (t1, b) = w[1];
        if t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            return a + (b - a) * f;
        }
    }
    keys[keys.len() - 1].1
}

/// GPU blend mode for the particle draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParticleBlend {
    /// Standard over-blend (premultiplied): particles occlude the background by
    /// their alpha.
    Alpha,
    /// Additive: particle colour adds to the background. Emissive glow look.
    Additive,
    /// Premultiplied over-blend. Same blend state as `Alpha` given the shader
    /// already outputs premultiplied colour; kept distinct for intent.
    Premultiplied,
}

impl Default for ParticleBlend {
    fn default() -> Self {
        ParticleBlend::Additive
    }
}

/// A particle attribute an init modifier can set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attribute {
    /// World-space position (`vec3`).
    Position,
    /// Velocity (`vec3`).
    Velocity,
    /// Lifetime in seconds (`f32`); also seeds `max_lifetime`.
    Lifetime,
    /// RGB colour (`vec3`); alpha comes from the lifetime fade at draw.
    Colour,
    /// Billboard size (`f32`).
    Size,
}

/// An init modifier: set an attribute from an expression at spawn.
#[derive(Clone, Debug)]
pub struct SetAttribute {
    /// Which attribute to write.
    pub attribute: Attribute,
    /// Expression evaluated per particle. Its type must match the attribute
    /// (`vec3` for position/velocity/colour, `f32` for lifetime/size).
    pub value: ExprHandle,
}

/// An update modifier: contributes to the per-step motion.
#[derive(Clone, Debug)]
pub enum UpdateOp {
    /// Add a `vec3` acceleration (integrated into velocity each step). Force
    /// modifiers are expressed this way; the expression may read attributes such
    /// as `position` to build attractors.
    Accelerate(ExprHandle),
}

/// A CPU-updatable value the host sets per frame, read in expressions via
/// [`Expr::Property`](crate::Expr::Property). The variant fixes the property's
/// type; a read yields `f32`, `vec3`, or `vec4` accordingly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PropertyValue {
    /// A scalar property.
    F32(f32),
    /// A 3-component vector property.
    Vec3([f32; 3]),
    /// A 4-component vector property.
    Vec4([f32; 4]),
}

impl PropertyValue {
    /// Pack the value into a `vec4` lane for the property uniform (the GPU stores
    /// every property as a `vec4<f32>`, swizzled on read to its declared type).
    pub(crate) fn to_vec4(self) -> [f32; 4] {
        match self {
            PropertyValue::F32(v) => [v, 0.0, 0.0, 0.0],
            PropertyValue::Vec3([x, y, z]) => [x, y, z, 0.0],
            PropertyValue::Vec4(v) => v,
        }
    }
}

/// A named, typed property declared on an [`EffectProgram`]. Its
/// [`default`](Self::default) supplies the value until the host overrides it per
/// frame via [`ParticleItem`](crate::ParticleItem).
#[derive(Clone, Debug)]
pub struct PropertyDecl {
    /// Property name, referenced by [`Expr::Property`](crate::Expr::Property) and
    /// used as the uniform field name; must be a valid WGSL identifier.
    pub name: &'static str,
    /// Default value (and, via its variant, the property's type).
    pub default: PropertyValue,
}

/// A codegen program: an expression [`Module`] plus the init and update
/// modifiers that reference it. An [`EffectAsset`] carrying a program compiles
/// to per-effect emit and simulate WGSL kernels instead of using the
/// fixed-function [`Emitter`] path.
#[derive(Clone, Debug)]
pub struct EffectProgram {
    /// The expression graph the modifiers reference.
    pub module: Module,
    /// Emission schedule.
    pub rate: SpawnRate,
    /// Spawn-time attribute writers, applied in order.
    pub init: Vec<SetAttribute>,
    /// Per-step contributions, applied in order.
    pub update: Vec<UpdateOp>,
    /// Per-particle lifetime range, sampled uniformly at spawn unless an init
    /// modifier writes `Lifetime` explicitly.
    pub lifetime: (f32, f32),
    /// CPU-updatable properties readable in the expression graph. Packed into a
    /// per-effect uniform in declaration order and updated from the per-frame
    /// submission.
    pub properties: Vec<PropertyDecl>,
}

impl Default for EffectProgram {
    fn default() -> Self {
        Self {
            module: Module::new(),
            rate: SpawnRate::default(),
            init: Vec::new(),
            update: Vec::new(),
            lifetime: (1.0, 2.0),
            properties: Vec::new(),
        }
    }
}

impl EffectProgram {
    /// A program with an empty module and the default rate and lifetime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the emission schedule.
    pub fn with_rate(mut self, rate: SpawnRate) -> Self {
        self.rate = rate;
        self
    }

    /// Append an init modifier.
    pub fn set(mut self, attribute: Attribute, value: ExprHandle) -> Self {
        self.init.push(SetAttribute { attribute, value });
        self
    }

    /// Append an update modifier.
    pub fn update(mut self, op: UpdateOp) -> Self {
        self.update.push(op);
        self
    }

    /// Set the spawn lifetime range.
    pub fn with_lifetime(mut self, min: f32, max: f32) -> Self {
        self.lifetime = (min, max);
        self
    }

    /// Declare a CPU-updatable property with a default value. Reference it in the
    /// expression graph with [`Module::property`](crate::Module::property) and
    /// override it per frame via
    /// [`ParticleItem::with_property`](crate::ParticleItem::with_property).
    pub fn property(mut self, name: &'static str, default: PropertyValue) -> Self {
        self.properties.push(PropertyDecl { name, default });
        self
    }
}

/// Fixed-function emitter configuration.
#[derive(Clone, Copy, Debug)]
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

/// The authored description of one effect.
///
/// Carries no GPU state. Register it with a
/// [`ParticlePlugin`](crate::ParticlePlugin) to get an [`EffectId`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct EffectAsset {
    /// Human-readable name, surfaced in debug labels.
    pub name: String,
    /// Maximum simultaneously-live particles. Memory scales linearly.
    pub capacity: u32,
    /// Blend mode for the draw.
    pub blend: ParticleBlend,
    /// Fixed-function emitter. Ignored when [`program`](Self::program) is set.
    pub emitter: Emitter,
    /// Forces summed each simulation step. Ignored when
    /// [`program`](Self::program) is set.
    pub forces: Vec<ForceModifier>,
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
}

impl Default for EffectAsset {
    fn default() -> Self {
        Self {
            name: "effect".to_string(),
            capacity: 10_000,
            blend: ParticleBlend::default(),
            emitter: Emitter::default(),
            forces: Vec::new(),
            program: None,
            gradient: None,
            render: ParticleRender::default(),
            texture: None,
            texture_mode: TextureMode::default(),
            flipbook: None,
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

    /// Append a force.
    pub fn force(mut self, force: ForceModifier) -> Self {
        self.forces.push(force);
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
}
