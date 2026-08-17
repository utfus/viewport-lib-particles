//! The particle [`ItemTypePlugin`] and its per-frame item collection.
//!
//! [`ParticlePlugin`] owns every registered effect's compute/draw pipelines and
//! its persistent particle buffer. The host registers one plugin with the
//! renderer via
//! [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
//! then submits a [`ParticleItems`] collection each frame under
//! [`ParticlePlugin::TYPE_NAME`].
//!
//! Each frame the renderer calls [`prepare`](ItemTypePlugin::prepare), which
//! encodes an emit compute pass (recycle dead slots into new particles) and a
//! simulate compute pass (integrate forces, age particles) per live effect,
//! then [`paint`](ItemTypePlugin::paint), which draws the live particles as
//! camera-facing billboards in the HDR scene pass. No per-particle work happens
//! on the CPU.
//!
//! There is one simulation per registered effect, driven by the first non-hidden
//! instance of that effect (its position is the emitter origin). Per-frame
//! overrides on that instance tune the emitter, forces, gradient, and program
//! properties without re-registering the effect.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use viewport_lib::plugin_api::shared_wgsl::{
    SHARED_BINDINGS_WGSL, SHARED_MASK_WGSL, SHARED_OIT_WGSL, SHARED_PICK_WGSL,
};
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OIT_ACCUM_BLEND, OIT_REVEAL_BLEND, OutlineMaskContext,
    PaintContext, PickPassContext, PluginItemCollection, ShadowCastContext, SharedBindings,
};
use viewport_lib::resources::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT,
    SCENE_DEPTH_FORMAT, SHADOW_DEPTH_FORMAT,
};
use viewport_lib::scene::material::ItemSettings;
use viewport_lib::wgpu as vwgpu;
use viewport_lib::Frustum;

use crate::codegen;
use crate::effect::{
    EffectAsset, EffectId, EffectProgram, Emitter, EventCondition, ForceModifier, Gradient,
    MeshAlign, ParticleBlend, ParticleMeshId, ParticleRender, ParticleTextureId, PropertyDecl,
    PropertyValue, SimulationSpace, SpawnRate, SpawnShape, TextureMode, VelocityDist,
};

/// MSAA sample count the plugin builds its pipelines for. A plugin cannot read
/// the renderer's configured sample count from `SharedBindings` today, so this
/// assumes the default single-sampled HDR pass (see the plan).
const SAMPLE_COUNT: u32 = 1;

/// High bit of an [`InstanceKey`]'s `key`, reserved for internal transient
/// (burst) instances (Phase 25) so they never collide with host-supplied keys.
/// Host keys must stay below this (node ids / indices always do).
const TRANSIENT_KEY_BIT: u64 = 1 << 63;

/// Default cap on concurrent transient (burst) instances per effect.
const DEFAULT_MAX_TRANSIENTS: usize = 64;

/// Max forces per effect, matching the fixed array in `particle_sim.wgsl`.
const MAX_FORCES: usize = 8;

/// Trail history samples kept per particle. Matches `HISTORY_LEN` usage in the
/// trail shaders (passed to them as `HistParams::history_len`). Longer trails
/// cost `capacity * HISTORY_LEN` vec4 of storage per trail effect.
const HISTORY_LEN: u32 = 24;

/// One live instance of an effect in the scene this frame.
///
/// Carries the emitter origin and per-item settings; the authored behavior
/// lives in the [`EffectAsset`] the [`EffectId`] points at.
///
/// Each submitted item with a distinct [`key`](Self::key) simulates as its own
/// independent particle set at its own position (many braziers from one brazier
/// effect), so a host holding a stable key can move / hide / retire one instance
/// across frames without disturbing the others. Items sharing an `(effect, key)`
/// collapse to one instance driven by the first; the default key is `0`, so the
/// common single-instance case is unchanged. (Sub-emitter parent/child effects
/// stay single-instance — their key is forced to `0`.)
///
/// Per-frame overrides tune the driving instance without re-registering the
/// effect: [`with_emitter`](Self::with_emitter) / [`with_forces`](Self::with_forces)
/// replace the fixed-function emitter and forces for live sliders,
/// [`with_gradient`](Self::with_gradient) re-bakes the lifetime ramp, and
/// [`with_property`](Self::with_property) sets a codegen property's value. They
/// are read from the driving item of each `(effect, key)` instance.
#[derive(Clone, Debug)]
pub struct ParticleItem {
    /// Which registered effect to simulate and draw.
    pub effect: EffectId,
    /// Stable per-instance identity within an effect. Items of the same effect
    /// with different keys simulate independently; the same key across frames is
    /// the same live particle set. Defaults to `0`.
    pub key: u64,
    /// World-space emitter origin, added to the effect's spawn shape.
    pub position: [f32; 3],
    /// Shared per-item flags (hidden, selected, pick id, opacity, ...).
    pub settings: ItemSettings,
    /// Per-frame emitter override for the fixed-function path. `None` uses the
    /// effect's registered emitter. Ignored by effects carrying a program.
    pub emitter: Option<Emitter>,
    /// Per-frame force override for the fixed-function path. `None` uses the
    /// effect's registered forces. Ignored by effects carrying a program.
    pub forces: Option<Vec<ForceModifier>>,
    /// Per-frame lifetime-ramp override. When set, the effect's ramp LUT is
    /// re-baked from these keys this frame. `None` keeps the registered gradient.
    pub gradient: Option<Gradient>,
    /// Per-frame property values, applied over the program's declared defaults by
    /// name. Empty leaves every property at its default.
    pub properties: Vec<(&'static str, PropertyValue)>,
    /// When `true`, this is a fire-and-forget transient burst: submit it once at
    /// a world point and the plugin plays it in a pooled transient slot, then
    /// self-retires it when its particles have aged out — no need to keep
    /// submitting it. See [`burst`](Self::burst). Ignored for sub-emitter effects.
    pub one_shot: bool,
    /// Optional host RNG seed for deterministic spawning (Phase 28). When set,
    /// this instance's per-frame emit seed is a fixed function of the seed, the
    /// `(effect, key)`, and the instance's own simulation-frame counter — so a
    /// fixed seed + `dt` sequence reproduces the same seed stream run to run
    /// (preview vs runtime, captures, golden tests). `None` derives the seed from
    /// the global frame counter (not reproducible across runs). See
    /// [`with_seed`](Self::with_seed). GPU atomic spawn order still limits exact
    /// particle-set reproducibility.
    pub seed: Option<u64>,
}

impl ParticleItem {
    /// A visible instance of `effect` at the world origin.
    pub fn new(effect: EffectId) -> Self {
        Self {
            effect,
            key: 0,
            position: [0.0, 0.0, 0.0],
            settings: ItemSettings::default(),
            emitter: None,
            forces: None,
            gradient: None,
            properties: Vec::new(),
            one_shot: false,
            seed: None,
        }
    }

    /// Place the emitter at `position`.
    pub fn at(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }

    /// Mark this as a fire-and-forget transient burst: submit it once (e.g. a
    /// muzzle flash on a shot) and the plugin plays it in a pooled transient
    /// slot and self-retires it when its particles die. No need to keep
    /// submitting it or hold a key. Best paired with `SpawnRate::Burst`. Ignored
    /// for sub-emitter effects.
    pub fn burst(mut self) -> Self {
        self.one_shot = true;
        self
    }

    /// Give this instance a stable per-effect identity. Different keys simulate
    /// as independent particle sets; the same key across frames is the same
    /// live set (so the host can move / hide / retire one of many). No effect on
    /// sub-emitter parent/child effects, which stay single-instance.
    pub fn with_key(mut self, key: u64) -> Self {
        self.key = key;
        self
    }

    /// Override the fixed-function emitter for this frame (live tuning of rate,
    /// lifetime, shape, velocity, colour, size). No effect on program effects.
    pub fn with_emitter(mut self, emitter: Emitter) -> Self {
        self.emitter = Some(emitter);
        self
    }

    /// Override the fixed-function forces for this frame. No effect on program
    /// effects.
    pub fn with_forces(mut self, forces: Vec<ForceModifier>) -> Self {
        self.forces = Some(forces);
        self
    }

    /// Re-bake the lifetime ramp from `gradient` this frame (live gradient-key
    /// editing).
    pub fn with_gradient(mut self, gradient: Gradient) -> Self {
        self.gradient = Some(gradient);
        self
    }

    /// Set a codegen property's value for this frame.
    pub fn with_property(mut self, name: &'static str, value: PropertyValue) -> Self {
        self.properties.push((name, value));
        self
    }

    /// Set a deterministic host RNG seed for this instance (Phase 28). A fixed
    /// seed + `dt` sequence reproduces the same spawn-seed stream run to run.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// A runtime effect-lifecycle command (Phase 27), queued on [`ParticleItems`] and
/// applied by the plugin in `prepare` before it simulates. This is the delivery
/// path for hot-reload since the renderer owns the plugin after installation, so
/// the host cannot call [`ParticlePlugin::replace_effect`] / `remove_effect`
/// directly. (Registering a *new* effect still needs
/// [`ParticlePlugin::add_effect`] at startup, which returns the [`EffectId`].)
#[derive(Clone, Debug)]
enum EffectCommand {
    Replace(EffectId, Box<EffectAsset>),
    Remove(EffectId),
}

/// The per-frame collection of live effect instances.
///
/// Submit via
/// [`SceneFrame::submit_plugin_items`](viewport_lib::renderer::SceneFrame::submit_plugin_items)
/// under [`ParticlePlugin::TYPE_NAME`]. Set [`dt`](Self::dt) to the frame delta
/// so the simulation advances at the right rate.
#[derive(Clone, Debug)]
pub struct ParticleItems {
    items: Vec<ParticleItem>,
    /// Queued lifecycle commands applied at the start of `prepare`.
    commands: Vec<EffectCommand>,
    /// Simulation time step in seconds for this frame. Defaults to 1/60.
    pub dt: f32,
    /// Whether to depth-sort non-additive (alpha / premultiplied) effects
    /// back-to-front this frame. Additive effects are order-independent and are
    /// never sorted. Defaults to `true`; set `false` to see the ordering
    /// artefact.
    pub sort_transparent: bool,
    /// Distance cull (Phase 26): instances whose emitter origin is farther than
    /// this from the camera are not emitted, simulated, or drawn, and drop their
    /// live particles (resuming fresh when back in range). `None` disables the
    /// distance cull. Sub-emitter effects are never distance-culled.
    pub cull_distance: Option<f32>,
    /// Frustum cull (Phase 26): when `true`, instances whose (approximate)
    /// bounding sphere is fully outside the camera frustum are culled like the
    /// distance cull. The bound is estimated from the effect (spawn extent +
    /// particle travel), so it is conservative. `false` disables it.
    pub frustum_cull: bool,
    /// Simulation budget (Phase 26): cap on the number of instances simulated
    /// this frame. When more are visible, the nearest are kept and the rest are
    /// frozen (not simulated or drawn; particles retained). Sub-emitter instances
    /// are always kept. `None` is unlimited.
    pub max_sim_instances: Option<usize>,
    /// Per-effect cap on concurrent transient (burst) instances (Phase 25). When
    /// a new burst would exceed it, the oldest transient of that effect is
    /// recycled. `None` uses the default (64).
    pub max_transients: Option<usize>,
}

impl Default for ParticleItems {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            commands: Vec::new(),
            dt: 1.0 / 60.0,
            sort_transparent: true,
            cull_distance: None,
            frustum_cull: false,
            max_sim_instances: None,
            max_transients: None,
        }
    }
}

impl ParticleItems {
    /// An empty collection with the default 1/60 s time step.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the simulation time step for this frame.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Enable or disable back-to-front depth sorting of non-additive effects for
    /// this frame.
    pub fn with_sort_transparent(mut self, sort: bool) -> Self {
        self.sort_transparent = sort;
        self
    }

    /// Set the distance cull: instances farther than `distance` from the camera
    /// are not simulated or drawn (Phase 26).
    pub fn with_cull_distance(mut self, distance: f32) -> Self {
        self.cull_distance = Some(distance);
        self
    }

    /// Enable frustum culling of off-screen instances this frame (Phase 26).
    pub fn with_frustum_cull(mut self, enable: bool) -> Self {
        self.frustum_cull = enable;
        self
    }

    /// Cap the number of instances simulated this frame; the nearest are kept
    /// (Phase 26).
    pub fn with_max_sim_instances(mut self, max: usize) -> Self {
        self.max_sim_instances = Some(max);
        self
    }

    /// Cap concurrent transient (burst) instances per effect (Phase 25).
    pub fn with_max_transients(mut self, max: usize) -> Self {
        self.max_transients = Some(max);
        self
    }

    /// Append an instance.
    pub fn push(&mut self, item: ParticleItem) {
        self.items.push(item);
    }

    /// Queue a hot-reload: replace `id`'s asset when the plugin next runs
    /// `prepare` (Phase 27). Preserves the live particle set for a uniform-only
    /// change; a structural change rebuilds the effect. See
    /// [`ParticlePlugin::replace_effect`].
    pub fn replace_effect(&mut self, id: EffectId, asset: EffectAsset) {
        self.commands
            .push(EffectCommand::Replace(id, Box::new(asset)));
    }

    /// Queue a removal: tombstone `id` and reclaim its GPU state when the plugin
    /// next runs `prepare` (Phase 27). See [`ParticlePlugin::remove_effect`].
    pub fn remove_effect(&mut self, id: EffectId) {
        self.commands.push(EffectCommand::Remove(id));
    }

    /// The submitted instances.
    pub fn items(&self) -> &[ParticleItem] {
        &self.items
    }
}

impl PluginItemCollection for ParticleItems {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn item_settings(&self, index: usize) -> &ItemSettings {
        &self.items[index].settings
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One particle on the GPU. Matches `Particle` in the WGSL shaders. 80 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParticle {
    position: [f32; 3],
    lifetime: f32,
    velocity: [f32; 3],
    max_lifetime: f32,
    colour: [f32; 4],
    size: f32,
    seed: f32,
    _pad: [f32; 2],
}

/// Emit-pass parameters. Matches `EmitParams` in `particle_emit.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EmitParamsGpu {
    origin: [f32; 4],
    spawn_a: [f32; 4],
    spawn_b: [f32; 4],
    vel_a: [f32; 4],
    vel_b: [f32; 4],
    colour: [f32; 4],
    misc: [f32; 4],
    ctrl: [u32; 4],
    kinds: [u32; 4],
}

/// One force. Matches `Force` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuForce {
    /// xyz vector (accel or attractor position), w = kind as f32.
    data0: [f32; 4],
    /// strength, falloff, drag, _.
    data1: [f32; 4],
}

/// Simulate-pass parameters. Matches `SimParams` in `particle_sim.wgsl` (and,
/// with the extra `event` lane read, `particle_sim_parent.wgsl`). The `event`
/// lane trails the forces array, so the non-parent sim shader, whose struct ends
/// at `forces`, reads the same buffer unchanged.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SimParamsGpu {
    /// dt, time, _, force_count.
    misc: [f32; 4],
    forces: [GpuForce; MAX_FORCES],
    /// Sub-emitter event control (parent effects only): condition
    /// (0 = every step, 1 = on death), events per trigger, child capacity, and
    /// an active flag (1 when this effect is a parent). All zero otherwise.
    event: [u32; 4],
    /// Simulation-space translation applied to every live particle this frame:
    /// `xyz` is the emitting instance's movement since last frame under
    /// [`SimulationSpace::Local`], zero under `Global`. Trails `event`, so the
    /// non-parent sim shader (whose struct now also declares these last two
    /// lanes) reads the same buffer.
    space: [f32; 4],
}

/// One GPU spawn event appended by a parent and consumed by its child. Matches
/// `SpawnEvent` in the sub-emitter shaders. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SpawnEventGpu {
    position: [f32; 4],
    velocity: [f32; 4],
}

/// Bytes of the event buffer header (an atomic count plus padding to the 16-byte
/// alignment the `SpawnEvent` array needs). Matches the `SpawnEvents` struct
/// prefix in the sub-emitter shaders.
const EVENT_HEADER_BYTES: u64 = 16;

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// An uploaded mesh for the mesh render route: local-space vertex positions and
/// a triangle index buffer.
struct ParticleMesh {
    vertex_buf: vwgpu::Buffer,
    index_buf: vwgpu::Buffer,
    index_count: u32,
}

/// An uploaded texture for the billboard texture / flipbook routes: the texture
/// and a default view, sampled in the billboard fragment (group 3).
struct ParticleTexture {
    #[allow(dead_code)]
    texture: vwgpu::Texture,
    view: vwgpu::TextureView,
}

/// Per-effect draw parameters (group 3). Matches `DrawParams` in the draw and
/// mesh shaders. 32 bytes. The billboard shaders read all eight fields; the mesh
/// and trail shaders read only the first four (their `DrawParams` struct is that
/// 16-byte prefix, valid against the larger buffer).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DrawParams {
    /// Billboard stretch factor (0 = round).
    stretch: f32,
    /// Mesh orientation: 0 = velocity-aligned, 1 = random tumble.
    align: u32,
    /// Trail ribbon half-width at the head (world units).
    trail_width: f32,
    /// Trail segments swept (clamped to `HISTORY_LEN - 1`).
    trail_segments: u32,
    /// Texture modulation: 0 = none (procedural dot), 1 = modulate rgb+a,
    /// 2 = modulate rgb, 3 = alpha-from-red mask.
    tex_mode: u32,
    /// Flipbook atlas columns (0 = no flipbook, sample the whole texture).
    flip_cols: u32,
    /// Flipbook atlas rows.
    flip_rows: u32,
    /// Flipbook playback: 0 stretches the sheet across life; > 0 loops at fps.
    flip_fps: f32,
}

/// Trail history parameters (group 1, binding 2 of the append and trail draw
/// passes). Matches `HistParams` in `particle_history.wgsl` /
/// `particle_trail.wgsl`. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HistoryParams {
    /// Ring slot written this frame; also the newest sample when drawing.
    head: u32,
    capacity: u32,
    history_len: u32,
    _pad: u32,
}

/// Generic per-effect params for a generated (codegen) effect. Matches
/// `GenParams` in the generated WGSL. 48 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GenParams {
    origin: [f32; 4],
    /// Simulation-space translation for this frame (see [`SimParamsGpu::space`]).
    space: [f32; 4],
    dt: f32,
    time: f32,
    spawn_count: u32,
    capacity: u32,
    rng_seed: u32,
    _pad: [u32; 3],
}

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// The variant follows the effect's authoring path: fixed-function emitter, or
/// a generated program with its own compiled emit/simulate pipelines.
enum EffectGpu {
    Fixed(FixedGpu),
    Gen(GenGpu),
}

impl EffectGpu {
    fn draw_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.draw_bg,
            EffectGpu::Gen(g) => &g.draw_bg,
        }
    }

    fn ramp_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.ramp_bg,
            EffectGpu::Gen(g) => &g.ramp_bg,
        }
    }

    fn drawparams_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.drawparams_bg,
            EffectGpu::Gen(g) => &g.drawparams_bg,
        }
    }

    /// The persistent particle buffer, for building the trail history bindings.
    fn particle_buf(&self) -> &vwgpu::Buffer {
        match self {
            EffectGpu::Fixed(g) => &g.particle_buf,
            EffectGpu::Gen(g) => &g.particle_buf,
        }
    }

    /// Swap in a freshly baked lifetime ramp (for per-frame gradient overrides).
    fn set_ramp(&mut self, bg: vwgpu::BindGroup, lut: vwgpu::Texture) {
        match self {
            EffectGpu::Fixed(g) => {
                g.ramp_bg = bg;
                g.ramp_lut = Some(lut);
            }
            EffectGpu::Gen(g) => {
                g.ramp_bg = bg;
                g.ramp_lut = Some(lut);
            }
        }
    }
}

/// Init-pass parameters for the depth sort. Matches `InitParams` in
/// `particle_sort_init.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SortInitParams {
    eye: [f32; 4],
    capacity: u32,
    n: u32,
    _pad: [u32; 2],
}

/// One bitonic stage's constants. Matches `SortParams` in `particle_sort.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SortStageParams {
    k: u32,
    j: u32,
    n: u32,
    _pad: u32,
}

/// Minimum dynamic-uniform-offset alignment (wgpu default), the stride between
/// the per-stage `SortStageParams` in the shared stage buffer.
const SORT_STAGE_STRIDE: u64 = 256;

/// Depth-sort GPU state for one non-additive effect: the key buffer, the sort
/// parameter buffers, and the init/stage bind groups over the effect's particle
/// and order buffers. The bitonic stage constants are fixed for the effect's
/// padded length, so they are written once at build time.
struct SortGpu {
    #[allow(dead_code)]
    keys_buf: vwgpu::Buffer,
    init_buf: vwgpu::Buffer,
    #[allow(dead_code)]
    stage_buf: vwgpu::Buffer,
    init_bg: vwgpu::BindGroup,
    stage_bg: vwgpu::BindGroup,
    /// Padded power-of-two length (>= capacity).
    n: u32,
    /// Number of bitonic stages for `n`.
    num_stages: u32,
}

/// Trail render-route GPU state for one effect: a per-particle position history
/// ring, its parameters, and the bind groups for the append (compute) and
/// ribbon (draw) passes. Built lazily alongside the effect's `EffectGpu` when
/// the effect uses `ParticleRender::Trail`.
struct TrailGpu {
    /// `capacity * HISTORY_LEN` vec4 samples (xyz = position, w = seed).
    #[allow(dead_code)]
    history_buf: vwgpu::Buffer,
    /// `HistoryParams`; `head` is rewritten each frame.
    params_buf: vwgpu::Buffer,
    /// Append pass bind group (particles, history, params).
    append_bg: vwgpu::BindGroup,
    /// Trail draw bind group at group 1 (particles, history, params).
    draw_bg: vwgpu::BindGroup,
}

/// Sub-emitter GPU state for one parent effect: the shared spawn-event buffer
/// and the bind groups the parent sim (which appends events) and the child emit
/// (which consumes them) use. Built once the parent is first driven.
struct SubEmitterGpu {
    /// When the parent produces events (matches the asset).
    condition: EventCondition,
    /// Events per trigger.
    count: u32,
    /// Fraction of parent velocity handed to each child particle.
    inherit_velocity: f32,
    /// The child's capacity, the cap on events held in the buffer.
    child_capacity: u32,
    /// `[count: atomic<u32>, pad*3, events: array<SpawnEvent>]`, reset each frame.
    event_buf: vwgpu::Buffer,
    /// Parent sim bind group (parent particles + sim params + event buffer).
    parent_sim_bg: vwgpu::BindGroup,
    /// Child emit bind group (child particles + emit params + budget + events).
    child_emit_bg: vwgpu::BindGroup,
}

/// Fixed-function effect GPU state (the emitter path).
struct FixedGpu {
    /// Persistent particle buffer; the bind groups alias it, so it is held but
    /// not read directly.
    #[allow(dead_code)]
    particle_buf: vwgpu::Buffer,
    emit_buf: vwgpu::Buffer,
    sim_buf: vwgpu::Buffer,
    budget_buf: vwgpu::Buffer,
    emit_bg: vwgpu::BindGroup,
    sim_bg: vwgpu::BindGroup,
    draw_bg: vwgpu::BindGroup,
    ramp_bg: vwgpu::BindGroup,
    drawparams_bg: vwgpu::BindGroup,
    /// Effect-specific ramp LUT, held alive when the effect has a gradient.
    /// `None` when the effect uses the shared identity ramp.
    #[allow(dead_code)]
    ramp_lut: Option<vwgpu::Texture>,
}

/// Generated (codegen) effect GPU state: one `GenParams` buffer feeds both the
/// emit and simulate kernels, which are this effect's own compiled pipelines.
struct GenGpu {
    #[allow(dead_code)]
    particle_buf: vwgpu::Buffer,
    params_buf: vwgpu::Buffer,
    /// Property uniform (one `vec4` lane per declared property, min one lane).
    /// Rewritten each frame from the driving item's property overrides.
    props_buf: vwgpu::Buffer,
    budget_buf: vwgpu::Buffer,
    emit_bg: vwgpu::BindGroup,
    sim_bg: vwgpu::BindGroup,
    draw_bg: vwgpu::BindGroup,
    ramp_bg: vwgpu::BindGroup,
    drawparams_bg: vwgpu::BindGroup,
    #[allow(dead_code)]
    ramp_lut: Option<vwgpu::Texture>,
    emit_pipeline: vwgpu::ComputePipeline,
    sim_pipeline: vwgpu::ComputePipeline,
}

/// A registered effect: the authored asset. Per-instance GPU and emission state
/// lives in [`EffectInstance`], keyed by [`InstanceKey`], since one effect can
/// drive many independent instances.
struct RegisteredEffect {
    asset: EffectAsset,
    /// Set by `remove_effect` (Phase 27): the slot is tombstoned (its
    /// `EffectId` index stays stable) and no longer simulated or drawn.
    removed: bool,
}

/// Identifies one live instance: a registered effect plus the host's stable
/// per-instance key ([`ParticleItem::key`]). Sub-emitter parent/child effects
/// always use key `0` (single instance).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct InstanceKey {
    effect: u32,
    key: u64,
}

/// One live instance of an effect: its own particle buffer + pipelines' bind
/// groups ([`EffectGpu`]), optional trail / sort state over that buffer, its
/// pick-id uniform, and its CPU emission bookkeeping. Built lazily the first
/// frame the instance is driven, and persists (a hidden instance keeps its
/// particles). Independent instances of one effect each own one of these.
struct EffectInstance {
    gpu: EffectGpu,
    /// Trail history state, present only for `ParticleRender::Trail` effects.
    trail: Option<TrailGpu>,
    /// Depth-sort state, present only for non-additive effects.
    sort: Option<SortGpu>,
    /// This instance's pick-id uniform (group 2 of the pick pipeline) and its
    /// bind group, written each frame from the driving item's pick id.
    pick_buf: vwgpu::Buffer,
    pick_bg: vwgpu::BindGroup,
    /// Fractional spawn accumulator for `SpawnRate::PerSecond`.
    spawn_accum: f32,
    /// Whether a `SpawnRate::Burst` has already fired.
    burst_done: bool,
    /// This instance's origin last frame, for the local-space translation delta.
    last_origin: Option<[f32; 3]>,
    /// Set when a distance/frustum cull has zeroed this instance's particle
    /// buffer, so the drop happens once per cull episode (not every frame).
    dropped: bool,
    /// Count of frames this instance has been simulated, mixed into the emit RNG
    /// seed under deterministic seeding (Phase 28) so each frame decorrelates.
    sim_frame: u32,
}

/// A live transient (burst) instance (Phase 25): a pooled particle set playing a
/// one-shot effect at a fixed world point, aged each frame and recycled when its
/// particles have died. Tracked separately from steady instances because the host
/// submits it once and the plugin advances it thereafter.
struct TransientInstance {
    /// Its `(effect, key)` in `instances`; the key carries [`TRANSIENT_KEY_BIT`].
    ikey: InstanceKey,
    /// Fixed world spawn point.
    origin: [f32; 3],
    /// Whether it casts shadows (captured from the spawning item).
    cast_shadows: bool,
    /// Seconds since spawn.
    age: f32,
    /// Age at which the instance is retired and its slot recycled (the effect's
    /// max particle lifetime plus a margin).
    retire_at: f32,
    /// Deterministic seed captured from the spawning item (Phase 28).
    seed: Option<u64>,
}

/// The particle system plugin.
///
/// Register effects with [`add_effect`](Self::add_effect) before handing the
/// plugin to the renderer, then install it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin).
#[derive(Default)]
pub struct ParticlePlugin {
    effects: Vec<RegisteredEffect>,

    // Fixed-function pipelines, shared across effects. Built in `init_gpu`.
    emit_pipeline: Option<vwgpu::ComputePipeline>,
    sim_pipeline: Option<vwgpu::ComputePipeline>,
    draw_pipeline_additive: Option<vwgpu::RenderPipeline>,
    // OIT variant of the billboard draw, for Alpha/Premultiplied effects. These
    // route through the lib's OIT pass (paint_transparent) instead of the main
    // HDR draw so they composite after the skybox and blend order-independently.
    draw_pipeline_oit: Option<vwgpu::RenderPipeline>,
    mesh_pipeline_additive: Option<vwgpu::RenderPipeline>,
    mesh_pipeline_over: Option<vwgpu::RenderPipeline>,
    emit_bgl: Option<vwgpu::BindGroupLayout>,
    sim_bgl: Option<vwgpu::BindGroupLayout>,
    // Sub-emitter variants: a child emit that also reads a spawn-event buffer
    // (binding 3) and a parent sim that also writes one (binding 2). Kept
    // separate so leaf effects keep the plain fixed-function pipelines.
    subemit_pipeline: Option<vwgpu::ComputePipeline>,
    sim_parent_pipeline: Option<vwgpu::ComputePipeline>,
    subemit_bgl: Option<vwgpu::BindGroupLayout>,
    sim_parent_bgl: Option<vwgpu::BindGroupLayout>,
    /// Per-parent sub-emitter GPU state (event buffer + the parent-sim and
    /// child-emit bind groups), keyed by the parent effect index. Built lazily
    /// the first frame a parent effect is driven.
    subemitters: HashMap<usize, SubEmitterGpu>,
    // Codegen-path variants carry an extra property uniform (emit binding 3,
    // sim binding 2). Kept separate so the fixed-function pipelines and bind
    // groups are unchanged.
    gen_emit_bgl: Option<vwgpu::BindGroupLayout>,
    gen_sim_bgl: Option<vwgpu::BindGroupLayout>,
    draw_bgl: Option<vwgpu::BindGroupLayout>,
    drawparams_bgl: Option<vwgpu::BindGroupLayout>,

    // Trail render route: the history-append compute pass and the ribbon draw.
    append_pipeline: Option<vwgpu::ComputePipeline>,
    trail_pipeline_additive: Option<vwgpu::RenderPipeline>,
    trail_pipeline_over: Option<vwgpu::RenderPipeline>,
    history_bgl: Option<vwgpu::BindGroupLayout>,
    trail_draw_bgl: Option<vwgpu::BindGroupLayout>,

    // Depth sort: the setup (key + identity order) pass and one bitonic stage.
    sort_init_pipeline: Option<vwgpu::ComputePipeline>,
    sort_stage_pipeline: Option<vwgpu::ComputePipeline>,
    sort_init_bgl: Option<vwgpu::BindGroupLayout>,
    sort_stage_bgl: Option<vwgpu::BindGroupLayout>,

    // Interaction: pick-id, outline-mask, and shadow-cast pipelines.
    pick_pipeline: Option<vwgpu::RenderPipeline>,
    mask_pipeline: Option<vwgpu::RenderPipeline>,
    shadow_pipeline: Option<vwgpu::RenderPipeline>,
    pick_bgl: Option<vwgpu::BindGroupLayout>,
    shadow_group0_bgl: Option<vwgpu::BindGroupLayout>,

    /// Live per-instance GPU + emission state, keyed by `(effect, key)`. Each
    /// entry owns an independent particle buffer, trail / sort state, and pick
    /// uniform. Built lazily the first frame an instance is driven.
    instances: HashMap<InstanceKey, EffectInstance>,

    /// Active transient (burst) instances (Phase 25), advanced every frame and
    /// recycled when they age out. Their GPU state lives in `instances` under
    /// `TRANSIENT_KEY_BIT` keys.
    transients: Vec<TransientInstance>,
    /// Recycled transient slot indices per effect (a free list).
    transient_free: HashMap<usize, Vec<u64>>,
    /// Next fresh transient slot index per effect (the pool high-water mark).
    transient_next: HashMap<usize, u64>,

    /// Meshes uploaded for the mesh render route, indexed by `ParticleMeshId`.
    meshes: Vec<ParticleMesh>,

    /// Textures uploaded for the billboard texture / flipbook routes, indexed by
    /// `ParticleTextureId`.
    textures: Vec<ParticleTexture>,
    /// A 1x1 white texture bound at group 3 for effects with no texture (so the
    /// billboard fragment always has a valid sampler). Built lazily in `prepare`.
    default_tex_view: Option<vwgpu::TextureView>,

    // Lifetime-ramp LUT (group 2 of the draw pipeline): the layout, the shared
    // linear sampler, and an identity LUT + bind group reused by effects with no
    // gradient. Built in `init_gpu`.
    ramp_bgl: Option<vwgpu::BindGroupLayout>,
    ramp_sampler: Option<vwgpu::Sampler>,
    #[allow(dead_code)]
    identity_ramp_lut: Option<vwgpu::Texture>,
    identity_ramp_bg: Option<vwgpu::BindGroup>,

    /// Compiled generated emit/simulate pipelines, keyed by the hash of their
    /// WGSL so effects with identical programs share pipelines.
    gen_cache: HashMap<u64, (vwgpu::ComputePipeline, vwgpu::ComputePipeline)>,

    /// Effects with live simulation this frame, drawn in `paint` and the
    /// interaction passes.
    draw_list: Vec<DrawEntry>,

    /// Elapsed simulation time in seconds, accumulated from each frame's `dt`.
    /// Scrolls time-varying fields such as the `CurlNoise` force.
    time: f32,
}

/// One instance drawn this frame, with the driving item's interaction flags.
#[derive(Copy, Clone)]
struct DrawEntry {
    key: InstanceKey,
    /// Pick id of the driving item (0 = `PickId::NONE`, not pickable).
    pick_id: u32,
    selected: bool,
    cast_shadows: bool,
}

/// One instance to consider simulating this frame, before culling / budget.
/// Steady instances carry `item_idx` (the source of per-frame overrides);
/// transient (burst) instances carry `None` and use the registered asset.
#[derive(Copy, Clone)]
struct DrivenInstance {
    ikey: InstanceKey,
    origin: [f32; 3],
    pick_id: u32,
    selected: bool,
    cast_shadows: bool,
    /// `Some(index)` for a steady submitted item; `None` for a transient.
    item_idx: Option<usize>,
    /// Deterministic seed for this instance's emit, if any (Phase 28).
    seed: Option<u64>,
}

/// Per-effect pick-id uniform (group 2 of the pick pipeline). 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PickUniform {
    id: u32,
    _pad: [u32; 3],
}

impl ParticlePlugin {
    /// The `type_name` this plugin registers under and that [`ParticleItems`]
    /// are keyed by on the scene frame.
    pub const TYPE_NAME: &'static str = "viewport_lib_particles";

    /// A plugin with no effects yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an effect, returning a handle to submit instances of it later.
    ///
    /// Register every effect before handing the plugin to the renderer; the
    /// renderer owns the plugin afterward. GPU buffers are allocated lazily the
    /// first frame the effect is simulated.
    pub fn add_effect(&mut self, _device: &vwgpu::Device, asset: EffectAsset) -> EffectId {
        let id = EffectId(self.effects.len() as u32);
        self.effects.push(RegisteredEffect {
            asset,
            removed: false,
        });
        id
    }

    /// Number of registered effects (including tombstoned / removed slots).
    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }

    /// Replace a registered effect's asset in place (Phase 27), for hot-reload.
    /// The [`EffectId`] stays valid and its live particle set is **preserved**
    /// when the change is uniform-only (emitter / forces / lifetime / sim-space):
    /// those are re-read per frame, so nothing rebuilds. A structural change
    /// (capacity, render route, blend, texture, gradient, program, sub-emitter)
    /// discards the effect's GPU state so it rebuilds lazily next frame (its
    /// particles reset). A no-op for an unknown or removed id.
    ///
    /// Callable at runtime via [`ParticleItems::replace_effect`] (which routes
    /// here in `prepare`), or directly when a handle to the plugin is held.
    pub fn replace_effect(&mut self, id: EffectId, asset: EffectAsset) {
        let idx = id.0 as usize;
        let Some(eff) = self.effects.get(idx) else {
            return;
        };
        if eff.removed {
            return;
        }
        if structural_sig(&eff.asset) != structural_sig(&asset) {
            self.purge_effect_gpu(idx);
        }
        self.effects[idx].asset = asset;
    }

    /// Remove a registered effect (Phase 27): tombstone its slot (the
    /// [`EffectId`] index stays stable but is inert) and reclaim its per-effect
    /// GPU buffers, bind groups, transient slots, and sub-emitter state. A no-op
    /// for an unknown id. Slots are not reused, so existing ids keep pointing at
    /// the same effects.
    pub fn remove_effect(&mut self, id: EffectId) {
        let idx = id.0 as usize;
        if idx >= self.effects.len() {
            return;
        }
        self.purge_effect_gpu(idx);
        self.effects[idx].removed = true;
    }

    /// Discard all GPU state belonging to effect `idx`: its live instances
    /// (steady + transient), transient pool bookkeeping, and any sub-emitter
    /// entry where it is the parent or the child (whose bind groups alias its
    /// buffers). The wgpu resources drop with the map entries.
    fn purge_effect_gpu(&mut self, idx: usize) {
        let effect = idx as u32;
        self.instances.retain(|k, _| k.effect != effect);
        self.transients.retain(|t| t.ikey.effect != effect);
        self.transient_free.remove(&idx);
        self.transient_next.remove(&idx);
        // Drop sub-emitter state where this effect is the parent, or is the child
        // of some parent (that parent's bind groups reference this buffer).
        let parents_to_drop: Vec<usize> = self
            .effects
            .iter()
            .enumerate()
            .filter_map(|(pi, e)| match e.asset.sub_emitter {
                Some(sub) if sub.child.0 as usize == idx => Some(pi),
                _ => None,
            })
            .collect();
        self.subemitters.remove(&idx);
        for pi in parents_to_drop {
            self.subemitters.remove(&pi);
        }
    }

    /// Upload a mesh for the mesh render route, returning a handle to reference
    /// it from `ParticleRender::Mesh`. `positions` are local-space vertices;
    /// `indices` are triangle indices into them. Upload meshes before handing
    /// the plugin to the renderer.
    pub fn upload_mesh(
        &mut self,
        device: &vwgpu::Device,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> ParticleMeshId {
        let vertex_buf = buffer_with_data(
            device,
            "particle_mesh_vertices",
            bytemuck::cast_slice(positions),
            vwgpu::BufferUsages::VERTEX,
        );
        let index_buf = buffer_with_data(
            device,
            "particle_mesh_indices",
            bytemuck::cast_slice(indices),
            vwgpu::BufferUsages::INDEX,
        );
        let id = ParticleMeshId(self.meshes.len() as u32);
        self.meshes.push(ParticleMesh {
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        });
        id
    }

    /// Upload an RGBA8 texture (sRGB) for the billboard texture / flipbook
    /// routes, returning a handle to reference from `EffectAsset::with_texture`.
    /// `rgba` is `width * height * 4` bytes, row-major, no padding. Upload
    /// textures before handing the plugin to the renderer.
    pub fn upload_texture(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> ParticleTextureId {
        let texture = device.create_texture(&vwgpu::TextureDescriptor {
            label: Some("particle_texture"),
            size: vwgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: vwgpu::TextureDimension::D2,
            format: vwgpu::TextureFormat::Rgba8UnormSrgb,
            usage: vwgpu::TextureUsages::TEXTURE_BINDING | vwgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            vwgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: vwgpu::Origin3d::ZERO,
                aspect: vwgpu::TextureAspect::All,
            },
            rgba,
            vwgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            vwgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&vwgpu::TextureViewDescriptor::default());
        let id = ParticleTextureId(self.textures.len() as u32);
        self.textures.push(ParticleTexture { texture, view });
        id
    }

    /// Build effect `idx`'s GPU state if it does not exist yet. Fixed-function
    /// effects get the shared fixed-function pipelines; effects with a program compile
    /// (or reuse from the cache) their own emit/simulate pipelines.
    fn ensure_instance_gpu(
        &mut self,
        idx: usize,
        key: u64,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        emit_bgl: &vwgpu::BindGroupLayout,
        sim_bgl: &vwgpu::BindGroupLayout,
        draw_bgl: &vwgpu::BindGroupLayout,
    ) {
        let ikey = InstanceKey {
            effect: idx as u32,
            key,
        };
        if self.instances.contains_key(&ikey) {
            return;
        }
        let capacity = self.effects[idx].asset.capacity;
        // Clone the program and gradient out so the `self.effects` borrow ends
        // before we touch the pipeline cache and assign the new GPU state.
        let program = self.effects[idx].asset.program.clone();
        let gradient = self.effects[idx].asset.gradient.clone();
        let render = self.effects[idx].asset.render;
        let blend = self.effects[idx].asset.blend;
        let texture = self.effects[idx].asset.texture;
        let texture_mode = self.effects[idx].asset.texture_mode;
        let flipbook = self.effects[idx].asset.flipbook;

        // Draw order buffer, padded to a power of two so the bitonic sort has a
        // clean length. Identity for now; sorted per frame for non-additive
        // effects.
        let n = next_pow2(capacity);
        let order_buf = build_order_buffer(device, n);

        // Ramp bind group: an effect with a gradient bakes its own LUT; others
        // share the identity ramp built in `prepare`.
        let (ramp_bg, ramp_lut) = match gradient {
            Some(g) => {
                let (ramp_bgl, ramp_sampler) = (
                    self.ramp_bgl.as_ref().unwrap(),
                    self.ramp_sampler.as_ref().unwrap(),
                );
                let lut = build_ramp_lut(device, queue, &g);
                let bg = make_ramp_bg(device, ramp_bgl, ramp_sampler, &lut);
                (bg, Some(lut))
            }
            None => (self.identity_ramp_bg.clone().unwrap(), None),
        };

        // Texture modulation + flipbook (group 3). Only meaningful with a
        // texture bound; `tex_mode == 0` draws the procedural dot.
        let tex_mode = if texture.is_some() {
            match texture_mode {
                TextureMode::Modulate => 1,
                TextureMode::ModulateRgb => 2,
                TextureMode::ModulateAlphaFromR => 3,
            }
        } else {
            0
        };
        let (flip_cols, flip_rows, flip_fps) = match flipbook {
            Some(fb) => (fb.columns, fb.rows, fb.fps),
            None => (0, 0, 0.0),
        };

        // Per-effect draw params (group 3), derived from the render mode.
        let (stretch, align, trail_width, trail_segments) = match render {
            ParticleRender::Billboard { stretch } => (stretch, 0, 0.0, 0),
            ParticleRender::Mesh { align, .. } => (
                0.0,
                match align {
                    MeshAlign::Velocity => 0,
                    MeshAlign::Random => 1,
                },
                0.0,
                0,
            ),
            ParticleRender::Trail { width, segments } => {
                (0.0, 0, width, segments.clamp(1, HISTORY_LEN - 1))
            }
        };
        let dp = DrawParams {
            stretch,
            align,
            trail_width,
            trail_segments,
            tex_mode,
            flip_cols,
            flip_rows,
            flip_fps,
        };
        // Bind the effect's texture, or the shared 1x1 white default. The sprite
        // sampler is the shared linear-clamp ramp sampler.
        let sprite_view = match texture.and_then(|t| self.textures.get(t.0 as usize)) {
            Some(t) => &t.view,
            None => self.default_tex_view.as_ref().unwrap(),
        };
        let sprite_sampler = self.ramp_sampler.as_ref().unwrap();
        let drawparams_bg = build_drawparams_bg(
            device,
            self.drawparams_bgl.as_ref().unwrap(),
            dp,
            sprite_view,
            sprite_sampler,
        );

        let gpu = match program {
            None => EffectGpu::Fixed(build_fixed_gpu(
                device,
                capacity,
                &order_buf,
                emit_bgl,
                sim_bgl,
                draw_bgl,
                ramp_bg,
                ramp_lut,
                drawparams_bg,
            )),
            Some(program) => {
                let gen_emit_bgl = self.gen_emit_bgl.clone().unwrap();
                let gen_sim_bgl = self.gen_sim_bgl.clone().unwrap();
                let (emit_pipeline, sim_pipeline) =
                    self.gen_pipelines(device, &program, &gen_emit_bgl, &gen_sim_bgl);
                EffectGpu::Gen(build_gen_gpu(
                    device,
                    capacity,
                    &order_buf,
                    &gen_emit_bgl,
                    &gen_sim_bgl,
                    draw_bgl,
                    ramp_bg,
                    ramp_lut,
                    drawparams_bg,
                    program.properties.len(),
                    emit_pipeline,
                    sim_pipeline,
                ))
            }
        };
        // Trail effects also get a per-particle history ring plus the append and
        // ribbon bind groups, built over the particle buffer just allocated.
        let trail = if let ParticleRender::Trail { .. } = render {
            let history_bgl = self.history_bgl.clone().unwrap();
            let trail_draw_bgl = self.trail_draw_bgl.clone().unwrap();
            Some(build_trail_gpu(
                device,
                capacity,
                gpu.particle_buf(),
                &order_buf,
                &history_bgl,
                &trail_draw_bgl,
            ))
        } else {
            None
        };

        // Non-additive effects need back-to-front draw order for correct alpha,
        // so they get the sort key + bitonic state over the particle and order
        // buffers. Additive effects are order-independent and keep identity.
        let sort = if !matches!(blend, ParticleBlend::Additive) {
            let sort_init_bgl = self.sort_init_bgl.clone().unwrap();
            let sort_stage_bgl = self.sort_stage_bgl.clone().unwrap();
            Some(build_sort_gpu(
                device,
                n,
                gpu.particle_buf(),
                &order_buf,
                &sort_init_bgl,
                &sort_stage_bgl,
            ))
        } else {
            None
        };

        // Per-instance pick-id uniform (group 2 of the pick pipeline). Written
        // each frame from the driving item's pick id.
        let pick_bgl = self.pick_bgl.clone().unwrap();
        let pick_buf = device.create_buffer(&vwgpu::BufferDescriptor {
            label: Some("particle_pick_id"),
            size: std::mem::size_of::<PickUniform>() as u64,
            usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pick_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
            label: Some("particle_pick_bg"),
            layout: &pick_bgl,
            entries: &[vwgpu::BindGroupEntry {
                binding: 0,
                resource: pick_buf.as_entire_binding(),
            }],
        });

        self.instances.insert(
            ikey,
            EffectInstance {
                gpu,
                trail,
                sort,
                pick_buf,
                pick_bg,
                spawn_accum: 0.0,
                burst_done: false,
                last_origin: None,
                dropped: false,
                sim_frame: 0,
            },
        );
    }

    /// Build parent `parent_idx`'s sub-emitter state if it does not exist yet:
    /// the shared event buffer plus the parent-sim and child-emit bind groups.
    /// Ensures both the parent and child effects have GPU state first. A no-op if
    /// the parent has no sub-emitter, or if either end carries a program (the
    /// feature is fixed-function only).
    fn ensure_subemitter_gpu(
        &mut self,
        parent_idx: usize,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        emit_bgl: &vwgpu::BindGroupLayout,
        sim_bgl: &vwgpu::BindGroupLayout,
        draw_bgl: &vwgpu::BindGroupLayout,
    ) {
        if self.subemitters.contains_key(&parent_idx) {
            return;
        }
        let Some(sub) = self.effects[parent_idx].asset.sub_emitter else {
            return;
        };
        let child_idx = sub.child.0 as usize;
        if child_idx >= self.effects.len() {
            return;
        }
        if self.effects[parent_idx].asset.program.is_some()
            || self.effects[child_idx].asset.program.is_some()
        {
            return;
        }

        // Both effects need their (single, key-0) instance GPU state so we can
        // bind their buffers. Sub-emitter effects are always single-instance.
        self.ensure_instance_gpu(parent_idx, 0, device, queue, emit_bgl, sim_bgl, draw_bgl);
        self.ensure_instance_gpu(child_idx, 0, device, queue, emit_bgl, sim_bgl, draw_bgl);
        let parent_ikey = InstanceKey {
            effect: parent_idx as u32,
            key: 0,
        };
        let child_ikey = InstanceKey {
            effect: child_idx as u32,
            key: 0,
        };

        let child_capacity = self.effects[child_idx].asset.capacity;
        let event_buf = build_event_buffer(device, child_capacity);

        let parent_sim_bg = {
            let Some(EffectGpu::Fixed(g)) = self.instances.get(&parent_ikey).map(|i| &i.gpu) else {
                return;
            };
            build_sim_parent_bg(
                device,
                self.sim_parent_bgl.as_ref().unwrap(),
                &g.particle_buf,
                &g.sim_buf,
                &event_buf,
            )
        };
        let child_emit_bg = {
            let Some(EffectGpu::Fixed(g)) = self.instances.get(&child_ikey).map(|i| &i.gpu) else {
                return;
            };
            build_subemit_bg(
                device,
                self.subemit_bgl.as_ref().unwrap(),
                &g.particle_buf,
                &g.emit_buf,
                &g.budget_buf,
                &event_buf,
            )
        };

        self.subemitters.insert(
            parent_idx,
            SubEmitterGpu {
                condition: sub.condition,
                count: sub.count,
                inherit_velocity: sub.inherit_velocity,
                child_capacity,
                event_buf,
                parent_sim_bg,
                child_emit_bg,
            },
        );
    }

    /// Acquire a transient (burst) slot index for `effect`: a recycled free slot,
    /// else a fresh one while under `max`, else the slot of the oldest active
    /// transient of this effect (drop-oldest). Returns `None` only when `max` is
    /// 0. The caller retires any dropped transient's tracking.
    fn acquire_transient_slot(&mut self, effect: usize, max: usize) -> Option<u64> {
        if let Some(slot) = self.transient_free.get_mut(&effect).and_then(|v| v.pop()) {
            return Some(slot);
        }
        let next = self.transient_next.entry(effect).or_insert(0);
        if (*next as usize) < max {
            let slot = *next;
            *next += 1;
            return Some(slot);
        }
        // Pool full: recycle the oldest active transient of this effect.
        let oldest = self
            .transients
            .iter()
            .enumerate()
            .filter(|(_, t)| t.ikey.effect as usize == effect)
            .max_by(|a, b| {
                a.1.age
                    .partial_cmp(&b.1.age)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);
        oldest.map(|i| self.transients.remove(i).ikey.key & !TRANSIENT_KEY_BIT)
    }

    /// Compile (or reuse) the emit/simulate pipelines for a program, keyed by
    /// the hash of its generated WGSL.
    fn gen_pipelines(
        &mut self,
        device: &vwgpu::Device,
        program: &EffectProgram,
        emit_bgl: &vwgpu::BindGroupLayout,
        sim_bgl: &vwgpu::BindGroupLayout,
    ) -> (vwgpu::ComputePipeline, vwgpu::ComputePipeline) {
        let shaders = codegen::generate_program(program);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        shaders.emit.hash(&mut hasher);
        shaders.sim.hash(&mut hasher);
        let key = hasher.finish();
        if let Some((emit, sim)) = self.gen_cache.get(&key) {
            return (emit.clone(), sim.clone());
        }
        let emit = build_compute_pipeline(device, &shaders.emit, "emit_main", "gen_emit", emit_bgl);
        let sim = build_compute_pipeline(device, &shaders.sim, "sim_main", "gen_sim", sim_bgl);
        self.gen_cache.insert(key, (emit.clone(), sim.clone()));
        (emit, sim)
    }
}

impl ItemTypePlugin for ParticlePlugin {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    fn init_gpu(&mut self, device: &vwgpu::Device, shared: &SharedBindings<'_>) {
        let storage_rw = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::COMPUTE,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::COMPUTE,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let emit_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_emit_bgl"),
            entries: &[storage_rw(0), uniform(1), storage_rw(2)],
        });
        let sim_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sim_bgl"),
            entries: &[storage_rw(0), uniform(1)],
        });
        // Codegen kernels add the property uniform: emit binding 3, sim binding 2.
        let gen_emit_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_gen_emit_bgl"),
            entries: &[storage_rw(0), uniform(1), storage_rw(2), uniform(3)],
        });
        let gen_sim_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_gen_sim_bgl"),
            entries: &[storage_rw(0), uniform(1), uniform(2)],
        });
        let draw_storage = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::VERTEX,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // binding 0 = particles, binding 1 = draw order (identity or sorted).
        let draw_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_draw_bgl"),
            entries: &[draw_storage(0), draw_storage(1)],
        });

        // Group 2: the lifetime-ramp LUT, sampled in the vertex stage. The
        // identity LUT + its bind group are built lazily in `prepare` (which has
        // a queue for the texture upload).
        let ramp_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_ramp_bgl"),
            entries: &[
                vwgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vwgpu::ShaderStages::VERTEX,
                    ty: vwgpu::BindingType::Texture {
                        sample_type: vwgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: vwgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vwgpu::ShaderStages::VERTEX,
                    ty: vwgpu::BindingType::Sampler(vwgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let ramp_sampler = device.create_sampler(&vwgpu::SamplerDescriptor {
            label: Some("particle_ramp_sampler"),
            address_mode_u: vwgpu::AddressMode::ClampToEdge,
            address_mode_v: vwgpu::AddressMode::ClampToEdge,
            address_mode_w: vwgpu::AddressMode::ClampToEdge,
            mag_filter: vwgpu::FilterMode::Linear,
            min_filter: vwgpu::FilterMode::Linear,
            mipmap_filter: vwgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Group 3: per-effect draw params (binding 0, read in the vertex stage of
        // every route and the fragment stage of the billboard routes) plus the
        // billboard texture (binding 1) and its sampler (binding 2). Every
        // effect binds a texture here (a 1x1 white default when it has none), so
        // the billboard fragment always has a valid sampler; the mesh and trail
        // shaders leave bindings 1/2 unused.
        let drawparams_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_drawparams_bgl"),
            entries: &[
                vwgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vwgpu::ShaderStages::VERTEX | vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Buffer {
                        ty: vwgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Texture {
                        sample_type: vwgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: vwgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Sampler(vwgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Trail render route bind group layouts. The append pass (compute) reads
        // the particles and writes the history ring; the ribbon draw reads both
        // in the vertex stage. Binding 2 is the shared `HistParams` uniform.
        let storage_ro_vertex = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::VERTEX,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_vertex = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::VERTEX,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_ro_compute = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::COMPUTE,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let history_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_history_bgl"),
            entries: &[storage_ro_compute(0), storage_rw(1), uniform(2)],
        });
        // binding 0 = particles, 1 = history, 2 = HistParams, 3 = draw order.
        let trail_draw_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_trail_draw_bgl"),
            entries: &[
                storage_ro_vertex(0),
                storage_ro_vertex(1),
                uniform_vertex(2),
                storage_ro_vertex(3),
            ],
        });

        // Compute pipelines.
        let compute = |src: &str, entry: &str, label: &str, bgl: &vwgpu::BindGroupLayout| {
            let module = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: vwgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = vwgpu::pipeline_layout(device, label, &[bgl]);
            device.create_compute_pipeline(&vwgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let emit_pipeline = compute(
            include_str!("shaders/particle_emit.wgsl"),
            "emit_main",
            "particle_emit",
            &emit_bgl,
        );
        // Prepend the shared curl-noise helpers so the CurlNoise force can call
        // them (same source of truth as the codegen noise ops).
        let sim_src = format!(
            "{}\n{}",
            codegen::NOISE_WGSL,
            include_str!("shaders/particle_sim.wgsl")
        );
        let sim_pipeline = compute(&sim_src, "sim_main", "particle_sim", &sim_bgl);

        // Sub-emitter kernels: a child emit that also reads a spawn-event buffer,
        // and a parent sim that also writes one.
        let subemit_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_subemit_bgl"),
            entries: &[
                storage_rw(0),
                uniform(1),
                storage_rw(2),
                storage_ro_compute(3),
            ],
        });
        let sim_parent_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sim_parent_bgl"),
            entries: &[storage_rw(0), uniform(1), storage_rw(2)],
        });
        let subemit_pipeline = compute(
            include_str!("shaders/particle_subemit.wgsl"),
            "emit_main",
            "particle_subemit",
            &subemit_bgl,
        );
        let sim_parent_src = format!(
            "{}\n{}",
            codegen::NOISE_WGSL,
            include_str!("shaders/particle_sim_parent.wgsl")
        );
        let sim_parent_pipeline =
            compute(&sim_parent_src, "sim_main", "particle_sim_parent", &sim_parent_bgl);

        let append_pipeline = compute(
            include_str!("shaders/particle_history.wgsl"),
            "history_main",
            "particle_history",
            &history_bgl,
        );

        // Depth sort: a setup pass (keys + identity order) and a bitonic stage.
        let sort_init_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sort_init_bgl"),
            entries: &[
                storage_ro_compute(0),
                storage_rw(1),
                storage_rw(2),
                uniform(3),
            ],
        });
        // The stage constants sit in a dynamic-offset uniform so one pass can be
        // dispatched per bitonic stage without re-uploading between passes.
        let sort_stage_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_sort_stage_bgl"),
            entries: &[
                storage_ro_compute(0),
                storage_rw(1),
                vwgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vwgpu::ShaderStages::COMPUTE,
                    ty: vwgpu::BindingType::Buffer {
                        ty: vwgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sort_init_pipeline = compute(
            include_str!("shaders/particle_sort_init.wgsl"),
            "init_main",
            "particle_sort_init",
            &sort_init_bgl,
        );
        let sort_stage_pipeline = compute(
            include_str!("shaders/particle_sort.wgsl"),
            "sort_main",
            "particle_sort",
            &sort_stage_bgl,
        );

        // Draw pipelines (additive and premultiplied over).
        let draw_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("shaders/particle_draw.wgsl")
        );
        let draw_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_draw_shader"),
            source: vwgpu::ShaderSource::Wgsl(draw_src.into()),
        });
        let draw_layout = vwgpu::pipeline_layout(
            device,
            "particle_draw_layout",
            &[shared.group0_layout, &draw_bgl, &ramp_bgl, &drawparams_bgl],
        );

        let one = vwgpu::BlendComponent {
            src_factor: vwgpu::BlendFactor::One,
            dst_factor: vwgpu::BlendFactor::One,
            operation: vwgpu::BlendOperation::Add,
        };
        let over = vwgpu::BlendComponent {
            src_factor: vwgpu::BlendFactor::One,
            dst_factor: vwgpu::BlendFactor::OneMinusSrcAlpha,
            operation: vwgpu::BlendOperation::Add,
        };
        let make_draw = |blend: vwgpu::BlendState, label: &str| {
            vwgpu::render_pipeline(
                device,
                vwgpu::RenderPipelineDesc {
                    label,
                    layout: &draw_layout,
                    vertex: vwgpu::VertexState {
                        module: &draw_shader,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(vwgpu::FragmentState {
                        module: &draw_shader,
                        entry_point: Some("fs"),
                        targets: &[Some(vwgpu::ColorTargetState {
                            format: HDR_COLOR_FORMAT,
                            blend: Some(blend),
                            write_mask: vwgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: vwgpu::PrimitiveState {
                        topology: vwgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    // Depth-test against opaque geometry, but do not write depth
                    // so overlapping particles blend instead of occluding.
                    depth_stencil: Some(vwgpu::depth_stencil(
                        SCENE_DEPTH_FORMAT,
                        false,
                        vwgpu::CompareFunction::LessEqual,
                    )),
                    multisample: vwgpu::MultisampleState {
                        count: SAMPLE_COUNT,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };

        self.draw_pipeline_additive = Some(make_draw(
            vwgpu::BlendState {
                color: one,
                alpha: one,
            },
            "particle_draw_additive",
        ));
        // OIT billboard variant: same vertex stage, but the fragment writes the
        // two weighted-blended MRT targets (accum + reveal). Drawn inside the
        // lib's OIT pass via paint_transparent, so it composites after the
        // skybox and needs no back-to-front sort.
        let oit_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{SHARED_OIT_WGSL}\n{}",
            include_str!("shaders/particle_oit.wgsl")
        );
        let oit_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_oit_shader"),
            source: vwgpu::ShaderSource::Wgsl(oit_src.into()),
        });
        self.draw_pipeline_oit = Some(vwgpu::render_pipeline(
            device,
            vwgpu::RenderPipelineDesc {
                label: "particle_draw_oit",
                layout: &draw_layout,
                vertex: vwgpu::VertexState {
                    module: &oit_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(vwgpu::FragmentState {
                    module: &oit_shader,
                    entry_point: Some("fs"),
                    targets: &[
                        // Target 0: accum (Rgba16Float, additive).
                        Some(vwgpu::ColorTargetState {
                            format: HDR_COLOR_FORMAT,
                            blend: Some(OIT_ACCUM_BLEND),
                            write_mask: vwgpu::ColorWrites::ALL,
                        }),
                        // Target 1: reveal (R8Unorm, multiplicative).
                        Some(vwgpu::ColorTargetState {
                            format: MASK_COLOR_FORMAT,
                            blend: Some(OIT_REVEAL_BLEND),
                            write_mask: vwgpu::ColorWrites::RED,
                        }),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: vwgpu::PrimitiveState {
                    topology: vwgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                // Depth-test against opaques, read-only (the OIT pass never
                // writes depth).
                depth_stencil: Some(vwgpu::depth_stencil(
                    SCENE_DEPTH_FORMAT,
                    false,
                    vwgpu::CompareFunction::LessEqual,
                )),
                multisample: vwgpu::MultisampleState {
                    count: SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        ));

        // Mesh render route: instance an uploaded mesh per particle. Same bind
        // group layouts plus a per-vertex position stream.
        let mesh_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("shaders/particle_mesh.wgsl")
        );
        let mesh_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_mesh_shader"),
            source: vwgpu::ShaderSource::Wgsl(mesh_src.into()),
        });
        let mesh_vertex_layout = vwgpu::VertexBufferLayout {
            array_stride: 12,
            step_mode: vwgpu::VertexStepMode::Vertex,
            attributes: &[vwgpu::VertexAttribute {
                format: vwgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            }],
        };
        let make_mesh = |blend: vwgpu::BlendState, label: &str| {
            vwgpu::render_pipeline(
                device,
                vwgpu::RenderPipelineDesc {
                    label,
                    layout: &draw_layout,
                    vertex: vwgpu::VertexState {
                        module: &mesh_shader,
                        entry_point: Some("vs"),
                        buffers: &[mesh_vertex_layout.clone()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(vwgpu::FragmentState {
                        module: &mesh_shader,
                        entry_point: Some("fs"),
                        targets: &[Some(vwgpu::ColorTargetState {
                            format: HDR_COLOR_FORMAT,
                            blend: Some(blend),
                            write_mask: vwgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: vwgpu::PrimitiveState {
                        topology: vwgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(vwgpu::depth_stencil(
                        SCENE_DEPTH_FORMAT,
                        false,
                        vwgpu::CompareFunction::LessEqual,
                    )),
                    multisample: vwgpu::MultisampleState {
                        count: SAMPLE_COUNT,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        self.mesh_pipeline_additive = Some(make_mesh(
            vwgpu::BlendState {
                color: one,
                alpha: one,
            },
            "particle_mesh_additive",
        ));
        self.mesh_pipeline_over = Some(make_mesh(
            vwgpu::BlendState {
                color: over,
                alpha: over,
            },
            "particle_mesh_over",
        ));

        // Trail render route: a camera-facing ribbon swept through the history
        // ring. Group 1 is the trail bindings (particles + history + params);
        // groups 0/2/3 match the other routes.
        let trail_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("shaders/particle_trail.wgsl")
        );
        let trail_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_trail_shader"),
            source: vwgpu::ShaderSource::Wgsl(trail_src.into()),
        });
        let trail_layout = vwgpu::pipeline_layout(
            device,
            "particle_trail_layout",
            &[
                shared.group0_layout,
                &trail_draw_bgl,
                &ramp_bgl,
                &drawparams_bgl,
            ],
        );
        let make_trail = |blend: vwgpu::BlendState, label: &str| {
            vwgpu::render_pipeline(
                device,
                vwgpu::RenderPipelineDesc {
                    label,
                    layout: &trail_layout,
                    vertex: vwgpu::VertexState {
                        module: &trail_shader,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(vwgpu::FragmentState {
                        module: &trail_shader,
                        entry_point: Some("fs"),
                        targets: &[Some(vwgpu::ColorTargetState {
                            format: HDR_COLOR_FORMAT,
                            blend: Some(blend),
                            write_mask: vwgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: vwgpu::PrimitiveState {
                        topology: vwgpu::PrimitiveTopology::TriangleList,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: Some(vwgpu::depth_stencil(
                        SCENE_DEPTH_FORMAT,
                        false,
                        vwgpu::CompareFunction::LessEqual,
                    )),
                    multisample: vwgpu::MultisampleState {
                        count: SAMPLE_COUNT,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        self.trail_pipeline_additive = Some(make_trail(
            vwgpu::BlendState {
                color: one,
                alpha: one,
            },
            "particle_trail_additive",
        ));
        self.trail_pipeline_over = Some(make_trail(
            vwgpu::BlendState {
                color: over,
                alpha: over,
            },
            "particle_trail_over",
        ));

        // Interaction pipelines: pick id, outline mask, and shadow casting. They
        // reuse the draw bind group (particles + order) at group 1.
        let pick_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_pick_bgl"),
            entries: &[vwgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vwgpu::ShaderStages::VERTEX,
                ty: vwgpu::BindingType::Buffer {
                    ty: vwgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        // The shadow pass binds a single cascade light-space matrix with a
        // dynamic offset (the lib's shadow_camera group), not the scene camera.
        let shadow_group0_bgl =
            device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
                label: Some("particle_shadow_group0_bgl"),
                entries: &[vwgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vwgpu::ShaderStages::VERTEX,
                    ty: vwgpu::BindingType::Buffer {
                        ty: vwgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pick_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{SHARED_PICK_WGSL}\n{}",
            include_str!("shaders/particle_pick.wgsl")
        );
        let pick_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_pick_shader"),
            source: vwgpu::ShaderSource::Wgsl(pick_src.into()),
        });
        let pick_layout = vwgpu::pipeline_layout(
            device,
            "particle_pick_layout",
            &[shared.group0_layout, &draw_bgl, &pick_bgl],
        );
        let pick_color = |format| {
            Some(vwgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: vwgpu::ColorWrites::ALL,
            })
        };
        let pick_pipeline = vwgpu::render_pipeline(
            device,
            vwgpu::RenderPipelineDesc {
                label: "particle_pick",
                layout: &pick_layout,
                vertex: vwgpu::VertexState {
                    module: &pick_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(vwgpu::FragmentState {
                    module: &pick_shader,
                    entry_point: Some("viewport_pick_fs"),
                    targets: &[
                        pick_color(PICK_COLOR_FORMAT),
                        pick_color(PICK_COLOR_FORMAT),
                        pick_color(PICK_DEPTH_CHANNEL_FORMAT),
                    ],
                    compilation_options: Default::default(),
                }),
                primitive: vwgpu::PrimitiveState {
                    topology: vwgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(vwgpu::depth_stencil(
                    SCENE_DEPTH_FORMAT,
                    true,
                    vwgpu::CompareFunction::LessEqual,
                )),
                multisample: vwgpu::MultisampleState {
                    count: SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        );

        let mask_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{SHARED_MASK_WGSL}\n{}",
            include_str!("shaders/particle_mask.wgsl")
        );
        let mask_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_mask_shader"),
            source: vwgpu::ShaderSource::Wgsl(mask_src.into()),
        });
        let mask_layout = vwgpu::pipeline_layout(
            device,
            "particle_mask_layout",
            &[shared.group0_layout, &draw_bgl],
        );
        let mask_pipeline = vwgpu::render_pipeline(
            device,
            vwgpu::RenderPipelineDesc {
                label: "particle_mask",
                layout: &mask_layout,
                vertex: vwgpu::VertexState {
                    module: &mask_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(vwgpu::FragmentState {
                    module: &mask_shader,
                    entry_point: Some("viewport_mask_fs"),
                    targets: &[Some(vwgpu::ColorTargetState {
                        format: MASK_COLOR_FORMAT,
                        blend: None,
                        write_mask: vwgpu::ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: vwgpu::PrimitiveState {
                    topology: vwgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(vwgpu::depth_stencil(
                    SCENE_DEPTH_FORMAT,
                    false,
                    vwgpu::CompareFunction::LessEqual,
                )),
                multisample: vwgpu::MultisampleState {
                    count: SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        );

        let shadow_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_shadow_shader"),
            source: vwgpu::ShaderSource::Wgsl(include_str!("shaders/particle_shadow.wgsl").into()),
        });
        let shadow_layout = vwgpu::pipeline_layout(
            device,
            "particle_shadow_layout",
            &[&shadow_group0_bgl, &draw_bgl],
        );
        let shadow_pipeline = vwgpu::render_pipeline(
            device,
            vwgpu::RenderPipelineDesc {
                label: "particle_shadow",
                layout: &shadow_layout,
                vertex: vwgpu::VertexState {
                    module: &shadow_shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: None,
                primitive: vwgpu::PrimitiveState {
                    topology: vwgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(vwgpu::DepthStencilState {
                    format: SHADOW_DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: vwgpu::CompareFunction::LessEqual,
                    stencil: vwgpu::StencilState::default(),
                    bias: vwgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: vwgpu::MultisampleState {
                    count: SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        );

        self.pick_pipeline = Some(pick_pipeline);
        self.mask_pipeline = Some(mask_pipeline);
        self.shadow_pipeline = Some(shadow_pipeline);
        self.pick_bgl = Some(pick_bgl);
        self.shadow_group0_bgl = Some(shadow_group0_bgl);

        self.emit_pipeline = Some(emit_pipeline);
        self.sim_pipeline = Some(sim_pipeline);
        self.subemit_pipeline = Some(subemit_pipeline);
        self.sim_parent_pipeline = Some(sim_parent_pipeline);
        self.subemit_bgl = Some(subemit_bgl);
        self.sim_parent_bgl = Some(sim_parent_bgl);
        self.append_pipeline = Some(append_pipeline);
        self.sort_init_pipeline = Some(sort_init_pipeline);
        self.sort_stage_pipeline = Some(sort_stage_pipeline);
        self.emit_bgl = Some(emit_bgl);
        self.sim_bgl = Some(sim_bgl);
        self.gen_emit_bgl = Some(gen_emit_bgl);
        self.gen_sim_bgl = Some(gen_sim_bgl);
        self.draw_bgl = Some(draw_bgl);
        self.drawparams_bgl = Some(drawparams_bgl);
        self.history_bgl = Some(history_bgl);
        self.trail_draw_bgl = Some(trail_draw_bgl);
        self.sort_init_bgl = Some(sort_init_bgl);
        self.sort_stage_bgl = Some(sort_stage_bgl);
        self.ramp_bgl = Some(ramp_bgl);
        self.ramp_sampler = Some(ramp_sampler);
    }

    fn prepare(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &dyn PluginItemCollection,
    ) -> Vec<vwgpu::CommandBuffer> {
        self.draw_list.clear();

        let Some(items) = items.as_any().downcast_ref::<ParticleItems>() else {
            return Vec::new();
        };
        let (Some(emit_bgl), Some(sim_bgl), Some(draw_bgl)) = (
            self.emit_bgl.as_ref(),
            self.sim_bgl.as_ref(),
            self.draw_bgl.as_ref(),
        ) else {
            return Vec::new();
        };
        let (Some(emit_pipeline), Some(sim_pipeline)) =
            (self.emit_pipeline.as_ref(), self.sim_pipeline.as_ref())
        else {
            return Vec::new();
        };
        // Cheap clones (wgpu handles are Arc-backed) so the borrow of `self`
        // ends before the per-effect loop mutates `self.effects`.
        let (emit_bgl, sim_bgl, draw_bgl) = (emit_bgl.clone(), sim_bgl.clone(), draw_bgl.clone());
        let (emit_pipeline, sim_pipeline) = (emit_pipeline.clone(), sim_pipeline.clone());
        let subemit_pipeline = self.subemit_pipeline.clone();
        let sim_parent_pipeline = self.sim_parent_pipeline.clone();
        let append_pipeline = self.append_pipeline.clone();
        let sort_init_pipeline = self.sort_init_pipeline.clone();
        let sort_stage_pipeline = self.sort_stage_pipeline.clone();
        // For per-frame gradient overrides (re-baking the ramp LUT in the loop).
        let ramp_bgl = self.ramp_bgl.clone();
        let ramp_sampler = self.ramp_sampler.clone();

        // Build the shared identity ramp on first use (needs a queue, so not in
        // `init_gpu`).
        if self.identity_ramp_bg.is_none() {
            if let (Some(ramp_bgl), Some(ramp_sampler)) =
                (self.ramp_bgl.as_ref(), self.ramp_sampler.as_ref())
            {
                let lut = build_ramp_lut(device, queue, &crate::effect::Gradient::default());
                let bg = make_ramp_bg(device, ramp_bgl, ramp_sampler, &lut);
                self.identity_ramp_lut = Some(lut);
                self.identity_ramp_bg = Some(bg);
            }
        }

        // Shared 1x1 white texture bound at group 3 for effects with no texture.
        if self.default_tex_view.is_none() {
            let tex = build_solid_texture(device, queue, [255, 255, 255, 255]);
            self.default_tex_view = Some(tex.create_view(&vwgpu::TextureViewDescriptor::default()));
        }

        let dt = items.dt.max(0.0);
        // Advance the shared simulation clock once per frame (drives time-varying
        // fields such as the CurlNoise force).
        self.time += dt;
        let time = self.time;

        // Apply queued lifecycle commands (Phase 27) before simulating, so a
        // hot-reloaded or removed effect takes effect this frame.
        for cmd in &items.commands {
            match cmd {
                EffectCommand::Replace(id, asset) => self.replace_effect(*id, (**asset).clone()),
                EffectCommand::Remove(id) => self.remove_effect(*id),
            }
        }

        // Sub-emitter wiring: which effect is whose child. Derived from the
        // assets, independent of GPU state. Used to order passes and to pick the
        // sub-emit / parent-sim pipelines below.
        let mut parent_of: HashMap<usize, usize> = HashMap::new();
        for (pi, eff) in self.effects.iter().enumerate() {
            if eff.removed {
                continue;
            }
            if let Some(sub) = eff.asset.sub_emitter {
                let ci = sub.child.0 as usize;
                if ci < self.effects.len() {
                    parent_of.insert(ci, pi);
                }
            }
        }

        let num_effects = self.effects.len();
        // Effects involved in a sub-emitter (parent or child) stay single-instance
        // (key 0) and are exempt from culling / budget / transients — their event
        // buffers bind one particle set. Precomputed into a set (not a closure
        // over `self`) so the per-instance loops below can mutate `self` freely.
        let mut single_effects: std::collections::HashSet<usize> =
            parent_of.keys().copied().collect();
        for (ei, e) in self.effects.iter().enumerate() {
            if e.asset.sub_emitter.is_some() {
                single_effects.insert(ei);
            }
        }
        let is_single = |idx: usize| single_effects.contains(&idx);

        // Steady instances: first non-hidden, non-one-shot item of each distinct
        // `(effect, key)`. One-shot items become transients below.
        let mut driven: Vec<DrivenInstance> = Vec::new();
        let mut seen: std::collections::HashSet<InstanceKey> = std::collections::HashSet::new();
        for (i, it) in items.items.iter().enumerate() {
            if it.settings.hidden || it.one_shot {
                continue;
            }
            let idx = it.effect.0 as usize;
            if idx >= num_effects || self.effects[idx].removed {
                continue;
            }
            let key = if is_single(idx) { 0 } else { it.key };
            let ikey = InstanceKey {
                effect: idx as u32,
                key,
            };
            if seen.insert(ikey) {
                driven.push(DrivenInstance {
                    ikey,
                    origin: it.position,
                    pick_id: it.settings.pick_id.0 as u32,
                    selected: it.settings.selected,
                    cast_shadows: it.settings.cast_shadows,
                    item_idx: Some(i),
                    seed: it.seed,
                });
            }
        }

        // Transient (burst) instances (Phase 25): age the live ones and retire
        // those past their lifetime (freeing pooled slots), then spawn a fresh
        // transient for each one-shot item submitted this frame.
        let mut ti = 0;
        while ti < self.transients.len() {
            self.transients[ti].age += dt;
            if self.transients[ti].age >= self.transients[ti].retire_at {
                let t = self.transients.remove(ti);
                let slot = t.ikey.key & !TRANSIENT_KEY_BIT;
                self.transient_free
                    .entry(t.ikey.effect as usize)
                    .or_default()
                    .push(slot);
            } else {
                ti += 1;
            }
        }
        let max_transients = items.max_transients.unwrap_or(DEFAULT_MAX_TRANSIENTS);
        for it in &items.items {
            if !it.one_shot || it.settings.hidden {
                continue;
            }
            let idx = it.effect.0 as usize;
            if idx >= num_effects || is_single(idx) || self.effects[idx].removed {
                continue;
            }
            let Some(slot) = self.acquire_transient_slot(idx, max_transients) else {
                continue;
            };
            let ikey = InstanceKey {
                effect: idx as u32,
                key: TRANSIENT_KEY_BIT | slot,
            };
            self.ensure_instance_gpu(idx, ikey.key, device, queue, &emit_bgl, &sim_bgl, &draw_bgl);
            let capacity = self.effects[idx].asset.capacity;
            {
                let inst = self.instances.get_mut(&ikey).unwrap();
                zero_particle_buffer(queue, inst.gpu.particle_buf(), capacity);
                inst.spawn_accum = 0.0;
                inst.burst_done = false;
                inst.last_origin = None;
                inst.dropped = false;
                inst.sim_frame = 0;
            }
            let retire_at = transient_lifetime(&self.effects[idx].asset);
            self.transients.push(TransientInstance {
                ikey,
                origin: it.position,
                cast_shadows: it.settings.cast_shadows,
                age: 0.0,
                retire_at,
                seed: it.seed,
            });
        }
        for t in &self.transients {
            driven.push(DrivenInstance {
                ikey: t.ikey,
                origin: t.origin,
                pick_id: 0,
                selected: false,
                cast_shadows: t.cast_shadows,
                item_idx: None,
                seed: t.seed,
            });
        }

        // Culling + simulation budget (Phase 26). Distance/frustum-cull instances
        // (sub-emitter effects exempt), dropping culled instances' particles; then
        // keep the nearest `max_sim_instances`.
        let eye = glam::Vec3::from(ctx.camera.eye_position);
        let frustum = if items.frustum_cull {
            Some(Frustum::from_view_proj(&ctx.camera.view_proj()))
        } else {
            None
        };
        let cull_dist_sq = items.cull_distance.map(|d| d * d);
        if cull_dist_sq.is_some() || frustum.is_some() {
            let mut kept = Vec::with_capacity(driven.len());
            for di in driven.drain(..) {
                let idx = di.ikey.effect as usize;
                if is_single(idx) {
                    kept.push(di);
                    continue;
                }
                let pos = glam::Vec3::from(di.origin);
                let too_far = cull_dist_sq.is_some_and(|c| (pos - eye).length_squared() > c);
                let off_screen = frustum
                    .as_ref()
                    .is_some_and(|f| f.cull_sphere(pos, effect_cull_radius(&self.effects[idx].asset)));
                if too_far || off_screen {
                    // Drop this instance's particles once per cull episode.
                    if let Some(inst) = self.instances.get_mut(&di.ikey) {
                        if !inst.dropped {
                            let cap = self.effects[idx].asset.capacity;
                            zero_particle_buffer(queue, inst.gpu.particle_buf(), cap);
                            inst.dropped = true;
                            inst.spawn_accum = 0.0;
                            inst.burst_done = false;
                            inst.last_origin = None;
                        }
                    }
                } else {
                    kept.push(di);
                }
            }
            driven = kept;
        }
        if let Some(max) = items.max_sim_instances {
            if driven.len() > max {
                // Sub-emitter instances first, then nearest-first; keep `max`.
                driven.sort_by(|a, b| {
                    let sa = is_single(a.ikey.effect as usize);
                    let sb = is_single(b.ikey.effect as usize);
                    let da = (glam::Vec3::from(a.origin) - eye).length_squared();
                    let db = (glam::Vec3::from(b.origin) - eye).length_squared();
                    sb.cmp(&sa)
                        .then(da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal))
                });
                driven.truncate(max);
            }
        }

        // Order parents-before-children so a parent's simulate (which appends
        // spawn events) is encoded before its child's emit within one command
        // buffer. Bounded walk; a misconfigured cycle cannot loop forever.
        let ancestor_depth = |mut idx: usize| -> usize {
            let mut d = 0;
            let mut guard = 0;
            while let Some(&p) = parent_of.get(&idx) {
                d += 1;
                idx = p;
                guard += 1;
                if guard > num_effects {
                    break;
                }
            }
            d
        };
        driven.sort_by_key(|di| ancestor_depth(di.ikey.effect as usize));

        // Reset every existing event buffer's append count to zero. Queue writes
        // apply before this frame's command buffer runs, so parents append from a
        // clean count and children consume only this frame's events. Buffers built
        // later this frame start zeroed.
        for se in self.subemitters.values() {
            queue.write_buffer(&se.event_buf, 0, bytemuck::bytes_of(&0u32));
        }

        let mut encoder = device.create_command_encoder(&vwgpu::CommandEncoderDescriptor {
            label: Some("particle_sim_encoder"),
        });
        let mut any = false;

        for &di in &driven {
            let ikey = di.ikey;
            let idx = ikey.effect as usize;
            // Steady instances read per-frame overrides from their item; transient
            // (burst) instances have none and use the registered asset.
            let item = di.item_idx.map(|i| &items.items[i]);
            let origin = di.origin;
            let pick_id = di.pick_id;
            let selected = di.selected;
            let cast_shadows = di.cast_shadows;

            // Build this instance's GPU state (fixed or generated) if new.
            self.ensure_instance_gpu(idx, ikey.key, device, queue, &emit_bgl, &sim_bgl, &draw_bgl);

            // If this effect is a parent, wire its sub-emitter over the key-0
            // instances (event buffer + parent-sim and child-emit bind groups),
            // building the child's instance too. Done here so the child, processed
            // later, finds it ready.
            if self.effects[idx].asset.sub_emitter.is_some() {
                self.ensure_subemitter_gpu(idx, device, queue, &emit_bgl, &sim_bgl, &draw_bgl);
            }

            // Local-space translation delta: how far this instance moved since
            // last frame. Under `Local`, this rigidly carries every live particle
            // with the emitter; under `Global` it stays zero.
            let space_delta = match self.effects[idx].asset.sim_space {
                SimulationSpace::Local => {
                    let last = self.instances.get(&ikey).unwrap().last_origin.unwrap_or(origin);
                    [origin[0] - last[0], origin[1] - last[1], origin[2] - last[2]]
                }
                SimulationSpace::Global => [0.0; 3],
            };

            // Per-frame gradient override: re-bake the ramp LUT and swap the draw
            // group-2 bind group. Cheap (a 64-texel upload) so it runs whenever an
            // override is present, no diffing.
            if let (Some(g), Some(ramp_bgl), Some(ramp_sampler)) = (
                item.and_then(|it| it.gradient.as_ref()),
                ramp_bgl.as_ref(),
                ramp_sampler.as_ref(),
            ) {
                let lut = build_ramp_lut(device, queue, g);
                let bg = make_ramp_bg(device, ramp_bgl, ramp_sampler, &lut);
                self.instances.get_mut(&ikey).unwrap().gpu.set_ramp(bg, lut);
            }

            // Effective spawn rate: a program takes its rate from the program; a
            // fixed effect from the per-frame emitter override, else its emitter.
            let rate = match &self.effects[idx].asset.program {
                Some(p) => p.rate,
                None => item
                    .and_then(|it| it.emitter)
                    .map(|e| e.rate)
                    .unwrap_or(self.effects[idx].asset.emitter.rate),
            };
            let capacity = self.effects[idx].asset.capacity;
            let groups = capacity.div_ceil(64).max(1);

            // Advance this instance's own spawn state and record its origin (for
            // the next frame's local-space delta). Simulating clears the culled
            // "dropped" flag so a future cull drops its particles again, and bumps
            // the per-instance sim-frame counter used by the deterministic seed.
            let (spawn_count, sim_frame) = {
                let inst = self.instances.get_mut(&ikey).unwrap();
                inst.last_origin = Some(origin);
                inst.dropped = false;
                let f = inst.sim_frame;
                inst.sim_frame = inst.sim_frame.wrapping_add(1);
                (
                    next_spawn_count(&mut inst.spawn_accum, &mut inst.burst_done, rate, dt),
                    f,
                )
            };
            // Emit RNG seed: deterministic from the host seed (Phase 28) when set,
            // else derived from the global frame counter.
            let rng_seed = instance_rng_seed(di.seed, ctx.frame_index, idx, ikey.key, sim_frame);

            // Publish this instance's pick id for the pick pass.
            {
                let inst = self.instances.get(&ikey).unwrap();
                let pu = PickUniform {
                    id: pick_id,
                    _pad: [0; 3],
                };
                queue.write_buffer(&inst.pick_buf, 0, bytemuck::bytes_of(&pu));
            }

            // Sub-emitter links (cloned out so the borrows end before the buffer
            // reads below). `child_link` present when this effect consumes a
            // parent's events; `parent_link` when it produces events for a child.
            // Only key-0 instances participate (sub-emitter effects are single).
            let child_link = parent_of
                .get(&idx)
                .and_then(|p| self.subemitters.get(p))
                .map(|se| (se.child_emit_bg.clone(), se.inherit_velocity));
            let parent_link = self.subemitters.get(&idx).map(|se| {
                (
                    se.parent_sim_bg.clone(),
                    se.condition,
                    se.count,
                    se.child_capacity,
                )
            });

            match &self.instances.get(&ikey).unwrap().gpu {
                EffectGpu::Fixed(g) => {
                    let asset = &self.effects[idx].asset;
                    // Per-frame emitter / force overrides fall back to the asset.
                    let emitter = item
                        .and_then(|it| it.emitter.as_ref())
                        .unwrap_or(&asset.emitter);
                    let forces = item
                        .and_then(|it| it.forces.as_deref())
                        .unwrap_or(&asset.forces);
                    let inherit = child_link.as_ref().map(|(_, s)| *s).unwrap_or(0.0);
                    let emit = build_emit_params(
                        emitter,
                        origin,
                        capacity,
                        spawn_count,
                        rng_seed,
                        child_link.is_some(),
                        inherit,
                    );
                    // Event control for a parent: condition, count, child cap, active.
                    let event = match &parent_link {
                        Some((_, condition, count, child_cap)) => [
                            match condition {
                                EventCondition::EveryStep => 0,
                                EventCondition::OnDeath => 1,
                            },
                            *count,
                            *child_cap,
                            1,
                        ],
                        None => [0; 4],
                    };
                    let sim = build_sim_params(forces, dt, time, event, space_delta);
                    queue.write_buffer(&g.emit_buf, 0, bytemuck::bytes_of(&emit));
                    queue.write_buffer(&g.sim_buf, 0, bytemuck::bytes_of(&sim));
                    queue.write_buffer(&g.budget_buf, 0, bytemuck::bytes_of(&0u32));
                    // Pick the emit / sim pipelines and bind groups: sub-emit when
                    // consuming events, parent-sim when producing them.
                    let (emit_pipe, emit_bg) = match (child_link.as_ref(), subemit_pipeline.as_ref())
                    {
                        (Some((bg, _)), Some(pipe)) => (pipe, bg),
                        _ => (&emit_pipeline, &g.emit_bg),
                    };
                    let (sim_pipe, sim_bg) = match (parent_link.as_ref(), sim_parent_pipeline.as_ref())
                    {
                        (Some((bg, ..)), Some(pipe)) => (pipe, bg),
                        _ => (&sim_pipeline, &g.sim_bg),
                    };
                    dispatch_sim(&mut encoder, emit_pipe, emit_bg, sim_pipe, sim_bg, groups);
                }
                EffectGpu::Gen(g) => {
                    let gp = GenParams {
                        origin: [origin[0], origin[1], origin[2], 0.0],
                        space: [space_delta[0], space_delta[1], space_delta[2], 0.0],
                        dt,
                        time,
                        spawn_count,
                        capacity,
                        rng_seed,
                        _pad: [0; 3],
                    };
                    queue.write_buffer(&g.params_buf, 0, bytemuck::bytes_of(&gp));
                    // Property uniform: declared defaults overlaid with this
                    // frame's overrides, in declaration order.
                    let decls: &[PropertyDecl] = self.effects[idx]
                        .asset
                        .program
                        .as_ref()
                        .map(|p| p.properties.as_slice())
                        .unwrap_or(&[]);
                    let props = build_props_data(
                        decls,
                        item.map(|it| it.properties.as_slice()).unwrap_or(&[]),
                    );
                    queue.write_buffer(&g.props_buf, 0, bytemuck::cast_slice(&props));
                    queue.write_buffer(&g.budget_buf, 0, bytemuck::bytes_of(&0u32));
                    dispatch_sim(
                        &mut encoder,
                        &g.emit_pipeline,
                        &g.emit_bg,
                        &g.sim_pipeline,
                        &g.sim_bg,
                        groups,
                    );
                }
            }

            // Trail effects record this frame's positions into the history ring,
            // after the simulate pass has moved the particles.
            if let ParticleRender::Trail { .. } = self.effects[idx].asset.render {
                if let (Some(append_pipeline), Some(trail)) = (
                    append_pipeline.as_ref(),
                    self.instances.get(&ikey).and_then(|i| i.trail.as_ref()),
                ) {
                    let head = (ctx.frame_index as u32) % HISTORY_LEN;
                    let hp = HistoryParams {
                        head,
                        capacity,
                        history_len: HISTORY_LEN,
                        _pad: 0,
                    };
                    queue.write_buffer(&trail.params_buf, 0, bytemuck::bytes_of(&hp));
                    let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
                        label: Some("particle_history_pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(append_pipeline);
                    pass.set_bind_group(0, &trail.append_bg, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
            }

            // Non-additive effects sort their draw order back-to-front, after
            // the simulate pass has set final positions. Skipped when the frame
            // disables sorting, leaving the identity order (to show the artefact).
            if items.sort_transparent {
                if let (Some(init_pipeline), Some(stage_pipeline), Some(sort)) = (
                    sort_init_pipeline.as_ref(),
                    sort_stage_pipeline.as_ref(),
                    self.instances.get(&ikey).and_then(|i| i.sort.as_ref()),
                ) {
                    let eye = ctx.camera.eye_position;
                    let ip = SortInitParams {
                        eye: [eye[0], eye[1], eye[2], 0.0],
                        capacity,
                        n: sort.n,
                        _pad: [0; 2],
                    };
                    queue.write_buffer(&sort.init_buf, 0, bytemuck::bytes_of(&ip));
                    dispatch_sort(&mut encoder, init_pipeline, stage_pipeline, sort);
                }
            }

            self.draw_list.push(DrawEntry {
                key: ikey,
                pick_id,
                selected,
                cast_shadows,
            });
            any = true;
        }

        if any {
            vec![encoder.finish()]
        } else {
            Vec::new()
        }
    }

    fn paint<'a>(
        &'a self,
        pass: &mut vwgpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
        let (Some(bb_add), Some(mesh_add), Some(mesh_over), Some(trail_add), Some(trail_over)) = (
            self.draw_pipeline_additive.as_ref(),
            self.mesh_pipeline_additive.as_ref(),
            self.mesh_pipeline_over.as_ref(),
            self.trail_pipeline_additive.as_ref(),
            self.trail_pipeline_over.as_ref(),
        ) else {
            return;
        };
        // Group 0 (camera + scene) is already bound by the lib.
        for entry in &self.draw_list {
            let eff = &self.effects[entry.key.effect as usize];
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            let gpu = &inst.gpu;
            let additive = matches!(eff.asset.blend, ParticleBlend::Additive);
            // Groups 2 (ramp) and 3 (draw params) are shared by every route;
            // group 1 is route-specific (particles, or the trail bindings).
            pass.set_bind_group(2, gpu.ramp_bg(), &[]);
            pass.set_bind_group(3, gpu.drawparams_bg(), &[]);

            match eff.asset.render {
                ParticleRender::Billboard { .. } => {
                    // Alpha/Premultiplied billboards route through the OIT pass
                    // (paint_transparent). Only additive draws here.
                    if !additive {
                        continue;
                    }
                    pass.set_pipeline(bb_add);
                    pass.set_bind_group(1, gpu.draw_bg(), &[]);
                    pass.draw(0..6, 0..eff.asset.capacity);
                }
                ParticleRender::Mesh { mesh, .. } => {
                    let Some(m) = self.meshes.get(mesh.0 as usize) else {
                        continue;
                    };
                    pass.set_pipeline(if additive { mesh_add } else { mesh_over });
                    pass.set_bind_group(1, gpu.draw_bg(), &[]);
                    pass.set_vertex_buffer(0, m.vertex_buf.slice(..));
                    pass.set_index_buffer(m.index_buf.slice(..), vwgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..m.index_count, 0, 0..eff.asset.capacity);
                }
                ParticleRender::Trail { segments, .. } => {
                    let Some(trail) = inst.trail.as_ref() else {
                        continue;
                    };
                    pass.set_pipeline(if additive { trail_add } else { trail_over });
                    pass.set_bind_group(1, &trail.draw_bg, &[]);
                    // Six vertices per segment; the shader clamps to the live
                    // history and discards stale or dead segments.
                    let segs = segments.clamp(1, HISTORY_LEN - 1);
                    pass.draw(0..segs * 6, 0..eff.asset.capacity);
                }
            }
        }
    }

    fn paint_transparent<'a>(
        &'a self,
        pass: &mut vwgpu::RenderPass<'a>,
        _ctx: &PaintContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
        let Some(oit) = self.draw_pipeline_oit.as_ref() else {
            return;
        };
        // Group 0 (camera + scene) is bound by the lib inside the OIT pass.
        // Draw only Alpha/Premultiplied billboards; additive billboards and the
        // mesh/trail routes stay in the main HDR draw (paint).
        pass.set_pipeline(oit);
        for entry in &self.draw_list {
            let eff = &self.effects[entry.key.effect as usize];
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            let gpu = &inst.gpu;
            if matches!(eff.asset.blend, ParticleBlend::Additive) {
                continue;
            }
            if !matches!(eff.asset.render, ParticleRender::Billboard { .. }) {
                continue;
            }
            pass.set_bind_group(1, gpu.draw_bg(), &[]);
            pass.set_bind_group(2, gpu.ramp_bg(), &[]);
            pass.set_bind_group(3, gpu.drawparams_bg(), &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }

    fn render_pick<'a>(
        &'a self,
        pass: &mut vwgpu::RenderPass<'a>,
        _ctx: &PickPassContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
        let Some(pipeline) = self.pick_pipeline.as_ref() else {
            return;
        };
        // Group 0 (camera) is bound by the lib. Draw each pickable effect's live
        // particles as billboards writing the driving item's pick id.
        for entry in &self.draw_list {
            if entry.pick_id == 0 {
                continue;
            }
            let eff = &self.effects[entry.key.effect as usize];
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
            pass.set_bind_group(2, &inst.pick_bg, &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }

    fn outline_mask<'a>(
        &'a self,
        pass: &mut vwgpu::RenderPass<'a>,
        _ctx: &OutlineMaskContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
        let Some(pipeline) = self.mask_pipeline.as_ref() else {
            return;
        };
        // Group 0 (camera) is bound by the lib. Draw selected effects' particles
        // into the R8 mask; the lib's edge pass turns coverage into an outline.
        for entry in &self.draw_list {
            if !entry.selected {
                continue;
            }
            let eff = &self.effects[entry.key.effect as usize];
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }

    fn cast_shadow_pass<'a>(
        &'a self,
        pass: &mut vwgpu::RenderPass<'a>,
        _ctx: &ShadowCastContext<'a>,
        _items: &'a dyn PluginItemCollection,
    ) {
        let Some(pipeline) = self.shadow_pipeline.as_ref() else {
            return;
        };
        // Group 0 (the cascade light matrix), viewport, and scissor are set by
        // the lib. Draw each shadow-casting effect's particles as depth-only
        // world-axis quads into the cascade tile.
        for entry in &self.draw_list {
            if !entry.cast_shadows {
                continue;
            }
            let eff = &self.effects[entry.key.effect as usize];
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }
}

/// A hash of the parts of an [`EffectAsset`] that are baked into GPU state at
/// [`ensure_instance_gpu`](ParticlePlugin::ensure_instance_gpu) time — capacity,
/// blend, render route, texture, flipbook, gradient, sub-emitter presence, and
/// the generated program (its WGSL). Two assets with the same signature differ
/// only in per-frame-consumed fields (emitter / forces / lifetime / sim-space),
/// so `replace_effect` can keep the live particle set when they match.
fn structural_sig(asset: &EffectAsset) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    asset.capacity.hash(&mut h);
    format!("{:?}", asset.blend).hash(&mut h);
    format!("{:?}", asset.render).hash(&mut h);
    asset.texture.map(|t| t.0).hash(&mut h);
    format!("{:?}", asset.texture_mode).hash(&mut h);
    format!("{:?}", asset.flipbook).hash(&mut h);
    format!("{:?}", asset.gradient).hash(&mut h);
    asset.sub_emitter.map(|s| s.child.0).hash(&mut h);
    match &asset.program {
        None => 0u8.hash(&mut h),
        Some(p) => {
            1u8.hash(&mut h);
            let shaders = codegen::generate_program(p);
            shaders.emit.hash(&mut h);
            shaders.sim.hash(&mut h);
        }
    }
    h.finish()
}

/// SplitMix64 finalizer, for deriving a well-mixed 64-bit value from a seed.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// The per-frame emit RNG seed for one instance. With a host `seed` (Phase 28) it
/// is a fixed function of the seed, the `(effect, key)`, and the instance's
/// sim-frame counter — reproducible run to run for a fixed seed + `dt` sequence
/// (GPU atomic spawn order still limits exact per-particle reproducibility).
/// Without one it derives from the global frame counter (the pre-Phase-28
/// behaviour, not reproducible across runs).
fn instance_rng_seed(
    seed: Option<u64>,
    frame_index: u64,
    effect: usize,
    key: u64,
    sim_frame: u32,
) -> u32 {
    match seed {
        Some(s) => splitmix64(
            s ^ (effect as u64).wrapping_mul(0x9E3779B97F4A7C15)
                ^ key.wrapping_mul(0xD1B54A32D192ED03)
                ^ (sim_frame as u64).wrapping_mul(0x2545F4914F6CDD1D),
        ) as u32,
        None => (frame_index as u32)
            .wrapping_mul(2654435761)
            .wrapping_add(effect as u32)
            .wrapping_add((key as u32).wrapping_mul(40503)),
    }
}

/// Overwrite a particle buffer with zeros, killing every slot (used to drop a
/// culled instance's particles so it resumes fresh when back in range).
fn zero_particle_buffer(queue: &vwgpu::Queue, buf: &vwgpu::Buffer, capacity: u32) {
    let bytes = (capacity as usize) * std::mem::size_of::<GpuParticle>();
    queue.write_buffer(buf, 0, &vec![0u8; bytes]);
}

/// The age at which a transient (burst) instance of `asset` is retired: the
/// effect's maximum particle lifetime plus a small margin, so all burst
/// particles have died.
fn transient_lifetime(asset: &EffectAsset) -> f32 {
    let life = match &asset.program {
        Some(p) => p.lifetime.1,
        None => asset.emitter.lifetime.1,
    };
    life.max(0.05) + 0.1
}

/// A conservative bounding radius for an effect's live particles about its
/// emitter origin, for frustum culling. Estimated from how far a particle can
/// travel (max speed x lifetime) plus the spawn extent and particle size — the
/// codegen path is unknown, so it gets a generous default.
fn effect_cull_radius(asset: &EffectAsset) -> f32 {
    if asset.program.is_some() {
        return 20.0;
    }
    let e = &asset.emitter;
    let life = e.lifetime.1.max(0.0);
    let speed = match e.velocity {
        VelocityDist::Fixed(v) => (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt(),
        VelocityDist::UniformBox { min, max } => {
            let m = |a: f32, b: f32| a.abs().max(b.abs());
            (m(min[0], max[0]).powi(2) + m(min[1], max[1]).powi(2) + m(min[2], max[2]).powi(2))
                .sqrt()
        }
        VelocityDist::UniformCone { max_speed, .. } => max_speed,
    };
    let spawn_extent = match e.spawn {
        SpawnShape::Point => 0.0,
        SpawnShape::Box { min, max } => {
            let d = |a: f32, b: f32| (a - b).abs();
            0.5 * (d(min[0], max[0]).powi(2) + d(min[1], max[1]).powi(2) + d(min[2], max[2]).powi(2))
                .sqrt()
        }
        SpawnShape::Sphere { radius, .. } | SpawnShape::Circle { radius, .. } => radius,
        SpawnShape::Cone {
            base_radius,
            top_radius,
            height,
            ..
        } => base_radius.max(top_radius).max(height),
    };
    speed * life + spawn_extent + e.size + 0.5
}

/// Allocate a zeroed persistent particle buffer of `capacity` slots.
fn build_particle_buffer(device: &vwgpu::Device, capacity: u32) -> vwgpu::Buffer {
    let bytes = (capacity as u64) * std::mem::size_of::<GpuParticle>() as u64;
    let buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_buffer"),
        size: bytes.max(std::mem::size_of::<GpuParticle>() as u64),
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Zero the buffer so all slots are dead.
    buf.slice(..).get_mapped_range_mut().fill(0);
    buf.unmap();
    buf
}

/// One-`u32` atomic spawn-budget buffer.
fn build_budget_buffer(device: &vwgpu::Device) -> vwgpu::Buffer {
    device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_spawn_budget"),
        size: std::mem::size_of::<u32>() as u64,
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Encode this frame's emit and simulate compute passes for one effect.
fn dispatch_sim(
    encoder: &mut vwgpu::CommandEncoder,
    emit_pipeline: &vwgpu::ComputePipeline,
    emit_bg: &vwgpu::BindGroup,
    sim_pipeline: &vwgpu::ComputePipeline,
    sim_bg: &vwgpu::BindGroup,
    groups: u32,
) {
    {
        let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
            label: Some("particle_emit_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(emit_pipeline);
        pass.set_bind_group(0, emit_bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
            label: Some("particle_sim_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(sim_pipeline);
        pass.set_bind_group(0, sim_bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// Build a compute pipeline from WGSL source against a single bind group layout.
fn build_compute_pipeline(
    device: &vwgpu::Device,
    src: &str,
    entry: &str,
    label: &str,
    bgl: &vwgpu::BindGroupLayout,
) -> vwgpu::ComputePipeline {
    let module = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: vwgpu::ShaderSource::Wgsl(src.into()),
    });
    let layout = vwgpu::pipeline_layout(device, label, &[bgl]);
    device.create_compute_pipeline(&vwgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create a buffer initialized with `data` via `mapped_at_creation` (no queue).
fn buffer_with_data(
    device: &vwgpu::Device,
    label: &str,
    data: &[u8],
    usage: vwgpu::BufferUsages,
) -> vwgpu::Buffer {
    let size = (data.len() as u64).max(4);
    let buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut()[..data.len()].copy_from_slice(data);
    buf.unmap();
    buf
}

/// Bind group (group 3): the `DrawParams` uniform (binding 0) plus the effect's
/// billboard texture (binding 1) and sampler (binding 2).
fn build_drawparams_bg(
    device: &vwgpu::Device,
    drawparams_bgl: &vwgpu::BindGroupLayout,
    params: DrawParams,
    texture_view: &vwgpu::TextureView,
    sampler: &vwgpu::Sampler,
) -> vwgpu::BindGroup {
    let buf = buffer_with_data(
        device,
        "particle_drawparams",
        bytemuck::bytes_of(&params),
        vwgpu::BufferUsages::UNIFORM,
    );
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_drawparams_bg"),
        layout: drawparams_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: vwgpu::BindingResource::TextureView(texture_view),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: vwgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Draw bind group binding the particle buffer (group 1, binding 0) and the draw
/// order buffer (binding 1).
fn build_draw_bg(
    device: &vwgpu::Device,
    draw_bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    order_buf: &vwgpu::Buffer,
) -> vwgpu::BindGroup {
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_draw_bg"),
        layout: draw_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: order_buf.as_entire_binding(),
            },
        ],
    })
}

/// Allocate a draw-order index buffer of `n` slots, initialised to identity
/// (`order[i] = i`). Unsorted effects keep this identity; sorted effects
/// overwrite it each frame in the sort setup pass.
fn build_order_buffer(device: &vwgpu::Device, n: u32) -> vwgpu::Buffer {
    let indices: Vec<u32> = (0..n).collect();
    buffer_with_data(
        device,
        "particle_order",
        bytemuck::cast_slice(&indices),
        vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
    )
}

/// Smallest power of two >= `v` (and >= 1).
fn next_pow2(v: u32) -> u32 {
    let mut n = 1u32;
    while n < v {
        n <<= 1;
    }
    n
}

/// Texels in a lifetime-ramp LUT.
const RAMP_WIDTH: u32 = 64;

/// Encode an `f32` as an IEEE-754 half-precision `u16`. Dependency-free so the
/// crate needs no `half`; handles the ranges gradients produce (small positive
/// colours and sizes), with round-to-nearest and flush-to-zero on underflow.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exp >= 0x1f {
        // Overflow / inf / nan -> largest finite (or inf for nan-free inputs).
        return sign | 0x7bff;
    }
    if exp <= 0 {
        // Subnormal or underflow -> flush to zero (adequate for ramps).
        return sign;
    }
    sign | ((exp as u16) << 10) | ((mantissa >> 13) as u16)
}

/// Bake a gradient into a `RAMP_WIDTH` x 1 `Rgba16Float` LUT: rgb = colour,
/// a = size scale, sampled by normalized age.
fn build_ramp_lut(
    device: &vwgpu::Device,
    queue: &vwgpu::Queue,
    gradient: &crate::effect::Gradient,
) -> vwgpu::Texture {
    let mut texels: Vec<u16> = Vec::with_capacity(RAMP_WIDTH as usize * 4);
    for i in 0..RAMP_WIDTH {
        let t = i as f32 / (RAMP_WIDTH - 1) as f32;
        let c = gradient.sample_colour(t);
        let s = gradient.sample_size(t);
        texels.push(f32_to_f16(c[0]));
        texels.push(f32_to_f16(c[1]));
        texels.push(f32_to_f16(c[2]));
        texels.push(f32_to_f16(s));
    }

    let texture = device.create_texture(&vwgpu::TextureDescriptor {
        label: Some("particle_ramp_lut"),
        size: vwgpu::Extent3d {
            width: RAMP_WIDTH,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: vwgpu::TextureDimension::D2,
        format: vwgpu::TextureFormat::Rgba16Float,
        usage: vwgpu::TextureUsages::TEXTURE_BINDING | vwgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        vwgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: vwgpu::Origin3d::ZERO,
            aspect: vwgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        vwgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(RAMP_WIDTH * 4 * 2),
            rows_per_image: Some(1),
        },
        vwgpu::Extent3d {
            width: RAMP_WIDTH,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// Build a 1x1 `Rgba8UnormSrgb` texture of a single colour (the default sprite
/// texture for untextured effects).
fn build_solid_texture(
    device: &vwgpu::Device,
    queue: &vwgpu::Queue,
    rgba: [u8; 4],
) -> vwgpu::Texture {
    let texture = device.create_texture(&vwgpu::TextureDescriptor {
        label: Some("particle_default_texture"),
        size: vwgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: vwgpu::TextureDimension::D2,
        format: vwgpu::TextureFormat::Rgba8UnormSrgb,
        usage: vwgpu::TextureUsages::TEXTURE_BINDING | vwgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        vwgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: vwgpu::Origin3d::ZERO,
            aspect: vwgpu::TextureAspect::All,
        },
        &rgba,
        vwgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        vwgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// Bind group for the ramp LUT (group 2): texture view + sampler.
fn make_ramp_bg(
    device: &vwgpu::Device,
    ramp_bgl: &vwgpu::BindGroupLayout,
    sampler: &vwgpu::Sampler,
    lut: &vwgpu::Texture,
) -> vwgpu::BindGroup {
    let view = lut.create_view(&vwgpu::TextureViewDescriptor::default());
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_ramp_bg"),
        layout: ramp_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: vwgpu::BindingResource::TextureView(&view),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: vwgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Build a generated (codegen) effect's GPU state: a `GenParams` uniform, the
/// budget buffer, and the emit/sim/draw bind groups over the generated
/// pipelines.
fn build_gen_gpu(
    device: &vwgpu::Device,
    capacity: u32,
    order_buf: &vwgpu::Buffer,
    emit_bgl: &vwgpu::BindGroupLayout,
    sim_bgl: &vwgpu::BindGroupLayout,
    draw_bgl: &vwgpu::BindGroupLayout,
    ramp_bg: vwgpu::BindGroup,
    ramp_lut: Option<vwgpu::Texture>,
    drawparams_bg: vwgpu::BindGroup,
    property_count: usize,
    emit_pipeline: vwgpu::ComputePipeline,
    sim_pipeline: vwgpu::ComputePipeline,
) -> GenGpu {
    let particle_buf = build_particle_buffer(device, capacity);
    let budget_buf = build_budget_buffer(device);
    let params_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_gen_params"),
        size: std::mem::size_of::<GenParams>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // One `vec4` lane per property; at least one so the binding is never empty
    // (the generated `Properties` struct always has a lane).
    let props_lanes = property_count.max(1) as u64;
    let props_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_gen_props"),
        size: props_lanes * 16,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let emit_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_gen_emit_bg"),
        layout: emit_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: budget_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: props_buf.as_entire_binding(),
            },
        ],
    });
    let sim_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_gen_sim_bg"),
        layout: sim_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: params_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: props_buf.as_entire_binding(),
            },
        ],
    });
    let draw_bg = build_draw_bg(device, draw_bgl, &particle_buf, order_buf);

    GenGpu {
        particle_buf,
        params_buf,
        props_buf,
        budget_buf,
        emit_bg,
        sim_bg,
        draw_bg,
        ramp_bg,
        drawparams_bg,
        ramp_lut,
        emit_pipeline,
        sim_pipeline,
    }
}

/// Allocate one fixed-function effect's persistent particle buffer, its
/// parameter buffers, and the emit/sim/draw bind groups. The particle buffer
/// starts zeroed, so every slot reads as dead until the first emit pass.
fn build_fixed_gpu(
    device: &vwgpu::Device,
    capacity: u32,
    order_buf: &vwgpu::Buffer,
    emit_bgl: &vwgpu::BindGroupLayout,
    sim_bgl: &vwgpu::BindGroupLayout,
    draw_bgl: &vwgpu::BindGroupLayout,
    ramp_bg: vwgpu::BindGroup,
    ramp_lut: Option<vwgpu::Texture>,
    drawparams_bg: vwgpu::BindGroup,
) -> FixedGpu {
    let particle_buf = build_particle_buffer(device, capacity);
    let budget_buf = build_budget_buffer(device);

    let emit_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_emit_params"),
        size: std::mem::size_of::<EmitParamsGpu>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sim_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_sim_params"),
        size: std::mem::size_of::<SimParamsGpu>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let emit_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_emit_bg"),
        layout: emit_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: emit_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: budget_buf.as_entire_binding(),
            },
        ],
    });
    let sim_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_sim_bg"),
        layout: sim_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: sim_buf.as_entire_binding(),
            },
        ],
    });
    let draw_bg = build_draw_bg(device, draw_bgl, &particle_buf, order_buf);

    FixedGpu {
        particle_buf,
        emit_buf,
        sim_buf,
        budget_buf,
        emit_bg,
        sim_bg,
        draw_bg,
        ramp_bg,
        drawparams_bg,
        ramp_lut,
    }
}

/// Build a trail effect's history ring, its params uniform, and the append and
/// ribbon bind groups over the effect's persistent particle buffer. The history
/// starts zeroed, so every sample reads as invalid (`w = 0` never matches a live
/// seed) until the append pass fills it in over the first frames.
fn build_trail_gpu(
    device: &vwgpu::Device,
    capacity: u32,
    particle_buf: &vwgpu::Buffer,
    order_buf: &vwgpu::Buffer,
    history_bgl: &vwgpu::BindGroupLayout,
    trail_draw_bgl: &vwgpu::BindGroupLayout,
) -> TrailGpu {
    let sample_count = (capacity as u64) * HISTORY_LEN as u64;
    let bytes = (sample_count * 16).max(16);
    let history_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_trail_history"),
        size: bytes,
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    history_buf.slice(..).get_mapped_range_mut().fill(0);
    history_buf.unmap();

    let params_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_trail_params"),
        size: std::mem::size_of::<HistoryParams>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let append_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_trail_append_bg"),
        layout: history_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: history_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let draw_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_trail_draw_bg"),
        layout: trail_draw_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: history_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: order_buf.as_entire_binding(),
            },
        ],
    });

    TrailGpu {
        history_buf,
        params_buf,
        append_bg,
        draw_bg,
    }
}

/// Allocate a zeroed spawn-event buffer sized to the child's capacity: a 16-byte
/// header (the atomic append count) followed by `child_capacity` `SpawnEvent`s.
fn build_event_buffer(device: &vwgpu::Device, child_capacity: u32) -> vwgpu::Buffer {
    let bytes = EVENT_HEADER_BYTES
        + (child_capacity.max(1) as u64) * std::mem::size_of::<SpawnEventGpu>() as u64;
    let buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_spawn_events"),
        size: bytes,
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    buf.slice(..).get_mapped_range_mut().fill(0);
    buf.unmap();
    buf
}

/// Parent-sim bind group: parent particles (binding 0), sim params (1), and the
/// event buffer the parent appends to (2).
fn build_sim_parent_bg(
    device: &vwgpu::Device,
    bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    sim_buf: &vwgpu::Buffer,
    event_buf: &vwgpu::Buffer,
) -> vwgpu::BindGroup {
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_sim_parent_bg"),
        layout: bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: sim_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: event_buf.as_entire_binding(),
            },
        ],
    })
}

/// Child sub-emit bind group: child particles (binding 0), emit params (1), the
/// spawn budget atomic (2), and the event buffer the child consumes (3).
fn build_subemit_bg(
    device: &vwgpu::Device,
    bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    emit_buf: &vwgpu::Buffer,
    budget_buf: &vwgpu::Buffer,
    event_buf: &vwgpu::Buffer,
) -> vwgpu::BindGroup {
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_subemit_bg"),
        layout: bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: emit_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: budget_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: event_buf.as_entire_binding(),
            },
        ],
    })
}

/// Build a non-additive effect's depth-sort state: the key buffer, the per-frame
/// init uniform, the fixed bitonic stage constants, and the init/stage bind
/// groups over the effect's particle and order buffers.
fn build_sort_gpu(
    device: &vwgpu::Device,
    n: u32,
    particle_buf: &vwgpu::Buffer,
    order_buf: &vwgpu::Buffer,
    sort_init_bgl: &vwgpu::BindGroupLayout,
    sort_stage_bgl: &vwgpu::BindGroupLayout,
) -> SortGpu {
    let keys_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_sort_keys"),
        size: (n as u64) * 4,
        usage: vwgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let init_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_sort_init_params"),
        size: std::mem::size_of::<SortInitParams>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Precompute the bitonic (k, j) stages for this padded length. They depend
    // only on `n`, so they are baked into a dynamic-offset uniform once.
    let mut stages: Vec<SortStageParams> = Vec::new();
    let mut k = 2u32;
    while k <= n {
        let mut j = k >> 1;
        while j > 0 {
            stages.push(SortStageParams { k, j, n, _pad: 0 });
            j >>= 1;
        }
        k <<= 1;
    }
    let num_stages = stages.len() as u32;
    let mut stage_bytes = vec![0u8; num_stages as usize * SORT_STAGE_STRIDE as usize];
    for (s, params) in stages.iter().enumerate() {
        let off = s * SORT_STAGE_STRIDE as usize;
        stage_bytes[off..off + std::mem::size_of::<SortStageParams>()]
            .copy_from_slice(bytemuck::bytes_of(params));
    }
    let stage_buf = buffer_with_data(
        device,
        "particle_sort_stages",
        &stage_bytes,
        vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
    );

    let init_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_sort_init_bg"),
        layout: sort_init_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: keys_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: order_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: init_buf.as_entire_binding(),
            },
        ],
    });
    // The stage uniform is bound with a dynamic offset; each dispatch supplies
    // its stage's offset, so bind just one struct's worth here.
    let stage_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_sort_stage_bg"),
        layout: sort_stage_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: keys_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: order_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: vwgpu::BindingResource::Buffer(vwgpu::BufferBinding {
                    buffer: &stage_buf,
                    offset: 0,
                    size: std::num::NonZeroU64::new(std::mem::size_of::<SortStageParams>() as u64),
                }),
            },
        ],
    });

    SortGpu {
        keys_buf,
        init_buf,
        stage_buf,
        init_bg,
        stage_bg,
        n,
        num_stages,
    }
}

/// Encode the depth sort: the setup pass, then one compute pass per bitonic
/// stage (separate passes so each stage sees the previous stage's writes).
fn dispatch_sort(
    encoder: &mut vwgpu::CommandEncoder,
    init_pipeline: &vwgpu::ComputePipeline,
    stage_pipeline: &vwgpu::ComputePipeline,
    sort: &SortGpu,
) {
    let groups = sort.n.div_ceil(64).max(1);
    {
        let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
            label: Some("particle_sort_init_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(init_pipeline);
        pass.set_bind_group(0, &sort.init_bg, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    for s in 0..sort.num_stages {
        let offset = s * SORT_STAGE_STRIDE as u32;
        let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
            label: Some("particle_sort_stage_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(stage_pipeline);
        pass.set_bind_group(0, &sort.stage_bg, &[offset]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// How many particles to spawn this frame, advancing the effect's accumulator.
/// The caller resolves the effective `rate` (program, per-frame emitter
/// override, or the registered emitter).
fn next_spawn_count(spawn_accum: &mut f32, burst_done: &mut bool, rate: SpawnRate, dt: f32) -> u32 {
    match rate {
        SpawnRate::PerSecond(rate) => {
            *spawn_accum += rate.max(0.0) * dt;
            let count = spawn_accum.floor();
            *spawn_accum -= count;
            count as u32
        }
        SpawnRate::Burst { count } => {
            if *burst_done {
                0
            } else {
                *burst_done = true;
                count
            }
        }
    }
}

/// Pack a program's property values for the GPU: one `vec4` lane per declared
/// property in declaration order, each the item's override for that name or the
/// declared default. Effects with no properties get a single zero lane (the
/// generated `Properties` struct always declares one).
fn build_props_data(
    decls: &[PropertyDecl],
    overrides: &[(&'static str, PropertyValue)],
) -> Vec<[f32; 4]> {
    if decls.is_empty() {
        return vec![[0.0; 4]];
    }
    decls
        .iter()
        .map(|d| {
            overrides
                .iter()
                .find(|(name, _)| *name == d.name.as_str())
                .map(|(_, v)| *v)
                .unwrap_or(d.default)
                .to_vec4()
        })
        .collect()
}

fn build_emit_params(
    emitter: &Emitter,
    origin: [f32; 3],
    capacity: u32,
    spawn_count: u32,
    rng_seed: u32,
    has_events: bool,
    inherit_velocity: f32,
) -> EmitParamsGpu {
    let mut p = EmitParamsGpu::zeroed();
    p.origin = [origin[0], origin[1], origin[2], 0.0];
    p.colour = emitter.colour;
    p.misc = [
        emitter.lifetime.0,
        emitter.lifetime.1,
        emitter.size,
        inherit_velocity,
    ];
    p.ctrl = [rng_seed, spawn_count, capacity, 0];

    let (spawn_kind, spawn_a, spawn_b) = match emitter.spawn {
        SpawnShape::Point => (0u32, [0.0; 4], [0.0; 4]),
        SpawnShape::Box { min, max } => (
            1u32,
            [min[0], min[1], min[2], 0.0],
            [max[0], max[1], max[2], 0.0],
        ),
        SpawnShape::Sphere { radius, volume } => (
            2u32,
            [radius, if volume { 1.0 } else { 0.0 }, 0.0, 0.0],
            [0.0; 4],
        ),
        SpawnShape::Circle { radius, volume } => (
            3u32,
            [radius, if volume { 1.0 } else { 0.0 }, 0.0, 0.0],
            [0.0; 4],
        ),
        SpawnShape::Cone {
            base_radius,
            top_radius,
            height,
            volume,
        } => (
            4u32,
            [
                base_radius,
                top_radius,
                height,
                if volume { 1.0 } else { 0.0 },
            ],
            [0.0; 4],
        ),
    };
    p.spawn_a = spawn_a;
    p.spawn_b = spawn_b;

    let (vel_kind, vel_a, vel_b) = match emitter.velocity {
        VelocityDist::Fixed(v) => (0u32, [v[0], v[1], v[2], 0.0], [0.0; 4]),
        VelocityDist::UniformBox { min, max } => (
            1u32,
            [min[0], min[1], min[2], 0.0],
            [max[0], max[1], max[2], 0.0],
        ),
        VelocityDist::UniformCone {
            axis,
            half_angle,
            min_speed,
            max_speed,
        } => (
            2u32,
            [axis[0], axis[1], axis[2], 0.0],
            [half_angle, min_speed, max_speed, 0.0],
        ),
    };
    p.vel_a = vel_a;
    p.vel_b = vel_b;
    p.kinds = [spawn_kind, vel_kind, has_events as u32, 0];
    p
}

fn build_sim_params(
    forces: &[ForceModifier],
    dt: f32,
    time: f32,
    event: [u32; 4],
    space: [f32; 3],
) -> SimParamsGpu {
    let mut p = SimParamsGpu::zeroed();
    p.event = event;
    p.space = [space[0], space[1], space[2], 0.0];
    let n = forces.len().min(MAX_FORCES);
    for (i, f) in forces.iter().take(n).enumerate() {
        p.forces[i] = match *f {
            ForceModifier::Accel(a) => GpuForce {
                data0: [a[0], a[1], a[2], 0.0],
                data1: [0.0; 4],
            },
            ForceModifier::Drag(c) => GpuForce {
                data0: [0.0, 0.0, 0.0, 1.0],
                data1: [0.0, 0.0, c, 0.0],
            },
            ForceModifier::PointAttractor {
                position,
                strength,
                falloff,
            } => GpuForce {
                data0: [position[0], position[1], position[2], 2.0],
                data1: [strength, falloff, 0.0, 0.0],
            },
            ForceModifier::CurlNoise {
                scale,
                strength,
                speed,
            } => GpuForce {
                data0: [0.0, 0.0, 0.0, 3.0],
                data1: [scale, strength, speed, 0.0],
            },
        };
    }
    p.misc = [dt, time, 0.0, n as f32];
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_rng_is_deterministic_and_frame_independent() {
        // Phase 28: with a host seed the emit seed is a fixed function of
        // (seed, effect, key, sim_frame) and ignores the global frame counter, so
        // it reproduces run to run.
        assert_eq!(
            instance_rng_seed(Some(42), 100, 3, 7, 5),
            instance_rng_seed(Some(42), 999, 3, 7, 5),
            "seeded path must ignore the global frame index"
        );
        // Each simulated frame decorrelates.
        assert_ne!(
            instance_rng_seed(Some(42), 0, 3, 7, 5),
            instance_rng_seed(Some(42), 0, 3, 7, 6),
        );
        // Distinct instances (key) get distinct streams.
        assert_ne!(
            instance_rng_seed(Some(42), 0, 3, 7, 5),
            instance_rng_seed(Some(42), 0, 3, 8, 5),
        );
        // A different seed gives a different stream.
        assert_ne!(
            instance_rng_seed(Some(1), 0, 3, 7, 5),
            instance_rng_seed(Some(2), 0, 3, 7, 5),
        );
        // Unseeded path varies with the frame counter (the pre-Phase-28 behaviour).
        assert_ne!(
            instance_rng_seed(None, 1, 3, 7, 5),
            instance_rng_seed(None, 2, 3, 7, 5),
        );
    }
}
