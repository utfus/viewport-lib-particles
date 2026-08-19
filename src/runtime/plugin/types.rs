use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::{Pod, Zeroable};
use viewport_lib::wgpu as vwgpu;

use crate::effect::{
    EffectAsset, EventCondition,
};

use super::*;

/// One particle on the GPU. Matches `Particle` in the WGSL shaders. 80 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuParticle {
    pub(crate) position: [f32; 3],
    pub(crate) lifetime: f32,
    pub(crate) velocity: [f32; 3],
    pub(crate) max_lifetime: f32,
    pub(crate) colour: [f32; 4],
    pub(crate) size: f32,
    pub(crate) seed: f32,
    pub(crate) _pad: [f32; 2],
}

/// Emit-pass parameters. Matches `EmitParams` in `particle_emit.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct EmitParamsGpu {
    pub(crate) origin: [f32; 4],
    pub(crate) spawn_a: [f32; 4],
    pub(crate) spawn_b: [f32; 4],
    pub(crate) vel_a: [f32; 4],
    pub(crate) vel_b: [f32; 4],
    pub(crate) colour: [f32; 4],
    pub(crate) misc: [f32; 4],
    pub(crate) ctrl: [u32; 4],
    pub(crate) kinds: [u32; 4],
}

/// One force. Matches `Force` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuForce {
    /// xyz vector (accel or attractor position), w = kind as f32.
    pub(crate) data0: [f32; 4],
    /// strength, falloff, drag, _.
    pub(crate) data1: [f32; 4],
}

/// Simulate-pass parameters. Matches `SimParams` in `particle_sim.wgsl` (and,
/// with the extra `event` lane read, `particle_sim_parent.wgsl`). The `event`
/// lane trails the forces array, so the non-parent sim shader, whose struct ends
/// at `forces`, reads the same buffer unchanged.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SimParamsGpu {
    /// dt, time, _, force_count.
    pub(crate) misc: [f32; 4],
    pub(crate) forces: [GpuForce; MAX_FORCES],
    /// Sub-emitter event control (parent effects only): condition
    /// (0 = every step, 1 = on death), events per trigger, child capacity, and
    /// an active flag (1 when this effect is a parent). All zero otherwise.
    pub(crate) event: [u32; 4],
    /// Simulation-space translation applied to every live particle this frame:
    /// `xyz` is the emitting instance's movement since last frame under
    /// [`SimulationSpace::Local`], zero under `Global`. Trails `event`, so the
    /// non-parent sim shader (whose struct now also declares these last two
    /// lanes) reads the same buffer.
    pub(crate) space: [f32; 4],
}

/// One GPU spawn event appended by a parent and consumed by its child. Matches
/// `SpawnEvent` in the sub-emitter shaders. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SpawnEventGpu {
    pub(crate) position: [f32; 4],
    pub(crate) velocity: [f32; 4],
}

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// An uploaded mesh for the mesh render route: local-space vertex positions and
/// a triangle index buffer.
pub(crate) struct ParticleMesh {
    pub(crate) vertex_buf: vwgpu::Buffer,
    pub(crate) index_buf: vwgpu::Buffer,
    pub(crate) index_count: u32,
    /// CPU copy of the local-space vertex positions, kept so `MeshSurface` /
    /// `PointCloud` spawn shapes (Phase 22) can bake an area-weighted sample pool
    /// at effect-registration time.
    pub(crate) positions: Vec<[f32; 3]>,
    /// CPU copy of the triangle indices (empty for a pure point cloud).
    pub(crate) indices: Vec<u32>,
}

/// One baked spawn sample for the `MeshSurface` / `PointCloud` shapes (Phase 22).
/// Matches `SpawnSample` in `particle_emit.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SpawnSampleGpu {
    /// Instance-relative sample position (xyz; w unused).
    pub(crate) position: [f32; 4],
    /// Surface (or radial) normal at the sample (xyz; w unused).
    pub(crate) normal: [f32; 4],
}

/// An uploaded texture for the billboard texture / flipbook routes: the texture
/// and a default view, sampled in the billboard fragment (group 3).
pub(crate) struct ParticleTexture {
    #[allow(dead_code)]
    pub(crate) texture: vwgpu::Texture,
    pub(crate) view: vwgpu::TextureView,
}

/// Per-effect draw parameters (group 3). Matches `DrawParams` in the draw and
/// mesh shaders. 48 bytes. The billboard shaders read all twelve fields; the
/// mesh and trail shaders read only the first four (their `DrawParams` struct is
/// that 16-byte prefix, valid against the larger buffer). The size stays a
/// multiple of 16 so the uniform layout is valid.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct DrawParams {
    /// Billboard stretch factor (0 = round).
    pub(crate) stretch: f32,
    /// Mesh orientation: 0 = velocity-aligned, 1 = random tumble.
    pub(crate) align: u32,
    /// Trail ribbon half-width at the head (world units).
    pub(crate) trail_width: f32,
    /// Trail segments swept (clamped to `HISTORY_LEN - 1`).
    pub(crate) trail_segments: u32,
    /// Texture modulation: 0 = none (procedural dot), 1 = modulate rgb+a,
    /// 2 = modulate rgb, 3 = alpha-from-red mask.
    pub(crate) tex_mode: u32,
    /// Flipbook atlas columns (0 = no flipbook, sample the whole texture).
    pub(crate) flip_cols: u32,
    /// Flipbook atlas rows.
    pub(crate) flip_rows: u32,
    /// Flipbook playback: 0 stretches the sheet across life; > 0 loops at fps.
    pub(crate) flip_fps: f32,
    /// Billboard orientation (Phase 15): 0 = parallel camera plane, 1 = face
    /// camera position, 2 = along velocity, 3 = fixed screen roll.
    pub(crate) orient: u32,
    /// Orientation parameter: the fixed roll angle (radians) for orient == 3.
    pub(crate) orient_param: f32,
    /// Screen-space size flag (Phase 15): 1 keeps constant apparent size.
    pub(crate) screen_size: u32,
    /// Soft-particle fade distance in world units (Phase 9); 0 = not soft. Read
    /// only by the soft (depth-read) billboard shader.
    pub(crate) soft_dist: f32,
}

/// Trail history parameters (group 1, binding 2 of the append and trail draw
/// passes). Matches `HistParams` in `particle_history.wgsl` /
/// `particle_trail.wgsl`. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct HistoryParams {
    /// Ring slot written this frame; also the newest sample when drawing.
    pub(crate) head: u32,
    pub(crate) capacity: u32,
    pub(crate) history_len: u32,
    pub(crate) _pad: u32,
}

/// Generic per-effect params for a generated (codegen) effect. Matches
/// `GenParams` in the generated WGSL. 48 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GenParams {
    pub(crate) origin: [f32; 4],
    /// Simulation-space translation for this frame (see [`SimParamsGpu::space`]).
    pub(crate) space: [f32; 4],
    pub(crate) dt: f32,
    pub(crate) time: f32,
    pub(crate) spawn_count: u32,
    pub(crate) capacity: u32,
    pub(crate) rng_seed: u32,
    pub(crate) _pad: [u32; 3],
}

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// The variant follows the effect's authoring path: fixed-function emitter, or
/// a generated program with its own compiled emit/simulate pipelines.
pub(crate) enum EffectGpu {
    Fixed(FixedGpu),
    Gen(GenGpu),
}

impl EffectGpu {
    pub(crate) fn draw_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.draw_bg,
            EffectGpu::Gen(g) => &g.draw_bg,
        }
    }

    pub(crate) fn ramp_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.ramp_bg,
            EffectGpu::Gen(g) => &g.ramp_bg,
        }
    }

    pub(crate) fn drawparams_bg(&self) -> &vwgpu::BindGroup {
        match self {
            EffectGpu::Fixed(g) => &g.drawparams_bg,
            EffectGpu::Gen(g) => &g.drawparams_bg,
        }
    }

    /// The persistent particle buffer, for building the trail history bindings.
    pub(crate) fn particle_buf(&self) -> &vwgpu::Buffer {
        match self {
            EffectGpu::Fixed(g) => &g.particle_buf,
            EffectGpu::Gen(g) => &g.particle_buf,
        }
    }

    /// Swap in a freshly baked lifetime ramp (for per-frame gradient overrides).
    pub(crate) fn set_ramp(&mut self, bg: vwgpu::BindGroup, lut: vwgpu::Texture) {
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
pub(crate) struct SortInitParams {
    pub(crate) eye: [f32; 4],
    pub(crate) capacity: u32,
    pub(crate) n: u32,
    pub(crate) _pad: [u32; 2],
}

/// One bitonic stage's constants. Matches `SortParams` in `particle_sort.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SortStageParams {
    pub(crate) k: u32,
    pub(crate) j: u32,
    pub(crate) n: u32,
    pub(crate) _pad: u32,
}

/// Depth-sort GPU state for one non-additive effect: the key buffer, the sort
/// parameter buffers, and the init/stage bind groups over the effect's particle
/// and order buffers. The bitonic stage constants are fixed for the effect's
/// padded length, so they are written once at build time.
pub(crate) struct SortGpu {
    #[allow(dead_code)]
    pub(crate) keys_buf: vwgpu::Buffer,
    pub(crate) init_buf: vwgpu::Buffer,
    #[allow(dead_code)]
    pub(crate) stage_buf: vwgpu::Buffer,
    pub(crate) init_bg: vwgpu::BindGroup,
    pub(crate) stage_bg: vwgpu::BindGroup,
    /// Padded power-of-two length (>= capacity).
    pub(crate) n: u32,
    /// Number of bitonic stages for `n`.
    pub(crate) num_stages: u32,
}

/// Trail render-route GPU state for one effect: a per-particle position history
/// ring, its parameters, and the bind groups for the append (compute) and
/// ribbon (draw) passes. Built lazily alongside the effect's `EffectGpu` when
/// the effect uses `ParticleRender::Trail`.
pub(crate) struct TrailGpu {
    /// `capacity * HISTORY_LEN` vec4 samples (xyz = position, w = seed).
    #[allow(dead_code)]
    pub(crate) history_buf: vwgpu::Buffer,
    /// `HistoryParams`; `head` is rewritten each frame.
    pub(crate) params_buf: vwgpu::Buffer,
    /// Append pass bind group (particles, history, params).
    pub(crate) append_bg: vwgpu::BindGroup,
    /// Trail draw bind group at group 1 (particles, history, params).
    pub(crate) draw_bg: vwgpu::BindGroup,
}

/// Sub-emitter GPU state for one parent effect: the shared spawn-event buffer
/// and the bind groups the parent sim (which appends events) and the child emit
/// (which consumes them) use. Built once the parent is first driven.
pub(crate) struct SubEmitterGpu {
    /// When the parent produces events (matches the asset).
    pub(crate) condition: EventCondition,
    /// Events per trigger.
    pub(crate) count: u32,
    /// Fraction of parent velocity handed to each child particle.
    pub(crate) inherit_velocity: f32,
    /// The child's capacity, the cap on events held in the buffer.
    pub(crate) child_capacity: u32,
    /// `[count: atomic<u32>, pad*3, events: array<SpawnEvent>]`, reset each frame.
    pub(crate) event_buf: vwgpu::Buffer,
    /// Parent sim bind group (parent particles + sim params + event buffer).
    pub(crate) parent_sim_bg: vwgpu::BindGroup,
    /// Child emit bind group (child particles + emit params + budget + events).
    pub(crate) child_emit_bg: vwgpu::BindGroup,
}

/// Fixed-function effect GPU state (the emitter path).
pub(crate) struct FixedGpu {
    /// Persistent particle buffer; the bind groups alias it, so it is held but
    /// not read directly.
    #[allow(dead_code)]
    pub(crate) particle_buf: vwgpu::Buffer,
    pub(crate) emit_buf: vwgpu::Buffer,
    pub(crate) sim_buf: vwgpu::Buffer,
    pub(crate) budget_buf: vwgpu::Buffer,
    pub(crate) emit_bg: vwgpu::BindGroup,
    pub(crate) sim_bg: vwgpu::BindGroup,
    pub(crate) draw_bg: vwgpu::BindGroup,
    pub(crate) ramp_bg: vwgpu::BindGroup,
    pub(crate) drawparams_bg: vwgpu::BindGroup,
    /// Effect-specific ramp LUT, held alive when the effect has a gradient.
    /// `None` when the effect uses the shared identity ramp.
    #[allow(dead_code)]
    pub(crate) ramp_lut: Option<vwgpu::Texture>,
}

/// Generated (codegen) effect GPU state: one `GenParams` buffer feeds both the
/// emit and simulate kernels, which are this effect's own compiled pipelines.
pub(crate) struct GenGpu {
    #[allow(dead_code)]
    pub(crate) particle_buf: vwgpu::Buffer,
    pub(crate) params_buf: vwgpu::Buffer,
    /// Property uniform (one `vec4` lane per declared property, min one lane).
    /// Rewritten each frame from the driving item's property overrides.
    pub(crate) props_buf: vwgpu::Buffer,
    pub(crate) budget_buf: vwgpu::Buffer,
    pub(crate) emit_bg: vwgpu::BindGroup,
    pub(crate) sim_bg: vwgpu::BindGroup,
    pub(crate) draw_bg: vwgpu::BindGroup,
    pub(crate) ramp_bg: vwgpu::BindGroup,
    pub(crate) drawparams_bg: vwgpu::BindGroup,
    #[allow(dead_code)]
    pub(crate) ramp_lut: Option<vwgpu::Texture>,
    pub(crate) emit_pipeline: vwgpu::ComputePipeline,
    pub(crate) sim_pipeline: vwgpu::ComputePipeline,
}

/// A registered effect: the authored asset. Per-instance GPU and emission state
/// lives in [`EffectInstance`], keyed by [`InstanceKey`], since one effect can
/// drive many independent instances.
pub(crate) struct RegisteredEffect {
    pub(crate) asset: EffectAsset,
    /// Set by `remove_effect` (Phase 27): the slot is tombstoned (its
    /// `EffectId` index stays stable) and no longer simulated or drawn.
    pub(crate) removed: bool,
    /// Baked `MeshSurface` / `PointCloud` spawn-sample pool (Phase 22), built at
    /// registration when the emitter's spawn shape needs it. Bound at fixed-emit
    /// binding 3; effects without a mesh spawn shape use the shared 1-entry
    /// default and never index it.
    pub(crate) spawn_samples: Option<vwgpu::Buffer>,
    /// Number of samples in `spawn_samples` (0 when it is the shared default).
    pub(crate) spawn_sample_count: u32,
}

/// Identifies one live instance: a registered effect plus the host's stable
/// per-instance key ([`ParticleItem::key`]). Sub-emitter parent/child effects
/// always use key `0` (single instance).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct InstanceKey {
    pub(crate) effect: u32,
    pub(crate) key: u64,
}

/// One live instance of an effect: its own particle buffer + pipelines' bind
/// groups ([`EffectGpu`]), optional trail / sort state over that buffer, its
/// pick-id uniform, and its CPU emission bookkeeping. Built lazily the first
/// frame the instance is driven, and persists (a hidden instance keeps its
/// particles). Independent instances of one effect each own one of these.
pub(crate) struct EffectInstance {
    pub(crate) gpu: EffectGpu,
    /// Trail history state, present only for `ParticleRender::Trail` effects.
    pub(crate) trail: Option<TrailGpu>,
    /// Soft-particle draw bind group (Phase 9): group 1 of the depth-read
    /// pipeline, combining this instance's particles + order + draw params +
    /// sprite (folded into one group so group 3 is free for the scene depth).
    /// Present only for soft billboard effects.
    pub(crate) soft_draw_bg: Option<vwgpu::BindGroup>,
    /// Depth-sort state, present only for non-additive effects.
    pub(crate) sort: Option<SortGpu>,
    /// This instance's pick-id uniform (group 2 of the pick pipeline) and its
    /// bind group, written each frame from the driving item's pick id.
    pub(crate) pick_buf: vwgpu::Buffer,
    pub(crate) pick_bg: vwgpu::BindGroup,
    /// Fractional spawn accumulator for `SpawnRate::PerSecond`.
    pub(crate) spawn_accum: f32,
    /// Whether a `SpawnRate::Burst` has already fired.
    pub(crate) burst_done: bool,
    /// This instance's origin last frame, for the local-space translation delta.
    pub(crate) last_origin: Option<[f32; 3]>,
    /// Set when a distance/frustum cull has zeroed this instance's particle
    /// buffer, so the drop happens once per cull episode (not every frame).
    pub(crate) dropped: bool,
    /// Count of frames this instance has been simulated, mixed into the emit RNG
    /// seed under deterministic seeding (Phase 28) so each frame decorrelates.
    pub(crate) sim_frame: u32,
}

/// A live transient (burst) instance (Phase 25): a pooled particle set playing a
/// one-shot effect at a fixed world point, aged each frame and recycled when its
/// particles have died. Tracked separately from steady instances because the host
/// submits it once and the plugin advances it thereafter.
pub(crate) struct TransientInstance {
    /// Its `(effect, key)` in `instances`; the key carries [`TRANSIENT_KEY_BIT`].
    pub(crate) ikey: InstanceKey,
    /// Fixed world spawn point.
    pub(crate) origin: [f32; 3],
    /// Whether it casts shadows (captured from the spawning item).
    pub(crate) cast_shadows: bool,
    /// Seconds since spawn.
    pub(crate) age: f32,
    /// Age at which the instance is retired and its slot recycled (the effect's
    /// max particle lifetime plus a margin).
    pub(crate) retire_at: f32,
    /// Deterministic seed captured from the spawning item (Phase 28).
    pub(crate) seed: Option<u64>,
}

/// The particle system plugin.
///
/// Register effects with [`add_effect`](Self::add_effect) before handing the
/// plugin to the renderer, then install it with
/// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin).
#[derive(Default)]
pub struct ParticlePlugin {
    pub(crate) effects: Vec<RegisteredEffect>,

    // Fixed-function pipelines, shared across effects. Built in `init_gpu`.
    pub(crate) emit_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) sim_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) draw_pipeline_additive: Option<vwgpu::RenderPipeline>,
    // OIT variant of the billboard draw, for Alpha/Premultiplied effects. These
    // route through the lib's OIT pass (paint_transparent) instead of the main
    // HDR draw so they composite after the skybox and blend order-independently.
    pub(crate) draw_pipeline_oit: Option<vwgpu::RenderPipeline>,
    pub(crate) mesh_pipeline_additive: Option<vwgpu::RenderPipeline>,
    pub(crate) mesh_pipeline_over: Option<vwgpu::RenderPipeline>,
    // Soft-particle (Phase 9) billboard draw for the read-only-depth pass. Its
    // group 1 folds particles+order+draw-params+sprite so group 3 carries the
    // scene depth; `depth_read_bgl` replicates the renderer's `depth_read_bgl`
    // so the context's prebuilt `scene_depth_bind_group` binds against it. Two
    // blend variants (over / additive) so soft alpha smoke and soft additive
    // flame both fade against geometry; chosen per effect by its blend.
    pub(crate) soft_pipeline_over: Option<vwgpu::RenderPipeline>,
    pub(crate) soft_pipeline_additive: Option<vwgpu::RenderPipeline>,
    pub(crate) soft_draw_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) depth_read_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) emit_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) sim_bgl: Option<vwgpu::BindGroupLayout>,
    // Sub-emitter variants: a child emit that also reads a spawn-event buffer
    // (binding 3) and a parent sim that also writes one (binding 2). Kept
    // separate so leaf effects keep the plain fixed-function pipelines.
    pub(crate) subemit_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) sim_parent_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) subemit_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) sim_parent_bgl: Option<vwgpu::BindGroupLayout>,
    /// Per-parent sub-emitter GPU state (event buffer + the parent-sim and
    /// child-emit bind groups), keyed by the parent effect index. Built lazily
    /// the first frame a parent effect is driven.
    pub(crate) subemitters: HashMap<usize, SubEmitterGpu>,
    // Codegen-path variants carry an extra property uniform (emit binding 3,
    // sim binding 2). Kept separate so the fixed-function pipelines and bind
    // groups are unchanged.
    pub(crate) gen_emit_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) gen_sim_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) draw_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) drawparams_bgl: Option<vwgpu::BindGroupLayout>,

    // Trail render route: the history-append compute pass and the ribbon draw.
    pub(crate) append_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) trail_pipeline_additive: Option<vwgpu::RenderPipeline>,
    pub(crate) trail_pipeline_over: Option<vwgpu::RenderPipeline>,
    pub(crate) history_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) trail_draw_bgl: Option<vwgpu::BindGroupLayout>,

    // Depth sort: the setup (key + identity order) pass and one bitonic stage.
    pub(crate) sort_init_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) sort_stage_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) sort_init_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) sort_stage_bgl: Option<vwgpu::BindGroupLayout>,

    // Interaction: pick-id, outline-mask, and shadow-cast pipelines.
    pub(crate) pick_pipeline: Option<vwgpu::RenderPipeline>,
    pub(crate) mask_pipeline: Option<vwgpu::RenderPipeline>,
    pub(crate) shadow_pipeline: Option<vwgpu::RenderPipeline>,
    pub(crate) pick_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) shadow_group0_bgl: Option<vwgpu::BindGroupLayout>,

    /// Live per-instance GPU + emission state, keyed by `(effect, key)`. Each
    /// entry owns an independent particle buffer, trail / sort state, and pick
    /// uniform. Built lazily the first frame an instance is driven.
    pub(crate) instances: HashMap<InstanceKey, EffectInstance>,

    /// Active transient (burst) instances (Phase 25), advanced every frame and
    /// recycled when they age out. Their GPU state lives in `instances` under
    /// `TRANSIENT_KEY_BIT` keys.
    pub(crate) transients: Vec<TransientInstance>,
    /// Recycled transient slot indices per effect (a free list).
    pub(crate) transient_free: HashMap<usize, Vec<u64>>,
    /// Next fresh transient slot index per effect (the pool high-water mark).
    pub(crate) transient_next: HashMap<usize, u64>,

    /// Meshes uploaded for the mesh render route, indexed by `ParticleMeshId`.
    pub(crate) meshes: Vec<ParticleMesh>,

    /// Textures uploaded for the billboard texture / flipbook routes, indexed by
    /// `ParticleTextureId`.
    pub(crate) textures: Vec<ParticleTexture>,
    /// A 1x1 white texture bound at group 3 for effects with no texture (so the
    /// billboard fragment always has a valid sampler). Built lazily in `prepare`.
    pub(crate) default_tex_view: Option<vwgpu::TextureView>,
    /// A 1-entry spawn-sample buffer bound at fixed-emit binding 3 for effects
    /// that do not use a `MeshSurface` / `PointCloud` spawn shape (Phase 22), so
    /// the emit bind group always has a valid buffer there. Built in `build_gpu`.
    pub(crate) default_spawn_samples: Option<vwgpu::Buffer>,

    // Lifetime-ramp LUT (group 2 of the draw pipeline): the layout, the shared
    // linear sampler, and an identity LUT + bind group reused by effects with no
    // gradient. Built in `init_gpu`.
    pub(crate) ramp_bgl: Option<vwgpu::BindGroupLayout>,
    pub(crate) ramp_sampler: Option<vwgpu::Sampler>,
    #[allow(dead_code)]
    pub(crate) identity_ramp_lut: Option<vwgpu::Texture>,
    pub(crate) identity_ramp_bg: Option<vwgpu::BindGroup>,

    /// Compiled generated emit/simulate pipelines, keyed by the hash of their
    /// WGSL so effects with identical programs share pipelines.
    pub(crate) gen_cache: HashMap<u64, (vwgpu::ComputePipeline, vwgpu::ComputePipeline)>,

    /// Effects with live simulation this frame, drawn in `paint` and the
    /// interaction passes.
    pub(crate) draw_list: Vec<DrawEntry>,

    /// Elapsed simulation time in seconds, accumulated from each frame's `dt`.
    /// Scrolls time-varying fields such as the `CurlNoise` force.
    pub(crate) time: f32,
}

/// One instance drawn this frame, with the driving item's interaction flags.
#[derive(Copy, Clone)]
pub(crate) struct DrawEntry {
    pub(crate) key: InstanceKey,
    /// Pick id of the driving item (0 = `PickId::NONE`, not pickable).
    pub(crate) pick_id: u32,
    pub(crate) selected: bool,
    pub(crate) cast_shadows: bool,
}

/// One instance to consider simulating this frame, before culling / budget.
/// Steady instances carry `item_idx` (the source of per-frame overrides);
/// transient (burst) instances carry `None` and use the registered asset.
#[derive(Copy, Clone)]
pub(crate) struct DrivenInstance {
    pub(crate) ikey: InstanceKey,
    pub(crate) origin: [f32; 3],
    pub(crate) pick_id: u32,
    pub(crate) selected: bool,
    pub(crate) cast_shadows: bool,
    /// `Some(index)` for a steady submitted item; `None` for a transient.
    pub(crate) item_idx: Option<usize>,
    /// Deterministic seed for this instance's emit, if any (Phase 28).
    pub(crate) seed: Option<u64>,
}

/// Per-effect pick-id uniform (group 2 of the pick pipeline). 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct PickUniform {
    pub(crate) id: u32,
    pub(crate) _pad: [u32; 3],
}
