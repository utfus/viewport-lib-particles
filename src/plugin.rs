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
//! Phase 2 is fixed-function: one simulation per registered effect, driven by
//! the first non-hidden instance of that effect (its position is the emitter
//! origin). Multiple instances of one effect, and per-effect expression codegen,
//! come later.

use std::any::Any;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use bytemuck::{Pod, Zeroable};
use viewport_lib::plugin_api::shared_wgsl::{
    SHARED_BINDINGS_WGSL, SHARED_MASK_WGSL, SHARED_PICK_WGSL,
};
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, OutlineMaskContext, PaintContext, PickPassContext,
    PluginItemCollection, ShadowCastContext, SharedBindings,
};
use viewport_lib::resources::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT,
    SCENE_DEPTH_FORMAT, SHADOW_DEPTH_FORMAT,
};
use viewport_lib::scene::material::ItemSettings;
use viewport_lib::wgpu as vwgpu;

use crate::codegen;
use crate::effect::{
    EffectAsset, EffectId, EffectProgram, Emitter, ForceModifier, MeshAlign, ParticleBlend,
    ParticleMeshId, ParticleRender, SpawnRate, SpawnShape, VelocityDist,
};

/// MSAA sample count the plugin builds its pipelines for. A plugin cannot read
/// the renderer's configured sample count from `SharedBindings` today, so this
/// assumes the default single-sampled HDR pass (see the plan).
const SAMPLE_COUNT: u32 = 1;

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
#[derive(Clone, Debug)]
pub struct ParticleItem {
    /// Which registered effect to simulate and draw.
    pub effect: EffectId,
    /// World-space emitter origin, added to the effect's spawn shape.
    pub position: [f32; 3],
    /// Shared per-item flags (hidden, selected, pick id, opacity, ...).
    pub settings: ItemSettings,
}

impl ParticleItem {
    /// A visible instance of `effect` at the world origin.
    pub fn new(effect: EffectId) -> Self {
        Self {
            effect,
            position: [0.0, 0.0, 0.0],
            settings: ItemSettings::default(),
        }
    }

    /// Place the emitter at `position`.
    pub fn at(mut self, position: [f32; 3]) -> Self {
        self.position = position;
        self
    }
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
    /// Simulation time step in seconds for this frame. Defaults to 1/60.
    pub dt: f32,
    /// Whether to depth-sort non-additive (alpha / premultiplied) effects
    /// back-to-front this frame. Additive effects are order-independent and are
    /// never sorted. Defaults to `true`; set `false` to see the ordering
    /// artefact.
    pub sort_transparent: bool,
}

impl Default for ParticleItems {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            dt: 1.0 / 60.0,
            sort_transparent: true,
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

    /// Append an instance.
    pub fn push(&mut self, item: ParticleItem) {
        self.items.push(item);
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

/// Simulate-pass parameters. Matches `SimParams` in `particle_sim.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SimParamsGpu {
    /// dt, time, _, force_count.
    misc: [f32; 4],
    forces: [GpuForce; MAX_FORCES],
}

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// An uploaded mesh for the mesh render route: local-space vertex positions and
/// a triangle index buffer.
struct ParticleMesh {
    vertex_buf: vwgpu::Buffer,
    index_buf: vwgpu::Buffer,
    index_count: u32,
}

/// Per-effect draw parameters (group 3). Matches `DrawParams` in the draw and
/// mesh shaders. 16 bytes.
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

/// Fixed-function effect GPU state (Phase 2 emitter path).
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

/// A registered effect plus its CPU-side emission bookkeeping.
struct RegisteredEffect {
    asset: EffectAsset,
    gpu: Option<EffectGpu>,
    /// Fractional spawn accumulator for `SpawnRate::PerSecond`.
    spawn_accum: f32,
    /// Whether a `SpawnRate::Burst` has already fired.
    burst_done: bool,
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
    draw_pipeline_over: Option<vwgpu::RenderPipeline>,
    mesh_pipeline_additive: Option<vwgpu::RenderPipeline>,
    mesh_pipeline_over: Option<vwgpu::RenderPipeline>,
    emit_bgl: Option<vwgpu::BindGroupLayout>,
    sim_bgl: Option<vwgpu::BindGroupLayout>,
    draw_bgl: Option<vwgpu::BindGroupLayout>,
    drawparams_bgl: Option<vwgpu::BindGroupLayout>,

    // Trail render route: the history-append compute pass and the ribbon draw.
    append_pipeline: Option<vwgpu::ComputePipeline>,
    trail_pipeline_additive: Option<vwgpu::RenderPipeline>,
    trail_pipeline_over: Option<vwgpu::RenderPipeline>,
    history_bgl: Option<vwgpu::BindGroupLayout>,
    trail_draw_bgl: Option<vwgpu::BindGroupLayout>,
    /// Per-effect trail state, keyed by effect index. Present only for effects
    /// whose render route is `ParticleRender::Trail`.
    trails: HashMap<usize, TrailGpu>,

    // Depth sort: the setup (key + identity order) pass and one bitonic stage.
    sort_init_pipeline: Option<vwgpu::ComputePipeline>,
    sort_stage_pipeline: Option<vwgpu::ComputePipeline>,
    sort_init_bgl: Option<vwgpu::BindGroupLayout>,
    sort_stage_bgl: Option<vwgpu::BindGroupLayout>,
    /// Per-effect sort state, keyed by effect index. Present only for
    /// non-additive effects, whose draw order needs back-to-front sorting.
    sorts: HashMap<usize, SortGpu>,

    // Interaction: pick-id, outline-mask, and shadow-cast pipelines.
    pick_pipeline: Option<vwgpu::RenderPipeline>,
    mask_pipeline: Option<vwgpu::RenderPipeline>,
    shadow_pipeline: Option<vwgpu::RenderPipeline>,
    pick_bgl: Option<vwgpu::BindGroupLayout>,
    shadow_group0_bgl: Option<vwgpu::BindGroupLayout>,
    /// Per-effect pick-id uniform + bind group (group 2 of the pick pipeline),
    /// keyed by effect index. Written each frame from the driving item's id.
    pick_gpu: HashMap<usize, (vwgpu::Buffer, vwgpu::BindGroup)>,

    /// Meshes uploaded for the mesh render route, indexed by `ParticleMeshId`.
    meshes: Vec<ParticleMesh>,

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
}

/// One effect drawn this frame, with the driving item's interaction flags.
#[derive(Copy, Clone)]
struct DrawEntry {
    idx: usize,
    /// Pick id of the driving item (0 = `PickId::NONE`, not pickable).
    pick_id: u32,
    selected: bool,
    cast_shadows: bool,
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
            gpu: None,
            spawn_accum: 0.0,
            burst_done: false,
        });
        id
    }

    /// Number of registered effects.
    pub fn effect_count(&self) -> usize {
        self.effects.len()
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

    /// Build effect `idx`'s GPU state if it does not exist yet. Fixed-function
    /// effects get the shared Phase 2 pipelines; effects with a program compile
    /// (or reuse from the cache) their own emit/simulate pipelines.
    fn ensure_effect_gpu(
        &mut self,
        idx: usize,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        emit_bgl: &vwgpu::BindGroupLayout,
        sim_bgl: &vwgpu::BindGroupLayout,
        draw_bgl: &vwgpu::BindGroupLayout,
    ) {
        if self.effects[idx].gpu.is_some() {
            return;
        }
        let capacity = self.effects[idx].asset.capacity;
        // Clone the program and gradient out so the `self.effects` borrow ends
        // before we touch the pipeline cache and assign the new GPU state.
        let program = self.effects[idx].asset.program.clone();
        let gradient = self.effects[idx].asset.gradient.clone();
        let render = self.effects[idx].asset.render;
        let blend = self.effects[idx].asset.blend;

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

        // Per-effect draw params (group 3), derived from the render mode.
        let dp = match render {
            ParticleRender::Billboard { stretch } => DrawParams {
                stretch,
                align: 0,
                trail_width: 0.0,
                trail_segments: 0,
            },
            ParticleRender::Mesh { align, .. } => DrawParams {
                stretch: 0.0,
                align: match align {
                    MeshAlign::Velocity => 0,
                    MeshAlign::Random => 1,
                },
                trail_width: 0.0,
                trail_segments: 0,
            },
            ParticleRender::Trail { width, segments } => DrawParams {
                stretch: 0.0,
                align: 0,
                trail_width: width,
                trail_segments: segments.clamp(1, HISTORY_LEN - 1),
            },
        };
        let drawparams_bg = build_drawparams_bg(device, self.drawparams_bgl.as_ref().unwrap(), dp);

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
                let (emit_pipeline, sim_pipeline) =
                    self.gen_pipelines(device, &program, emit_bgl, sim_bgl);
                EffectGpu::Gen(build_gen_gpu(
                    device,
                    capacity,
                    &order_buf,
                    emit_bgl,
                    sim_bgl,
                    draw_bgl,
                    ramp_bg,
                    ramp_lut,
                    drawparams_bg,
                    emit_pipeline,
                    sim_pipeline,
                ))
            }
        };
        self.effects[idx].gpu = Some(gpu);

        // Trail effects also get a per-particle history ring plus the append and
        // ribbon bind groups, built over the particle buffer just allocated.
        if let ParticleRender::Trail { .. } = render {
            let history_bgl = self.history_bgl.clone().unwrap();
            let trail_draw_bgl = self.trail_draw_bgl.clone().unwrap();
            let particle_buf = self.effects[idx].gpu.as_ref().unwrap().particle_buf();
            let trail = build_trail_gpu(
                device,
                capacity,
                particle_buf,
                &order_buf,
                &history_bgl,
                &trail_draw_bgl,
            );
            self.trails.insert(idx, trail);
        }

        // Non-additive effects need back-to-front draw order for correct alpha,
        // so they get the sort key + bitonic state over the particle and order
        // buffers. Additive effects are order-independent and keep identity.
        if !matches!(blend, ParticleBlend::Additive) {
            let sort_init_bgl = self.sort_init_bgl.clone().unwrap();
            let sort_stage_bgl = self.sort_stage_bgl.clone().unwrap();
            let particle_buf = self.effects[idx].gpu.as_ref().unwrap().particle_buf();
            let sort = build_sort_gpu(
                device,
                n,
                particle_buf,
                &order_buf,
                &sort_init_bgl,
                &sort_stage_bgl,
            );
            self.sorts.insert(idx, sort);
        }

        // Per-effect pick-id uniform (group 2 of the pick pipeline). Written each
        // frame from the driving item's pick id.
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
        self.pick_gpu.insert(idx, (pick_buf, pick_bg));
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

        // Group 3: per-effect draw params (stretch / mesh align), read in the
        // vertex stage of both the billboard and mesh pipelines.
        let drawparams_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_drawparams_bgl"),
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
        let sim_pipeline = compute(
            include_str!("shaders/particle_sim.wgsl"),
            "sim_main",
            "particle_sim",
            &sim_bgl,
        );
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
        self.draw_pipeline_over = Some(make_draw(
            vwgpu::BlendState {
                color: over,
                alpha: over,
            },
            "particle_draw_over",
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
        self.append_pipeline = Some(append_pipeline);
        self.sort_init_pipeline = Some(sort_init_pipeline);
        self.sort_stage_pipeline = Some(sort_stage_pipeline);
        self.emit_bgl = Some(emit_bgl);
        self.sim_bgl = Some(sim_bgl);
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
        let append_pipeline = self.append_pipeline.clone();
        let sort_init_pipeline = self.sort_init_pipeline.clone();
        let sort_stage_pipeline = self.sort_stage_pipeline.clone();

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

        let dt = items.dt.max(0.0);

        // First non-hidden instance per effect drives that effect's simulation.
        // First non-hidden instance per effect drives the simulation and supplies
        // the interaction flags (pick id / selected / cast shadows).
        let mut driver: Vec<Option<([f32; 3], u32, bool, bool)>> = vec![None; self.effects.len()];
        for it in &items.items {
            if it.settings.hidden {
                continue;
            }
            let idx = it.effect.0 as usize;
            if idx < driver.len() && driver[idx].is_none() {
                driver[idx] = Some((
                    it.position,
                    it.settings.pick_id.0 as u32,
                    it.settings.selected,
                    it.settings.cast_shadows,
                ));
            }
        }

        let mut encoder = device.create_command_encoder(&vwgpu::CommandEncoderDescriptor {
            label: Some("particle_sim_encoder"),
        });
        let mut any = false;

        for (idx, origin) in driver.iter().enumerate() {
            let Some((origin, pick_id, selected, cast_shadows)) = *origin else {
                continue;
            };

            // Lazily build this effect's GPU state (fixed or generated).
            self.ensure_effect_gpu(idx, device, queue, &emit_bgl, &sim_bgl, &draw_bgl);

            // Publish the driving item's pick id for the pick pass.
            if let Some((buf, _)) = self.pick_gpu.get(&idx) {
                let pu = PickUniform {
                    id: pick_id,
                    _pad: [0; 3],
                };
                queue.write_buffer(buf, 0, bytemuck::bytes_of(&pu));
            }

            let spawn_count = next_spawn_count(&mut self.effects[idx], dt);
            let capacity = self.effects[idx].asset.capacity;
            let rng_seed = (ctx.frame_index as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(idx as u32);
            let groups = capacity.div_ceil(64).max(1);

            match self.effects[idx].gpu.as_ref().unwrap() {
                EffectGpu::Fixed(g) => {
                    let asset = &self.effects[idx].asset;
                    let emit =
                        build_emit_params(&asset.emitter, origin, capacity, spawn_count, rng_seed);
                    let sim = build_sim_params(&asset.forces, dt);
                    queue.write_buffer(&g.emit_buf, 0, bytemuck::bytes_of(&emit));
                    queue.write_buffer(&g.sim_buf, 0, bytemuck::bytes_of(&sim));
                    queue.write_buffer(&g.budget_buf, 0, bytemuck::bytes_of(&0u32));
                    dispatch_sim(
                        &mut encoder,
                        &emit_pipeline,
                        &g.emit_bg,
                        &sim_pipeline,
                        &g.sim_bg,
                        groups,
                    );
                }
                EffectGpu::Gen(g) => {
                    let gp = GenParams {
                        origin: [origin[0], origin[1], origin[2], 0.0],
                        dt,
                        time: 0.0,
                        spawn_count,
                        capacity,
                        rng_seed,
                        _pad: [0; 3],
                    };
                    queue.write_buffer(&g.params_buf, 0, bytemuck::bytes_of(&gp));
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
                if let (Some(append_pipeline), Some(trail)) =
                    (append_pipeline.as_ref(), self.trails.get(&idx))
                {
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
                    self.sorts.get(&idx),
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
                idx,
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
        let (
            Some(bb_add),
            Some(bb_over),
            Some(mesh_add),
            Some(mesh_over),
            Some(trail_add),
            Some(trail_over),
        ) = (
            self.draw_pipeline_additive.as_ref(),
            self.draw_pipeline_over.as_ref(),
            self.mesh_pipeline_additive.as_ref(),
            self.mesh_pipeline_over.as_ref(),
            self.trail_pipeline_additive.as_ref(),
            self.trail_pipeline_over.as_ref(),
        )
        else {
            return;
        };
        // Group 0 (camera + scene) is already bound by the lib.
        for entry in &self.draw_list {
            let idx = entry.idx;
            let eff = &self.effects[idx];
            let Some(gpu) = eff.gpu.as_ref() else {
                continue;
            };
            let additive = matches!(eff.asset.blend, ParticleBlend::Additive);
            // Groups 2 (ramp) and 3 (draw params) are shared by every route;
            // group 1 is route-specific (particles, or the trail bindings).
            pass.set_bind_group(2, gpu.ramp_bg(), &[]);
            pass.set_bind_group(3, gpu.drawparams_bg(), &[]);

            match eff.asset.render {
                ParticleRender::Billboard { .. } => {
                    pass.set_pipeline(if additive { bb_add } else { bb_over });
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
                    let Some(trail) = self.trails.get(&idx) else {
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
            let eff = &self.effects[entry.idx];
            let Some(gpu) = eff.gpu.as_ref() else {
                continue;
            };
            let Some((_, pick_bg)) = self.pick_gpu.get(&entry.idx) else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, gpu.draw_bg(), &[]);
            pass.set_bind_group(2, pick_bg, &[]);
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
            let eff = &self.effects[entry.idx];
            let Some(gpu) = eff.gpu.as_ref() else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, gpu.draw_bg(), &[]);
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
            let eff = &self.effects[entry.idx];
            let Some(gpu) = eff.gpu.as_ref() else {
                continue;
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, gpu.draw_bg(), &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }
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

/// Uniform bind group (group 3) holding a `DrawParams`.
fn build_drawparams_bg(
    device: &vwgpu::Device,
    drawparams_bgl: &vwgpu::BindGroupLayout,
    params: DrawParams,
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
        entries: &[vwgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
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
        ],
    });
    let draw_bg = build_draw_bg(device, draw_bgl, &particle_buf, order_buf);

    GenGpu {
        particle_buf,
        params_buf,
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
/// A generated effect takes its rate from the program; a fixed effect from the
/// emitter.
fn next_spawn_count(effect: &mut RegisteredEffect, dt: f32) -> u32 {
    let rate = match &effect.asset.program {
        Some(program) => program.rate,
        None => effect.asset.emitter.rate,
    };
    match rate {
        SpawnRate::PerSecond(rate) => {
            effect.spawn_accum += rate.max(0.0) * dt;
            let count = effect.spawn_accum.floor();
            effect.spawn_accum -= count;
            count as u32
        }
        SpawnRate::Burst { count } => {
            if effect.burst_done {
                0
            } else {
                effect.burst_done = true;
                count
            }
        }
    }
}

fn build_emit_params(
    emitter: &Emitter,
    origin: [f32; 3],
    capacity: u32,
    spawn_count: u32,
    rng_seed: u32,
) -> EmitParamsGpu {
    let mut p = EmitParamsGpu::zeroed();
    p.origin = [origin[0], origin[1], origin[2], 0.0];
    p.colour = emitter.colour;
    p.misc = [emitter.lifetime.0, emitter.lifetime.1, emitter.size, 0.0];
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
    p.kinds = [spawn_kind, vel_kind, 0, 0];
    p
}

fn build_sim_params(forces: &[ForceModifier], dt: f32) -> SimParamsGpu {
    let mut p = SimParamsGpu::zeroed();
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
        };
    }
    p.misc = [dt, 0.0, 0.0, n as f32];
    p
}
