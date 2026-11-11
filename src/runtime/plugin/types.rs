use std::collections::HashMap;
use std::hash::Hash;

use bytemuck::{Pod, Zeroable};
use viewport_lib::wgpu as vwgpu;

use std::sync::{Arc, Mutex};

use crate::effect::{EffectAsset, EffectId, EventCondition, ParticleBlend, ParticleTextureId};

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
    /// Spawn roll base, roll variance, size variance, unused.
    pub(crate) rot: [f32; 4],
    /// World-space spawn offset (xyz; w unused).
    pub(crate) spawn_offset: [f32; 4],
    /// Plane lock (see [`plane_lane`]).
    pub(crate) plane: [f32; 4],
}

/// One packed modifier. Matches `Modifier` in the shared `MODIFIERS_WGSL` block.
/// `kind` selects the behaviour; the three `data` lanes carry that kind's
/// parameters (see `build_sim_params` / `eval_modifier` for the layout per kind).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct GpuModifier {
    /// Kind discriminant (matches the `Modifier` enum order).
    pub(crate) kind: u32,
    pub(crate) _pad: [u32; 3],
    pub(crate) data0: [f32; 4],
    pub(crate) data1: [f32; 4],
    pub(crate) data2: [f32; 4],
}

/// Simulate-pass parameters. Matches `SimParams` in `particle_sim.wgsl` (and,
/// with the extra `event` lane read, `particle_sim_parent.wgsl`). The `event`
/// lane trails the modifiers array, so both sim shaders read the same buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SimParamsGpu {
    /// dt, time, _, modifier_count.
    pub(crate) misc: [f32; 4],
    pub(crate) modifiers: [GpuModifier; MAX_MODIFIERS],
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
    /// Plane lock (see [`plane_lane`]).
    pub(crate) plane: [f32; 4],
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
    /// `PointCloud` spawn shapes can bake an area-weighted sample pool
    /// at effect-registration time.
    pub(crate) positions: Vec<[f32; 3]>,
    /// CPU copy of the triangle indices (empty for a pure point cloud).
    pub(crate) indices: Vec<u32>,
}

/// One baked spawn sample for the `MeshSurface` / `PointCloud` shapes.
/// Matches `SpawnSample` in `particle_emit.wgsl`. 32 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct SpawnSampleGpu {
    /// Instance-relative sample position (xyz; w unused).
    pub(crate) position: [f32; 4],
    /// Surface (or radial) normal at the sample (xyz; w unused).
    pub(crate) normal: [f32; 4],
}

/// Count-pass parameters. Matches `CountParams` in `particle_count.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct CountParamsGpu {
    pub(crate) capacity: u32,
    pub(crate) _pad: [u32; 3],
}

/// Live simulation statistics for one instance (or one effect, summed over its
/// instances), read back from the GPU a couple of frames late. Suitable for an
/// editor readout: a live particle count and an over-budget flag, without
/// stalling the frame.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ParticleStats {
    /// Live particle count at the last completed readback.
    pub live_count: u32,
    /// The instance's (or effect's) particle-buffer capacity.
    pub capacity: u32,
    /// True when the buffer is full (`live_count >= capacity`), so the emitter
    /// cannot spawn more this frame: the over-budget state an editor shows.
    pub over_budget: bool,
}

/// One instance's latest readback: live count and its buffer capacity.
#[derive(Copy, Clone)]
pub(crate) struct StatEntry {
    pub(crate) live: u32,
    pub(crate) capacity: u32,
}

/// Shared live-count store, keyed by `(effect, key)`. The plugin writes each
/// instance's completed readback here inside `prepare`; a [`ParticleStatsHandle`]
/// cloned out before install reads it back on the host side. The handle is how a
/// host observes these counts after the renderer takes ownership of the plugin.
#[derive(Default)]
pub(crate) struct StatsShared {
    pub(crate) entries: HashMap<(u32, u64), StatEntry>,
}

/// A cheap, cloneable handle to a [`ParticlePlugin`]'s live particle-count stats.
/// Obtain it with [`ParticlePlugin::stats_handle`] before handing the plugin to
/// the renderer, then read counts from it each frame. The plugin updates the
/// shared store inside its `prepare`, which the renderer drives, so the handle
/// stays live after install.
///
/// Enable the readback per frame with
/// [`ParticleItems::with_stats_readback`](crate::ParticleItems::with_stats_readback).
/// The values lag about two frames (an async, non-stalling GPU read), which is
/// fine for an editor readout. Counts are per instance, so
/// [`instance_stats`](Self::instance_stats) addresses a single `(effect, key)`
/// and [`effect_stats`](Self::effect_stats) sums an effect's instances.
#[derive(Clone, Default)]
pub struct ParticleStatsHandle {
    pub(crate) inner: Arc<Mutex<StatsShared>>,
}

impl ParticleStatsHandle {
    /// Latest readback for one instance (`key` from
    /// [`ParticleItem::with_key`](crate::ParticleItem::with_key)), or `None`
    /// until a readback for it has completed (the first frames after enabling are
    /// empty, since the read is deferred).
    pub fn instance_stats(&self, effect: EffectId, key: u64) -> Option<ParticleStats> {
        let g = self.inner.lock().ok()?;
        let e = g.entries.get(&(effect.0, key))?;
        Some(ParticleStats {
            live_count: e.live,
            capacity: e.capacity,
            over_budget: e.capacity > 0 && e.live >= e.capacity,
        })
    }

    /// Latest readback for an effect, summed over its instances that have a
    /// completed readback (each owns a `capacity`-sized buffer, so `capacity`
    /// here is that sum). `None` until at least one instance has a readback.
    pub fn effect_stats(&self, effect: EffectId) -> Option<ParticleStats> {
        let g = self.inner.lock().ok()?;
        let mut live = 0u32;
        let mut capacity = 0u32;
        let mut any = false;
        for ((eff, _), e) in &g.entries {
            if *eff == effect.0 {
                live = live.saturating_add(e.live);
                capacity = capacity.saturating_add(e.capacity);
                any = true;
            }
        }
        any.then_some(ParticleStats {
            live_count: live,
            capacity,
            over_budget: capacity > 0 && live >= capacity,
        })
    }
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
    /// Billboard orientation: 0 = parallel camera plane, 1 = face
    /// camera position, 2 = along velocity, 3 = fixed screen roll, 4 = in
    /// the lock plane.
    pub(crate) orient: u32,
    /// Orientation parameter: the fixed roll angle (radians) for orient == 3,
    /// the plane (0 = XY, 1 = XZ) for orient == 4.
    pub(crate) orient_param: f32,
    /// Screen-space size flag: 1 keeps constant apparent size.
    pub(crate) screen_size: u32,
    /// Soft-particle fade distance in world units; 0 = not soft. Read
    /// only by the soft (depth-read) billboard shader.
    pub(crate) soft_dist: f32,
    /// Billboard spin in radians per second, advanced by particle age. Billboard
    /// shaders only; 0 = no spin. The particle's spawn roll rides its `pad.x`.
    pub(crate) roll_rate: f32,
    /// Padding to a 16-byte multiple (uniform size requirement). The mesh and
    /// trail shaders read only the 16-byte prefix, so growth here is invisible to
    /// them.
    pub(crate) _pad: [f32; 3],
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
/// `GenParams` in the generated WGSL. 80 bytes.
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
    /// Plane lock (see [`plane_lane`]).
    pub(crate) plane: [f32; 4],
}

/// Per-effect GPU state, built lazily the first time an effect is simulated.
/// The variant follows the effect's authoring path: fixed-function emitter, or
/// a generated program with its own compiled emit/simulate pipelines.
pub(crate) enum EffectGpu {
    Fixed(FixedGpu),
    Gen(GenGpu),
}

impl EffectGpu {
    /// GPU buffer bytes this effect's state holds. Bind groups are views over
    /// these buffers, so counting the buffers counts everything once.
    pub(crate) fn resident_bytes(&self) -> u64 {
        match self {
            EffectGpu::Fixed(g) => {
                g.particle_buf.size() + g.emit_buf.size() + g.sim_buf.size() + g.budget_buf.size()
            }
            EffectGpu::Gen(g) => {
                g.particle_buf.size()
                    + g.params_buf.size()
                    + g.props_buf.size()
                    + g.budget_buf.size()
            }
        }
    }

    /// The effect's own gradient LUT, when it has one.
    pub(crate) fn ramp_lut(&self) -> Option<&vwgpu::Texture> {
        match self {
            EffectGpu::Fixed(g) => g.ramp_lut.as_ref(),
            EffectGpu::Gen(g) => g.ramp_lut.as_ref(),
        }
    }

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
    /// Set by `remove_effect`: the slot is tombstoned (its
    /// `EffectId` index stays stable) and no longer simulated or drawn.
    pub(crate) removed: bool,
    /// Baked `MeshSurface` / `PointCloud` spawn-sample pool, built at
    /// registration when the emitter's spawn shape needs it. Bound at fixed-emit
    /// binding 3; effects without a mesh spawn shape use the shared 1-entry
    /// default and never index it.
    pub(crate) spawn_samples: Option<vwgpu::Buffer>,
    /// Number of samples in `spawn_samples` (0 when it is the shared default).
    pub(crate) spawn_sample_count: u32,
    /// Texture ids for a custom fragment hook's inputs, uploaded from
    /// the hook's inline bytes during `drain_pending_textures`. Empty for effects
    /// without a hook or with no hook textures.
    pub(crate) hook_texture_ids: Vec<ParticleTextureId>,
    /// The inline texture already uploaded for this slot, as
    /// `(content signature, id)`. Lets `replace_effect` reuse a resident texture
    /// when the incoming bytes are unchanged: no re-upload, no structural rebuild.
    /// `None` until an inline texture has been uploaded for this effect.
    pub(crate) texture_upload: Option<(u64, ParticleTextureId)>,
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
    /// Soft-particle draw bind group: group 1 of the depth-read
    /// pipeline, combining this instance's particles + order + draw params +
    /// sprite (folded into one group so group 3 is free for the scene depth).
    /// Present only for soft billboard effects.
    pub(crate) soft_draw_bg: Option<vwgpu::BindGroup>,
    /// Depth-sort state, present only for non-additive effects.
    pub(crate) sort: Option<SortGpu>,
    /// Live-count readback state, built lazily the first frame the host requests
    /// stats for this instance and retained thereafter.
    pub(crate) stats: Option<StatsGpu>,
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
    /// seed under deterministic seeding so each frame decorrelates.
    pub(crate) sim_frame: u32,
    /// Seconds this instance has simulated, for firing timed bursts.
    pub(crate) elapsed: f32,
    /// Bitmask of timed bursts (by index in the effect's `spawn_schedule`) that
    /// have already fired, so each fires once. Bursts past index 63 share bit 63.
    pub(crate) schedule_fired: u64,
    /// Frame index this instance was last *submitted* by the host.
    /// The idle GC reclaims a steady instance whose key has gone
    /// [`IDLE_RETIRE_FRAMES`] frames without submission. Submission (not
    /// simulation) resets it, so a paused / culled / budget-frozen instance the
    /// host still submits is kept.
    pub(crate) last_seen: u64,
    /// Seconds this instance has simulated, scaled by its time scale.
    /// Drives the time-varying-force `time` uniform per instance (so a time-scaled
    /// or prewarmed instance advances its noise fields at its own rate) and is
    /// seeded by the prewarm fast-forward.
    pub(crate) local_time: f32,
    /// Whether this instance's one-time prewarm fast-forward has run.
    /// Set once on first activation (or immediately when no warmup applies) so it
    /// never re-warms.
    pub(crate) warmed: bool,
    /// Custom fragment-hook draw state, present only for a billboard
    /// effect carrying a [`FragmentHook`](crate::effect::FragmentHook): the cached
    /// per-hook pipelines plus this instance's group-3 hook bind group and its
    /// per-frame params buffer. `None` uses the shared built-in pipelines.
    pub(crate) custom_draw: Option<CustomDraw>,
}

/// Cached custom draw pipelines for one fragment hook, shared by every
/// effect with the same composed source. wgpu pipelines are `Arc`-backed, so
/// clones are cheap.
#[derive(Clone)]
pub(crate) struct CustomDrawPipelines {
    /// Additive HDR billboard pipeline (drawn in `paint`).
    pub(crate) additive: vwgpu::RenderPipeline,
    /// Multiply and screen HDR billboard pipelines (drawn in `paint`).
    pub(crate) blends: BlendPipelines,
    /// Weighted-blended OIT pipeline (drawn in `paint_transparent`).
    pub(crate) oit: vwgpu::RenderPipeline,
}

/// A route's multiply and screen pipelines.
#[derive(Clone)]
pub(crate) struct BlendPipelines {
    pub(crate) multiply: vwgpu::RenderPipeline,
    pub(crate) screen: vwgpu::RenderPipeline,
}

impl BlendPipelines {
    /// The pipeline for an order-independent `blend`, given the route's
    /// additive one.
    pub(crate) fn pick<'a>(
        &'a self,
        blend: ParticleBlend,
        additive: &'a vwgpu::RenderPipeline,
    ) -> &'a vwgpu::RenderPipeline {
        match blend {
            ParticleBlend::Multiply => &self.multiply,
            ParticleBlend::Screen => &self.screen,
            _ => additive,
        }
    }
}

/// Per-instance custom-shading draw state.
pub(crate) struct CustomDraw {
    /// The hook's cached pipelines (additive + OIT).
    pub(crate) pipelines: CustomDrawPipelines,
    /// Group 3 for the hooked pipeline: draw params + sprite + sampler +
    /// `hook_params` + hook textures.
    pub(crate) bind_group: vwgpu::BindGroup,
    /// The `hook_params` uniform (lane 0 engine-reserved, `.w` = time), rewritten
    /// each frame in `prepare`.
    pub(crate) params_buf: vwgpu::Buffer,
}

/// A live transient (burst) instance: a pooled particle set playing a
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
    /// Deterministic seed captured from the spawning item.
    pub(crate) seed: Option<u64>,
}

/// Live-count readback state for one instance. Present only when the host enables
/// readback (`ParticleItems::with_stats_readback(true)`); built lazily the first
/// such frame. Each frame a small count pass tallies live slots into `stats_buf`,
/// which is copied into one of a small ring of mappable buffers; a buffer copied
/// on a previous frame is mapped and read here, so the count is about two frames
/// late but never stalls the frame.
pub(crate) struct StatsGpu {
    /// Single `atomic<u32>` live-count, reset to 0 each frame before the pass.
    pub(crate) stats_buf: vwgpu::Buffer,
    /// `CountParams` uniform (capacity), written once at build.
    #[allow(dead_code)]
    pub(crate) params_buf: vwgpu::Buffer,
    /// Count-pass bind group (particles read + stats read_write + params).
    pub(crate) count_bg: vwgpu::BindGroup,
    /// Ring of mappable readback buffers, cycled so a copy and its later map do
    /// not alias.
    pub(crate) ring: Vec<StatsReadback>,
}

/// One mappable readback buffer in a [`StatsGpu`] ring, with its async-map state.
pub(crate) struct StatsReadback {
    /// `MAP_READ | COPY_DST`, four bytes (the live count).
    pub(crate) buf: vwgpu::Buffer,
    pub(crate) state: ReadbackState,
    /// Set true by the `map_async` callback when the map completes.
    pub(crate) done: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Async-map lifecycle of one [`StatsReadback`] buffer.
#[derive(PartialEq, Eq)]
pub(crate) enum ReadbackState {
    /// Free to receive a copy this frame.
    Idle,
    /// A copy was encoded this frame; next frame (once submitted) start the map.
    Copied,
    /// `map_async` is in flight; poll `done`, then read + unmap + return to idle.
    Mapping,
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
    /// Multiply and screen members of the billboard, mesh, trail and soft routes.
    pub(crate) draw_blends: Option<BlendPipelines>,
    pub(crate) mesh_blends: Option<BlendPipelines>,
    pub(crate) trail_blends: Option<BlendPipelines>,
    pub(crate) soft_blends: Option<BlendPipelines>,
    pub(crate) mesh_pipeline_additive: Option<vwgpu::RenderPipeline>,
    pub(crate) mesh_pipeline_over: Option<vwgpu::RenderPipeline>,
    // Soft-particle billboard draw for the read-only-depth pass. Its
    // group 1 folds particles+order+draw-params+sprite so group 3 carries the
    // scene depth, in the renderer's depth-read layout so the context's
    // prebuilt `scene_depth_bind_group` binds against it. Two
    // blend variants (over / additive) so soft alpha smoke and soft additive
    // flame both fade against geometry; chosen per effect by its blend.
    pub(crate) soft_pipeline_over: Option<vwgpu::RenderPipeline>,
    pub(crate) soft_pipeline_additive: Option<vwgpu::RenderPipeline>,
    pub(crate) soft_draw_bgl: Option<vwgpu::BindGroupLayout>,
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

    // Live-count readback: a compute pass tallying live particles into a
    // per-instance atomic, run only for instances whose host requested stats.
    pub(crate) count_pipeline: Option<vwgpu::ComputePipeline>,
    pub(crate) count_bgl: Option<vwgpu::BindGroupLayout>,
    /// Shared live-count store, written in `prepare` and read by any
    /// [`ParticleStatsHandle`] cloned out via [`stats_handle`](Self::stats_handle).
    pub(crate) stats: Arc<Mutex<StatsShared>>,

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
    /// The world origin the shadow pass's cascade matrix is relative to, at
    /// group 2 of the shadow pipeline, and the value last written to it.
    pub(crate) shadow_origin_buf: Option<vwgpu::Buffer>,
    pub(crate) shadow_origin_bg: Option<vwgpu::BindGroup>,
    pub(crate) shadow_origin_written: Option<[f32; 3]>,

    /// Live per-instance GPU + emission state, keyed by `(effect, key)`. Each
    /// entry owns an independent particle buffer, trail / sort state, and pick
    /// uniform. Built lazily the first frame an instance is driven.
    pub(crate) instances: HashMap<InstanceKey, EffectInstance>,

    /// Active transient (burst) instances, advanced every frame and
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
    /// Effect indices with an inline texture awaiting GPU upload.
    /// Queued by `add_effect` / `replace_effect` (which lack a `queue`) and
    /// drained at the top of `prepare`, where the plugin has `device`/`queue`.
    pub(crate) pending_textures: Vec<usize>,
    /// A 1x1 white texture bound at group 3 for effects with no texture (so the
    /// billboard fragment always has a valid sampler). Built lazily in `prepare`.
    pub(crate) default_tex_view: Option<vwgpu::TextureView>,
    /// A 1-entry spawn-sample buffer bound at fixed-emit binding 3 for effects
    /// that do not use a `MeshSurface` / `PointCloud` spawn shape, so
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

    /// Custom fragment-hook draw pipelines, keyed by the hash of the
    /// composed billboard source so effects with identical hooks share pipelines
    /// (the analogue of `gen_cache` for the fragment stage).
    pub(crate) custom_draw_cache: HashMap<u64, CustomDrawPipelines>,
    /// The renderer's pipeline builder, kept from `build_gpu` so custom draw
    /// pipelines can be built lazily in `ensure_instance_gpu`.
    pub(crate) builder: Option<viewport_lib::plugin_api::PipelineBuilder>,
    /// Group-3 layout for hooked billboard effects: draw params (0), sprite (1),
    /// sampler (2), the `hook_params` uniform (3), and [`HOOK_TEXTURES`] hook
    /// textures (4..). Fixed, so every hooked pipeline shares one layout.
    pub(crate) hook_drawparams_bgl: Option<vwgpu::BindGroupLayout>,

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
    /// Deterministic seed for this instance's emit, if any.
    pub(crate) seed: Option<u64>,
    /// Effective per-instance time scale: the item's `time_scale`
    /// times the effect's `simulation_speed` (asset-only for transients),
    /// clamped at `0`.
    pub(crate) time_scale: f32,
    /// Effective prewarm duration in seconds: the item's `warmup`
    /// override, else one particle lifetime when the effect sets `prewarm`, else
    /// `0`. Transients never prewarm.
    pub(crate) warmup: f32,
}

/// Per-effect pick-id uniform (group 2 of the pick pipeline). 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(crate) struct PickUniform {
    pub(crate) id: u32,
    pub(crate) _pad: [u32; 3],
}
