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

use bytemuck::{Pod, Zeroable};
use viewport_lib::plugin_api::shared_wgsl::SHARED_BINDINGS_WGSL;
use viewport_lib::plugin_api::{
    ItemFrameContext, ItemTypePlugin, PaintContext, PluginItemCollection, SharedBindings,
};
use viewport_lib::resources::{HDR_COLOR_FORMAT, SCENE_DEPTH_FORMAT};
use viewport_lib::scene::material::ItemSettings;
use viewport_lib::wgpu as vwgpu;

use crate::effect::{
    EffectAsset, EffectId, Emitter, ForceModifier, ParticleBlend, SpawnRate, SpawnShape,
    VelocityDist,
};

/// MSAA sample count the plugin builds its pipelines for. A plugin cannot read
/// the renderer's configured sample count from `SharedBindings` today, so this
/// assumes the default single-sampled HDR pass (see the plan).
const SAMPLE_COUNT: u32 = 1;

/// Max forces per effect, matching the fixed array in `particle_sim.wgsl`.
const MAX_FORCES: usize = 8;

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
}

impl Default for ParticleItems {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            dt: 1.0 / 60.0,
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
struct EffectGpu {
    /// The persistent particle buffer. Held to own it for the effect's lifetime;
    /// the emit/sim/draw bind groups alias it, so it is not read directly.
    #[allow(dead_code)]
    particle_buf: vwgpu::Buffer,
    emit_buf: vwgpu::Buffer,
    sim_buf: vwgpu::Buffer,
    budget_buf: vwgpu::Buffer,
    emit_bg: vwgpu::BindGroup,
    sim_bg: vwgpu::BindGroup,
    draw_bg: vwgpu::BindGroup,
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
    emit_bgl: Option<vwgpu::BindGroupLayout>,
    sim_bgl: Option<vwgpu::BindGroupLayout>,
    draw_bgl: Option<vwgpu::BindGroupLayout>,

    /// Effects with live simulation this frame, drawn in `paint`.
    draw_list: Vec<usize>,
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
        let draw_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_draw_bgl"),
            entries: &[vwgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: vwgpu::ShaderStages::VERTEX,
                ty: vwgpu::BindingType::Buffer {
                    ty: vwgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
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

        // Draw pipelines (additive and premultiplied over).
        let draw_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("shaders/particle_draw.wgsl")
        );
        let draw_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_draw_shader"),
            source: vwgpu::ShaderSource::Wgsl(draw_src.into()),
        });
        let draw_layout =
            vwgpu::pipeline_layout(device, "particle_draw_layout", &[shared.group0_layout, &draw_bgl]);

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

        self.emit_pipeline = Some(emit_pipeline);
        self.sim_pipeline = Some(sim_pipeline);
        self.emit_bgl = Some(emit_bgl);
        self.sim_bgl = Some(sim_bgl);
        self.draw_bgl = Some(draw_bgl);
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

        let dt = items.dt.max(0.0);

        // First non-hidden instance per effect drives that effect's simulation.
        let mut driver: Vec<Option<[f32; 3]>> = vec![None; self.effects.len()];
        for it in &items.items {
            if it.settings.hidden {
                continue;
            }
            let idx = it.effect.0 as usize;
            if idx < driver.len() && driver[idx].is_none() {
                driver[idx] = Some(it.position);
            }
        }

        let mut encoder = device.create_command_encoder(&vwgpu::CommandEncoderDescriptor {
            label: Some("particle_sim_encoder"),
        });
        let mut any = false;

        for (idx, origin) in driver.iter().enumerate() {
            let Some(origin) = *origin else { continue };

            // Lazily build this effect's GPU state.
            if self.effects[idx].gpu.is_none() {
                let gpu = build_effect_gpu(
                    device,
                    self.effects[idx].asset.capacity,
                    &emit_bgl,
                    &sim_bgl,
                    &draw_bgl,
                );
                self.effects[idx].gpu = Some(gpu);
            }

            let spawn_count = next_spawn_count(&mut self.effects[idx], dt);
            let asset = &self.effects[idx].asset;
            let rng_seed = (ctx.frame_index as u32).wrapping_mul(2654435761).wrapping_add(idx as u32);

            let emit = build_emit_params(&asset.emitter, origin, asset.capacity, spawn_count, rng_seed);
            let sim = build_sim_params(&asset.forces, dt);

            let gpu = self.effects[idx].gpu.as_ref().unwrap();
            queue.write_buffer(&gpu.emit_buf, 0, bytemuck::bytes_of(&emit));
            queue.write_buffer(&gpu.sim_buf, 0, bytemuck::bytes_of(&sim));
            queue.write_buffer(&gpu.budget_buf, 0, bytemuck::bytes_of(&0u32));

            let groups = self.effects[idx].asset.capacity.div_ceil(64).max(1);
            {
                let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
                    label: Some("particle_emit_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&emit_pipeline);
                pass.set_bind_group(0, &gpu.emit_bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
                    label: Some("particle_sim_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&sim_pipeline);
                pass.set_bind_group(0, &gpu.sim_bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }

            self.draw_list.push(idx);
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
        let (Some(additive), Some(over)) = (
            self.draw_pipeline_additive.as_ref(),
            self.draw_pipeline_over.as_ref(),
        ) else {
            return;
        };
        // Group 0 (camera + scene) is already bound by the lib.
        for &idx in &self.draw_list {
            let eff = &self.effects[idx];
            let Some(gpu) = eff.gpu.as_ref() else { continue };
            let pipeline = match eff.asset.blend {
                ParticleBlend::Additive => additive,
                ParticleBlend::Alpha | ParticleBlend::Premultiplied => over,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &gpu.draw_bg, &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }
}

/// Allocate one effect's persistent particle buffer, its parameter buffers, and
/// the emit/sim/draw bind groups. The particle buffer starts zeroed, so every
/// slot reads as dead until the first emit pass.
fn build_effect_gpu(
    device: &vwgpu::Device,
    capacity: u32,
    emit_bgl: &vwgpu::BindGroupLayout,
    sim_bgl: &vwgpu::BindGroupLayout,
    draw_bgl: &vwgpu::BindGroupLayout,
) -> EffectGpu {
    let particle_bytes = (capacity as u64) * std::mem::size_of::<GpuParticle>() as u64;
    let particle_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_buffer"),
        size: particle_bytes.max(std::mem::size_of::<GpuParticle>() as u64),
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    // Zero the buffer so all slots are dead.
    particle_buf.slice(..).get_mapped_range_mut().fill(0);
    particle_buf.unmap();

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
    let budget_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_spawn_budget"),
        size: std::mem::size_of::<u32>() as u64,
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
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
    let draw_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_draw_bg"),
        layout: draw_bgl,
        entries: &[vwgpu::BindGroupEntry {
            binding: 0,
            resource: particle_buf.as_entire_binding(),
        }],
    });

    EffectGpu {
        particle_buf,
        emit_buf,
        sim_buf,
        budget_buf,
        emit_bg,
        sim_bg,
        draw_bg,
    }
}

/// How many particles to spawn this frame, advancing the effect's accumulator.
fn next_spawn_count(effect: &mut RegisteredEffect, dt: f32) -> u32 {
    match effect.asset.emitter.rate {
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
