use std::hash::{Hash, Hasher};

use bytemuck::Zeroable;
use viewport_lib::wgpu as vwgpu;

use crate::codegen;
use crate::effect::{
    EffectAsset, Emitter, Modifier, PropertyDecl,
    PropertyValue, SpawnRate, SpawnShape, VelocityDist,
};

use super::*;

/// A hash of the parts of an [`EffectAsset`] that are baked into GPU state at
/// [`ensure_instance_gpu`](ParticlePlugin::ensure_instance_gpu) time — capacity,
/// blend, render route, texture, flipbook, gradient, sub-emitter presence, and
/// the generated program (its WGSL). Two assets with the same signature differ
/// only in per-frame-consumed fields (emitter / forces / lifetime / sim-space),
/// so `replace_effect` can keep the live particle set when they match.
pub(crate) fn structural_sig(asset: &EffectAsset) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    asset.capacity.hash(&mut h);
    format!("{:?}", asset.blend).hash(&mut h);
    format!("{:?}", asset.render).hash(&mut h);
    format!("{:?}", asset.orient).hash(&mut h);
    asset.screen_space_size.hash(&mut h);
    asset.soft.map(|d| d.to_bits()).hash(&mut h);
    // Spin rides DrawParams, which is baked at instance build, so a change must
    // rebuild. Roll base/variance, size variance, spawn offset, and the burst
    // schedule are re-read per frame, so they need no structural rebuild.
    asset.rotation_rate.to_bits().hash(&mut h);
    asset.texture.map(|t| t.0).hash(&mut h);
    // Custom fragment hook (Phase 33): the composed source (name + body + texture
    // count) determines the pipeline, so it is structural; param *values* ride the
    // per-frame `hook_params` uniform, so they are not.
    match &asset.fragment_hook {
        None => 0u8.hash(&mut h),
        Some(hook) => {
            1u8.hash(&mut h);
            hook.name.hash(&mut h);
            hook.body.hash(&mut h);
            hook.textures.len().hash(&mut h);
        }
    }
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

/// Content signature of an inline texture (issue 017): dimensions plus a hash of
/// the pixel bytes, so `replace_effect` can tell whether an incoming inline
/// texture matches the one already resident for a slot and skip a re-upload.
pub(crate) fn inline_tex_sig(t: &crate::effect::InlineTexture) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    t.width.hash(&mut h);
    t.height.hash(&mut h);
    t.rgba.hash(&mut h);
    h.finish()
}

/// SplitMix64 finalizer, for deriving a well-mixed 64-bit value from a seed.
pub(crate) fn splitmix64(x: u64) -> u64 {
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
pub(crate) fn instance_rng_seed(
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
pub(crate) fn zero_particle_buffer(queue: &vwgpu::Queue, buf: &vwgpu::Buffer, capacity: u32) {
    let bytes = (capacity as usize) * std::mem::size_of::<GpuParticle>();
    queue.write_buffer(buf, 0, &vec![0u8; bytes]);
}

/// The age at which a transient (burst) instance of `asset` is retired: the
/// effect's maximum particle lifetime plus a small margin, so all burst
/// particles have died.
/// One particle's maximum lifetime for `asset` (the upper bound of the emitter's
/// or program's lifetime range). Used as the default prewarm duration (issue
/// 015): warming an instance by this long reaches steady state.
pub(crate) fn max_particle_lifetime(asset: &EffectAsset) -> f32 {
    match &asset.program {
        Some(p) => p.lifetime.1,
        None => asset.emitter.lifetime.1,
    }
}

pub(crate) fn transient_lifetime(asset: &EffectAsset) -> f32 {
    let life = match &asset.program {
        Some(p) => p.lifetime.1,
        None => asset.emitter.lifetime.1,
    };
    // A timed burst can spawn late, so the transient must live past the last
    // burst plus a full particle lifetime, or those particles get cut off.
    let last_burst = asset
        .spawn_schedule
        .iter()
        .map(|b| b.time)
        .fold(0.0, f32::max);
    life.max(0.05) + last_burst + 0.1
}

/// A conservative bounding radius for an effect's live particles about its
/// emitter origin, for frustum culling. Estimated from how far a particle can
/// travel (max speed x lifetime) plus the spawn extent and particle size — the
/// codegen path is unknown, so it gets a generous default.
pub(crate) fn effect_cull_radius(asset: &EffectAsset) -> f32 {
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
        // Mesh-surface / point-cloud extent is unknown here without the geometry;
        // use a generous default so a mesh spawn is not culled prematurely.
        SpawnShape::MeshSurface { .. } | SpawnShape::PointCloud { .. } => 5.0,
    };
    speed * life + spawn_extent + e.size + 0.5
}

/// Allocate a zeroed persistent particle buffer of `capacity` slots.
pub(crate) fn build_particle_buffer(device: &vwgpu::Device, capacity: u32) -> vwgpu::Buffer {
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
pub(crate) fn build_budget_buffer(device: &vwgpu::Device) -> vwgpu::Buffer {
    device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_spawn_budget"),
        size: std::mem::size_of::<u32>() as u64,
        usage: vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Live-count readback state for one instance: the single-atomic stats buffer, a
/// `CountParams` uniform (capacity, written once), the count bind group over this
/// instance's particle buffer, and a small ring of mappable readback buffers.
/// `ring_len` buffers pipeline the copy-then-map-a-frame-later cycle without
/// aliasing.
pub(crate) fn build_stats_gpu(
    device: &vwgpu::Device,
    count_bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    capacity: u32,
    ring_len: usize,
) -> StatsGpu {
    let stats_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_stats"),
        size: std::mem::size_of::<u32>() as u64,
        usage: vwgpu::BufferUsages::STORAGE
            | vwgpu::BufferUsages::COPY_SRC
            | vwgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Capacity uniform, seeded at creation (it never changes for this instance).
    let params = CountParamsGpu {
        capacity,
        _pad: [0; 3],
    };
    let params_buf = device.create_buffer(&vwgpu::BufferDescriptor {
        label: Some("particle_count_params"),
        size: std::mem::size_of::<CountParamsGpu>() as u64,
        usage: vwgpu::BufferUsages::UNIFORM,
        mapped_at_creation: true,
    });
    params_buf
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::bytes_of(&params));
    params_buf.unmap();
    let count_bg = device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_count_bg"),
        layout: count_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: stats_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let ring = (0..ring_len)
        .map(|_| StatsReadback {
            buf: device.create_buffer(&vwgpu::BufferDescriptor {
                label: Some("particle_stats_readback"),
                size: std::mem::size_of::<u32>() as u64,
                usage: vwgpu::BufferUsages::MAP_READ | vwgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: ReadbackState::Idle,
            done: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
        .collect();
    StatsGpu {
        stats_buf,
        params_buf,
        count_bg,
        ring,
    }
}

/// A `[0, 1)` draw from a SplitMix64 stream, advancing `state`.
fn sample_rand(state: &mut u64) -> f32 {
    *state = splitmix64(*state);
    ((*state >> 40) as f32) / ((1u64 << 24) as f32)
}

fn vsub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn vcross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn vnorm(a: [f32; 3]) -> [f32; 3] {
    let l = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if l > 1e-6 {
        [a[0] / l, a[1] / l, a[2] / l]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Bake a `MESH_SAMPLE_POOL`-entry spawn-sample buffer for the `MeshSurface` /
/// `PointCloud` spawn shapes (Phase 22). For a triangle mesh, samples are
/// area-weighted (larger triangles get proportionally more) with a face normal
/// per sample. For a point cloud (`point_cloud` set, or no triangles), the
/// vertices are cycled through and the normal points radially outward from the
/// local origin. Returns `None` when there is no geometry to sample. Sampling is
/// deterministic (a fixed SplitMix64 stream) so registration is reproducible.
pub(crate) fn build_spawn_samples(
    device: &vwgpu::Device,
    positions: &[[f32; 3]],
    indices: &[u32],
    point_cloud: bool,
) -> Option<(vwgpu::Buffer, u32)> {
    let samples = compute_spawn_samples(positions, indices, point_cloud)?;
    let buf = buffer_with_data(
        device,
        "particle_spawn_samples",
        bytemuck::cast_slice(&samples),
        vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
    );
    let n = samples.len() as u32;
    Some((buf, n))
}

/// The device-free core of [`build_spawn_samples`]: bake the sample pool from
/// geometry. Split out so the area-weighting is unit-testable without a GPU.
pub(crate) fn compute_spawn_samples(
    positions: &[[f32; 3]],
    indices: &[u32],
    point_cloud: bool,
) -> Option<Vec<SpawnSampleGpu>> {
    if positions.is_empty() {
        return None;
    }
    let mut samples: Vec<SpawnSampleGpu> = Vec::with_capacity(MESH_SAMPLE_POOL as usize);
    let use_points = point_cloud || indices.len() < 3;

    if use_points {
        for i in 0..MESH_SAMPLE_POOL as usize {
            let p = positions[i % positions.len()];
            let n = vnorm(p);
            samples.push(SpawnSampleGpu {
                position: [p[0], p[1], p[2], 0.0],
                normal: [n[0], n[1], n[2], 0.0],
            });
        }
    } else {
        let tris = indices.len() / 3;
        let tri = |t: usize| -> ([f32; 3], [f32; 3], [f32; 3]) {
            (
                positions[indices[t * 3] as usize],
                positions[indices[t * 3 + 1] as usize],
                positions[indices[t * 3 + 2] as usize],
            )
        };
        // Cumulative triangle areas for area-weighted picking.
        let mut cum = Vec::with_capacity(tris);
        let mut total = 0.0f32;
        for t in 0..tris {
            let (a, b, c) = tri(t);
            let area = 0.5 * {
                let cr = vcross(vsub(b, a), vsub(c, a));
                (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt()
            };
            total += area;
            cum.push(total);
        }
        if total <= 0.0 {
            return None;
        }
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..MESH_SAMPLE_POOL {
            let r = sample_rand(&mut state) * total;
            // Binary search into the CDF for the picked triangle.
            let (mut lo, mut hi) = (0usize, tris - 1);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if cum[mid] < r {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let (a, b, c) = tri(lo);
            // Uniform barycentric sample over the triangle.
            let (mut u, mut v) = (sample_rand(&mut state), sample_rand(&mut state));
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            let w = 1.0 - u - v;
            let pos = [
                a[0] * w + b[0] * u + c[0] * v,
                a[1] * w + b[1] * u + c[1] * v,
                a[2] * w + b[2] * u + c[2] * v,
            ];
            let n = vnorm(vcross(vsub(b, a), vsub(c, a)));
            samples.push(SpawnSampleGpu {
                position: [pos[0], pos[1], pos[2], 0.0],
                normal: [n[0], n[1], n[2], 0.0],
            });
        }
    }

    Some(samples)
}

/// A shared 1-entry spawn-sample buffer bound at fixed-emit binding 3 for effects
/// that do not use a mesh spawn shape (they never index it).
pub(crate) fn build_default_spawn_samples(device: &vwgpu::Device) -> vwgpu::Buffer {
    buffer_with_data(
        device,
        "particle_spawn_samples_default",
        bytemuck::bytes_of(&SpawnSampleGpu::zeroed()),
        vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
    )
}

/// Encode this frame's emit and simulate compute passes for one effect.
pub(crate) fn dispatch_sim(
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
pub(crate) fn build_compute_pipeline(
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
pub(crate) fn buffer_with_data(
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
pub(crate) fn build_drawparams_bg(
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

/// Group-3 bind group for a custom fragment hook (Phase 33): the draw-params
/// uniform (0), the sprite texture (1) + sampler (2), the per-frame `hook_params`
/// uniform (3), and the hook's textures (4..), padded with the shared white
/// default so all [`HOOK_TEXTURES`] slots are bound. Matches `hook_drawparams_bgl`.
pub(crate) fn build_hook_draw_bg(
    device: &vwgpu::Device,
    bgl: &vwgpu::BindGroupLayout,
    params: DrawParams,
    sprite_view: &vwgpu::TextureView,
    sampler: &vwgpu::Sampler,
    hook_params_buf: &vwgpu::Buffer,
    hook_views: &[&vwgpu::TextureView],
) -> vwgpu::BindGroup {
    let dp_buf = buffer_with_data(
        device,
        "particle_hook_drawparams",
        bytemuck::bytes_of(&params),
        vwgpu::BufferUsages::UNIFORM,
    );
    let mut entries = vec![
        vwgpu::BindGroupEntry {
            binding: 0,
            resource: dp_buf.as_entire_binding(),
        },
        vwgpu::BindGroupEntry {
            binding: 1,
            resource: vwgpu::BindingResource::TextureView(sprite_view),
        },
        vwgpu::BindGroupEntry {
            binding: 2,
            resource: vwgpu::BindingResource::Sampler(sampler),
        },
        vwgpu::BindGroupEntry {
            binding: 3,
            resource: hook_params_buf.as_entire_binding(),
        },
    ];
    for (i, v) in hook_views.iter().enumerate() {
        entries.push(vwgpu::BindGroupEntry {
            binding: 4 + i as u32,
            resource: vwgpu::BindingResource::TextureView(v),
        });
    }
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_hook_draw_bg"),
        layout: bgl,
        entries: &entries,
    })
}

/// Soft-particle draw bind group (Phase 9): group 1 of the depth-read pipeline,
/// folding particles (0), draw order (1), the `DrawParams` uniform (2), the
/// billboard texture (3), and its sampler (4) into one group — so group 3 stays
/// free for the scene depth. Owns a private `DrawParams` buffer.
pub(crate) fn build_soft_draw_bg(
    device: &vwgpu::Device,
    soft_draw_bgl: &vwgpu::BindGroupLayout,
    particle_buf: &vwgpu::Buffer,
    order_buf: &vwgpu::Buffer,
    params: DrawParams,
    texture_view: &vwgpu::TextureView,
    sampler: &vwgpu::Sampler,
) -> vwgpu::BindGroup {
    let buf = buffer_with_data(
        device,
        "particle_soft_drawparams",
        bytemuck::bytes_of(&params),
        vwgpu::BufferUsages::UNIFORM,
    );
    device.create_bind_group(&vwgpu::BindGroupDescriptor {
        label: Some("particle_soft_draw_bg"),
        layout: soft_draw_bgl,
        entries: &[
            vwgpu::BindGroupEntry {
                binding: 0,
                resource: particle_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 1,
                resource: order_buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 2,
                resource: buf.as_entire_binding(),
            },
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: vwgpu::BindingResource::TextureView(texture_view),
            },
            vwgpu::BindGroupEntry {
                binding: 4,
                resource: vwgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

/// Draw bind group binding the particle buffer (group 1, binding 0) and the draw
/// order buffer (binding 1).
pub(crate) fn build_draw_bg(
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
pub(crate) fn build_order_buffer(device: &vwgpu::Device, n: u32) -> vwgpu::Buffer {
    let indices: Vec<u32> = (0..n).collect();
    buffer_with_data(
        device,
        "particle_order",
        bytemuck::cast_slice(&indices),
        vwgpu::BufferUsages::STORAGE | vwgpu::BufferUsages::COPY_DST,
    )
}

/// Smallest power of two >= `v` (and >= 1).
pub(crate) fn next_pow2(v: u32) -> u32 {
    let mut n = 1u32;
    while n < v {
        n <<= 1;
    }
    n
}

/// Encode an `f32` as an IEEE-754 half-precision `u16`. Dependency-free so the
/// crate needs no `half`; handles the ranges gradients produce (small positive
/// colours and sizes), with round-to-nearest and flush-to-zero on underflow.
pub(crate) fn f32_to_f16(value: f32) -> u16 {
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

/// Bake a gradient into a `RAMP_WIDTH` x 2 `Rgba16Float` LUT sampled by
/// normalized age. Row 0 (v = 0.25) holds rgb = colour, a = size scale; row 1
/// (v = 0.75) holds r = alpha (Phase 15). The billboard shaders sample both
/// rows; the mesh and trail shaders sample row 0 only. Two rows so the alpha
/// ramp gets its own channel without disturbing the colour/size packing.
pub(crate) fn build_ramp_lut(
    device: &vwgpu::Device,
    queue: &vwgpu::Queue,
    gradient: &crate::effect::Gradient,
) -> vwgpu::Texture {
    // Row 0: colour + size. Row 1: alpha in r, zero elsewhere.
    let mut texels: Vec<u16> = Vec::with_capacity(RAMP_WIDTH as usize * 4 * 2);
    for i in 0..RAMP_WIDTH {
        let t = i as f32 / (RAMP_WIDTH - 1) as f32;
        let c = gradient.sample_colour(t);
        let s = gradient.sample_size(t);
        texels.push(f32_to_f16(c[0]));
        texels.push(f32_to_f16(c[1]));
        texels.push(f32_to_f16(c[2]));
        texels.push(f32_to_f16(s));
    }
    for i in 0..RAMP_WIDTH {
        let t = i as f32 / (RAMP_WIDTH - 1) as f32;
        let a = gradient.sample_alpha(t);
        texels.push(f32_to_f16(a));
        texels.push(0);
        texels.push(0);
        texels.push(0);
    }

    let texture = device.create_texture(&vwgpu::TextureDescriptor {
        label: Some("particle_ramp_lut"),
        size: vwgpu::Extent3d {
            width: RAMP_WIDTH,
            height: 2,
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
            rows_per_image: Some(2),
        },
        vwgpu::Extent3d {
            width: RAMP_WIDTH,
            height: 2,
            depth_or_array_layers: 1,
        },
    );
    texture
}

/// Build a 1x1 `Rgba8UnormSrgb` texture of a single colour (the default sprite
/// texture for untextured effects).
pub(crate) fn build_solid_texture(
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
pub(crate) fn make_ramp_bg(
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
pub(crate) fn build_gen_gpu(
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
pub(crate) fn build_fixed_gpu(
    device: &vwgpu::Device,
    capacity: u32,
    order_buf: &vwgpu::Buffer,
    emit_bgl: &vwgpu::BindGroupLayout,
    sim_bgl: &vwgpu::BindGroupLayout,
    draw_bgl: &vwgpu::BindGroupLayout,
    ramp_bg: vwgpu::BindGroup,
    ramp_lut: Option<vwgpu::Texture>,
    drawparams_bg: vwgpu::BindGroup,
    spawn_samples: &vwgpu::Buffer,
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
            vwgpu::BindGroupEntry {
                binding: 3,
                resource: spawn_samples.as_entire_binding(),
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
pub(crate) fn build_trail_gpu(
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
pub(crate) fn build_event_buffer(device: &vwgpu::Device, child_capacity: u32) -> vwgpu::Buffer {
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
pub(crate) fn build_sim_parent_bg(
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
pub(crate) fn build_subemit_bg(
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
pub(crate) fn build_sort_gpu(
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
pub(crate) fn dispatch_sort(
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
pub(crate) fn next_spawn_count(spawn_accum: &mut f32, burst_done: &mut bool, rate: SpawnRate, dt: f32) -> u32 {
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
pub(crate) fn build_props_data(
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_emit_params(
    emitter: &Emitter,
    origin: [f32; 3],
    capacity: u32,
    spawn_count: u32,
    rng_seed: u32,
    has_events: bool,
    inherit_velocity: f32,
    sample_count: u32,
    rotation: f32,
    rotation_variance: f32,
    size_variance: f32,
    spawn_offset: [f32; 3],
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
    // ctrl.w carries the mesh spawn-sample pool size (0 for non-mesh shapes).
    p.ctrl = [rng_seed, spawn_count, capacity, sample_count];
    p.rot = [rotation, rotation_variance, size_variance, 0.0];
    p.spawn_offset = [spawn_offset[0], spawn_offset[1], spawn_offset[2], 0.0];

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
        // Mesh-surface / point-cloud (Phase 22): the emit kernel indexes the
        // spawn-sample pool; spawn_a.x is the normal lift-off speed.
        SpawnShape::MeshSurface {
            normal_velocity, ..
        }
        | SpawnShape::PointCloud {
            normal_velocity, ..
        } => (5u32, [normal_velocity, 0.0, 0.0, 0.0], [0.0; 4]),
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

/// Pack one [`Modifier`] into its GPU slot. The `kind` discriminant and the
/// `data` lane layout here must match the `case` arm in `particle_modifiers.wgsl`.
/// This and that `case` are the two places a new modifier touches on either side.
pub(crate) fn pack_modifier(m: &Modifier) -> GpuModifier {
    let flag = |b: bool| if b { 1.0 } else { 0.0 };
    let mut g = GpuModifier::zeroed();
    match *m {
        Modifier::Accel(a) => {
            g.kind = 0;
            g.data0 = [a[0], a[1], a[2], 0.0];
        }
        Modifier::Drag(c) => {
            g.kind = 1;
            g.data0 = [c, 0.0, 0.0, 0.0];
        }
        Modifier::PointAttractor {
            position,
            strength,
            falloff,
        } => {
            g.kind = 2;
            g.data0 = [position[0], position[1], position[2], 0.0];
            g.data1 = [strength, falloff, 0.0, 0.0];
        }
        Modifier::CurlNoise {
            scale,
            strength,
            speed,
        } => {
            g.kind = 3;
            g.data1 = [scale, strength, speed, 0.0];
        }
        Modifier::Radial {
            center,
            strength,
            falloff,
        } => {
            g.kind = 4;
            g.data0 = [center[0], center[1], center[2], 0.0];
            g.data1 = [strength, falloff, 0.0, 0.0];
        }
        Modifier::Vortex {
            center,
            axis,
            strength,
            falloff,
        } => {
            g.kind = 5;
            g.data0 = [center[0], center[1], center[2], 0.0];
            g.data1 = [axis[0], axis[1], axis[2], 0.0];
            g.data2 = [strength, falloff, 0.0, 0.0];
        }
        Modifier::Conform {
            center,
            radius,
            attraction,
            influence,
            sticky,
        } => {
            g.kind = 6;
            g.data0 = [center[0], center[1], center[2], radius];
            g.data1 = [attraction, influence, sticky, 0.0];
        }
        Modifier::KillSphere {
            center,
            radius,
            kill_inside,
        } => {
            g.kind = 7;
            g.data0 = [center[0], center[1], center[2], radius];
            g.data1 = [flag(kill_inside), 0.0, 0.0, 0.0];
        }
        Modifier::KillAabb {
            min,
            max,
            kill_inside,
        } => {
            g.kind = 8;
            g.data0 = [min[0], min[1], min[2], 0.0];
            g.data1 = [max[0], max[1], max[2], 0.0];
            g.data2 = [flag(kill_inside), 0.0, 0.0, 0.0];
        }
    }
    g
}

pub(crate) fn build_sim_params(
    modifiers: &[Modifier],
    dt: f32,
    time: f32,
    event: [u32; 4],
    space: [f32; 3],
) -> SimParamsGpu {
    let mut p = SimParamsGpu::zeroed();
    p.event = event;
    p.space = [space[0], space[1], space[2], 0.0];
    let n = modifiers.len().min(MAX_MODIFIERS);
    for (i, m) in modifiers.iter().take(n).enumerate() {
        p.modifiers[i] = pack_modifier(m);
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

    #[test]
    fn point_cloud_samples_cycle_vertices_with_radial_normals() {
        // Phase 22: a point cloud reuses its vertices, normals point outward.
        let pts = vec![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, -3.0]];
        let s = compute_spawn_samples(&pts, &[], true).unwrap();
        assert_eq!(s.len() as u32, MESH_SAMPLE_POOL);
        // First three samples are the three vertices in order.
        assert_eq!(&s[0].position[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&s[1].position[..3], &[0.0, 2.0, 0.0]);
        // Normal of vertex 0 is +X (radial from origin).
        assert!((s[0].normal[0] - 1.0).abs() < 1e-5);
        // Vertex 1's radial normal is +Y.
        assert!((s[1].normal[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mesh_surface_sampling_is_area_weighted() {
        // Two triangles sharing a plane: triangle A is far larger than B, so it
        // should receive the large majority of area-weighted samples.
        // Triangle A: big, in +X. Triangle B: tiny, in -X.
        let positions = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0], // A (area 50)
            [-1.0, 0.0, 0.0],
            [-1.1, 0.0, 0.0],
            [-1.0, 0.1, 0.0], // B (area ~0.005)
        ];
        let indices = vec![0, 1, 2, 3, 4, 5];
        let s = compute_spawn_samples(&positions, &indices, false).unwrap();
        assert_eq!(s.len() as u32, MESH_SAMPLE_POOL);
        let in_a = s.iter().filter(|x| x.position[0] > -0.5).count();
        // Overwhelmingly from the large triangle; normals are +Z (or -Z) face.
        assert!(
            in_a as f32 / s.len() as f32 > 0.98,
            "area weighting should favour the large triangle, got {in_a}"
        );
        assert!(s[0].normal[2].abs() > 0.99, "face normal is along Z");
    }
}
