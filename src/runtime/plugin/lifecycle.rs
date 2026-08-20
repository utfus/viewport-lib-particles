use std::hash::{Hash, Hasher};

use viewport_lib::plugin_api::shared_wgsl::{
    SHARED_BINDINGS_WGSL, SHARED_DEPTH_READ_WGSL, SHARED_MASK_WGSL, SHARED_OIT_WGSL,
    SHARED_PICK_WGSL,
};
use viewport_lib::plugin_api::{OIT_ACCUM_BLEND, OIT_REVEAL_BLEND, SharedBindings};
use viewport_lib::resources::{
    HDR_COLOR_FORMAT, MASK_COLOR_FORMAT, PICK_COLOR_FORMAT, PICK_DEPTH_CHANNEL_FORMAT,
    SCENE_DEPTH_FORMAT, SHADOW_DEPTH_FORMAT,
};
use viewport_lib::wgpu as vwgpu;

use crate::codegen;
use crate::effect::{
    EffectAsset, EffectId, EffectProgram, MeshAlign, Orient, ParticleBlend, ParticleMeshId,
    ParticleRender, ParticleTextureId, SpawnShape, TextureMode,
};

use super::*;

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
    pub fn add_effect(&mut self, device: &vwgpu::Device, asset: EffectAsset) -> EffectId {
        // Phase 22: bake a spawn-sample pool if the (fixed-function) emitter draws
        // its positions from an uploaded mesh's surface or vertices. Meshes must
        // be uploaded before the effect is registered.
        let (spawn_samples, spawn_sample_count) = match (&asset.program, asset.emitter.spawn) {
            (None, SpawnShape::MeshSurface { mesh, .. }) => self
                .meshes
                .get(mesh.0 as usize)
                .and_then(|m| build_spawn_samples(device, &m.positions, &m.indices, false))
                .map(|(b, n)| (Some(b), n))
                .unwrap_or((None, 0)),
            (None, SpawnShape::PointCloud { mesh, .. }) => self
                .meshes
                .get(mesh.0 as usize)
                .and_then(|m| build_spawn_samples(device, &m.positions, &m.indices, true))
                .map(|(b, n)| (Some(b), n))
                .unwrap_or((None, 0)),
            _ => (None, 0),
        };
        let id = EffectId(self.effects.len() as u32);
        self.effects.push(RegisteredEffect {
            asset,
            removed: false,
            spawn_samples,
            spawn_sample_count,
        });
        id
    }

    /// Number of registered effects (including tombstoned / removed slots).
    pub fn effect_count(&self) -> usize {
        self.effects.len()
    }

    /// A cheap, cloneable handle to this plugin's live particle-count stats.
    /// Clone it out before installing the plugin with
    /// [`ViewportRenderer::with_item_type_plugin`](viewport_lib::renderer::ViewportRenderer::with_item_type_plugin),
    /// which takes ownership; the plugin updates the shared store inside its
    /// `prepare`, so the handle keeps reporting after install. Enable the readback
    /// per frame with [`ParticleItems::with_stats_readback`]. See
    /// [`ParticleStatsHandle`].
    pub fn stats_handle(&self) -> ParticleStatsHandle {
        ParticleStatsHandle {
            inner: self.stats.clone(),
        }
    }

    /// Live simulation stats for one instance, from the shared store this plugin
    /// fills each frame. Equivalent to [`ParticleStatsHandle::instance_stats`];
    /// usable when a plugin handle is held directly, before install. Returns
    /// `None` until a readback for the `(effect, key)` has completed.
    pub fn instance_stats(&self, effect: EffectId, key: u64) -> Option<ParticleStats> {
        self.stats_handle().instance_stats(effect, key)
    }

    /// Live simulation stats for an effect, summed over its instances. Equivalent
    /// to [`ParticleStatsHandle::effect_stats`].
    pub fn effect_stats(&self, effect: EffectId) -> Option<ParticleStats> {
        self.stats_handle().effect_stats(effect)
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
    pub(crate) fn purge_effect_gpu(&mut self, idx: usize) {
        let effect = idx as u32;
        self.instances.retain(|k, _| k.effect != effect);
        // Drop this effect's live-count readbacks from the shared store.
        if let Ok(mut g) = self.stats.lock() {
            g.entries.retain(|(eff, _), _| *eff != effect);
        }
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
            // CPU copies for Phase 22 mesh-surface / point-cloud spawn sampling.
            positions: positions.to_vec(),
            indices: indices.to_vec(),
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
    pub(crate) fn ensure_instance_gpu(
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
        let orient_mode = self.effects[idx].asset.orient;
        let screen_space_size = self.effects[idx].asset.screen_space_size;
        let soft = self.effects[idx].asset.soft;
        let rotation_rate = self.effects[idx].asset.rotation_rate;

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
        // Billboard orientation + screen-space size (Phase 15). Only the billboard
        // route reads these; the mesh and trail shaders ignore them.
        let is_billboard = matches!(render, ParticleRender::Billboard { .. });
        let (orient, orient_param) = if is_billboard {
            match orient_mode {
                Orient::ParallelCameraPlane => (0u32, 0.0),
                Orient::FaceCamera => (1u32, 0.0),
                Orient::AlongVelocity => (2u32, 0.0),
                Orient::Fixed { radians } => (3u32, radians),
            }
        } else {
            (0u32, 0.0)
        };
        let screen_size = if is_billboard && screen_space_size { 1u32 } else { 0u32 };
        let soft_dist = if is_billboard { soft.unwrap_or(0.0) } else { 0.0 };
        let roll_rate = if is_billboard { rotation_rate } else { 0.0 };
        let dp = DrawParams {
            stretch,
            align,
            trail_width,
            trail_segments,
            tex_mode,
            flip_cols,
            flip_rows,
            flip_fps,
            orient,
            orient_param,
            screen_size,
            soft_dist,
            roll_rate,
            _pad: [0.0; 3],
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
            None => {
                // Phase 22: bind this effect's baked spawn-sample pool, or the
                // shared 1-entry default when it has none. Borrow is confined to
                // this arm (the `Some` arm needs `&mut self` for the pipeline cache).
                let default_samples = self.default_spawn_samples.as_ref().unwrap();
                let spawn_samples = self.effects[idx]
                    .spawn_samples
                    .as_ref()
                    .unwrap_or(default_samples);
                EffectGpu::Fixed(build_fixed_gpu(
                    device,
                    capacity,
                    &order_buf,
                    emit_bgl,
                    sim_bgl,
                    draw_bgl,
                    ramp_bg,
                    ramp_lut,
                    drawparams_bg,
                    spawn_samples,
                ))
            }
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

        // Soft-particle (Phase 9) group-1 bind group: built when the effect is a
        // soft billboard, folding particles + order + draw params + sprite so the
        // depth-read pipeline's group 3 carries the scene depth.
        let soft_draw_bg = if soft.is_some() && is_billboard {
            let soft_draw_bgl = self.soft_draw_bgl.clone().unwrap();
            let sprite_view = match texture.and_then(|t| self.textures.get(t.0 as usize)) {
                Some(t) => &t.view,
                None => self.default_tex_view.as_ref().unwrap(),
            };
            let sprite_sampler = self.ramp_sampler.as_ref().unwrap();
            Some(build_soft_draw_bg(
                device,
                &soft_draw_bgl,
                gpu.particle_buf(),
                &order_buf,
                dp,
                sprite_view,
                sprite_sampler,
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
                soft_draw_bg,
                sort,
                stats: None,
                pick_buf,
                pick_bg,
                spawn_accum: 0.0,
                burst_done: false,
                last_origin: None,
                dropped: false,
                sim_frame: 0,
                elapsed: 0.0,
                schedule_fired: 0,
                last_seen: 0,
                local_time: 0.0,
                warmed: false,
            },
        );
    }

    /// Build parent `parent_idx`'s sub-emitter state if it does not exist yet:
    /// the shared event buffer plus the parent-sim and child-emit bind groups.
    /// Ensures both the parent and child effects have GPU state first. A no-op if
    /// the parent has no sub-emitter, or if either end carries a program (the
    /// feature is fixed-function only).
    pub(crate) fn ensure_subemitter_gpu(
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
    pub(crate) fn acquire_transient_slot(&mut self, effect: usize, max: usize) -> Option<u64> {
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
    pub(crate) fn gen_pipelines(
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

impl ParticlePlugin {
    pub(crate) fn build_gpu(&mut self, device: &vwgpu::Device, shared: &SharedBindings<'_>) {
        // MSAA sample count of the renderer's HDR scene target. The HDR
        // colour-pass pipelines below build with this; the interaction passes
        // (shadow, pick, mask) stay single-sampled. Captured into a local so the
        // pipeline-building closures don't borrow `self`.
        let sample_count = shared.sample_count.max(1);
        self.sample_count = sample_count;
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

        // Binding 3 (Phase 22): the read-only mesh spawn-sample pool. Every fixed
        // emit bind group binds one (a shared 1-entry default when unused).
        let storage_ro = |binding| vwgpu::BindGroupLayoutEntry {
            binding,
            visibility: vwgpu::ShaderStages::COMPUTE,
            ty: vwgpu::BindingType::Buffer {
                ty: vwgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let emit_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_emit_bgl"),
            entries: &[storage_rw(0), uniform(1), storage_rw(2), storage_ro(3)],
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
            include_str!("../shaders/particle_emit.wgsl"),
            "emit_main",
            "particle_emit",
            &emit_bgl,
        );
        // Prepend the shared curl-noise helpers (so CurlNoise can call them) and
        // the shared modifier definitions (the `Modifier` struct + `eval_modifier`)
        // that both simulate shaders use.
        let sim_src = format!(
            "{}\n{}\n{}",
            codegen::NOISE_WGSL,
            include_str!("../shaders/particle_modifiers.wgsl"),
            include_str!("../shaders/particle_sim.wgsl")
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
            include_str!("../shaders/particle_subemit.wgsl"),
            "emit_main",
            "particle_subemit",
            &subemit_bgl,
        );
        let sim_parent_src = format!(
            "{}\n{}\n{}",
            codegen::NOISE_WGSL,
            include_str!("../shaders/particle_modifiers.wgsl"),
            include_str!("../shaders/particle_sim_parent.wgsl")
        );
        let sim_parent_pipeline =
            compute(&sim_parent_src, "sim_main", "particle_sim_parent", &sim_parent_bgl);

        let append_pipeline = compute(
            include_str!("../shaders/particle_history.wgsl"),
            "history_main",
            "particle_history",
            &history_bgl,
        );

        // Live-count readback: particles (read) + a single-atomic stats buffer
        // (read_write) + a CountParams uniform.
        let count_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_count_bgl"),
            entries: &[storage_ro_compute(0), storage_rw(1), uniform(2)],
        });
        let count_pipeline = compute(
            include_str!("../shaders/particle_count.wgsl"),
            "count_main",
            "particle_count",
            &count_bgl,
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
            include_str!("../shaders/particle_sort_init.wgsl"),
            "init_main",
            "particle_sort_init",
            &sort_init_bgl,
        );
        let sort_stage_pipeline = compute(
            include_str!("../shaders/particle_sort.wgsl"),
            "sort_main",
            "particle_sort",
            &sort_stage_bgl,
        );

        // Draw pipelines (additive and premultiplied over).
        let draw_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("../shaders/particle_draw.wgsl")
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
                        count: sample_count,
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
            include_str!("../shaders/particle_oit.wgsl")
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
                    count: sample_count,
                    ..Default::default()
                },
                cache: None,
            },
        ));

        // Soft-particle (Phase 9) billboard pipeline for the read-only-depth
        // pass. It folds particles+order+draw-params+sprite into group 1 so
        // group 3 can carry the scene depth (we are already at four groups), and
        // its group-3 layout replicates the renderer's `depth_read_bgl` so the
        // context's prebuilt `scene_depth_bind_group` binds against it (wgpu
        // deduplicates identical layout descriptors).
        let soft_draw_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_soft_draw_bgl"),
            entries: &[
                draw_storage(0),
                draw_storage(1),
                vwgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vwgpu::ShaderStages::VERTEX | vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Buffer {
                        ty: vwgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Texture {
                        sample_type: vwgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: vwgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Sampler(vwgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        // Replica of the renderer's depth_read_bgl: binding 0 = depth texture,
        // binding 1 = non-filtering sampler, both fragment-visible.
        let depth_read_bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("particle_depth_read_bgl"),
            entries: &[
                vwgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Texture {
                        sample_type: vwgpu::TextureSampleType::Depth,
                        view_dimension: vwgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Sampler(vwgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let soft_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{SHARED_DEPTH_READ_WGSL}\n{}",
            include_str!("../shaders/particle_soft.wgsl")
        );
        let soft_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_soft_shader"),
            source: vwgpu::ShaderSource::Wgsl(soft_src.into()),
        });
        let soft_layout = vwgpu::pipeline_layout(
            device,
            "particle_soft_layout",
            &[shared.group0_layout, &soft_draw_bgl, &ramp_bgl, &depth_read_bgl],
        );
        let make_soft = |blend: vwgpu::BlendState, label: &str| {
            vwgpu::render_pipeline(
                device,
                vwgpu::RenderPipelineDesc {
                    label,
                    layout: &soft_layout,
                    vertex: vwgpu::VertexState {
                        module: &soft_shader,
                        entry_point: Some("vs"),
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(vwgpu::FragmentState {
                        module: &soft_shader,
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
                    // Test against opaque depth, read-only (the depth-read pass
                    // binds scene depth read-only and never writes it).
                    depth_stencil: Some(vwgpu::depth_stencil(
                        SCENE_DEPTH_FORMAT,
                        false,
                        vwgpu::CompareFunction::LessEqual,
                    )),
                    multisample: vwgpu::MultisampleState {
                        count: sample_count,
                        ..Default::default()
                    },
                    cache: None,
                },
            )
        };
        self.soft_pipeline_over = Some(make_soft(
            vwgpu::BlendState {
                color: over,
                alpha: over,
            },
            "particle_soft_over",
        ));
        self.soft_pipeline_additive = Some(make_soft(
            vwgpu::BlendState {
                color: one,
                alpha: one,
            },
            "particle_soft_additive",
        ));
        self.soft_draw_bgl = Some(soft_draw_bgl);
        self.depth_read_bgl = Some(depth_read_bgl);

        // Mesh render route: instance an uploaded mesh per particle. Same bind
        // group layouts plus a per-vertex position stream.
        let mesh_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{}",
            include_str!("../shaders/particle_mesh.wgsl")
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
                        count: sample_count,
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
            include_str!("../shaders/particle_trail.wgsl")
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
                        count: sample_count,
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
            include_str!("../shaders/particle_pick.wgsl")
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
                    count: INTERACTION_SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        );

        let mask_src = format!(
            "{SHARED_BINDINGS_WGSL}\n{SHARED_MASK_WGSL}\n{}",
            include_str!("../shaders/particle_mask.wgsl")
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
                    count: INTERACTION_SAMPLE_COUNT,
                    ..Default::default()
                },
                cache: None,
            },
        );

        let shadow_shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("particle_shadow_shader"),
            source: vwgpu::ShaderSource::Wgsl(include_str!("../shaders/particle_shadow.wgsl").into()),
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
                // Depth-only, but with a fragment stage that discards for a round,
                // alpha-dithered footprint (no colour targets).
                fragment: Some(vwgpu::FragmentState {
                    module: &shadow_shader,
                    entry_point: Some("fs"),
                    targets: &[],
                    compilation_options: Default::default(),
                }),
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
                    count: INTERACTION_SAMPLE_COUNT,
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
        self.count_pipeline = Some(count_pipeline);
        self.count_bgl = Some(count_bgl);
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
        self.default_spawn_samples = Some(build_default_spawn_samples(device));
    }

}
