use std::collections::HashMap;

use viewport_lib::plugin_api::{
    DepthReadContext, ItemFrameContext, ItemTypePlugin, OutlineMaskContext,
    PaintContext, PickPassContext, PluginItemCollection, ShadowCastContext, SharedBindings,
};
use viewport_lib::wgpu as vwgpu;
use viewport_lib::Frustum;

use crate::effect::{
    EventCondition, ParticleBlend, ParticleRender, PropertyDecl, SimulationSpace, TimedBurst,
};

use super::*;

impl ItemTypePlugin for ParticlePlugin {
    fn type_name(&self) -> &'static str {
        Self::TYPE_NAME
    }

    fn init_gpu(&mut self, device: &vwgpu::Device, shared: &SharedBindings<'_>) {
        self.build_gpu(device, shared);
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
        // Live-count readback, used only when the host enables it.
        let count_pipeline = self.count_pipeline.clone();
        let count_bgl = self.count_bgl.clone();
        // Shared store a `ParticleStatsHandle` reads; written on readback below.
        let stats_shared = self.stats.clone();
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
        // Advance the shared simulation clock once per frame. Kept for reference;
        // each instance now drives time-varying fields (e.g. the CurlNoise force)
        // from its own `local_time`, which advances by the instance's time-scaled
        // dt (issue 014) and is seeded by any prewarm (issue 015).
        self.time += dt;

        // Apply queued lifecycle commands (Phase 27) before simulating, so a
        // hot-reloaded or removed effect takes effect this frame.
        for cmd in &items.commands {
            match cmd {
                EffectCommand::Replace(id, asset) => self.replace_effect(*id, (**asset).clone()),
                EffectCommand::Remove(id) => self.remove_effect(*id),
            }
        }

        // Upload any inline textures (issue 017) queued by add / replace, now that
        // a queue is available and before instances build their draw bind groups.
        self.drain_pending_textures(device, queue);

        let (driven, parent_of) =
            self.build_frame_plan(device, queue, ctx, items, dt, &emit_bgl, &sim_bgl, &draw_bgl);

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
            // Per-instance simulation dt (issue 014): the frame dt scaled by this
            // instance's effective time scale (item `time_scale` × effect
            // `simulation_speed`). Everything downstream that integrates over time
            // — spawn accumulation, burst-schedule times, forces, ageing — uses
            // this in place of the raw frame dt.
            let inst_dt = dt * di.time_scale;

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
            // Timed bursts layered on the continuous rate (fixed-function only).
            // Copied out so the immutable `self.effects` borrow ends before the
            // mutable `self.instances` borrow below.
            let schedule: Vec<TimedBurst> = if self.effects[idx].asset.program.is_none() {
                self.effects[idx].asset.spawn_schedule.clone()
            } else {
                Vec::new()
            };

            // Prewarm on first activation (issue 015): fast-forward a fresh
            // instance to steady state before its first draw so ambient set
            // dressing appears full, not filling in. Sub-emitter parents/children
            // are exempt — their shared event buffer makes a mid-stream warmup
            // unsafe — and simply mark themselves warmed.
            if !self.instances.get(&ikey).unwrap().warmed {
                let is_sub = self.effects[idx].asset.sub_emitter.is_some()
                    || parent_of.contains_key(&idx);
                if di.warmup > 0.0 && !is_sub {
                    self.prewarm_instance(
                        device,
                        queue,
                        ikey,
                        idx,
                        di.warmup,
                        di.seed,
                        ctx.frame_index,
                        origin,
                        capacity,
                        groups,
                        &emit_pipeline,
                        &sim_pipeline,
                    );
                } else {
                    self.instances.get_mut(&ikey).unwrap().warmed = true;
                }
            }

            // Advance this instance's own spawn state and record its origin (for
            // the next frame's local-space delta). Simulating clears the culled
            // "dropped" flag so a future cull drops its particles again, bumps the
            // per-instance sim-frame counter used by the deterministic seed, marks
            // the instance seen this frame (issue 013 idle GC), and advances its
            // time-scaled local clock (issue 014).
            let (spawn_count, sim_frame, time) = {
                let inst = self.instances.get_mut(&ikey).unwrap();
                inst.last_origin = Some(origin);
                inst.dropped = false;
                inst.last_seen = ctx.frame_index;
                let f = inst.sim_frame;
                inst.sim_frame = inst.sim_frame.wrapping_add(1);
                let mut count = next_spawn_count(&mut inst.spawn_accum, &mut inst.burst_done, rate, inst_dt);
                // Fire any timed burst whose time this instance has now reached,
                // once each (tracked by a per-instance bitmask).
                if !schedule.is_empty() {
                    inst.elapsed += inst_dt;
                    for (bi, b) in schedule.iter().enumerate() {
                        let bit = 1u64 << bi.min(63);
                        if inst.schedule_fired & bit == 0 && inst.elapsed >= b.time {
                            count = count.saturating_add(b.count);
                            inst.schedule_fired |= bit;
                        }
                    }
                }
                inst.local_time += inst_dt;
                (count, f, inst.local_time)
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

            // Custom fragment hook (Phase 33): refresh the engine-reserved params
            // lane (lane 0 `.w` = time) each frame. The effect's own params sit in
            // later lanes, written once when the instance was built.
            if let Some(cd) = self.instances.get(&ikey).and_then(|i| i.custom_draw.as_ref()) {
                let lane0 = [0.0f32, 0.0, 0.0, time];
                queue.write_buffer(&cd.params_buf, 0, bytemuck::bytes_of(&lane0));
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
                    let modifiers = item
                        .and_then(|it| it.modifiers.as_deref())
                        .unwrap_or(&asset.modifiers);
                    let inherit = child_link.as_ref().map(|(_, s)| *s).unwrap_or(0.0);
                    // Phase 22: size of this effect's baked spawn-sample pool
                    // (0 for non-mesh shapes; the emit kernel skips the mesh path).
                    let sample_count = self.effects[idx].spawn_sample_count;
                    let emit = build_emit_params(
                        emitter,
                        origin,
                        capacity,
                        spawn_count,
                        rng_seed,
                        child_link.is_some(),
                        inherit,
                        sample_count,
                        asset.rotation,
                        asset.rotation_variance,
                        asset.size_variance,
                        asset.spawn_offset,
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
                    let sim = build_sim_params(modifiers, inst_dt, time, event, space_delta);
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
                        dt: inst_dt,
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

            // Live particle-count readback. Build the per-instance stats state on
            // first request; each frame drain any completed map, advance a prior
            // copy into a map, then reset, count, and copy into a free ring
            // buffer. The read lags about two frames but never stalls.
            if items.stats_readback {
                if let (Some(count_pipeline), Some(count_bgl)) =
                    (count_pipeline.as_ref(), count_bgl.as_ref())
                {
                    let inst = self.instances.get_mut(&ikey).unwrap();
                    if inst.stats.is_none() {
                        inst.stats = Some(build_stats_gpu(
                            device,
                            count_bgl,
                            inst.gpu.particle_buf(),
                            capacity,
                            STATS_RING,
                        ));
                    }
                    let stats = inst.stats.as_mut().unwrap();

                    // Drain completed maps and advance prior copies into maps.
                    for rb in &mut stats.ring {
                        match rb.state {
                            ReadbackState::Mapping
                                if rb.done.load(std::sync::atomic::Ordering::Acquire) =>
                            {
                                let live = {
                                    let view = rb.buf.slice(..).get_mapped_range();
                                    u32::from_le_bytes(view[0..4].try_into().unwrap())
                                };
                                rb.buf.unmap();
                                rb.done.store(false, std::sync::atomic::Ordering::Release);
                                rb.state = ReadbackState::Idle;
                                // Publish to the shared store the host handle reads.
                                if let Ok(mut g) = stats_shared.lock() {
                                    g.entries.insert(
                                        (ikey.effect, ikey.key),
                                        StatEntry { live, capacity },
                                    );
                                }
                            }
                            ReadbackState::Copied => {
                                // The copy was encoded on a previous frame and has
                                // since been submitted, so mapping is safe now.
                                let done = rb.done.clone();
                                rb.buf.slice(..).map_async(
                                    vwgpu::MapMode::Read,
                                    move |res| {
                                        if res.is_ok() {
                                            done.store(true, std::sync::atomic::Ordering::Release);
                                        }
                                    },
                                );
                                rb.state = ReadbackState::Mapping;
                            }
                            _ => {}
                        }
                    }

                    // Count this frame's live particles into an idle ring buffer.
                    if let Some(slot) = stats
                        .ring
                        .iter()
                        .position(|rb| rb.state == ReadbackState::Idle)
                    {
                        queue.write_buffer(&stats.stats_buf, 0, bytemuck::bytes_of(&0u32));
                        {
                            let mut pass =
                                encoder.begin_compute_pass(&vwgpu::ComputePassDescriptor {
                                    label: Some("particle_count_pass"),
                                    timestamp_writes: None,
                                });
                            pass.set_pipeline(count_pipeline);
                            pass.set_bind_group(0, &stats.count_bg, &[]);
                            pass.dispatch_workgroups(groups, 1, 1);
                        }
                        encoder.copy_buffer_to_buffer(
                            &stats.stats_buf,
                            0,
                            &stats.ring[slot].buf,
                            0,
                            std::mem::size_of::<u32>() as u64,
                        );
                        stats.ring[slot].state = ReadbackState::Copied;
                    }
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

        // Idle GC (issue 013): reclaim steady instances the host has stopped
        // submitting for `IDLE_RETIRE_FRAMES` frames, bounding `self.instances` to
        // the live working set across many scene reloads. Reaped instances are, by
        // definition, not in `driven`/`draw_list` this frame, so their GPU buffers
        // are safe to drop. Pooled transients and sub-emitter instances are exempt
        // (they retire via their own paths / stay single-instance).
        let frame = ctx.frame_index;
        let effects = &self.effects;
        let mut reaped: Vec<(u32, u64)> = Vec::new();
        self.instances.retain(|k, inst| {
            if k.key & TRANSIENT_KEY_BIT != 0 {
                return true;
            }
            let idx = k.effect as usize;
            if idx < effects.len()
                && (effects[idx].asset.sub_emitter.is_some() || parent_of.contains_key(&idx))
            {
                return true;
            }
            let keep = frame.saturating_sub(inst.last_seen) <= IDLE_RETIRE_FRAMES;
            if !keep {
                reaped.push((k.effect, k.key));
            }
            keep
        });
        // Drop any live-count readback entries for reclaimed instances so the
        // stats handle stops reporting a dead `(effect, key)`.
        if !reaped.is_empty() {
            if let Ok(mut g) = stats_shared.lock() {
                for key in &reaped {
                    g.entries.remove(key);
                }
            }
        }

        if any {
            vec![encoder.finish()]
        } else {
            Vec::new()
        }
    }

    fn paint(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
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
                    // Soft billboards draw in the read-only-depth pass
                    // (paint_depth_read), never here.
                    if eff.asset.soft.is_some() {
                        continue;
                    }
                    // Alpha/Premultiplied billboards route through the OIT pass
                    // (paint_transparent). Only additive draws here.
                    if !additive {
                        continue;
                    }
                    // Custom fragment hook (Phase 33): swap in the effect's hooked
                    // pipeline and group-3 hook bind group; else the shared built-in.
                    if let Some(cd) = inst.custom_draw.as_ref() {
                        pass.set_pipeline(&cd.pipelines.additive);
                        pass.set_bind_group(3, &cd.bind_group, &[]);
                    } else {
                        pass.set_pipeline(bb_add);
                    }
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

    fn paint_transparent(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &PaintContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let Some(oit) = self.draw_pipeline_oit.as_ref() else {
            return;
        };
        // Group 0 (camera + scene) is bound by the lib inside the OIT pass.
        // Draw only Alpha/Premultiplied billboards; additive billboards and the
        // mesh/trail routes stay in the main HDR draw (paint). The pipeline is set
        // per draw because a custom-hook effect uses its own OIT pipeline.
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
            // Soft billboards draw in the read-only-depth pass, not the OIT pass.
            if eff.asset.soft.is_some() {
                continue;
            }
            pass.set_bind_group(1, gpu.draw_bg(), &[]);
            pass.set_bind_group(2, gpu.ramp_bg(), &[]);
            // Custom fragment hook (Phase 33): hooked effects draw with their own
            // OIT pipeline + group-3 hook bind group.
            if let Some(cd) = inst.custom_draw.as_ref() {
                pass.set_pipeline(&cd.pipelines.oit);
                pass.set_bind_group(3, &cd.bind_group, &[]);
            } else {
                pass.set_pipeline(oit);
                pass.set_bind_group(3, gpu.drawparams_bg(), &[]);
            }
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }

    fn draws_depth_read(&self) -> bool {
        // Open the read-only-depth pass only when a soft billboard effect is
        // actually being drawn this frame (Phase 9).
        self.draw_list.iter().any(|e| {
            let eff = &self.effects[e.key.effect as usize];
            eff.asset.soft.is_some() && matches!(eff.asset.render, ParticleRender::Billboard { .. })
        })
    }

    fn paint_depth_read(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        ctx: &DepthReadContext<'_>,
        _items: &dyn PluginItemCollection,
    ) {
        let (Some(soft_over), Some(soft_add)) = (
            self.soft_pipeline_over.as_ref(),
            self.soft_pipeline_additive.as_ref(),
        ) else {
            return;
        };
        // Group 0 (camera + scene) is bound by the lib on pass entry. Group 3 is
        // the scene depth, from the context's prebuilt bind group (its layout
        // matches our replicated `depth_read_bgl`). Each soft effect picks the
        // over / additive blend by its own blend mode.
        pass.set_bind_group(3, ctx.scene_depth_bind_group, &[]);
        for entry in &self.draw_list {
            let eff = &self.effects[entry.key.effect as usize];
            if eff.asset.soft.is_none()
                || !matches!(eff.asset.render, ParticleRender::Billboard { .. })
            {
                continue;
            }
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            let Some(soft_bg) = inst.soft_draw_bg.as_ref() else {
                continue;
            };
            let pipeline = if matches!(eff.asset.blend, ParticleBlend::Additive) {
                soft_add
            } else {
                soft_over
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, soft_bg, &[]);
            pass.set_bind_group(2, inst.gpu.ramp_bg(), &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }

    fn render_pick(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        _items: &dyn PluginItemCollection,
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

    fn outline_mask(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &OutlineMaskContext<'_>,
        _items: &dyn PluginItemCollection,
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

    fn cast_shadow_pass(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &ShadowCastContext<'_>,
        _items: &dyn PluginItemCollection,
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

impl ParticlePlugin {
    /// Build this frame's ordered list of instances to simulate: collect steady
    /// instances, age/spawn transients, then cull and apply the simulation budget.
    fn build_frame_plan(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        ctx: &ItemFrameContext<'_>,
        items: &ParticleItems,
        dt: f32,
        emit_bgl: &vwgpu::BindGroupLayout,
        sim_bgl: &vwgpu::BindGroupLayout,
        draw_bgl: &vwgpu::BindGroupLayout,
    ) -> (Vec<DrivenInstance>, HashMap<usize, usize>) {
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
                let asset = &self.effects[idx].asset;
                let time_scale = (it.time_scale * asset.simulation_speed).max(0.0);
                let warmup = it.warmup.unwrap_or_else(|| {
                    if asset.prewarm {
                        max_particle_lifetime(asset)
                    } else {
                        0.0
                    }
                });
                driven.push(DrivenInstance {
                    ikey,
                    origin: it.position,
                    pick_id: it.settings.pick_id.0 as u32,
                    selected: it.settings.selected,
                    cast_shadows: it.settings.cast_shadows,
                    item_idx: Some(i),
                    seed: it.seed,
                    time_scale,
                    warmup,
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
            self.ensure_instance_gpu(idx, ikey.key, device, queue, emit_bgl, sim_bgl, draw_bgl);
            let capacity = self.effects[idx].asset.capacity;
            {
                let inst = self.instances.get_mut(&ikey).unwrap();
                zero_particle_buffer(queue, inst.gpu.particle_buf(), capacity);
                inst.spawn_accum = 0.0;
                inst.burst_done = false;
                inst.last_origin = None;
                inst.dropped = false;
                inst.sim_frame = 0;
                // Reset the timed-burst schedule so a recycled slot replays it
                // from the start instead of treating its bursts as already fired.
                inst.elapsed = 0.0;
                inst.schedule_fired = 0;
                // A recycled slot plays fresh from t=0; transients never prewarm.
                inst.local_time = 0.0;
                inst.warmed = true;
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
            // Transients inherit only the effect-level simulation_speed (no
            // per-item time scale) and never prewarm.
            let time_scale = self.effects[t.ikey.effect as usize]
                .asset
                .simulation_speed
                .max(0.0);
            driven.push(DrivenInstance {
                ikey: t.ikey,
                origin: t.origin,
                pick_id: 0,
                selected: false,
                cast_shadows: t.cast_shadows,
                item_idx: None,
                seed: t.seed,
                time_scale,
                warmup: 0.0,
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

        // Idle-GC bookkeeping (issue 013): mark every submitted steady instance
        // seen this frame — whether or not it survived culling or the sim budget —
        // so the sweep only reaps instances the host has stopped submitting.
        // Instances created for the first time this frame don't exist yet; they
        // stamp themselves in the simulate loop.
        for ikey in &seen {
            if let Some(inst) = self.instances.get_mut(ikey) {
                inst.last_seen = ctx.frame_index;
            }
        }
        (driven, parent_of)
    }

    /// Fast-forward a fresh instance to steady state (issue 015): step its
    /// emit+simulate kernels over `warmup` seconds before its first draw, so a
    /// looping ambient effect presents full instead of visibly filling in.
    ///
    /// Runs at most [`MAX_PREWARM_STEPS`] sub-steps, each its own tiny command
    /// buffer submitted immediately (a single uniform buffer per instance can't
    /// carry distinct per-step params within one command buffer, since queued
    /// writes all land before it executes). Seeds from the *registered* asset, so
    /// it ignores this frame's per-frame overrides — a one-time activation seed.
    #[allow(clippy::too_many_arguments)]
    fn prewarm_instance(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        ikey: InstanceKey,
        idx: usize,
        warmup: f32,
        seed: Option<u64>,
        frame_index: u64,
        origin: [f32; 3],
        capacity: u32,
        groups: u32,
        emit_pipeline: &vwgpu::ComputePipeline,
        sim_pipeline: &vwgpu::ComputePipeline,
    ) {
        // Snapshot everything the loop needs from the effect so the mutable
        // per-step borrow of `self.instances` has no live borrow of `self.effects`.
        let asset = self.effects[idx].asset.clone();
        let sample_count = self.effects[idx].spawn_sample_count;
        let rate = match &asset.program {
            Some(p) => p.rate,
            None => asset.emitter.rate,
        };

        let steps = ((warmup / (1.0 / 60.0)).ceil() as u32).clamp(1, MAX_PREWARM_STEPS);
        let sub_dt = warmup / steps as f32;

        for _ in 0..steps {
            let (spawn_count, sim_frame, time) = {
                let inst = self.instances.get_mut(&ikey).unwrap();
                let f = inst.sim_frame;
                inst.sim_frame = inst.sim_frame.wrapping_add(1);
                let mut count =
                    next_spawn_count(&mut inst.spawn_accum, &mut inst.burst_done, rate, sub_dt);
                if asset.program.is_none() && !asset.spawn_schedule.is_empty() {
                    inst.elapsed += sub_dt;
                    for (bi, b) in asset.spawn_schedule.iter().enumerate() {
                        let bit = 1u64 << bi.min(63);
                        if inst.schedule_fired & bit == 0 && inst.elapsed >= b.time {
                            count = count.saturating_add(b.count);
                            inst.schedule_fired |= bit;
                        }
                    }
                }
                inst.local_time += sub_dt;
                (count, f, inst.local_time)
            };
            let rng_seed = instance_rng_seed(seed, frame_index, idx, ikey.key, sim_frame);

            let mut encoder = device.create_command_encoder(&vwgpu::CommandEncoderDescriptor {
                label: Some("particle_prewarm_encoder"),
            });
            let inst = self.instances.get(&ikey).unwrap();
            match &inst.gpu {
                EffectGpu::Fixed(g) => {
                    let emit = build_emit_params(
                        &asset.emitter,
                        origin,
                        capacity,
                        spawn_count,
                        rng_seed,
                        false,
                        0.0,
                        sample_count,
                        asset.rotation,
                        asset.rotation_variance,
                        asset.size_variance,
                        asset.spawn_offset,
                    );
                    let sim = build_sim_params(&asset.modifiers, sub_dt, time, [0; 4], [0.0; 3]);
                    queue.write_buffer(&g.emit_buf, 0, bytemuck::bytes_of(&emit));
                    queue.write_buffer(&g.sim_buf, 0, bytemuck::bytes_of(&sim));
                    queue.write_buffer(&g.budget_buf, 0, bytemuck::bytes_of(&0u32));
                    dispatch_sim(
                        &mut encoder,
                        emit_pipeline,
                        &g.emit_bg,
                        sim_pipeline,
                        &g.sim_bg,
                        groups,
                    );
                }
                EffectGpu::Gen(g) => {
                    let gp = GenParams {
                        origin: [origin[0], origin[1], origin[2], 0.0],
                        space: [0.0; 4],
                        dt: sub_dt,
                        time,
                        spawn_count,
                        capacity,
                        rng_seed,
                        _pad: [0; 3],
                    };
                    queue.write_buffer(&g.params_buf, 0, bytemuck::bytes_of(&gp));
                    let decls: &[PropertyDecl] = asset
                        .program
                        .as_ref()
                        .map(|p| p.properties.as_slice())
                        .unwrap_or(&[]);
                    let props = build_props_data(decls, &[]);
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
            queue.submit([encoder.finish()]);
        }

        self.instances.get_mut(&ikey).unwrap().warmed = true;
    }
}
