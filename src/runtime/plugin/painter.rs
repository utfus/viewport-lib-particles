//! Painter's-order drawing: instances with a [`PainterPlacement`] report one
//! batch each into viewport-lib's painter sequence and draw there, sorted with
//! sprites and tile layers by layer and key, instead of in the 3D passes.
//!
//! Billboards draw through the folded billboard shader with the renderer's
//! depth-read group at group 3, which carries the 2D light layers, so a lit
//! placement takes its range's light. The mesh and trail routes and hooked
//! billboards keep their own layouts, whose group 3 is taken, and draw unlit.
//!
//! [`PainterPlacement`]: viewport_lib_types::render_item::PainterPlacement

use viewport_lib::plugin_api::depth::DEPTH_COMPARE;
use viewport_lib::plugin_api::{PainterBatch, PainterContext, PainterRun, PickPassContext};
use viewport_lib::wgpu as vwgpu;

use crate::effect::{ParticleBlend, ParticleRender};

use super::*;

impl ParticlePlugin {
    /// The painter pipeline key for a draw-list entry, or `None` when the
    /// entry draws in the 3D passes.
    fn painter_key(&self, entry: &DrawEntry) -> Option<PainterPipelineKey> {
        entry.painter?;
        let asset = &self.effects[entry.key.effect as usize].asset;
        let route = match asset.render {
            ParticleRender::Billboard { .. } => match self
                .instances
                .get(&entry.key)
                .and_then(|i| i.custom_draw.as_ref())
            {
                Some(cd) => PainterRoute::Hook(cd.pipelines.key),
                None => PainterRoute::Billboard,
            },
            ParticleRender::Mesh { .. } => PainterRoute::Mesh,
            ParticleRender::Trail { .. } => PainterRoute::Trail,
        };
        Some(PainterPipelineKey {
            route,
            blend: asset.blend,
            depth_test: asset.painter_depth_test,
        })
    }

    /// Build the painter pipelines this frame's placements need, and write each
    /// painter billboard's layer and unlit flag into its draw params. Called at
    /// the end of `prepare`, once the draw list is final.
    pub(crate) fn prepare_painter(&mut self, device: &vwgpu::Device, queue: &vwgpu::Queue) {
        for i in 0..self.draw_list.len() {
            let entry = &self.draw_list[i];
            let (Some(key), Some(placement)) = (self.painter_key(entry), entry.painter) else {
                continue;
            };
            let ikey = entry.key;
            if !self.painter_pipelines.contains_key(&key) {
                if let Some(p) = self.build_painter_pipeline(device, key) {
                    self.painter_pipelines.insert(key, p);
                }
            }
            if let Some((buf, params)) = self
                .instances
                .get_mut(&ikey)
                .and_then(|inst| inst.folded_params.as_mut())
            {
                let layer = placement.layer;
                let unlit = placement.unlit as u32;
                if params.painter_layer != layer || params.painter_unlit != unlit {
                    params.painter_layer = layer;
                    params.painter_unlit = unlit;
                    queue.write_buffer(buf, 0, bytemuck::bytes_of(params));
                }
            }
        }
    }

    fn build_painter_pipeline(
        &self,
        device: &vwgpu::Device,
        key: PainterPipelineKey,
    ) -> Option<vwgpu::RenderPipeline> {
        let builder = self.builder.as_ref()?;
        let shaders = self.painter_shaders.as_ref()?;
        let ramp = self.ramp_bgl.as_ref()?;
        let multiply = key.blend == ParticleBlend::Multiply;
        let mesh_layout = [mesh_vertex_layout()];
        let (shader, fs, vertex_layouts, extra): (_, _, &[vwgpu::VertexBufferLayout<'_>], _) =
            match key.route {
                PainterRoute::Billboard => (
                    &shaders.billboard,
                    if multiply { "fs_painter_multiply" } else { "fs_painter" },
                    &[],
                    [self.soft_draw_bgl.as_ref()?, ramp, self.depth_read_bgl.as_ref()?],
                ),
                PainterRoute::Mesh => (
                    &shaders.mesh,
                    if multiply { "fs_multiply" } else { "fs" },
                    &mesh_layout,
                    [self.draw_bgl.as_ref()?, ramp, self.drawparams_bgl.as_ref()?],
                ),
                PainterRoute::Trail => (
                    &shaders.trail,
                    if multiply { "fs_multiply" } else { "fs" },
                    &[],
                    [self.trail_draw_bgl.as_ref()?, ramp, self.drawparams_bgl.as_ref()?],
                ),
                PainterRoute::Hook(hook) => (
                    &self.custom_draw_cache.get(&hook)?.draw_shader,
                    if multiply { "fs_hook_multiply" } else { "fs_hook" },
                    &[],
                    [self.draw_bgl.as_ref()?, ramp, self.hook_drawparams_bgl.as_ref()?],
                ),
            };
        let mut opts = route_opts(
            "particle_painter",
            shader,
            fs,
            vertex_layouts,
            &extra,
            Some(blend_state(key.blend)),
        );
        opts.depth_compare = if key.depth_test {
            DEPTH_COMPARE
        } else {
            vwgpu::CompareFunction::Always
        };
        Some(builder.build_depth_read_pipeline(device, &opts))
    }

    /// One batch per painter instance, its handle the index in the draw list.
    pub(crate) fn report_painter_batches(&self, out: &mut Vec<PainterBatch>) {
        for (i, entry) in self.draw_list.iter().enumerate() {
            if let Some(placement) = entry.painter {
                out.push(PainterBatch::at(placement, i as u32));
            }
        }
    }

    /// Draw the painter batches `ctx` names, in its order.
    pub(crate) fn paint_painter_batches(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        ctx: &PainterContext<'_>,
    ) {
        // Painter effects draw on the HDR path only: the LDR scene pass has no
        // pipelines of theirs.
        let (Some(batches), true) = (ctx.batches, ctx.is_hdr_pass()) else {
            return;
        };
        for &handle in batches {
            let Some(entry) = self.draw_list.get(handle as usize) else {
                continue;
            };
            let Some(pipeline) = self
                .painter_key(entry)
                .and_then(|k| self.painter_pipelines.get(&k))
            else {
                continue;
            };
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            let eff = &self.effects[entry.key.effect as usize];
            pass.set_pipeline(pipeline);
            pass.set_bind_group(2, inst.gpu.ramp_bg(), &[]);
            match eff.asset.render {
                ParticleRender::Billboard { .. } => {
                    if let Some(cd) = inst.custom_draw.as_ref() {
                        pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
                        pass.set_bind_group(3, &cd.bind_group, &[]);
                    } else {
                        let (Some(folded), Some(depth_read)) =
                            (inst.soft_draw_bg.as_ref(), ctx.depth_read_bind_group)
                        else {
                            continue;
                        };
                        pass.set_bind_group(1, folded, &[]);
                        pass.set_bind_group(3, depth_read, &[]);
                    }
                    pass.draw(0..6, 0..eff.asset.capacity);
                }
                ParticleRender::Mesh { mesh, .. } => {
                    let Some(m) = self.meshes.get(mesh.0 as usize) else {
                        continue;
                    };
                    pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
                    pass.set_bind_group(3, inst.gpu.drawparams_bg(), &[]);
                    pass.set_vertex_buffer(0, m.vertex_buf.slice(..));
                    pass.set_index_buffer(m.index_buf.slice(..), vwgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..m.index_count, 0, 0..eff.asset.capacity);
                }
                ParticleRender::Trail { segments, .. } => {
                    let Some(trail) = inst.trail.as_ref() else {
                        continue;
                    };
                    pass.set_bind_group(1, &trail.draw_bg, &[]);
                    pass.set_bind_group(3, inst.gpu.drawparams_bg(), &[]);
                    pass.draw(0..segments.clamp(1, HISTORY_LEN - 1) * 6, 0..eff.asset.capacity);
                }
            }
        }
    }

    /// Pick the painter batches of `run`, in its order, so the batch drawn last
    /// on screen wins the pick.
    pub(crate) fn pick_painter_batches(
        &self,
        pass: &mut vwgpu::RenderPass<'_>,
        _ctx: &PickPassContext<'_>,
        run: &PainterRun<'_>,
    ) {
        let (Some(pipeline), Some(batches)) = (self.pick_pipeline.as_ref(), run.batches) else {
            return;
        };
        for &handle in batches {
            let Some(entry) = self.draw_list.get(handle as usize) else {
                continue;
            };
            if entry.pick_id == 0 {
                continue;
            }
            let Some(inst) = self.instances.get(&entry.key) else {
                continue;
            };
            let eff = &self.effects[entry.key.effect as usize];
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, inst.gpu.draw_bg(), &[]);
            pass.set_bind_group(2, &inst.pick_bg, &[]);
            pass.draw(0..6, 0..eff.asset.capacity);
        }
    }
}
