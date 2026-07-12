//! Showcase 5: Refraction.
//!
//! The companion `RefractionPlugin` is a `GpuPlugin`, a different plugin type
//! from the particle `ItemTypePlugin`: it runs after the scene is painted and samples the
//! rendered colour. This showcase drives it directly (eframe paints through the
//! renderer's pass path, so there is no runtime `post_paint` call to hook): each
//! frame it renders a textured backdrop into an offscreen target, runs the
//! plugin to distort it with an expanding shockwave ring, and blits the result.
//!
//! `viewport-lib` does not composite a plugin's output back into the scene for
//! you, which is exactly why the plugin owns its output texture and the host
//! (here, this callback) reads it back out.

use eframe::egui;
use eframe::wgpu;
use glam::Vec2;
use viewport_lib::Camera;
use viewport_lib::resources::SCENE_DEPTH_FORMAT;
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, PostPaintTargets};
use viewport_lib_particles::RefractionPlugin;

/// Fullscreen triangle plus a procedural backdrop (checkerboard tinted by a
/// gradient) so the refraction wavefront is clearly visible.
const BACKDROP_WGSL: &str = r#"
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var xy = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let p = xy[vi];
    var o: VsOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return o;
}
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let g = uv * 16.0;
    let c = (floor(g.x) + floor(g.y)) % 2.0;
    let base = mix(vec3<f32>(0.08, 0.10, 0.16), vec3<f32>(0.85, 0.55, 0.25), c);
    let tint = mix(vec3<f32>(0.2, 0.4, 0.9), vec3<f32>(0.95, 0.85, 0.4), uv.y);
    let col = mix(base, tint, 0.4);
    return vec4<f32>(col, 1.0);
}
"#;

/// Fullscreen triangle that samples the plugin's distorted output onto the egui
/// target.
const BLIT_WGSL: &str = r#"
struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
    var xy = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let p = xy[vi];
    var o: VsOut;
    o.pos = vec4<f32>(p, 0.0, 1.0);
    o.uv = vec2<f32>(p.x * 0.5 + 0.5, 1.0 - (p.y * 0.5 + 0.5));
    return o;
}
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, in.uv, 0.0);
}
"#;

/// GPU resources for the refraction showcase, stored in the egui callback
/// resource map so both `prepare` and `paint` can reach them.
pub struct RefractionResources {
    plugin: RefractionPlugin,
    format: wgpu::TextureFormat,
    backdrop_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    depth_dummy: wgpu::TextureView,
    /// Offscreen backdrop target (texture, view, width, height), resized to fit.
    backdrop: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// Bind group over the plugin's output, rebuilt each frame in `prepare` and
    /// consumed by the blit in `paint`.
    blit_bg: Option<wgpu::BindGroup>,
}

impl RefractionResources {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let fullscreen = |src: &str,
                          label: &str,
                          bgl: Option<&wgpu::BindGroupLayout>,
                          depth: Option<wgpu::TextureFormat>| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &bgl.into_iter().collect::<Vec<_>>(),
                push_constant_ranges: &[],
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                // The blit draws into egui's render pass, which carries a
                // depth-stencil attachment (eframe was asked for a depth buffer),
                // so its pipeline must declare a matching depth state. The
                // offscreen backdrop pass owns no depth attachment.
                depth_stencil: depth.map(|format| wgpu::DepthStencilState {
                    format,
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::Always,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let backdrop_pipeline = fullscreen(BACKDROP_WGSL, "refraction_backdrop", None, None);

        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("refraction_blit_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let blit_pipeline = fullscreen(
            BLIT_WGSL,
            "refraction_blit",
            Some(&blit_bgl),
            Some(SCENE_DEPTH_FORMAT),
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("refraction_blit_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("refraction_depth_dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SCENE_DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_dummy = depth.create_view(&wgpu::TextureViewDescriptor::default());

        let mut plugin = RefractionPlugin::new();
        plugin.init_gpu(device);

        Self {
            plugin,
            format,
            backdrop_pipeline,
            blit_pipeline,
            blit_bgl,
            sampler,
            depth_dummy,
            backdrop: None,
            blit_bg: None,
        }
    }

    fn ensure_backdrop(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let ok = self
            .backdrop
            .as_ref()
            .is_some_and(|(_, _, w, h)| *w == width && *h == height);
        if ok {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("refraction_backdrop"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.backdrop = Some((texture, view, width, height));
    }
}

/// The egui paint callback for the refraction showcase.
pub struct RefractionCallback {
    pub center: [f32; 2],
    pub radius: f32,
    pub width: f32,
    pub strength: f32,
}

impl eframe::egui_wgpu::CallbackTrait for RefractionCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &eframe::egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut eframe::egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get_mut::<RefractionResources>() else {
            return Vec::new();
        };
        let [w, h] = screen.size_in_pixels;
        res.ensure_backdrop(device, w, h);
        let backdrop_view = res.backdrop.as_ref().unwrap().1.clone();

        // Draw the backdrop into the offscreen target.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("refraction_backdrop_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("refraction_backdrop_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &backdrop_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&res.backdrop_pipeline);
            pass.draw(0..3, 0..1);
        }

        // Run the plugin over the backdrop.
        res.plugin.set_aspect(w as f32 / h.max(1) as f32);
        res.plugin
            .set_ring(self.center, self.radius, self.width, self.strength);
        let camera = Camera::default();
        let ctx = GpuFrameContext::new(&camera, Vec2::new(w as f32, h as f32), 1.0 / 60.0, 0);
        let depth = res.depth_dummy.clone();
        let targets = PostPaintTargets::new(&backdrop_view, &depth, res.format);
        let plugin_cmds = res.plugin.post_paint(device, queue, &targets, &ctx);

        // Bind the plugin output for the blit in `paint` (no device there).
        let output = res.plugin.output_view().unwrap().clone();
        res.blit_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("refraction_blit_bg"),
            layout: &res.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
            ],
        }));

        let mut out = vec![encoder.finish()];
        out.extend(plugin_cmds);
        out
    }

    fn paint(
        &self,
        _info: eframe::egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &eframe::egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<RefractionResources>() else {
            return;
        };
        let Some(blit_bg) = res.blit_bg.as_ref() else {
            return;
        };
        render_pass.set_pipeline(&res.blit_pipeline);
        render_pass.set_bind_group(0, blit_bg, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// UI state for the refraction showcase.
pub struct State {
    pub strength: f32,
    pub speed: f32,
    pub width: f32,
    /// Seconds elapsed, driving the ring radius.
    pub time: f32,
}

impl Default for State {
    fn default() -> Self {
        Self {
            strength: 0.06,
            speed: 0.35,
            width: 0.05,
            time: 0.0,
        }
    }
}

impl State {
    /// The ring radius for this frame: expands from 0 and loops.
    pub fn radius(&self) -> f32 {
        (self.time * self.speed) % 0.9
    }
}

/// Left-panel controls.
pub fn controls(state: &mut State, ui: &mut egui::Ui) {
    ui.label("Shockwave");
    ui.add(egui::Slider::new(&mut state.strength, 0.0..=0.2).text("strength"));
    ui.add(egui::Slider::new(&mut state.speed, 0.05..=1.5).text("speed"));
    ui.add(egui::Slider::new(&mut state.width, 0.01..=0.15).text("width"));
    ui.separator();
    ui.label("An expanding ring re-samples the scene colour, bending the");
    ui.label("backdrop like a heat haze. Runs as a GpuPlugin (post_paint),");
    ui.label("separate from the particle draw.");
}
