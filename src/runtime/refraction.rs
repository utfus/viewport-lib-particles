//! Screen-space refraction as a companion [`GpuPlugin`].
//!
//! Particles that bend the scene behind them (shockwaves, heat haze) cannot be
//! done in the [`ParticlePlugin`](crate::ParticlePlugin) draw: that pass has no
//! readable copy of the scene colour. Refraction is a screen-space effect, so it
//! uses the other plugin type `viewport-lib` exposes: a [`GpuPlugin`] whose
//! `post_paint` runs after the scene is painted and can sample the rendered
//! colour.
//!
//! [`RefractionPlugin`] samples the scene colour handed to `post_paint` and
//! re-samples it with a radial displacement centered on an expanding ring, then
//! writes the distorted result into a texture it owns. `viewport-lib` does not
//! composite a plugin's output back into the scene colour for you, so the host
//! reads [`RefractionPlugin::output_view`] and blits or composites it (the
//! showcase does exactly this).
//!
//! The plugin is self-contained: a `GpuPlugin` receives only the wgpu device,
//! not the lib's shared bindings, so it builds its own sampler, uniform, and
//! fullscreen pipeline. The pipeline's colour target format follows the scene
//! colour handed in at `post_paint` (HDR `Rgba16Float` on the HDR path, an 8-bit
//! sRGB target on an LDR path), so it is built lazily on the first frame and
//! rebuilt if the format changes.

use bytemuck::{Pod, Zeroable};
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, PostPaintTargets, gpu_phase};
use viewport_lib::wgpu as vwgpu;

/// The expanding-ring parameters, matching `Refract` in `refraction.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RefractUniform {
    center: [f32; 2],
    radius: f32,
    width: f32,
    strength: f32,
    aspect: f32,
    _pad: [f32; 2],
}

/// The output texture the plugin distorts into, sized to the viewport.
struct Output {
    texture: vwgpu::Texture,
    view: vwgpu::TextureView,
    width: u32,
    height: u32,
    format: vwgpu::TextureFormat,
}

/// The fullscreen pipeline, rebuilt if the scene colour format changes.
struct Pipeline {
    pipeline: vwgpu::RenderPipeline,
    bgl: vwgpu::BindGroupLayout,
    format: vwgpu::TextureFormat,
}

/// A screen-space refraction post-process driven by an expanding shockwave ring.
///
/// Register on the runtime with
/// [`ViewportRuntime::with_gpu_plugin`](viewport_lib::runtime::ViewportRuntime::with_gpu_plugin),
/// or drive it directly by calling [`post_paint`](GpuPlugin::post_paint) after
/// the scene is painted (the showcase does the latter, since eframe paints
/// through the renderer's pass path). Set the ring with [`set_ring`](Self::set_ring)
/// and the viewport aspect with [`set_aspect`](Self::set_aspect) each frame, then
/// read the distorted image from [`output_view`](Self::output_view).
#[derive(Default)]
pub struct RefractionPlugin {
    center: [f32; 2],
    radius: f32,
    width: f32,
    strength: f32,
    aspect: f32,
    sampler: Option<vwgpu::Sampler>,
    uniform_buf: Option<vwgpu::Buffer>,
    pipeline: Option<Pipeline>,
    output: Option<Output>,
}

impl RefractionPlugin {
    /// A plugin with a disabled ring (`strength` 0, so it passes the scene
    /// through until [`set_ring`](Self::set_ring) is called).
    pub fn new() -> Self {
        Self {
            center: [0.5, 0.5],
            radius: 0.0,
            width: 0.05,
            strength: 0.0,
            aspect: 1.0,
            ..Default::default()
        }
    }

    /// Set the shockwave ring: `center` in uv space (0..1, origin top-left),
    /// `radius` and `width` in uv units, and `strength` (the peak displacement,
    /// in uv units, applied at the wavefront). A `strength` of 0 passes the
    /// scene through unchanged.
    pub fn set_ring(&mut self, center: [f32; 2], radius: f32, width: f32, strength: f32) {
        self.center = center;
        self.radius = radius;
        self.width = width.max(1e-4);
        self.strength = strength;
    }

    /// Set the viewport aspect ratio (width / height) so the ring stays circular
    /// on non-square viewports.
    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect.max(1e-4);
    }

    /// The distorted output view, sized to the last `post_paint` viewport, for a
    /// host that composites by sampling it. `None` until the first `post_paint`.
    pub fn output_view(&self) -> Option<&vwgpu::TextureView> {
        self.output.as_ref().map(|o| &o.view)
    }

    /// The distorted output texture (usage `RENDER_ATTACHMENT | TEXTURE_BINDING |
    /// COPY_SRC`), for a host that composites by copying it. `None` until the
    /// first `post_paint`.
    pub fn output_texture(&self) -> Option<&vwgpu::Texture> {
        self.output.as_ref().map(|o| &o.texture)
    }

    /// The format of the output texture (matches the scene colour handed to the
    /// last `post_paint`). `None` until the first `post_paint`.
    pub fn output_format(&self) -> Option<vwgpu::TextureFormat> {
        self.output.as_ref().map(|o| o.format)
    }

    /// Build (or reuse) the fullscreen pipeline for a colour target format.
    fn ensure_pipeline(&mut self, device: &vwgpu::Device, format: vwgpu::TextureFormat) {
        if self.pipeline.as_ref().is_some_and(|p| p.format == format) {
            return;
        }
        let bgl = device.create_bind_group_layout(&vwgpu::BindGroupLayoutDescriptor {
            label: Some("refraction_bgl"),
            entries: &[
                vwgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Texture {
                        sample_type: vwgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: vwgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Sampler(vwgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                vwgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vwgpu::ShaderStages::FRAGMENT,
                    ty: vwgpu::BindingType::Buffer {
                        ty: vwgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(vwgpu::ShaderModuleDescriptor {
            label: Some("refraction_shader"),
            source: vwgpu::ShaderSource::Wgsl(include_str!("shaders/refraction.wgsl").into()),
        });
        let layout = vwgpu::pipeline_layout(device, "refraction_layout", &[&bgl]);
        let pipeline = vwgpu::render_pipeline(
            device,
            vwgpu::RenderPipelineDesc {
                label: "refraction_pipeline",
                layout: &layout,
                vertex: vwgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(vwgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(vwgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: vwgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: vwgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: vwgpu::MultisampleState::default(),
                cache: None,
            },
        );
        self.pipeline = Some(Pipeline {
            pipeline,
            bgl,
            format,
        });
    }

    /// Allocate (or reuse) the output texture for a viewport size and format.
    fn ensure_output(
        &mut self,
        device: &vwgpu::Device,
        width: u32,
        height: u32,
        format: vwgpu::TextureFormat,
    ) {
        let ok = self
            .output
            .as_ref()
            .is_some_and(|o| o.width == width && o.height == height && o.format == format);
        if ok {
            return;
        }
        let texture = device.create_texture(&vwgpu::TextureDescriptor {
            label: Some("refraction_output"),
            size: vwgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: vwgpu::TextureDimension::D2,
            format,
            usage: vwgpu::TextureUsages::RENDER_ATTACHMENT
                | vwgpu::TextureUsages::TEXTURE_BINDING
                | vwgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&vwgpu::TextureViewDescriptor::default());
        self.output = Some(Output {
            texture,
            view,
            width,
            height,
            format,
        });
    }
}

impl GpuPlugin for RefractionPlugin {
    fn priority(&self) -> i32 {
        gpu_phase::POST_PAINT
    }

    fn type_name(&self) -> &'static str {
        "viewport_lib_particles::refraction"
    }

    fn init_gpu(&mut self, device: &vwgpu::Device) {
        if self.sampler.is_none() {
            self.sampler = Some(device.create_sampler(&vwgpu::SamplerDescriptor {
                label: Some("refraction_sampler"),
                address_mode_u: vwgpu::AddressMode::ClampToEdge,
                address_mode_v: vwgpu::AddressMode::ClampToEdge,
                address_mode_w: vwgpu::AddressMode::ClampToEdge,
                mag_filter: vwgpu::FilterMode::Linear,
                min_filter: vwgpu::FilterMode::Linear,
                mipmap_filter: vwgpu::FilterMode::Nearest,
                ..Default::default()
            }));
        }
        if self.uniform_buf.is_none() {
            self.uniform_buf = Some(device.create_buffer(&vwgpu::BufferDescriptor {
                label: Some("refraction_uniform"),
                size: std::mem::size_of::<RefractUniform>() as u64,
                usage: vwgpu::BufferUsages::UNIFORM | vwgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
    }

    fn on_device_recreated(&mut self, _device: &vwgpu::Device, _queue: &vwgpu::Queue) {
        // Drop everything built against the old device; init_gpu rebuilds the
        // sampler and uniform, and post_paint rebuilds the pipeline and output.
        self.sampler = None;
        self.uniform_buf = None;
        self.pipeline = None;
        self.output = None;
    }

    fn post_paint(
        &mut self,
        device: &vwgpu::Device,
        queue: &vwgpu::Queue,
        targets: &PostPaintTargets<'_>,
        ctx: &GpuFrameContext<'_>,
    ) -> Vec<vwgpu::CommandBuffer> {
        // init_gpu runs before the first pre_prepare, but a host that only drives
        // post_paint (like the showcase) may never call it, so make sure the
        // sampler and uniform exist.
        self.init_gpu(device);

        let width = ctx.viewport_size.x.max(1.0) as u32;
        let height = ctx.viewport_size.y.max(1.0) as u32;
        let format = targets.color_format;

        self.ensure_pipeline(device, format);
        self.ensure_output(device, width, height, format);

        let uniform = RefractUniform {
            center: self.center,
            radius: self.radius,
            width: self.width,
            strength: self.strength,
            aspect: self.aspect,
            _pad: [0.0; 2],
        };
        let uniform_buf = self.uniform_buf.as_ref().unwrap();
        queue.write_buffer(uniform_buf, 0, bytemuck::bytes_of(&uniform));

        let pipeline = self.pipeline.as_ref().unwrap();
        let sampler = self.sampler.as_ref().unwrap();
        let output = self.output.as_ref().unwrap();

        let bind_group = device.create_bind_group(&vwgpu::BindGroupDescriptor {
            label: Some("refraction_bg"),
            layout: &pipeline.bgl,
            entries: &[
                vwgpu::BindGroupEntry {
                    binding: 0,
                    resource: vwgpu::BindingResource::TextureView(targets.color_view),
                },
                vwgpu::BindGroupEntry {
                    binding: 1,
                    resource: vwgpu::BindingResource::Sampler(sampler),
                },
                vwgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&vwgpu::CommandEncoderDescriptor {
            label: Some("refraction_encoder"),
        });
        {
            let mut pass = encoder.begin_render_pass(&vwgpu::RenderPassDescriptor {
                label: Some("refraction_pass"),
                color_attachments: &[Some(vwgpu::RenderPassColorAttachment {
                    view: &output.view,
                    resolve_target: None,
                    ops: vwgpu::Operations {
                        load: vwgpu::LoadOp::Clear(vwgpu::Color::BLACK),
                        store: vwgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        vec![encoder.finish()]
    }
}
