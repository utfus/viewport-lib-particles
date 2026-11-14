//! Soft particles.
//!
//! A soft billboard effect fades its alpha as it nears the opaque scene surface,
//! via the renderer's read-only-depth plugin pass (`paint_depth_read`). With an
//! opaque quad sitting just behind the particles, a **soft** effect reads much
//! dimmer at centre than an otherwise-identical **non-soft** effect; the only
//! difference is the soft fade against scene depth. This also exercises the
//! read-only-depth pass end to end (our replicated `depth_read_bgl` binding the renderer's
//! prebuilt `scene_depth_bind_group`, and the gated pass running).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::resources::MeshData;
use viewport_lib::scene::material::ShadingModel;
use viewport_lib::renderer::MeshItem;
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin, SpawnRate,
    SpawnShape, VelocityDist,
};

const SIZE: u32 = 64;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-soft-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    let render_camera = RenderCamera::from_camera(&cam);
    frame.camera.render_camera = render_camera;
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

/// A large double-sided quad in the XY plane, so an occluder faces the camera
/// regardless of winding.
fn quad_mesh() -> MeshData {
    let s = 20.0;
    let mut mesh = MeshData::default();
    mesh.positions = vec![
        [-s, -s, 0.0],
        [s, -s, 0.0],
        [s, s, 0.0],
        [-s, s, 0.0],
    ];
    mesh.normals = vec![[0.0, 0.0, 1.0]; 4];
    // Front and back windings so the quad is visible (and writes depth) either way.
    mesh.indices = vec![0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2];
    mesh
}

fn smoke(soft: bool) -> EffectAsset {
    let mut e = EffectAsset::new("soft-smoke")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(6_000.0),
            lifetime: (2.0, 3.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 1.5,
        });
    if soft {
        e = e.with_soft(1.0);
    }
    e
}

/// Centre brightness of the effect drawn just in front of an opaque quad.
fn center_over_occluder(soft: bool) -> i32 {
    let (device, queue) = match headless_device() {
        Some(dq) => dq,
        None => return -1,
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mesh_id = renderer
        .resources_mut()
        .upload_mesh_data(&device, &quad_mesh())
        .expect("upload occluder mesh");

    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, smoke(soft));
    renderer.with_item_type_plugin(&device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-soft-target"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    // Opaque quad just behind the particle plane (farther from the +Z camera), a
    // stable dark baseline present in both runs so its colour cancels out.
    let mut occluder_material = viewport_lib::Material::default();
    occluder_material.shading_model = ShadingModel::Pbr;
    occluder_material.base_colour = viewport_lib::Colour::linear_rgb(0.12, 0.12, 0.12);
    let occluder_material = renderer
        .resources_mut()
        .upload_material(occluder_material)
        .expect("upload occluder material");
    let mut occluder = MeshItem::new(mesh_id, occluder_material);
    occluder.model = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.0, -0.05)).into();

    for _ in 0..2 {
        let mut frame = frame_looking_at_origin();
        frame.scene.surfaces = SurfaceSubmission::Flat(vec![occluder.clone()].into());
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let i = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

#[test]
fn soft_particles_fade_into_geometry() {
    if headless_device().is_none() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }
    let hard = center_over_occluder(false);
    let soft = center_over_occluder(true);
    assert!(
        hard > soft + 60,
        "soft particles should fade near the surface, reading dimmer than the hard-edged effect: hard={hard}, soft={soft}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-soft-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-soft-copy"),
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(SIZE),
            },
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(std::iter::once(enc.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv().unwrap().unwrap();

    let mapped = staging.slice(..).get_mapped_range();
    mapped.to_vec()
}
