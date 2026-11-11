//! Mesh-surface spawn.
//!
//! `SpawnShape::MeshSurface` draws spawn positions from an uploaded mesh's
//! triangles. An effect whose mesh quad is centred at the origin lights the
//! centre pixel; the same effect whose mesh quad is far off to the side leaves
//! the centre dark, proving the positions come from the mesh surface (not the
//! emitter origin, which is the same in both cases).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticleMeshId,
    ParticlePlugin, SpawnRate, SpawnShape, VelocityDist,
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
        label: Some("particles-mesh-spawn-test"),
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

/// A small quad in the XY plane centred at `(cx, 0, 0)`.
fn quad(cx: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let s = 0.6;
    let positions = vec![
        [cx - s, -s, 0.0],
        [cx + s, -s, 0.0],
        [cx + s, s, 0.0],
        [cx - s, s, 0.0],
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (positions, indices)
}

fn mesh_effect(mesh: ParticleMeshId) -> EffectAsset {
    EffectAsset::new("mesh-spawn")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(8_000.0),
            lifetime: (2.0, 3.0),
            spawn: SpawnShape::MeshSurface {
                mesh,
                normal_velocity: 0.0,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.3,
        })
}

/// Centre brightness when the effect's mesh quad is centred at `cx`.
fn center_brightness(cx: f32) -> i32 {
    let (device, queue) = match headless_device() {
        Some(dq) => dq,
        None => return -1,
    };
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let (positions, indices) = quad(cx);
    let mesh = plugin.upload_mesh(&device, &positions, &indices);
    let id = plugin.add_effect(&device, mesh_effect(mesh));
    renderer.with_item_type_plugin(&device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-mesh-target"),
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

    for _ in 0..3 {
        let mut frame = frame_looking_at_origin();
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
fn spawns_from_mesh_surface_not_origin() {
    if headless_device().is_none() {
        eprintln!("skipping: no GPU adapter available");
        return;
    }
    // When the mesh is at the origin the centre pixel gets extra brightness from
    // the particles on top of the background; when the mesh is far off-screen
    // only the background remains, so the centre is markedly dimmer. This proves
    // the spawn positions come from the mesh surface, not the emitter origin
    // (identical in both cases).
    let centered = center_brightness(0.0);
    let offset = center_brightness(40.0);
    assert!(
        centered > offset + 100,
        "mesh at origin should light the centre well above the off-screen mesh (background only): centered={centered}, offset={offset}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-mesh-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-mesh-copy"),
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
