//! Phase 18: simulation space (local vs global) and the new spawn shapes.
//!
//! `local_vs_global_extent` moves an emitter horizontally across the view over
//! many frames. Under `Global` (world space) the already-spawned particles stay
//! where they were emitted, so they smear into a wide trail along the path;
//! under `Local` they rigidly follow the emitter, so the whole cloud stays a
//! compact blob near the current position. The global trail must therefore cover
//! a clearly wider horizontal span than the local blob.
//!
//! `spawn_shapes_draw` renders the new `Circle` and `Cone` spawn shapes and
//! confirms they emit visible particles.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SimulationSpace, SpawnRate, SpawnShape, VelocityDist,
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
        label: Some("particles-simspace-test"),
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

fn target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-simspace-target"),
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
    })
}

/// A steady cloud whose particles live long enough to persist across the sweep.
fn sweep_cloud(space: SimulationSpace) -> EffectAsset {
    EffectAsset::new("sweep")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_simulation_space(space)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (2.0, 2.5),
            spawn: SpawnShape::Sphere {
                radius: 0.05,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.08,
        })
}

/// Sweep the emitter across the view and return the horizontal span (max-min x,
/// in pixels) of the lit particle pixels.
fn horizontal_extent(device: &wgpu::Device, queue: &wgpu::Queue, space: SimulationSpace) -> i32 {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, sweep_cloud(space));
    renderer.with_item_type_plugin(device, Box::new(plugin)).expect("register item type plugin");

    let target = target(device);
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let frames = 30u32;
    for f in 0..frames {
        let x = -1.4 + 2.8 * (f as f32) / ((frames - 1) as f32);
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([x, 0.0, 0.0]));
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(device, queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    let (mut min_x, mut max_x) = (SIZE as i32, -1i32);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if luma(((y * SIZE + x) * 4) as usize) > corner + 60 {
                min_x = min_x.min(x as i32);
                max_x = max_x.max(x as i32);
            }
        }
    }
    if max_x < 0 { 0 } else { max_x - min_x }
}

#[test]
fn local_vs_global_extent() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let global = horizontal_extent(&device, &queue, SimulationSpace::Global);
    let local = horizontal_extent(&device, &queue, SimulationSpace::Local);

    assert!(global > 0, "global sweep did not draw ({global} px extent)");
    assert!(local > 0, "local sweep did not draw ({local} px extent)");
    assert!(
        global > local + local / 2,
        "expected the global sweep to smear wider than the local blob: global {global} px, local {local} px"
    );
}

#[test]
fn spawn_shapes_draw() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    for spawn in [
        SpawnShape::Circle {
            radius: 0.4,
            volume: true,
        },
        SpawnShape::Cone {
            base_radius: 0.4,
            top_radius: 0.05,
            height: 0.6,
            volume: true,
        },
    ] {
        let asset = EffectAsset::new("shape")
            .with_capacity(10_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(3_000.0),
                lifetime: (1.0, 1.5),
                spawn,
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.06,
            });

        let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
        let mut plugin = ParticlePlugin::new();
        let id = plugin.add_effect(&device, asset);
        renderer
            .with_item_type_plugin(&device, Box::new(plugin))
            .expect("register item type plugin");

        let target = target(&device);
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        for _ in 0..12 {
            let mut frame = frame_looking_at_origin();
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
            frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
        }

        let pixels = read_back(&device, &queue, &target);
        let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
        let corner = luma(((SIZE + 1) * 4) as usize);
        let lit = (0..(SIZE * SIZE) as usize)
            .filter(|p| luma(p * 4) > corner + 60)
            .count();
        assert!(lit > 0, "spawn shape {spawn:?} drew no particles");
    }
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-simspace-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-simspace-copy"),
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
