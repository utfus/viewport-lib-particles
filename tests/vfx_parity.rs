//! Fixed-function parity features: timed spawn bursts, spawn offset, per-particle
//! size variance, and per-particle billboard rotation.
//!
//! - `schedule_burst_fires_after_delay`: an effect with no continuous rate and a
//!   burst scheduled at t = 0.3 s stays dark before that time and lights up after.
//! - `spawn_offset_moves_glow`: a large spawn offset pushes the glow off the
//!   emitter anchor, so the centre darkens relative to no offset.
//! - `size_variance_spawns_particles`: with a base size of zero, a size variance
//!   still gives some particles a positive size (so the effect draws), where zero
//!   variance draws nothing.
//! - `rotation_renders`: an effect with a spin rate draws (exercises the rotated
//!   billboard path).
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SpawnRate, SpawnShape, VelocityDist,
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
        label: Some("particles-vfx-parity-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    let mut render_camera = RenderCamera::from_camera(&cam);
    render_camera.forward = [0.0, 0.0, -1.0];
    render_camera.far = cam.zfar;
    render_camera.aspect = 1.0;
    frame.camera.render_camera = render_camera;
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-vfx-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
    rx.recv().unwrap().unwrap();
    staging.slice(..).get_mapped_range().to_vec()
}

/// Render `total` frames of `effect`, submitting `build(frame, id)` each, and
/// return the readback pixels.
fn render(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    effect: EffectAsset,
    total: usize,
    build: impl Fn(usize, EffectId) -> ParticleItems,
) -> Vec<u8> {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, effect);
    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-vfx-target"),
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

    for f in 0..total {
        let mut frame = frame_looking_at_origin();
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, build(f, id));
        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }
    read_back(device, queue, &target)
}

fn lit_pixels(pixels: &[u8]) -> usize {
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    (0..(SIZE * SIZE) as usize)
        .filter(|p| luma(p * 4) > corner + 60)
        .count()
}

fn center_luma(pixels: &[u8]) -> i32 {
    let mid = ((SIZE / 2) * SIZE + SIZE / 2) as usize * 4;
    pixels[mid] as i32 + pixels[mid + 1] as i32 + pixels[mid + 2] as i32
}

fn dense_cloud() -> EffectAsset {
    EffectAsset::new("cloud")
        .with_capacity(4_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(2_000.0),
            lifetime: (2.0, 2.5),
            spawn: SpawnShape::Sphere {
                radius: 0.1,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.12,
        })
}

#[test]
fn schedule_burst_fires_after_delay() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // No continuous rate; one burst scheduled at 0.3 s.
    let effect = || {
        EffectAsset::new("burst")
            .with_capacity(4_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(0.0),
                lifetime: (3.0, 3.0),
                spawn: SpawnShape::Sphere {
                    radius: 0.1,
                    volume: true,
                },
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.12,
            })
            .add_burst(600, 0.3)
    };
    let items = |_f: usize, id: EffectId| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    };

    // 10 frames: elapsed = 0.166 s < 0.3, burst not yet fired.
    let before = render(&device, &queue, effect(), 10, items);
    assert_eq!(lit_pixels(&before), 0, "burst fired before its scheduled time");

    // 30 frames: elapsed = 0.5 s > 0.3, burst has fired and particles are alive.
    let after = render(&device, &queue, effect(), 30, items);
    assert!(
        lit_pixels(&after) > 0,
        "burst did not fire after its scheduled time"
    );
}

#[test]
fn spawn_offset_moves_glow() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let items = |_f: usize, id: EffectId| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    };

    let centered = render(&device, &queue, dense_cloud(), 16, items);
    // A large offset pushes the whole cloud off the anchor, so the centre darkens.
    let offset = render(
        &device,
        &queue,
        dense_cloud().with_spawn_offset([2.0, 0.0, 0.0]),
        16,
        items,
    );

    assert!(
        center_luma(&centered) > center_luma(&offset) + 100,
        "spawn offset did not move the glow off centre (centered {}, offset {})",
        center_luma(&centered),
        center_luma(&offset)
    );
}

#[test]
fn size_variance_spawns_particles() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let base_zero = |variance: f32| {
        EffectAsset::new("sized")
            .with_capacity(4_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(2_000.0),
                lifetime: (2.0, 2.5),
                spawn: SpawnShape::Sphere {
                    radius: 0.1,
                    volume: true,
                },
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.0,
            })
            .with_size_variance(variance)
    };
    let items = |_f: usize, id: EffectId| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    };

    // Base size 0 with no variance: every particle collapses to zero area.
    let flat = render(&device, &queue, base_zero(0.0), 16, items);
    assert_eq!(lit_pixels(&flat), 0, "zero size drew something");

    // Base size 0 with variance: about half the particles get a positive size.
    let varied = render(&device, &queue, base_zero(0.3), 16, items);
    assert!(
        lit_pixels(&varied) > 0,
        "size variance produced no visible particles"
    );
}

#[test]
fn rotation_renders() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // A spinning effect still draws (exercises the rotated billboard path).
    let effect = dense_cloud().with_rotation(0.3, 0.5, 4.0);
    let lit = lit_pixels(&render(&device, &queue, effect, 16, |_f, id| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    }));
    assert!(lit > 0, "rotating effect drew nothing");
}
