//! Live particle-count readback.
//!
//! With `ParticleItems::with_stats_readback(true)`, each simulated instance runs
//! an extra count compute pass, copies the tally into a mappable ring buffer, and
//! kicks an async map a frame later, publishing the live count into a shared store
//! that a `ParticleStatsHandle` reads. This test clones the handle out before
//! installing the plugin (so the plugin the renderer takes ownership of stays
//! observable), renders several frames, and asserts both that the effect draws and
//! that the handle reports a nonzero live count with the expected capacity.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin, SpawnRate,
    SpawnShape, VelocityDist,
};

// 64 so the readback row (64 * 4 = 256) respects COPY_BYTES_PER_ROW_ALIGNMENT.
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
        label: Some("particles-stats-test"),
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

fn poll_wait(device: &wgpu::Device) {
    let _ = device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(5)),
    });
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-stats-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
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
    poll_wait(device);
    rx.recv().unwrap().unwrap();
    staging.slice(..).get_mapped_range().to_vec()
}

#[test]
fn readback_path_encodes_and_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let effect = EffectAsset::new("cloud")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (2.0, 2.5),
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        });

    let capacity = 8_000;
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, effect);
    // Clone the stats handle out before the renderer takes ownership.
    let stats = plugin.stats_handle();
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-stats-target"),
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

    // Several frames so the readback ring cycles through copy -> map -> read.
    for _ in 0..12 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new()
            .with_dt(1.0 / 60.0)
            .with_stats_readback(true);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
        // Flush any in-flight async map so the ring advances next frame.
        poll_wait(&device);
    }

    let pixels = read_back(&device, &queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    let lit = (0..(SIZE * SIZE) as usize)
        .filter(|p| luma(p * 4) > corner + 60)
        .count();
    assert!(lit > 0, "effect drew nothing with readback enabled");

    // The handle should now report a live count (readback lags ~2 frames; 12
    // frames + a poll each is ample) with the effect's capacity, under budget.
    let s = stats
        .instance_stats(id, 0)
        .expect("no stats readback completed after 12 frames");
    assert!(
        s.live_count > 0,
        "readback reported zero live particles (expected a steady emitter)"
    );
    assert!(
        s.live_count <= s.capacity,
        "live_count {} exceeds capacity {}",
        s.live_count,
        s.capacity
    );
    assert_eq!(s.capacity, capacity, "capacity mismatch");
    assert!(!s.over_budget, "steady emitter should be under budget");

    // effect_stats sums the effect's single instance, matching instance_stats.
    let e = stats.effect_stats(id).expect("no effect stats");
    assert_eq!(e.live_count, s.live_count);
}
