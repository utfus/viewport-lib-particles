//! Phase 6 verification: the refraction GpuPlugin distorts the scene colour.
//!
//! Feeds the plugin a known horizontal gradient as the "scene colour", runs its
//! `post_paint` with an expanding ring centered on the frame, and reads back the
//! distorted output. Where the ring wavefront crosses the gradient the sampled
//! value is displaced; the untouched center passes straight through.
//!
//! Skips cleanly when no GPU adapter is available.

use glam::Vec2;
use viewport_lib::Camera;
use viewport_lib::resources::SCENE_DEPTH_FORMAT;
use viewport_lib::runtime::{GpuFrameContext, GpuPlugin, PostPaintTargets};
use viewport_lib::wgpu;

use viewport_lib_particles::RefractionPlugin;

const SIZE: u32 = 64;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("refraction-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

/// A texture holding a horizontal 0..1 gray gradient (value depends on column).
fn gradient_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture {
    let mut data = vec![0u8; (SIZE * SIZE * 4) as usize];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let v = (x as f32 / (SIZE - 1) as f32 * 255.0) as u8;
            let i = ((y * SIZE + x) * 4) as usize;
            data[i] = v;
            data[i + 1] = v;
            data[i + 2] = v;
            data[i + 3] = 255;
        }
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("refraction-scene"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 4),
            rows_per_image: Some(SIZE),
        },
        wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    tex
}

#[test]
fn refraction_distorts_scene() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let scene = gradient_scene(&device, &queue);
    let scene_view = scene.create_view(&wgpu::TextureViewDescriptor::default());

    // post_paint requires a depth view; the plugin does not read it.
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("refraction-depth"),
        size: wgpu::Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SCENE_DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let mut plugin = RefractionPlugin::new();
    plugin.set_aspect(1.0);
    // A strong ring at radius 0.4 (uv) around the center.
    plugin.set_ring([0.5, 0.5], 0.4, 0.06, 0.15);

    let camera = Camera::default();
    let ctx = GpuFrameContext::new(&camera, Vec2::new(SIZE as f32, SIZE as f32), 1.0 / 60.0, 0);
    let targets = PostPaintTargets::new(&scene_view, &depth_view, FORMAT);

    let cmds = plugin.post_paint(&device, &queue, &targets, &ctx);
    queue.submit(cmds);

    let out = plugin
        .output_texture()
        .expect("post_paint should allocate the output");
    let pixels = read_back(&device, &queue, out);
    let r = |x: u32, y: u32| pixels[((y * SIZE + x) * 4) as usize] as i32;
    // Expected passthrough value in a column is the input gradient at that column.
    let input_r = |x: u32| (x as f32 / (SIZE - 1) as f32 * 255.0) as i32;

    // Center column sits far inside the ring: the wavefront profile is ~0 there,
    // so it passes straight through.
    let center_dev = (r(SIZE / 2, SIZE / 2) - input_r(SIZE / 2)).abs();
    assert!(
        center_dev <= 6,
        "center should pass through, deviation {center_dev}"
    );

    // The left arm of the ring (uv.x ~ 0.1) displaces along the gradient axis, so
    // the sampled value differs from the straight-through gradient there.
    let mut ring_dev = 0;
    for x in 2..12 {
        ring_dev = ring_dev.max((r(x, SIZE / 2) - input_r(x)).abs());
    }
    assert!(
        ring_dev >= 12,
        "ring wavefront should distort the gradient, max deviation {ring_dev}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("refraction-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("refraction-copy"),
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
