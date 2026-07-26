//! The depth-sort path runs and draws.
//!
//! A non-additive (alpha) effect triggers the per-frame key + bitonic sort in
//! `prepare`, and the draw reads particles through the sorted order buffer. This
//! renders an alpha effect with sorting on and again with it off, confirming both
//! the sorted and identity-order draw paths encode without validation errors and
//! produce a center glow.
//!
//! Correct back-to-front compositing is hard to assert from a single pixel; this
//! is a smoke test of the sort machinery (setup pass + ~78 bitonic stages) plus
//! the order-buffer indirection shared by every render route.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::Camera;
use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;

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
        label: Some("particles-sort-test"),
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

fn alpha_effect() -> EffectAsset {
    EffectAsset::new("smoke")
        .with_capacity(4096)
        .with_blend(ParticleBlend::Alpha)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (1.0, 1.5),
            spawn: SpawnShape::Sphere {
                radius: 0.2,
                volume: true,
            },
            velocity: VelocityDist::UniformBox {
                min: [-0.4, -0.4, -0.4],
                max: [0.4, 0.4, 0.4],
            },
            colour: [0.9, 0.9, 1.0, 0.5],
            size: 0.25,
        })
}

fn render_center(sort: bool) -> Option<i32> {
    let (device, queue) = headless_device()?;

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, alpha_effect());
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-sort-target"),
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

    for _ in 0..6 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new()
            .with_dt(1.0 / 60.0)
            .with_sort_transparent(sort);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let sum = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
    };
    let mut center = 0;
    for y in (SIZE / 2 - 6)..(SIZE / 2 + 6) {
        for x in (SIZE / 2 - 6)..(SIZE / 2 + 6) {
            center = center.max(sum(x, y));
        }
    }
    Some(center)
}

#[test]
fn sorted_alpha_draws() {
    let Some(on) = render_center(true) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let off = render_center(false).expect("second device");

    // Both paths must render the alpha cloud (the sort machinery encodes cleanly
    // and the order indirection draws in either mode).
    assert!(on > 120, "sorted alpha too dark: {on}");
    assert!(off > 120, "unsorted alpha too dark: {off}");
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-sort-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-sort-copy"),
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
