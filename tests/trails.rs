//! Phase 5 verification: the trail render route draws.
//!
//! Registers a fountain whose particles are drawn as ribbon trails, runs enough
//! frames for the position history to fill in, and reads back the center pixel to
//! confirm the ribbons accumulate into a glow. A trail needs several frames of
//! history before any segment is valid, so this drives more frames than the
//! billboard/mesh tests.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::Camera;
use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;

use viewport_lib_particles::{
    EffectAsset, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    ParticleRender, SpawnRate, SpawnShape, VelocityDist,
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
        label: Some("particles-trails-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn frame_looking_at_origin() -> FrameData {
    let cam = Camera::default();
    let mut frame = FrameData::default();
    frame.camera.render_camera = RenderCamera {
        view: cam.view_matrix(),
        projection: cam.proj_matrix(),
        eye_position: cam.eye_position().to_array(),
        forward: [0.0, 0.0, -1.0],
        orientation: cam.orientation,
        near: cam.effective_znear(),
        far: cam.zfar,
        distance: cam.distance,
        fov: cam.fov_y,
        aspect: 1.0,
    };
    frame.camera.viewport_size = [SIZE as f32, SIZE as f32];
    frame.viewport.show_grid = false;
    frame.viewport.show_axes_indicator = false;
    frame.scene.surfaces = SurfaceSubmission::Flat(vec![].into());
    frame
}

#[test]
fn trail_route_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();

    let effect = EffectAsset::new("comets")
        .with_capacity(10_000)
        .with_blend(ParticleBlend::Additive)
        .with_render(ParticleRender::Trail {
            width: 0.15,
            segments: 20,
        })
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(4_000.0),
            lifetime: (1.0, 1.5),
            spawn: SpawnShape::Point,
            // Symmetric spread so the density peak stays over the emitter at
            // the screen center; trails sweep outward from there.
            velocity: VelocityDist::UniformBox {
                min: [-0.7, -0.7, -0.7],
                max: [0.7, 0.7, 0.7],
            },
            colour: [1.0, 0.6, 0.3, 1.0],
            size: 0.1,
        });
    let id = plugin.add_effect(&device, effect);
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-trails-target"),
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

    // Enough frames for the history ring to fill and trails to sweep.
    for _ in 0..30 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
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
    // Brightest pixel in a central box vs the corner background. Trails render a
    // saturated core over the emitter; the corner stays at the scene background.
    let mut center = 0;
    for y in (SIZE / 2 - 8)..(SIZE / 2 + 8) {
        for x in (SIZE / 2 - 8)..(SIZE / 2 + 8) {
            center = center.max(sum(x, y));
        }
    }
    let corner = sum(1, 1);

    assert!(
        center > 100,
        "center too dark; trails did not draw: {center}"
    );
    assert!(
        center > corner + 120,
        "center {center} not clearly brighter than corner {corner}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-trails-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-trails-copy"),
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
