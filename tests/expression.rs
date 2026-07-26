//! An effect authored via the expression graph compiles
//! to emit/simulate kernels and draws.
//!
//! Builds a program whose per-particle colour comes from a random draw (which
//! the fixed-function emitter cannot express), spawns it at the origin, runs a
//! few frames, and reads back the center pixel to confirm the additive glow.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    Attribute, EffectAsset, EffectProgram, Module, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlugin, SpawnRate, UpdateOp,
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
        label: Some("particles-expr-test"),
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

/// An effect whose colour is `colA + (colB - colA) * rand` per particle, with a
/// constant size and a small downward acceleration. Particles hold near the
/// origin over the few frames of the test.
fn expression_effect() -> EffectAsset {
    let mut m = Module::new();
    let ca = m.lit_vec3([1.0, 0.3, 0.1]);
    let cb = m.lit_vec3([0.2, 0.5, 1.0]);
    let r = m.rand();
    let sr = m.splat3(r);
    let diff = m.sub(cb, ca);
    let scaled = m.mul(diff, sr);
    let col = m.add(ca, scaled);
    let size = m.lit(1.5);
    let gravity = m.lit_vec3([0.0, 0.0, -1.0]);

    let mut program = EffectProgram::new()
        .with_rate(SpawnRate::PerSecond(6_000.0))
        .with_lifetime(2.0, 3.0);
    program.module = m;
    let program = program
        .set(Attribute::Colour, col)
        .set(Attribute::Size, size)
        .update(UpdateOp::Accelerate(gravity));

    EffectAsset::new("expr")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_program(program)
}

#[test]
fn generated_effect_emits_and_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, expression_effect());
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-expr-target"),
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
    let sum = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
    };
    let center = sum(SIZE / 2, SIZE / 2);
    let corner = sum(1, 1);

    assert!(center > 150, "center too dark; generated effect did not draw: {center}");
    assert!(
        center > corner + 100,
        "center {center} not clearly brighter than corner {corner}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-expr-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-expr-copy"),
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
