//! Phase 24: per-instance simulation.
//!
//! One registered effect drives several independent instances, each keyed by
//! `ParticleItem::with_key` and simulated at its own position. `two_keys_two_
//! clusters` submits two instances at separated x positions and asserts particles
//! draw in *both* screen regions (before Phase 24 only the first — the single
//! driver — simulated, so only one region lit). `same_key_collapses` submits two
//! items sharing a key and confirms they collapse to one instance (driven by the
//! first), lighting only one region.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, EffectId, Emitter, ParticleBlend, ParticleItem, ParticleItems, ParticlePlugin,
    SpawnRate, SpawnShape, VelocityDist,
};

const SIZE: u32 = 96;

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-per-instance-test"),
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

fn cloud_effect() -> EffectAsset {
    EffectAsset::new("cloud")
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
        })
}

/// Render the submission `build(id)` for several frames and return the count of
/// lit pixels in the left third and right third of the image.
fn lit_halves(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    build: impl Fn(EffectId) -> ParticleItems,
) -> (i32, i32) {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, cloud_effect());
    renderer.with_item_type_plugin(device, Box::new(plugin)).expect("register item type plugin");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-per-instance-target"),
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

    for _ in 0..16 {
        let mut frame = frame_looking_at_origin();
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, build(id));
        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(device, queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    let (mut left, mut right) = (0, 0);
    for y in 0..SIZE {
        for x in 0..SIZE {
            if luma(((y * SIZE + x) * 4) as usize) > corner + 60 {
                if x < SIZE / 3 {
                    left += 1;
                } else if x > 2 * SIZE / 3 {
                    right += 1;
                }
            }
        }
    }
    (left, right)
}

#[test]
fn two_keys_two_clusters() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (left, right) = lit_halves(&device, &queue, |id| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).with_key(0).at([-1.3, 0.0, 0.0]));
        items.push(ParticleItem::new(id).with_key(1).at([1.3, 0.0, 0.0]));
        items
    });
    assert!(
        left > 0 && right > 0,
        "two keyed instances should light both regions: left {left}, right {right}"
    );
}

#[test]
fn same_key_collapses() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    // Same key (default 0): the second item is ignored; only the first drives, so
    // only its region (left) lights.
    let (left, right) = lit_halves(&device, &queue, |id| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([-1.3, 0.0, 0.0]));
        items.push(ParticleItem::new(id).at([1.3, 0.0, 0.0]));
        items
    });
    assert!(left > 0, "the driving instance should light the left region");
    assert_eq!(
        right, 0,
        "a same-key duplicate must not simulate a second instance"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    // `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of
    // COPY_BYTES_PER_ROW_ALIGNMENT (256). SIZE*4 = 384 is not, so pad each row up
    // to the alignment for the copy and compact back to tight rows afterwards.
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = row_bytes.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-per-instance-readback"),
        size: (padded_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-per-instance-copy"),
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
                bytes_per_row: Some(padded_row),
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
    // Compact the padded rows back to tight `SIZE * 4`-byte rows.
    let mut tight = Vec::with_capacity((row_bytes * SIZE) as usize);
    for y in 0..SIZE {
        let start = (y * padded_row) as usize;
        tight.extend_from_slice(&mapped[start..start + row_bytes as usize]);
    }
    tight
}
