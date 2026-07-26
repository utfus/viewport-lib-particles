//! A parent effect's particles spawn a child effect's particles on the GPU.
//!
//! The child self-emits nothing (rate zero); every particle it draws must come
//! from a spawn event the parent appended. A parent of zero-size (invisible)
//! particles sits at the origin and, every step, appends events there; the child
//! spawns a bright additive glow from them. Comparing a run with the parent
//! present against one without isolates the event-driven spawn: only the parent
//! run lights the centre.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

use viewport_lib_particles::{
    EffectAsset, Emitter, EventCondition, ParticleBlend, ParticleItem, ParticleItems,
    ParticlePlugin, SpawnRate, SpawnShape, SubEmitter, VelocityDist,
};

const SIZE: u32 = 64; // 64 * 4 bytes = 256-byte rows, no padding.

fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::default_instance();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("particles-test"),
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

/// Render a few frames of the parent/child pair and read back the centre pixel
/// brightness. With `with_parent` false the parent instance is not submitted, so
/// the child receives no events.
fn center_brightness(device: &wgpu::Device, queue: &wgpu::Queue, with_parent: bool) -> i32 {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Child: draws a bright additive glow but self-emits nothing (rate zero), so
    // every particle must come from a parent event.
    let child = EffectAsset::new("sparks")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(0.0),
            lifetime: (2.0, 3.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 0.7, 0.3, 1.0],
            size: 1.5,
        });

    let mut plugin = ParticlePlugin::new();
    // Register the child first so the parent can name it.
    let child_id = plugin.add_effect(device, child);

    // Parent: zero-size (invisible) particles that live at the origin and append
    // spawn events there every step.
    let parent = EffectAsset::new("emitter")
        .with_capacity(256)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::Burst { count: 64 },
            lifetime: (10.0, 10.0),
            spawn: SpawnShape::Point,
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.0,
        })
        .with_sub_emitter(SubEmitter::new(child_id, EventCondition::EveryStep, 4));
    let parent_id = plugin.add_effect(device, parent);

    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-test-target"),
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

    for _ in 0..4 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        // The child is always submitted (so it draws); the parent only when the
        // scenario calls for it.
        items.push(ParticleItem::new(child_id).at([0.0, 0.0, 0.0]));
        if with_parent {
            items.push(ParticleItem::new(parent_id).at([0.0, 0.0, 0.0]));
        }
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

        let cmd = renderer.owned().render(device, queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(device, queue, &target);
    let i = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32
}

#[test]
fn parent_events_spawn_child_particles() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let with_parent = center_brightness(&device, &queue, true);
    let without_parent = center_brightness(&device, &queue, false);

    // With the parent, its events spawn a bright glow at the centre; without it,
    // the rate-zero child never spawns and the centre stays background-dark.
    assert!(
        with_parent > 150,
        "expected a glow from event-spawned sparks, got {with_parent}"
    );
    assert!(
        with_parent > without_parent + 100,
        "parent run {with_parent} not clearly brighter than child-only run {without_parent}"
    );
}

/// A three-level chain (shell -> sparks -> glitter) where the shell and sparks
/// also draw as ribbon trails. The sparks effect is at once a sub-emit child (of
/// the shell), a parent (of the glitter), and a trail effect, so this exercises
/// that those pipelines compose without a bind-group or validation error, and
/// that the depth-ordered passes still light the scene.
#[test]
fn three_level_trailed_chain_renders() {
    use viewport_lib_particles::ParticleRender;

    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();

    // Grandchild: a brief sparkle, spawned only from spark deaths.
    let glitter = plugin.add_effect(
        &device,
        EffectAsset::new("glitter")
            .with_capacity(10_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(0.0),
                lifetime: (0.2, 0.4),
                spawn: SpawnShape::Point,
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.6,
            }),
    );
    // Child + parent + trail: sparks hover near the origin (so they glow at the
    // centre) and break into glitter on death.
    let sparks = plugin.add_effect(
        &device,
        EffectAsset::new("sparks")
            .with_capacity(10_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(0.0),
                lifetime: (0.3, 0.5),
                spawn: SpawnShape::Point,
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [1.0, 0.8, 0.4, 1.0],
                size: 1.2,
            })
            .with_render(ParticleRender::Trail {
                width: 0.05,
                segments: 8,
            })
            .with_sub_emitter(SubEmitter::new(glitter, EventCondition::OnDeath, 4)),
    );
    // Parent + trail: a stream of short-lived shells at the origin that break
    // into sparks on death.
    let shell = plugin.add_effect(
        &device,
        EffectAsset::new("shell")
            .with_capacity(256)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(120.0),
                lifetime: (0.15, 0.2),
                spawn: SpawnShape::Point,
                velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
                colour: [0.5, 0.5, 0.5, 1.0],
                size: 0.2,
            })
            .with_render(ParticleRender::Trail {
                width: 0.05,
                segments: 8,
            })
            .with_sub_emitter(SubEmitter::new(sparks, EventCondition::OnDeath, 8)),
    );
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-test-target"),
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

    // Enough frames for shells to die into sparks (and sparks into glitter).
    for _ in 0..16 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 30.0);
        items.push(ParticleItem::new(shell).at([0.0, 0.0, 0.0]));
        items.push(ParticleItem::new(sparks).at([0.0, 0.0, 0.0]));
        items.push(ParticleItem::new(glitter).at([0.0, 0.0, 0.0]));
        frame.scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);

        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let i = (((SIZE / 2) * SIZE + (SIZE / 2)) * 4) as usize;
    let center = pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    // The shells break into hovering sparks, so the centre lights up. (The plain
    // check is that the trailed multi-level chain renders at all without error.)
    assert!(center > 80, "trailed three-level chain did not light the centre: {center}");
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-test-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-test-copy"),
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
