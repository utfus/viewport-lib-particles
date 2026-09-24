//! Phase 25 (transient bursts) and Phase 26 (culling + simulation budget).
//!
//! - `distance_cull_skips_emitter`: the same origin emitter draws with the cull
//!   distance off, and is skipped (dark) with a cull distance shorter than its
//!   distance to the camera.
//! - `budget_limits_instances`: three separated instances light three regions,
//!   but with `max_sim_instances = 1` only one is simulated.
//! - `burst_plays_then_retires`: a one-shot burst submitted a single frame lights
//!   up, then self-retires (goes dark) without being resubmitted.
//! - `a_plain_item_list_simulates_and_draws`: instances submitted as a bare
//!   `Vec<ParticleItem>` through the typed accessor draw like a `ParticleItems` does.
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
        label: Some("particles-lifecycle-test"),
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

fn cloud_effect(rate: SpawnRate, lifetime: (f32, f32)) -> EffectAsset {
    EffectAsset::new("cloud")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate,
            lifetime,
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        })
}

/// Render `total` frames, submitting `build(frame, id)` each, and count lit
/// pixels in the final image.
fn render_and_count(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    effect: EffectAsset,
    total: usize,
    build: impl Fn(usize, EffectId) -> ParticleItems,
) -> i32 {
    let mut renderer = ViewportRenderer::new(device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(device, effect);
    renderer.with_item_type_plugin(device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-lifecycle-target"),
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

    let pixels = read_back(device, queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    (0..(SIZE * SIZE) as usize)
        .filter(|p| luma(p * 4) > corner + 60)
        .count() as i32
}

#[test]
fn distance_cull_skips_emitter() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // No cull: the origin emitter draws.
    let lit = render_and_count(&device, &queue, cloud_effect(SpawnRate::PerSecond(3_000.0), (2.0, 2.5)), 16, |_, id| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    });
    assert!(lit > 0, "control (no cull) drew nothing");

    // Cull at 0.5 m: the camera is farther than that from the origin, so the
    // emitter is culled and nothing draws.
    let culled = render_and_count(&device, &queue, cloud_effect(SpawnRate::PerSecond(3_000.0), (2.0, 2.5)), 16, |_, id| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0).with_cull_distance(0.5);
        items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        items
    });
    assert_eq!(culled, 0, "distance-culled emitter should not draw ({culled} lit)");
}

#[test]
fn budget_limits_instances() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let three = |budget: Option<usize>| {
        move |_f: usize, id: EffectId| {
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            if let Some(b) = budget {
                items = items.with_max_sim_instances(b);
            }
            items.push(ParticleItem::new(id).with_key(0).at([-1.4, 0.0, 0.0]));
            items.push(ParticleItem::new(id).with_key(1).at([0.0, 0.0, 0.0]));
            items.push(ParticleItem::new(id).with_key(2).at([1.4, 0.0, 0.0]));
            items
        }
    };

    let all = render_and_count(&device, &queue, cloud_effect(SpawnRate::PerSecond(2_000.0), (2.0, 2.5)), 16, three(None));
    let one = render_and_count(&device, &queue, cloud_effect(SpawnRate::PerSecond(2_000.0), (2.0, 2.5)), 16, three(Some(1)));
    assert!(all > 0 && one > 0, "expected both runs to draw: all {all}, one {one}");
    assert!(
        one < all * 3 / 5,
        "budget of 1 should simulate far fewer particles than 3 instances: one {one}, all {all}"
    );
}

#[test]
fn burst_plays_then_retires() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // A burst submitted only on frame 0. It plays in a transient slot and must
    // self-retire once its ~0.4 s particles die, without being resubmitted.
    let burst = |_f: usize, id: EffectId| {
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        if _f == 0 {
            items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]).burst());
        }
        items
    };
    let effect = || cloud_effect(SpawnRate::Burst { count: 3_000 }, (0.3, 0.4));

    // ~5 frames in: particles are alive.
    let early = render_and_count(&device, &queue, effect(), 5, burst);
    assert!(early > 0, "burst should be visible shortly after firing ({early} lit)");

    // ~40 frames in (0.66 s > lifetime + margin): retired, nothing drawn.
    let late = render_and_count(&device, &queue, effect(), 40, burst);
    assert_eq!(late, 0, "burst should self-retire and go dark ({late} lit)");
}

#[test]
fn remove_effect_stops_drawing() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Control: the effect draws.
    let lit = render_and_count(
        &device,
        &queue,
        cloud_effect(SpawnRate::PerSecond(3_000.0), (2.0, 2.5)),
        6,
        |_, id| {
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
            items
        },
    );
    assert!(lit > 0, "control drew nothing");

    // Phase 27: a queued remove on frame 2 tombstones the effect and reclaims its
    // particles, so nothing draws thereafter even though the item is still sent.
    let removed = render_and_count(
        &device,
        &queue,
        cloud_effect(SpawnRate::PerSecond(3_000.0), (2.0, 2.5)),
        12,
        |f, id| {
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            if f == 2 {
                items.remove_effect(id);
            }
            items.push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
            items
        },
    );
    assert_eq!(removed, 0, "removed effect should not draw ({removed} lit)");
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    // `copy_texture_to_buffer` requires `bytes_per_row` to be a multiple of
    // COPY_BYTES_PER_ROW_ALIGNMENT (256). SIZE*4 = 384 is not, so pad each row up
    // to the alignment for the copy and compact back to tight rows afterwards.
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = row_bytes.div_ceil(align) * align;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-lifecycle-readback"),
        size: (padded_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-lifecycle-copy"),
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

#[test]
fn a_plain_item_list_simulates_and_draws() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // The terse submission form: a host with nothing to say about the frame's time step or
    // its culls pushes instances straight onto the typed collection and never names the
    // plugin's type string.
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(
        &device,
        cloud_effect(SpawnRate::PerSecond(3_000.0), (2.0, 2.5)),
    );
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-plain-target"),
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
            .items_mut::<ParticleItem>()
            .push(ParticleItem::new(id).at([0.0, 0.0, 0.0]));
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let pixels = read_back(&device, &queue, &target);
    let luma = |i: usize| pixels[i] as i32 + pixels[i + 1] as i32 + pixels[i + 2] as i32;
    let corner = luma(((SIZE + 1) * 4) as usize);
    let lit = (0..(SIZE * SIZE) as usize)
        .filter(|p| luma(p * 4) > corner + 60)
        .count();
    assert!(lit > 0, "the plain item list draws ({lit} lit)");
}
