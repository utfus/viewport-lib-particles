//! Instance time-scale (issue 014), prewarm (issue 015), and idle GC (issue 013).
//!
//! All three are observed through the live particle-count stats handle (the same
//! non-stalling readback `tests/stats.rs` uses), driving the plugin through the
//! real renderer `prepare` path:
//!
//! - **Time scale (014):** a `time_scale = 0` instance never spawns (its `dt` is
//!   zeroed), while a `time_scale = 2` instance outpaces a `1.0` instance.
//! - **Prewarm (015):** an instance given a warmup is already near steady state on
//!   its first frame, where a cold instance has barely begun.
//! - **Idle GC (013):** an instance the host stops submitting is reclaimed after
//!   `IDLE_RETIRE_FRAMES`, and its stats entry disappears.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::Camera;

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
        label: Some("particles-lifetime-test"),
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

fn offscreen_target(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-lifetime-target"),
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

/// A modest steady emitter: low enough rate and long enough lifetime that, over
/// the few frames these tests run, live particles accumulate linearly and none
/// die — so a live count is a faithful proxy for accumulated simulated time.
fn steady_effect() -> EffectAsset {
    EffectAsset::new("steady")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(2_000.0),
            lifetime: (100.0, 100.0),
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        })
}

#[test]
fn time_scale_scales_and_freezes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, steady_effect());
    let stats = plugin.stats_handle();
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let view = offscreen_target(&device).create_view(&wgpu::TextureViewDescriptor::default());

    // Three co-located instances of one effect, distinct keys: normal speed,
    // frozen (0x), and double speed.
    const NORMAL: u64 = 1;
    const FROZEN: u64 = 2;
    const DOUBLE: u64 = 3;
    for _ in 0..14 {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new()
            .with_dt(1.0 / 60.0)
            .with_stats_readback(true);
        items.push(ParticleItem::new(id).with_key(NORMAL).with_time_scale(1.0));
        items.push(ParticleItem::new(id).with_key(FROZEN).with_time_scale(0.0));
        items.push(ParticleItem::new(id).with_key(DOUBLE).with_time_scale(2.0));
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        queue.submit(std::iter::once(
            renderer.owned().render(&device, &queue, &view, &frame),
        ));
        poll_wait(&device);
    }

    let normal = stats.instance_stats(id, NORMAL).expect("no normal stats");
    let frozen = stats.instance_stats(id, FROZEN).expect("no frozen stats");
    let double = stats.instance_stats(id, DOUBLE).expect("no double stats");

    assert!(
        normal.live_count > 0,
        "the 1.0x instance should be spawning ({} live)",
        normal.live_count
    );
    assert_eq!(
        frozen.live_count, 0,
        "a time_scale = 0 instance must never spawn (got {} live)",
        frozen.live_count
    );
    // 2x accumulates simulated time twice as fast, so it holds roughly double the
    // live particles. Use a loose lower bound to stay robust to spawn jitter.
    assert!(
        double.live_count as f32 > normal.live_count as f32 * 1.5,
        "2x instance ({}) should far outpace 1x ({})",
        double.live_count,
        normal.live_count
    );
}

#[test]
fn prewarm_starts_near_steady_state() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Emitter that would take ~2s of real simulation to fill; a warmup should
    // reach that population on the first frame.
    let effect = EffectAsset::new("ambient")
        .with_capacity(20_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (2.0, 2.0),
            spawn: SpawnShape::Sphere {
                radius: 0.15,
                volume: true,
            },
            velocity: VelocityDist::Fixed([0.0, 0.0, 0.0]),
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.1,
        });

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, effect);
    let stats = plugin.stats_handle();
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let view = offscreen_target(&device).create_view(&wgpu::TextureViewDescriptor::default());

    const WARM: u64 = 1;
    const COLD: u64 = 2;
    // A few frames so the readback lands, but with a tiny dt so the cold instance
    // barely advances — the gap on frame one is what the warmup created.
    for f in 0..4 {
        let dt = if f == 0 { 1.0 / 60.0 } else { 1.0e-4 };
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(dt).with_stats_readback(true);
        // WARM prewarms by two full lifetimes on its first activation.
        items.push(ParticleItem::new(id).with_key(WARM).with_warmup(2.0));
        items.push(ParticleItem::new(id).with_key(COLD));
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        queue.submit(std::iter::once(
            renderer.owned().render(&device, &queue, &view, &frame),
        ));
        poll_wait(&device);
    }

    let warm = stats.instance_stats(id, WARM).expect("no warm stats");
    let cold = stats.instance_stats(id, COLD).expect("no cold stats");
    assert!(
        warm.live_count > cold.live_count * 10,
        "prewarmed instance ({}) should start far fuller than the cold one ({})",
        warm.live_count,
        cold.live_count
    );
}

#[test]
fn idle_instance_is_reclaimed() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, steady_effect());
    let stats = plugin.stats_handle();
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let view = offscreen_target(&device).create_view(&wgpu::TextureViewDescriptor::default());

    // A keep-alive instance (key 0) submitted every frame keeps `prepare` running
    // (and the frame index advancing) so the GC sweep executes.
    const KEEP: u64 = 0;
    const TRANSIENT: u64 = 42;

    let render_frame = |renderer: &mut ViewportRenderer, submit_transient: bool| {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new()
            .with_dt(1.0 / 60.0)
            .with_stats_readback(true);
        items.push(ParticleItem::new(id).with_key(KEEP));
        if submit_transient {
            items.push(ParticleItem::new(id).with_key(TRANSIENT).at([0.5, 0.0, 0.0]));
        }
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        queue.submit(std::iter::once(
            renderer.owned().render(&device, &queue, &view, &frame),
        ));
        poll_wait(&device);
    };

    // Submit the temporary instance for several frames, until it reports a count.
    for _ in 0..8 {
        render_frame(&mut renderer, true);
    }
    let before = stats.instance_stats(id, TRANSIENT);
    assert!(
        before.is_some_and(|s| s.live_count > 0),
        "temporary instance should have a live count while submitted"
    );

    // Stop submitting it. After more than IDLE_RETIRE_FRAMES (120) frames without
    // submission, the idle GC reclaims it and clears its stats entry.
    for _ in 0..140 {
        render_frame(&mut renderer, false);
    }
    assert!(
        stats.instance_stats(id, TRANSIENT).is_none(),
        "idle instance should have been reclaimed and its stats cleared"
    );
    // The keep-alive instance, submitted every frame, must survive.
    assert!(
        stats
            .instance_stats(id, KEEP)
            .is_some_and(|s| s.live_count > 0),
        "continuously-submitted instance must not be reclaimed"
    );
}
