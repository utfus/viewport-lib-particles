//! A particle system is GPU-pickable through `render_pick`.
//!
//! Registers a dense effect at the origin with a known pick id, renders a few
//! frames to populate the particle buffer, then runs the renderer's GPU pick at
//! the screen center and asserts the returned hit carries that id. This exercises
//! the plugin's `render_pick` hook end to end through the lib's pick pass.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::Camera;
use viewport_lib::renderer::{
    FrameData, PickBackend, PickId, PickMask, RenderCamera, SurfaceSubmission, ViewportRenderer,
};
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
        label: Some("particles-pick-test"),
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

fn dense_effect() -> EffectAsset {
    EffectAsset::new("pickable")
        .with_capacity(8_000)
        .with_blend(ParticleBlend::Additive)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(6_000.0),
            lifetime: (1.5, 2.0),
            spawn: SpawnShape::Sphere {
                radius: 0.25,
                volume: true,
            },
            velocity: VelocityDist::UniformBox {
                min: [-0.2, -0.2, -0.2],
                max: [0.2, 0.2, 0.2],
            },
            colour: [1.0, 1.0, 1.0, 1.0],
            size: 0.25,
        })
}

#[test]
fn particle_system_is_pickable() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, dense_effect());
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-pick-target"),
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

    let make_frame = || {
        let mut frame = frame_looking_at_origin();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        let mut item = ParticleItem::new(id).at([0.0, 0.0, 0.0]);
        item.settings.pick_id = PickId(42);
        items.push(item);
        frame
            .scene
            .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
        frame
    };

    // Warm up the particle buffer, then keep the last frame for the pick query.
    let mut frame = make_frame();
    for _ in 0..5 {
        frame = make_frame();
        let cmd = renderer.owned().render(&device, &queue, &view, &frame);
        queue.submit(std::iter::once(cmd));
    }

    let hit = renderer.pick_object(
        PickBackend::Gpu,
        glam::Vec2::new(SIZE as f32 / 2.0, SIZE as f32 / 2.0),
        &frame,
        &device,
        &queue,
        PickMask::OBJECT,
    );

    let hit = hit.expect("center should hit the particle system");
    assert_eq!(hit.id, 42, "picked the wrong id: {}", hit.id);
}
