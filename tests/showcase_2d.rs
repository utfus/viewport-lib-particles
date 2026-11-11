//! The showcase's 2D page, rendered headless: every demo draws particles among
//! its sprites without a validation error.
//!
//! Skips cleanly when no GPU adapter is available.

#[path = "../examples/showcase/showcase_25_2d.rs"]
#[allow(dead_code)]
mod showcase_25_2d;

use showcase_25_2d::Demo;
use viewport_lib::renderer::{CameraFrame, FrameData, SceneFrame, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib_particles::ParticlePlugin;
use viewport_lib_plugins::item_types::sprite::SpriteItem;

const SIZE: u32 = 128;

#[test]
fn every_2d_demo_draws() {
    let instance = wgpu::default_instance();
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
    else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let ids = showcase_25_2d::register(&mut plugin, &device, &queue);
    renderer
        .with_item_type_plugin(&device, Box::new(plugin))
        .expect("register item type plugin");
    viewport_lib_plugins::item_types::install(&mut renderer, &device);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
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
    let size = [SIZE as f32, SIZE as f32];
    for demo in [
        Demo::PlaneLock,
        Demo::Blends,
        Demo::Painter,
        Demo::PixelArt,
        Demo::Hooks,
        Demo::Rain,
    ] {
        let mut state = showcase_25_2d::State::default();
        state.demo = demo;
        let mut draws = 0;
        for _ in 0..60 {
            let (items, sprites) = showcase_25_2d::scene(&ids, &mut state, 1.0 / 60.0);
            let mut scene = SceneFrame::from_surface_items(Vec::new());
            *scene.items_mut::<SpriteItem>() = sprites;
            scene.submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
            let frame = FrameData::new(
                CameraFrame::from_camera(&showcase_25_2d::camera(size[0], size[1]), size),
                scene,
            );
            let order = renderer.debug_painter_sequence(&frame, true);
            draws = draws.max(
                order
                    .iter()
                    .filter(|(name, ..)| *name == ParticlePlugin::TYPE_NAME)
                    .count(),
            );
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
        }
        // Painter demos sort with the sprites; the others draw in the 3D passes
        // over the painter ground, which is fine for a smoke test.
        if matches!(demo, Demo::Blends | Demo::Painter | Demo::Hooks | Demo::Rain) {
            assert!(draws > 0, "{} should put particles in the painter sequence", demo_name(demo));
        }
    }
}

fn demo_name(demo: Demo) -> &'static str {
    match demo {
        Demo::PlaneLock => "plane lock",
        Demo::Blends => "blends",
        Demo::Painter => "painter",
        Demo::PixelArt => "pixel art",
        Demo::Hooks => "hooks",
        Demo::Rain => "rain",
    }
}
