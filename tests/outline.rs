//! A selected particle system drives the outline mask.
//!
//! With `interaction.outline_selected` on and the item marked `selected`, the lib
//! runs its outline offscreen pass, which dispatches the plugin's `outline_mask`.
//! (This relies on the viewport-lib gate change that lets plugin selections
//! trigger the pass; before it, plugin-only selections were skipped.) The test
//! confirms the mask pipeline is compatible with the mask pass by rendering
//! cleanly with a selected system.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::Camera;
use viewport_lib::renderer::{
    FrameData, PickId, RenderCamera, SurfaceSubmission, ViewportRenderer,
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
        label: Some("particles-outline-test"),
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
    // Turn on the selection outline in a distinct red so it is separable from the
    // blue particles.
    frame.interaction.outline_selected = true;
    frame.interaction.outline_colour = [1.0, 0.0, 0.0, 1.0];
    frame.interaction.outline_width_px = 4.0;
    frame
}

/// Count strongly-red pixels (the outline colour), which the blue particles do
/// not produce.
fn red_pixels(pixels: &[u8]) -> u32 {
    let mut n = 0;
    for i in (0..pixels.len()).step_by(4) {
        let (r, g, b) = (pixels[i] as i32, pixels[i + 1] as i32, pixels[i + 2] as i32);
        if r > 140 && r - g > 60 && r - b > 60 {
            n += 1;
        }
    }
    n
}

#[test]
fn selected_system_runs_outline_mask() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    // A blue cloud, so it never produces the red outline colour.
    let id = plugin.add_effect(
        &device,
        EffectAsset::new("selected")
            .with_capacity(6_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(4_000.0),
                lifetime: (1.5, 2.0),
                spawn: SpawnShape::Sphere {
                    radius: 0.25,
                    volume: true,
                },
                velocity: VelocityDist::UniformBox {
                    min: [-0.15, -0.15, -0.15],
                    max: [0.15, 0.15, 0.15],
                },
                colour: [0.2, 0.4, 1.0, 1.0],
                size: 0.22,
            }),
    );
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-outline-target"),
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

    let mut render = |selected: bool| -> u32 {
        for _ in 0..4 {
            let mut frame = frame_looking_at_origin();
            let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
            let mut item = ParticleItem::new(id).at([0.0, 0.0, 0.0]);
            item.settings.pick_id = PickId(9);
            item.settings.selected = selected;
            items.push(item);
            frame
                .scene
                .submit_plugin_items(ParticlePlugin::TYPE_NAME, items);
            let cmd = renderer.owned().render(&device, &queue, &view, &frame);
            queue.submit(std::iter::once(cmd));
        }
        red_pixels(&read_back(&device, &queue, &target))
    };

    let unselected = render(false);
    let selected = render(true);

    // The red outline appears only when the system is selected.
    assert!(
        unselected < 4,
        "unexpected red without selection: {unselected}"
    );
    assert!(
        selected > 12,
        "no red outline drawn for the selected system: {selected}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-outline-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-outline-copy"),
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
