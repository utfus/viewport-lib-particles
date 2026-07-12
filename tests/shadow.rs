//! The shadow-cast pipeline encodes cleanly.
//!
//! Enables a shadow-casting directional light and renders an effect whose items
//! cast shadows (the default). This exercises `cast_shadow_pass` once per
//! cascade; the risky part is the shadow pipeline's group-0 layout matching the
//! lib's cascade light-matrix bind group. The test confirms the frame renders
//! (no validation error) and the particles still draw.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::renderer::{FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer};
use viewport_lib::wgpu;
use viewport_lib::{Camera, LightKind, LightSource, LightingSettings};

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
        label: Some("particles-shadow-test"),
        ..Default::default()
    }))
    .ok()?;
    Some((device, queue))
}

fn shadowed_frame() -> FrameData {
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

    let mut ls = LightingSettings::default();
    let mut light = LightSource::default();
    light.kind = LightKind::Directional {
        direction: [0.3, 0.4, 0.9],
    };
    ls.lights = vec![light];
    ls.shadows_enabled = true;
    ls.shadow_cascade_count = 2;
    frame.effects.lighting = ls;
    frame
}

#[test]
fn shadow_cast_encodes() {
    let Some((device, queue)) = headless_device() else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);
    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(
        &device,
        EffectAsset::new("casters")
            .with_capacity(6_000)
            .with_blend(ParticleBlend::Additive)
            .with_emitter(Emitter {
                rate: SpawnRate::PerSecond(4_000.0),
                lifetime: (1.5, 2.0),
                spawn: SpawnShape::Sphere {
                    radius: 0.3,
                    volume: true,
                },
                velocity: VelocityDist::UniformBox {
                    min: [-0.2, -0.2, -0.2],
                    max: [0.2, 0.2, 0.2],
                },
                colour: [1.0, 1.0, 1.0, 1.0],
                size: 0.3,
            }),
    );
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-shadow-target"),
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
        let mut frame = shadowed_frame();
        let mut items = ParticleItems::new().with_dt(1.0 / 60.0);
        // cast_shadows defaults to true on ItemSettings.
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
    let center = sum(SIZE / 2, SIZE / 2);
    assert!(
        center > 100,
        "particles did not draw with shadows on: {center}"
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-shadow-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-shadow-copy"),
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
