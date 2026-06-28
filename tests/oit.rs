//! OIT compositing: alpha-blended billboards route through the lib's OIT pass
//! (`paint_transparent`) instead of the main HDR draw, so they composite after
//! the skybox instead of being overwritten by it.
//!
//! This renders an `Alpha` effect over a solid-red skybox and reads back the
//! center. Before OIT routing, the skybox drew after the plugin's main paint
//! and hid the particles; the center would read as pure skybox. With the
//! particles in the OIT pass they show over the skybox, so the center carries
//! their (blue) contribution that the red skybox lacks.
//!
//! Skips cleanly when no GPU adapter is available.

use viewport_lib::Camera;
use viewport_lib::renderer::{
    EnvironmentMap, FrameData, RenderCamera, SurfaceSubmission, ViewportRenderer,
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
        label: Some("particles-oit-test"),
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
    // Render the uploaded environment as a visible skybox background.
    frame.effects.environment = Some(EnvironmentMap {
        show_skybox: true,
        ..Default::default()
    });
    frame
}

// A blue cloud so its contribution is separable from the red skybox. The blend
// selects the compositing route: Alpha goes through the OIT pass, Additive
// stays in the main draw.
fn blue_effect(blend: ParticleBlend) -> EffectAsset {
    EffectAsset::new("smoke")
        .with_capacity(4096)
        .with_blend(blend)
        .with_emitter(Emitter {
            rate: SpawnRate::PerSecond(3_000.0),
            lifetime: (1.0, 1.5),
            spawn: SpawnShape::Sphere {
                radius: 0.2,
                volume: true,
            },
            velocity: VelocityDist::UniformBox {
                min: [-0.4, -0.4, -0.4],
                max: [0.4, 0.4, 0.4],
            },
            colour: [0.2, 0.3, 1.0, 0.7],
            size: 0.25,
        })
}

struct Readout {
    center_blue: i32,
    corner_blue: i32,
}

fn render(blend: ParticleBlend) -> Option<Readout> {
    let (device, queue) = headless_device()?;

    let mut renderer = ViewportRenderer::new(&device, wgpu::TextureFormat::Rgba8UnormSrgb);

    // Solid red equirectangular environment (RGBA f32). Red so the corners of
    // the frame (skybox only) carry almost no blue.
    let (env_w, env_h) = (4u32, 2u32);
    let mut env = Vec::with_capacity((env_w * env_h * 4) as usize);
    for _ in 0..(env_w * env_h) {
        env.extend_from_slice(&[1.5, 0.02, 0.02, 1.0]);
    }
    renderer
        .upload_environment_map(&device, &queue, &env, env_w, env_h)
        .expect("upload env map");

    let mut plugin = ParticlePlugin::new();
    let id = plugin.add_effect(&device, blue_effect(blend));
    renderer.with_item_type_plugin(&device, Box::new(plugin));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("particles-oit-target"),
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

    for _ in 0..6 {
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
    let blue = |x: u32, y: u32| {
        let i = ((y * SIZE + x) * 4) as usize;
        pixels[i + 2] as i32
    };

    // Center: where the cloud sits. Corner: skybox only.
    let mut center_blue = 0;
    for y in (SIZE / 2 - 6)..(SIZE / 2 + 6) {
        for x in (SIZE / 2 - 6)..(SIZE / 2 + 6) {
            center_blue = center_blue.max(blue(x, y));
        }
    }
    let corner_blue = blue(1, 1)
        .max(blue(SIZE - 2, 1))
        .max(blue(1, SIZE - 2))
        .max(blue(SIZE - 2, SIZE - 2));

    Some(Readout {
        center_blue,
        corner_blue,
    })
}

#[test]
fn alpha_composites_over_skybox() {
    let Some(r) = render(ParticleBlend::Alpha) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // The blue cloud composites over the skybox in the OIT pass (drawn after the
    // skybox), so the center is clearly bluer than the skybox-only corners.
    assert!(
        r.center_blue > r.corner_blue + 40,
        "alpha particles did not composite over skybox: center_blue={}, corner_blue={}",
        r.center_blue,
        r.corner_blue
    );
}

#[test]
fn additive_composites_over_skybox() {
    let Some(r) = render(ParticleBlend::Additive) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };

    // Additive particles draw in the main pass, which now runs after the skybox,
    // so they add over the sky instead of being painted over by it. The center
    // is clearly bluer than the skybox-only corners.
    assert!(
        r.center_blue > r.corner_blue + 40,
        "additive particles did not composite over skybox: center_blue={}, corner_blue={}",
        r.center_blue,
        r.corner_blue
    );
}

fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture) -> Vec<u8> {
    let row_bytes = SIZE * 4;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particles-oit-readback"),
        size: (row_bytes * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("particles-oit-copy"),
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
